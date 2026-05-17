from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image


CLASS_NAMES_19 = [
    "0 - кухня / столовая",
    "1 - кухня-гостиная",
    "2 - универсальная комната",
    "3 - гостиная",
    "4 - спальня",
    "5 - кабинет",
    "6 - детская",
    "7 - ванная комната",
    "8 - туалет",
    "9 - совмещенный санузел",
    "10 - коридор / прихожая",
    "11 - гардеробная / кладовая / постирочная",
    "12 - балкон / лоджия",
    "13 - вид из окна / с балкона",
    "14 - дом снаружи / двор",
    "15 - подъезд / лестничная площадка",
    "16 - другое",
    "17 - предметы интерьера / бытовая техника",
    "18 - комната без мебели",
]


class RoomClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_base",
        num_classes: int = 20,
        pretrained: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def default_weights_dir() -> Path:
    current = Path(__file__).resolve()
    prod_root = current.parents[1]
    return prod_root / "models" / "weights"


def resolve_weight_paths() -> List[Path]:
    raw = os.getenv("ROOM_MODEL_WEIGHTS")
    if raw:
        return [Path(p.strip()).expanduser().resolve() for p in raw.split(";") if p.strip()]

    weights_dir = Path(os.getenv("ROOM_MODEL_WEIGHTS_DIR", str(default_weights_dir()))).expanduser().resolve()
    return [
        weights_dir / "fold1" / "best_model.pth",
        weights_dir / "fold2" / "best_model.pth",
        weights_dir / "fold3" / "best_model.pth",
    ]


def build_preprocess(img_size: int = 224) -> T.Compose:
    resize_to = int(img_size * 256 / 224)
    return T.Compose(
        [
            T.Resize((resize_to, resize_to)),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu")
PREPROCESS = build_preprocess()
MODELS: List[RoomClassifier] = []
LOAD_ERROR: str | None = None
WEIGHT_PATHS = resolve_weight_paths()


def load_models() -> None:
    global LOAD_ERROR
    if MODELS or LOAD_ERROR:
        return

    missing = [str(p) for p in WEIGHT_PATHS if not p.exists()]
    if missing:
        LOAD_ERROR = "Не найдены веса модели:\n" + "\n".join(missing)
        return

    try:
        for path in WEIGHT_PATHS:
            model = RoomClassifier(model_name="convnext_base", num_classes=20, pretrained=False, dropout=0.0)
            state = torch.load(str(path), map_location=DEVICE)
            model.load_state_dict(state)
            model.to(DEVICE)
            model.eval()
            MODELS.append(model)
    except Exception as exc:
        LOAD_ERROR = f"Ошибка загрузки модели: {type(exc).__name__}: {exc}"


def apply_tta(batch: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return batch
    if mode == "hflip":
        return torch.flip(batch, dims=[3])
    raise ValueError(f"Unknown TTA mode: {mode}")


def convert_20_to_19(probs20: torch.Tensor) -> torch.Tensor:
    if probs20.shape[-1] == 19:
        return probs20
    if probs20.shape[-1] != 20:
        raise ValueError(f"Unexpected number of classes: {probs20.shape[-1]}")

    probs19 = probs20[..., :19].clone()
    probs19[..., 18] = probs20[..., 18] + probs20[..., 19]
    probs19 = probs19 / probs19.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs19


@torch.no_grad()
def predict(image: Image.Image, use_flip_tta: bool = True) -> Tuple[Dict[str, float], str]:
    load_models()
    if LOAD_ERROR:
        return {}, LOAD_ERROR
    if image is None:
        return {}, "Загрузите изображение комнаты."

    started = time.perf_counter()
    image = image.convert("RGB")
    batch = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    tta_modes = ["none", "hflip"] if use_flip_tta else ["none"]

    logits_sum = None
    for model in MODELS:
        model_logits = None
        for mode in tta_modes:
            logits = model(apply_tta(batch, mode))
            model_logits = logits if model_logits is None else model_logits + logits
        model_logits = model_logits / len(tta_modes)
        logits_sum = model_logits if logits_sum is None else logits_sum + model_logits

    probs20 = torch.softmax(logits_sum / len(MODELS), dim=1)[0]
    probs19 = convert_20_to_19(probs20).cpu()
    scores = {CLASS_NAMES_19[i]: float(probs19[i]) for i in range(len(CLASS_NAMES_19))}

    top_idx = int(torch.argmax(probs19).item())
    latency_ms = (time.perf_counter() - started) * 1000
    details = (
        f"Предсказание: {CLASS_NAMES_19[top_idx]}\n"
        f"Уверенность: {float(probs19[top_idx]):.3f}\n"
        f"Модель: 3-fold ConvNeXt Base ensemble\n"
        f"TTA: {', '.join(tta_modes)}\n"
        f"Устройство: {DEVICE}\n"
        f"Время обработки: {latency_ms:.0f} мс\n"
        f"Веса: {', '.join(str(p) for p in WEIGHT_PATHS)}"
    )
    return scores, details


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Room Type Classifier") as demo:
        gr.Markdown(
            """
            # Распознавание типа комнаты по фото

            Загрузите фотографию помещения. Сервис применит лучшую модель проекта:
            ансамбль трех `ConvNeXt Base` fold-моделей с локальным инференсом.
            """
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Фото комнаты")
            with gr.Column():
                use_flip_tta = gr.Checkbox(value=True, label="Использовать horizontal flip TTA")
                button = gr.Button("Определить тип комнаты", variant="primary")
                label = gr.Label(label="Топ классов", num_top_classes=5)
                details = gr.Textbox(label="Детали инференса", lines=8)

        gr.Markdown(
            """
            Модель работает локально. Загруженные изображения не отправляются во внешние сервисы.
            """
        )
        button.click(fn=predict, inputs=[image, use_flip_tta], outputs=[label, details])
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
