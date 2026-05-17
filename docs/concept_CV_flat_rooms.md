# Концепция решения: Система распознавания типа комнаты по изображению
## Кейс 3 — Авито | МИФИ Практический курс 2025–2026

---

## 1. Постановка задачи

**Бизнес-контекст.** Авито — крупнейшая платформа объявлений в мире (SimilarWeb): 230 млн активных объявлений, 72 млн пользователей в месяц, 2 млн новых объявлений в сутки. Для категории «Недвижимость» критически важна автоматическая классификация фотографий по типу комнаты — это улучшает поиск, повышает качество листингов и снижает ручной труд модераторов.

**Техническая задача.** Многоклассовая классификация изображений: по фото определить один из **19 типов комнаты/пространства**.

| ID | Класс |
|----|-------|
| 0 | Кухня / столовая |
| 1 | Кухня-гостиная |
| 2 | Универсальная комната |
| 3 | Гостиная |
| 4 | Спальня |
| 5 | Кабинет |
| 6 | Детская |
| 7 | Ванная комната |
| 8 | Туалет |
| 9 | Совмещённый санузел |
| 10 | Коридор / прихожая |
| 11 | Гардеробная / кладовая / постирочная |
| 12 | Балкон / лоджия |
| 13 | Вид из окна / с балкона |
| 14 | Дом снаружи / двор |
| 15 | Подъезд / лестничная площадка |
| 16 | Другое |
| 17 | Предметы интерьера / бытовая техника |
| 18 | Комната без мебели |

**Метрика оценки:** Macro F1-score — невзвешенное среднее F1 по всем 19 классам.

```python
from sklearn.metrics import f1_score
score = f1_score(y_true, y_pred, average='macro')
```

> **Ключевое следствие для стратегии:** Macro F1 — невзвешенное среднее, а не средневзвешенное. Это означает, что один слабо предсказываемый класс (например, F1=0.1) тянет общую метрику так же сильно, как один хорошо предсказываемый (F1=0.9). Улучшение редких и сложных классов даёт бо́льший прирост метрики, чем шлифовка частых классов. **Приоритет — классы с наименьшим F1 по confusion matrix.**

**Целевые пороги:**
- Macro F1 ≥ 0.60 → 20/20 баллов
- 0.50 ≤ F1 < 0.60 → 10/20 баллов
- 0.40 ≤ F1 < 0.50 → 5/20 баллов

---

## 2. Данные

### 2.1 Структура датасета

| Файл / Папка | Размер | Описание |
|---|---|---|
| `data/train_df.csv` | ~519 КБ | Обучающая выборка (~4 562 изображения, размечена на Толоке) |
| `data/val_df.csv` | ~56 КБ | Валидационная выборка (~500 изображений) |
| `data/test_df.csv` | ~7.4 МБ | Тестовая выборка (~48 903 изображения, без меток) |
| `data/room_type_sample_submission.csv` | — | Пример формата submission |
| `data/train_images/` | — | ~4 562 изображения для обучения |
| `data/val_images/` | — | ~500 изображений для валидации |
| `data/test_images/` | — | ~48 903 изображения для предсказания |
| `data/heuristics_images/` | — | ~49 983 дополнительных изображения |
| `data/heuristics_cabinet.csv` | ~29.7 МБ | Доп. данные для класса «кабинет» |
| `data/heuristics_detskaya.csv` | ~34.6 МБ | Доп. данные для класса «детская» |
| `data/heuristics_dressing_room.csv` | ~48.4 МБ | Доп. данные для класса «гардеробная» |

### 2.2 Схема полей

**train_df / val_df:**
```
item_id         — id объявления
image           — URL изображения
image_id_ext    — id изображения (ключ для submission)
result          — числовой id класса (0–18), цель предсказания
label           — текстовое название класса
ratio           — доля толокеров, выбравших данный ответ (0–1)
```

**test_df:**
```
item_id, image, image_id_ext  — идентификаторы
perform_top_microcat_name     — предсказание модели Avito Perform
perform_top_microcat_prob     — уверенность предсказания
perform_top_other_classes_value / prob — топ альтернативных предсказаний
```

**heuristics_*.csv (дополнительные поля):**
```
crop_coords, crop_area  — Smart-crop координаты и площадь
is_catalog              — признак каталожного изображения
n_texts                 — кол-во текстовых областей (Text Detector)
person_found            — наличие человека (Face Detector)
perform_*               — выходы Avito Perform
```

### 2.3 Особенности данных

- **Дисбаланс классов**: train содержит ~4 562 примеров на 19 классов — в среднем ~240 на класс, но распределение неравномерное. Редкие классы (кабинет=5, детская=6, гардеробная=11) имеют мало примеров → необходима балансировка. Из-за Macro F1 эти классы в приоритете.
- **Шумная разметка (ratio)**: часть train-примеров имеет неоднозначную разметку — толокеры не сошлись во мнении. При ratio < 0.6–0.7 метки ненадёжны. Простейшее решение — удалить такие примеры из обучения или снизить их вес.
- **Эвристические данные**: heuristics-файлы содержат дополнительные изображения для трёх слабых классов, подобранные по ключевым словам (microcat, текст объявления). Автоматическая разметка содержит много «мусора» — требуется фильтрация. Подход: использовать LLM (Claude, Гигачат) для анализа полей `title`, `description`, `microcat_name` и отсева нерелевантных примеров (см. Шаг 3б).
- **Ручной сбор данных**: команда договорилась дополнительно собрать по 100 изображений вручную для классов «кабинет» и «гардеробная» из любых источников (не обязательно Авито). Формат — CSV, аналогичный `train_df.csv` (см. Шаг 2а).
- **Perform-признаки в test_df**: выходы уже обученной модели Avito можно использовать как дополнительный признак (ансамбль / мета-классификатор).

---

## 3. Архитектура решения

### 3.1 Общая схема

```
[Изображение] 
    → [Предобработка + Аугментация]
    → [Backbone (pretrained CNN/ViT)]
    → [Head (FC + Dropout)]
    → [Логиты 19 классов]
    → [Softmax → Predicted class]
```

Дополнительно (опционально):
```
[test_df perform_* признаки]
    → [Meta-classifier / Ensemble blend]
    → [Итоговое предсказание]
```

### 3.2 Рекомендуемые архитектуры

| Архитектура | Библиотека | Обоснование |
|---|---|---|
| **EfficientNet-B4 / B5** | `timm` | Лучший баланс качество/скорость, хорошо работает на малых датасетах; **рекомендован для baseline** |
| **MaxViT-T / MaxViT-S** | `timm` | Гибрид CNN + ViT с multi-axis attention; высокая точность при умеренных вычислительных затратах; обсуждён на встрече 05-02 как кандидат для baseline |
| **ConvNeXt-Base / Large** | `timm` | Современный CNN, превосходит ViT на малых выборках |
| **ViT-Base/16** | `timm` | Vision Transformer, сильный с fine-tuning на достаточном данных |
| **Swin Transformer** | `timm` | Гибридный подход, хорошо для иерархических признаков |
| **CLIP (ViT-L/14)** | `transformers` | Мультимодальная модель, zero-shot + fine-tuning |

**Рекомендация для baseline:** `EfficientNet-B4` или `MaxViT-T` — быстрый цикл обучения, высокая стартовая точность.

```python
# Пример создания MaxViT через timm
import timm
model = timm.create_model('maxvit_tiny_tf_224', pretrained=True, num_classes=19)
```

### 3.3 Transfer Learning стратегия

1. **Заморозить backbone**, обучить только head (2–5 эпох) — "warm-up"
2. **Разморозить** последние N слоёв backbone + head — fine-tuning с малым lr
3. **Полный fine-tuning** при наличии времени и ресурсов

---

## 4. Пошаговый план реализации

### Шаг 1. Настройка окружения

```bash
# Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Windows

# Установить зависимости
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations pandas numpy scikit-learn matplotlib seaborn
pip install mlflow gradio jupyter opencv-python Pillow tqdm
```

Структура проекта:
```
room_type_classifier/
├── data/               ← ссылка на E:\Python\MIFI\prectice2curse\data\
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_baseline.ipynb    ← первый приоритет (срок: утро 3 мая)
│   ├── 03_experiments.ipynb
│   └── 04_inference.ipynb
├── src/                     ← рефакторинг из ноутбуков на втором этапе
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── configs/
│   └── config.yaml
├── app/
│   └── demo.py         ← Gradio/Streamlit интерфейс
├── data_extra/         ← ручные сборки CSV (кабинет, гардеробная)
├── requirements.txt
└── README.md
```

> **Workflow:** сначала всё реализуется в Jupyter Notebook для быстрых итераций. После получения работающего baseline ноутбуки рефакторятся в модульный `src/` проект.

**Создание Git репозитория (ответственный: AV):**

```bash
git init
git add .
git commit -m "initial project structure"
# Создать репозиторий на GitHub/GitLab и поделиться ссылкой с командой
git remote add origin <url>
git push -u origin main
```

---

### Шаг 2. Разведочный анализ данных (EDA)

**Файл:** `notebooks/01_EDA.ipynb`

**Что делаем:**
1. Загрузить `train_df.csv`, `val_df.csv` — проверить форму, типы, пропуски
2. Построить распределение классов (bar chart) — выявить дисбаланс
3. Визуализировать примеры изображений по каждому классу (grid 3×3)
4. Проанализировать `ratio` — сколько примеров с ratio < 0.7 (ненадёжная разметка)
5. Проверить пересечение `image_id_ext` между train/val/test (нет ли утечки)
6. Изучить `perform_*` признаки в test_df — насколько совпадают с реальными классами в val

```python
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('data/train_df.csv')
val_df = pd.read_csv('data/val_df.csv')

# Распределение классов
train_df['label'].value_counts().plot(kind='bar', figsize=(14, 5))
plt.title('Распределение классов в train')
plt.tight_layout()
plt.savefig('class_distribution.png')

# Примеры ненадёжной разметки
low_confidence = train_df[train_df['ratio'] < 0.7]
print(f"Примеров с ratio < 0.7: {len(low_confidence)} ({len(low_confidence)/len(train_df)*100:.1f}%)")
```

---

### Шаг 2а. Ручной сбор данных для проблемных классов

> **Решение с встречи 05-02:** команда вручную собирает дополнительные изображения, т.к. эвристики содержат много шума, а редкие классы критически тянут Macro F1.

**Задачи и ответственные:**

| Класс | ID | Кол-во | Ответственный | Срок |
|---|---|---|---|---|
| Кабинет | 5 | 100 изображений | Артем Власов (AV) | до 3 мая 2026 |
| Гардеробная / кладовая / постирочная | 11 | 100 изображений | Speaker 1 | до 3 мая 2026 |

**Требования к источникам:** любые открытые источники (Avito, Яндекс.Недвижимость, ЦИАН, Google Images и т.д.). Главное — валидная ссылка на изображение.

**Формат CSV** (аналогичен `train_df.csv`):

```
image_id_ext,item_id,image,result,label,ratio
manual_cabinet_001,,https://example.com/img1.jpg,5,кабинет,1.0
manual_cabinet_002,,https://example.com/img2.jpg,5,кабинет,1.0
```

Обязательные поля:
- `image_id_ext` — уникальный идентификатор (например, `manual_cabinet_001`)
- `image` — рабочая ссылка на изображение (проверить доступность)
- `result` — числовой класс (5 или 11)
- `label` — текстовое название класса
- `ratio` — ставить 1.0 (ручная разметка с уверенностью 100%)

> **Нерешённый вопрос (из встречи):** точный набор полей CSV для ручных данных требует согласования с командой. Поля `item_id` могут быть пустыми.

---

### Шаг 2б. LLM-фильтрация эвристических данных

> **Идея с встречи 05-02:** автоматически собранные эвристики содержат много нерелевантных изображений. Использовать LLM для анализа текстовых полей (`title`, `description`, `microcat_name`) и отсева мусора.

**Когда применять:** перед добавлением heuristics в обучающую выборку (Шаг 3).

**Подход: LLM-фильтрация через Claude API:**

```python
import anthropic
import pandas as pd

client = anthropic.Anthropic()

PROMPT_TEMPLATE = """Тебе дано объявление о недвижимости. 
Категория: {microcat_name}
Заголовок: {title}
Описание: {description}

Вопрос: относится ли это объявление к категории "{target_class}"?
Ответь только "да" или "нет".
"""

def llm_filter(row: pd.Series, target_class: str) -> bool:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # быстрая и дешёвая модель для фильтрации
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                microcat_name=row.get('microcat_name', ''),
                title=row.get('title', ''),
                description=str(row.get('description', ''))[:500],
                target_class=target_class
            )
        }]
    )
    answer = message.content[0].text.strip().lower()
    return answer.startswith('да')

# Пример: фильтрация кабинетов
heur_cabinet = pd.read_csv('data/heuristics_cabinet.csv')
mask = heur_cabinet.apply(lambda r: llm_filter(r, 'кабинет'), axis=1)
heur_cabinet_filtered = heur_cabinet[mask]
print(f"После фильтрации: {len(heur_cabinet_filtered)} / {len(heur_cabinet)}")
```

> **Замечание:** LLM-фильтрация — вычислительно дорогая операция на больших датасетах. Сначала применить быстрые правила (perform_prob > 0.7, n_texts < 3), затем LLM — только для оставшихся неоднозначных примеров.

**Альтернативный быстрый фильтр (без LLM):**

```python
# Фильтровать по уверенности Avito Perform + отсутствие текста/людей
def quick_filter(df, target_class_id, perform_thresh=0.7):
    return df[
        (df['perform_top_microcat_prob'] >= perform_thresh) &
        (df['n_texts'] < 5) &
        (df['person_found'] == False)
    ]
```

> **Нерешённый вопрос (из встречи):** нет назначенного ответственного и сроков для LLM-фильтрации. Требует отдельного согласования.

---

### Шаг 3. Подготовка данных и Dataset

**Файл:** `src/dataset.py`

**Требования:**
- Загрузка изображений по пути из `image_id_ext` + папки `train_images/`
- Применение аугментаций через `albumentations`
- Поддержка режимов `train` / `val` / `test`

**Ключевые аугментации (albumentations):**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.GaussNoise(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

**Балансировка классов (для DataLoader):**

```python
from torch.utils.data import WeightedRandomSampler

# Вычислить веса обратно пропорционально частоте класса
class_counts = train_df['result'].value_counts().sort_index().values
weights = 1.0 / class_counts
sample_weights = weights[train_df['result'].values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

**Добавление эвристических данных:**

```python
# Фильтровать только высококачественные эвристики
heur_cabinet = pd.read_csv('data/heuristics_cabinet.csv')
heur_detskaya = pd.read_csv('data/heuristics_detskaya.csv')
heur_dressing = pd.read_csv('data/heuristics_dressing_room.csv')

# Добавить колонку result с нужным классом
heur_cabinet['result'] = 5   # кабинет
heur_detskaya['result'] = 6  # детская
heur_dressing['result'] = 11 # гардеробная

# Ограничить до N примеров на класс для контроля шума
N = 500
heur_combined = pd.concat([
    heur_cabinet.sample(min(N, len(heur_cabinet))),
    heur_detskaya.sample(min(N, len(heur_detskaya))),
    heur_dressing.sample(min(N, len(heur_dressing))),
])

# Объединить с train
extended_train = pd.concat([train_df, heur_combined]).reset_index(drop=True)
```

---

### Шаг 4. Построение модели

**Файл:** `src/model.py`

```python
import timm
import torch
import torch.nn as nn

class RoomClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=19, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

---

### Шаг 5. Обучение модели

**Файл:** `src/train.py`

**Гиперпараметры (стартовые):**

| Параметр | Значение |
|---|---|
| `model_name` | `efficientnet_b4` |
| `img_size` | 224 |
| `batch_size` | 32 |
| `num_epochs` | 30 |
| `lr_head` | 1e-3 |
| `lr_backbone` | 1e-4 |
| `weight_decay` | 1e-4 |
| `scheduler` | CosineAnnealingLR |
| `loss` | CrossEntropyLoss + LabelSmoothing=0.1 |

**Работа с дисбалансом классов:**

```python
# Вариант 1: Взвешенный CrossEntropy
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Вариант 2: Focal Loss (если базовый CE не справляется)
# pip install focal-loss-torch
```

**Цикл обучения:**

```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
    
    # Валидация
    model.eval()
    with torch.no_grad():
        all_preds, all_labels = [], []
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch+1}: Macro F1 = {macro_f1:.4f}")
    
    # Сохранить лучшую модель
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), 'best_model.pth')
    
    scheduler.step()
```

**Логирование (MLflow / W&B):**

```python
import mlflow
mlflow.log_metric("macro_f1", macro_f1, step=epoch)
mlflow.log_metric("train_loss", avg_loss, step=epoch)
```

---

### Шаг 6. Эксперименты и улучшения

> **Порядок важен:** сначала зафиксировать baseline, проанализировать confusion matrix, затем целенаправленно улучшать слабые классы. Каждый эксперимент логируется в MLflow.

**Эксперимент 1 — Базовая линия** *(приоритет: срок 3 мая 2026)*
- Архитектура: `EfficientNet-B4` или `MaxViT-T` (через `timm`)
- Данные: только `train_df.csv`, без эвристик, без балансировки
- Цель: зафиксировать отправную точку Macro F1 и построить confusion matrix
- После обучения: выявить классы с наименьшим F1 → приоритизировать улучшения

```python
# После baseline — анализ проблемных классов
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
# Отсортировать классы по F1
per_class_f1 = {cls: report[cls]['f1-score'] for cls in CLASS_NAMES}
print(sorted(per_class_f1.items(), key=lambda x: x[1]))
```

**Эксперимент 2 — Очистка разметки**
- Удалить из train примеры с `ratio < 0.6` (шумная разметка)
- Ожидаемый эффект: +0.02–0.05 F1 за счёт снижения шума

**Эксперимент 3 — Балансировка классов**
- `WeightedRandomSampler` + взвешенный `CrossEntropyLoss`
- Ожидаемый прирост: +0.05–0.10 F1 (особенно для классов 5, 6, 11)

**Эксперимент 4 — Ручные данные + отфильтрованные эвристики**
- Добавить 100 ручных изображений для классов 5 и 11 (Шаг 2а)
- Добавить отфильтрованные heuristics для классов 5, 6, 11 (Шаг 2б)
- Отфильтровать примеры с `person_found=True` там, где люди не ожидаются

**Эксперимент 5 — Сравнение архитектур**
- Сравнить: `EfficientNet-B4` vs `MaxViT-T` vs `ConvNeXt-Base` vs `Swin-T`
- Зафиксировать val Macro F1 и время обучения для каждой
- Выбрать лучшую по соотношению качество/скорость

**Эксперимент 6 — TTA (Test Time Augmentation)**

```python
# При инференсе применить несколько аугментаций и усреднить вероятности
tta_transforms = [hflip_transform, crop_center, crop_5crops]
preds = torch.stack([model(aug(img)) for aug in tta_transforms]).mean(0)
```

**Эксперимент 7 — Ансамбль**

```python
# Усреднить вероятности от нескольких моделей (разные архитектуры или сиды)
ensemble_probs = (model1_probs + model2_probs + model3_probs) / 3
final_preds = ensemble_probs.argmax(dim=1)
```

**Эксперимент 8 — Мета-классификатор с Perform-признаками**
- Использовать `perform_*` из `test_df` как дополнительные признаки
- Обучить логистическую регрессию поверх вероятностей модели + perform признаков

---

### Шаг 7. Инференс и формирование submission

**Файл:** `src/inference.py`

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, ids in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            image_ids.extend(ids)
    
    return image_ids, predictions

# Загрузить лучшую модель
model.load_state_dict(torch.load('best_model.pth'))
image_ids, preds = predict(model, test_loader, device)

# Сформировать submission
submission = pd.DataFrame({
    'image_id_ext': image_ids,
    'Predicted': preds
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
# image_id_ext,Predicted
# 12345,11
# 12346,7
```

---

### Шаг 8. Интерпретация модели (Grad-CAM)

**Цель:** визуализировать, на какие области изображения опирается модель при предсказании.

```python
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.backbone.blocks[-1]]  # последний блок backbone
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=image_tensor)
visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
```

Сохранять примеры Grad-CAM для 3–5 изображений каждого класса.

---

### Шаг 9. Прототип интерфейса

**Файл:** `app/demo.py`

```python
import gradio as gr
import torch
from PIL import Image
from src.model import RoomClassifier

CLASS_NAMES = [
    "Кухня / столовая", "Кухня-гостиная", "Универсальная комната",
    "Гостиная", "Спальня", "Кабинет", "Детская", "Ванная комната",
    "Туалет", "Совмещённый санузел", "Коридор / прихожая",
    "Гардеробная / кладовая", "Балкон / лоджия", "Вид из окна",
    "Дом снаружи / двор", "Подъезд", "Другое",
    "Предметы интерьера", "Комната без мебели"
]

model = RoomClassifier()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

def classify_room(image):
    tensor = val_transforms(image=np.array(image))['image'].unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(19)}

gr.Interface(
    fn=classify_room,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Определение типа комнаты",
    description="Загрузите фото — модель определит тип комнаты"
).launch()
```

Запуск: `python app/demo.py`

---

### Шаг 10. Контейнеризация (Docker)

**Файл:** `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app/demo.py"]
```

```bash
docker build -t room-classifier .
docker run -p 7860:7860 room-classifier
```

---

## 5. Требования к финальному решению

### 5.1 Обязательные артефакты

| Артефакт | Описание |
|---|---|
| `submission.csv` | Предсказания на test_df в формате `image_id_ext,Predicted` |
| `best_model.pth` | Веса лучшей модели |
| `notebooks/01_EDA.ipynb` | Разведочный анализ данных |
| `notebooks/02_train.ipynb` | Обучение и эксперименты |
| `notebooks/03_inference.ipynb` | Инференс и формирование submission |
| `README.md` | Инструкция по запуску |
| Таблица экспериментов | Сравнение подходов (MLflow / W&B / Excel) |

### 5.2 Требования к коду

- Python 3.10+
- Фреймворк: PyTorch
- Зависимости зафиксированы в `requirements.txt`
- Код воспроизводим: зафиксированы `random.seed`, `torch.manual_seed`, `np.random.seed`
- Чёткое разделение `train` и `inference` скриптов
- Модели: open-source, лицензии MIT / Apache 2.0

### 5.3 Требования к документации

`README.md` должен содержать:
1. Описание задачи
2. Архитектура решения (схема / таблица)
3. Инструкция по установке (`pip install -r requirements.txt`)
4. Инструкция по обучению (`python src/train.py --config configs/config.yaml`)
5. Инструкция по инференсу (`python src/inference.py`)
6. Инструкция по запуску демо (`python app/demo.py`)
7. Итоговые метрики по экспериментам
8. Описание вклада каждого участника команды

---

## 6. Критерии качества и ориентиры

### 6.1 Баллы за модель (Macro F1)

| Macro F1 | Баллы |
|---|---|
| ≥ 0.60 | 20/20 |
| 0.50–0.60 | 10/20 |
| 0.40–0.50 | 5/20 |
| < 0.30 | 0/20 |

### 6.2 Типичные Macro F1 для этого типа задач

- **Baseline (ResNet-50, без балансировки):** ~0.40–0.45
- **EfficientNet-B4 + балансировка:** ~0.55–0.62
- **ConvNeXt + эвристики + TTA:** ~0.63–0.70
- **Ансамбль 3 моделей + мета-классификатор:** ~0.68–0.75

---

## 7. Риски и митигация

| Риск | Описание | Митигация |
|---|---|---|
| Дисбаланс классов | Классы 5, 6, 11 редкие; тянут Macro F1 вниз | WeightedSampler + ручные данные (Шаг 2а) + эвристики (Шаг 2б) + взвешенный loss |
| Шумная разметка (ratio < 0.7) | Модель обучается на неверных метках | Удалить примеры с ratio < 0.6 (Эксперимент 2) |
| Переобучение на малом train | 4 562 примера — мало для fine-tuning | Сильная аугментация + dropout + weight decay |
| Смешение похожих классов | Ванная / туалет / санузел похожи | Confusion matrix анализ после baseline, целенаправленные аугментации |
| Качество эвристик | Автоматическая разметка, возможны ошибки | Быстрый фильтр (perform_prob > 0.7) + LLM-фильтр для неоднозначных (Шаг 2б) |
| Нет формата ручных данных | Несогласованная структура CSV мешает объединению | Использовать шаблон из Шага 2а; согласовать поля с командой до начала сбора |
| Переход notebook → src | Код в ноутбуках сложно воспроизвести | После baseline провести рефакторинг в модульный `src/` проект |

---

## 8. Дедлайны и план работ

| Этап | Задача | Срок | Ответственный |
|---|---|---|---|
| 0 | Создать Git репозиторий и поделиться ссылкой | 2 мая 2026 | AV |
| 1 | EDA + baseline (EfficientNet-B4 или MaxViT) | **утро 3 мая 2026** | AV |
| 1б | Ручной сбор 100 изображений: кабинет | 3 мая 2026 | AV |
| 1в | Ручной сбор 100 изображений: гардеробная | 3 мая 2026 | Speaker 1 |
| — | **Созвон: обсуждение результатов baseline** | **3 мая 2026** | Вся команда |
| 2 | Балансировка + очистка разметки + ручные данные | ~5 мая 2026 | — |
| 3 | LLM-фильтрация эвристик (согласовать ответственного) | ~6 мая 2026 | TBD |
| 4 | Сравнение архитектур (EfficientNet / MaxViT / ConvNeXt) | ~8 мая 2026 | — |
| 5 | TTA + ансамблирование + мета-классификатор | ~11 мая 2026 | — |
| 6 | Рефакторинг ноутбуков в src/ (Docker опционально) | ~13 мая 2026 | — |
| 7 | Прототип интерфейса (Gradio) + Grad-CAM | ~14 мая 2026 | — |
| 8 | Документация (README) + финальный submission | ~16 мая 2026 | — |
| **Сдача** | Загрузка submission.csv + документа на платформу | **17 мая 2026, 23:59 МСК** | Тимлид |
| Защита | Онлайн-питч (8 мин) + Q&A (7 мин) | По расписанию | Вся команда |

---

## 9. Формат submission

Файл CSV с заголовком:

```
image_id_ext,Predicted
12345,11
12346,7
...
```

- `image_id_ext` — из `test_df.csv`
- `Predicted` — целое число от 0 до 18
- Файл должен содержать предсказание для **всех** строк `test_df`

---

## 10. Итоговая структура оценки

### Сдача (60 баллов)

| Критерий | Баллы |
|---|---|
| Все этапы реализованы (данные, модель, метрики, прототип) | 10 |
| Macro F1 ≥ 0.60 | 20 |
| Качественная работа с данными и эксперименты | 10 |
| Воспроизводимый, структурированный код | 10 |
| Обоснование архитектуры + интерпретация (Grad-CAM) | 5 |
| Документация и инструкция по запуску | 5 |

### Защита (40 баллов)

| Критерий | Баллы |
|---|---|
| Соответствие решения задаче заказчика | 10 |
| Практическая ценность и проработанность | 10 |
| Структура презентации (10–15 слайдов) | 10 |
| Ответы на вопросы | 10 |

---

## 11. Открытые вопросы (по итогам встречи 05-02)

Вопросы, не закрытые на встрече — требуют явного решения и фиксации ответственного:

| # | Вопрос | Текущий статус | Действие |
|---|---|---|---|
| 1 | Точный формат полей CSV для ручных данных (кабинет, гардеробная) | Не согласован | Согласовать на созвоне 3 мая; использовать шаблон из Шага 2а |
| 2 | LLM-фильтрация эвристик — назначить ответственного и срок | Нет исполнителя | Назначить на созвоне 3 мая; AV готов использовать подписку Claude |
| 3 | Переход от ноутбуков к структурированному `src/` проекту (Docker, Flask) | Отложен | Запланировать после baseline (~13 мая); Docker опционален |
