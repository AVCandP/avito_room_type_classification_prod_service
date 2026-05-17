# Avito Room Type Classification: Production Service

Продуктовая сборка сервиса распознавания типа комнаты по фотографии.

## Что внутри

- Gradio UI для загрузки одного изображения.
- Локальный inference на 3-fold ConvNeXt Base ensemble.
- Веса модели в `models/weights`.
- Dockerfile и docker-compose для демонстрационного запуска.
- Финальный `submission.csv` и stacking-артефакты в `artifacts/stacking`.
- Документация проекта в `docs`.

## Запуск локально

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_prod_service
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:ROOM_MODEL_WEIGHTS_DIR="$PWD\models\weights"
python app\app.py
```

Открыть: http://localhost:7860

## Docker

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_prod_service
docker compose up --build
```

Открыть: http://localhost:7860

## Лучший результат

OOF Macro F1: `0.7565167133953731`.

## Итоговый submission

`artifacts/stacking/submission.csv`
