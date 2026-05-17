# Инструкция по миграции проекта в продуктовый сервис

Целевая директория:

```text
.\prectice2curse\avito_room_type_classification_prod_service
```

Цель миграции - собрать из исследовательских и демонстрационных частей проекта единую продуктовую папку, которую можно развернуть локально или в Docker и показать заказчику/экспертам как работающий сервис распознавания типа комнаты по фотографии.

Инструкция учитывает все актуальные проектные материалы:

- `FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md`;
- `avito_room_type_classification_const/PROJECT_DESCRIPTION.md`;
- `avito_room_type_classification_const/EXPERIMENT_REPORT.md`;
- `avito_room_type_classification_const/micro_interface/*`;
- `avito_solution_bundle/*`;
- `avito_room_type_classification_av/README.md`;
- `avito_room_type_classification_av/conception/concept_CV_flat_rooms.md`.

## 1. Что должно получиться

После миграции продуктовая папка должна иметь такую структуру:

```text
avito_room_type_classification_prod_service/
|-- app/
|   |-- app.py
|   |-- requirements.txt
|   |-- README.md
|   `-- UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md
|
|-- src/
|   |-- dataset.py
|   |-- inference_ensemble_3fold.py
|   `-- model.py
|
|-- scripts/
|   `-- stacking_convnext_3fold.py
|
|-- configs/
|   |-- exp_convnext_base_clean_lr1e4_fold1.yaml
|   |-- exp_convnext_base_clean_lr1e4_fold2.yaml
|   `-- exp_convnext_base_clean_lr1e4_fold3.yaml
|
|-- models/
|   `-- weights/
|       |-- fold1/best_model.pth
|       |-- fold2/best_model.pth
|       `-- fold3/best_model.pth
|
|-- artifacts/
|   |-- stacking/
|   |   |-- submission.csv
|   |   |-- oof_meta.csv
|   |   |-- meta_model.pkl
|   |   `-- meta_info.json
|   `-- reports/
|
|-- docs/
|   |-- FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md
|   |-- PROJECT_DESCRIPTION.md
|   |-- EXPERIMENT_REPORT.md
|   |-- BUNDLE_PROJECT_DESCRIPTION.md
|   |-- AV_PHASE1_README.md
|   `-- concept_CV_flat_rooms.md
|
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- requirements.txt
|-- README.md
`-- MIGRATION_CHECKLIST.md
```

Ключевая идея: продуктовый сервис не должен зависеть от соседних директорий `avito_solution_bundle` или `avito_room_type_classification_const`. Все нужное для демонстрационного запуска должно лежать внутри `avito_room_type_classification_prod_service`.

## 2. Источники файлов

### 2.1 Лучшие веса модели

Источник:

```text
.\prectice2curse\avito_solution_bundle\weights
```

Назначение:

```text
.\prectice2curse\avito_room_type_classification_prod_service\models\weights
```

Файлы:

```text
fold1/best_model.pth
fold2/best_model.pth
fold3/best_model.pth
```

Это лучшие доступные веса проекта: `3-fold ConvNeXt Base ensemble`, итоговый OOF Macro F1 около `0.7565`.

### 2.2 UI-сервис

Источник:

```text
.\prectice2curse\avito_room_type_classification_const\micro_interface
```

Назначение:

```text
.\prectice2curse\avito_room_type_classification_prod_service\app
```

Копировать:

```text
app.py
requirements.txt
README.md
UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md
```

Не копировать:

```text
venv/
__pycache__/
```

### 2.3 Docker-файлы

Источник:

```text
.\prectice2curse\avito_room_type_classification_const\micro_interface
```

Назначение:

```text
.\prectice2curse\avito_room_type_classification_prod_service
```

Копировать:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

После копирования `docker-compose.yml` нужно исправить volume с весами на локальный путь внутри продуктовой папки.

### 2.4 Inference bundle

Источник:

```text
.\prectice2curse\avito_solution_bundle
```

Назначение:

```text
.\prectice2curse\avito_room_type_classification_prod_service
```

Копировать:

```text
src/
scripts/
configs/
stacking/
PROJECT_DESCRIPTION.md
BUNDLE_README.txt
requirements.txt
```

Папку `stacking` лучше положить в `artifacts/stacking`, чтобы отделить код от результатов.

### 2.5 Документация проекта

Копировать в `docs/`:

```text
.\prectice2curse\FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md
.\prectice2curse\avito_room_type_classification_const\PROJECT_DESCRIPTION.md
.\prectice2curse\avito_room_type_classification_const\EXPERIMENT_REPORT.md
.\prectice2curse\avito_solution_bundle\PROJECT_DESCRIPTION.md
.\prectice2curse\avito_room_type_classification_av\README.md
.\prectice2curse\avito_room_type_classification_av\conception\concept_CV_flat_rooms.md
```

## 3. Подготовка перед миграцией

Открыть PowerShell:

```powershell
cd .\prectice2curse
```

Проверить наличие источников:

```powershell
Test-Path .\avito_room_type_classification_const\micro_interface\app.py
Test-Path .\avito_solution_bundle\weights\fold1\best_model.pth
Test-Path .\avito_solution_bundle\weights\fold2\best_model.pth
Test-Path .\avito_solution_bundle\weights\fold3\best_model.pth
Test-Path .\avito_solution_bundle\stacking\submission.csv
Test-Path .\FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md
```

Все команды должны вернуть `True`.

Если целевая папка уже содержит файлы, сделать резервную копию:

```powershell
$target = ".\prectice2curse\avito_room_type_classification_prod_service"
$backup = ".\prectice2curse\avito_room_type_classification_prod_service_backup_$(Get-Date -Format yyyyMMdd_HHmmss)"
if (Test-Path $target) {
    if ((Get-ChildItem $target -Force | Measure-Object).Count -gt 0) {
        Copy-Item $target $backup -Recurse -Force
        Write-Host "Backup created: $backup"
    }
}
```

## 4. Создание продуктовой структуры

```powershell
$root = ".\prectice2curse"
$prod = Join-Path $root "avito_room_type_classification_prod_service"

New-Item -ItemType Directory -Force $prod | Out-Null
New-Item -ItemType Directory -Force "$prod\app" | Out-Null
New-Item -ItemType Directory -Force "$prod\src" | Out-Null
New-Item -ItemType Directory -Force "$prod\scripts" | Out-Null
New-Item -ItemType Directory -Force "$prod\configs" | Out-Null
New-Item -ItemType Directory -Force "$prod\models\weights" | Out-Null
New-Item -ItemType Directory -Force "$prod\artifacts\stacking" | Out-Null
New-Item -ItemType Directory -Force "$prod\artifacts\reports" | Out-Null
New-Item -ItemType Directory -Force "$prod\docs" | Out-Null
```

## 5. Копирование файлов

### 5.1 UI

```powershell
$root = ".\prectice2curse"
$prod = Join-Path $root "avito_room_type_classification_prod_service"
$ui = Join-Path $root "avito_room_type_classification_const\micro_interface"

Copy-Item "$ui\app.py" "$prod\app\app.py" -Force
Copy-Item "$ui\requirements.txt" "$prod\app\requirements.txt" -Force
Copy-Item "$ui\README.md" "$prod\app\README.md" -Force
Copy-Item "$ui\UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md" "$prod\app\UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md" -Force

Copy-Item "$ui\Dockerfile" "$prod\Dockerfile" -Force
Copy-Item "$ui\docker-compose.yml" "$prod\docker-compose.yml" -Force
Copy-Item "$ui\.dockerignore" "$prod\.dockerignore" -Force
```

### 5.2 Inference-код и конфиги

```powershell
$bundle = Join-Path $root "avito_solution_bundle"

Copy-Item "$bundle\src\*" "$prod\src" -Recurse -Force
Copy-Item "$bundle\scripts\*" "$prod\scripts" -Recurse -Force
Copy-Item "$bundle\configs\*" "$prod\configs" -Recurse -Force
```

### 5.3 Веса модели

```powershell
Copy-Item "$bundle\weights\*" "$prod\models\weights" -Recurse -Force
```

Проверка:

```powershell
Get-ChildItem "$prod\models\weights" -Recurse -Filter best_model.pth | Select-Object FullName,Length
```

Должно быть три файла.

### 5.4 Stacking-артефакты

```powershell
Copy-Item "$bundle\stacking\*" "$prod\artifacts\stacking" -Recurse -Force
```

Проверка:

```powershell
Test-Path "$prod\artifacts\stacking\submission.csv"
Test-Path "$prod\artifacts\stacking\meta_model.pkl"
Test-Path "$prod\artifacts\stacking\meta_info.json"
```

### 5.5 Документация

```powershell
Copy-Item "$root\FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md" "$prod\docs\FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md" -Force
Copy-Item "$root\avito_room_type_classification_const\PROJECT_DESCRIPTION.md" "$prod\docs\PROJECT_DESCRIPTION.md" -Force
Copy-Item "$root\avito_room_type_classification_const\EXPERIMENT_REPORT.md" "$prod\docs\EXPERIMENT_REPORT.md" -Force
Copy-Item "$root\avito_solution_bundle\PROJECT_DESCRIPTION.md" "$prod\docs\BUNDLE_PROJECT_DESCRIPTION.md" -Force
Copy-Item "$root\avito_room_type_classification_av\README.md" "$prod\docs\AV_PHASE1_README.md" -Force
Copy-Item "$root\avito_room_type_classification_av\conception\concept_CV_flat_rooms.md" "$prod\docs\concept_CV_flat_rooms.md" -Force
```

## 6. Настройка путей после копирования

### 6.1 Главная настройка весов для UI

`app/app.py` поддерживает переменную окружения:

```text
ROOM_MODEL_WEIGHTS_DIR
```

В продуктовой папке нужно запускать сервис с:

```powershell
$env:ROOM_MODEL_WEIGHTS_DIR=".\prectice2curse\avito_room_type_classification_prod_service\models\weights"
python app\app.py
```

Это позволит не менять код `app.py`.

### 6.2 Исправление `docker-compose.yml`

После копирования старый `docker-compose.yml` может ссылаться на:

```yaml
../../avito_solution_bundle/weights:/models/weights:ro
```

Для продуктового сервиса нужно заменить volume на локальный:

```yaml
./models/weights:/models/weights:ro
```

PowerShell-команда для замены:

```powershell
(Get-Content "$prod\docker-compose.yml") `
  -replace '\.\./\.\./avito_solution_bundle/weights:/models/weights:ro', './models/weights:/models/weights:ro' |
  Set-Content "$prod\docker-compose.yml" -Encoding UTF8
```

Итоговый `docker-compose.yml` должен выглядеть так:

```yaml
services:
  room-classifier-ui:
    build: .
    container_name: avito-room-classifier-ui
    ports:
      - "7860:7860"
    environment:
      ROOM_MODEL_WEIGHTS_DIR: /models/weights
      FORCE_CPU: "1"
    volumes:
      - ./models/weights:/models/weights:ro
```

### 6.3 Исправление Dockerfile

Исходный Dockerfile из `micro_interface` ожидает `app.py` и `requirements.txt` в корне build context. В продуктовой структуре они лежат в `app/`, поэтому Dockerfile нужно заменить на продуктовый вариант:

```powershell
@'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV ROOM_MODEL_WEIGHTS_DIR=/models/weights

WORKDIR /app

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/app.py ./app.py

EXPOSE 7860

CMD ["python", "app.py"]
'@ | Set-Content "$prod\Dockerfile" -Encoding UTF8
```

### 6.4 Корневой `requirements.txt`

Для удобства скопировать UI-зависимости в корень:

```powershell
Copy-Item "$prod\app\requirements.txt" "$prod\requirements.txt" -Force
```

### 6.5 Исправление `.dockerignore`

Продуктовый `.dockerignore` должен исключать окружения, кэши и тяжелые временные артефакты, но не должен исключать `models/weights`, если веса подключаются volume. Рекомендуемый вариант:

```powershell
@'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
outputs/
*.log
artifacts/stacking/oof_meta.csv
artifacts/stacking/submission.csv
models/weights/
'@ | Set-Content "$prod\.dockerignore" -Encoding UTF8
```

Примечание: `models/weights/` исключается из Docker build context, потому что веса подключаются через volume. Это снижает размер Docker image.

### 6.6 Batch inference и конфиги

Если продуктовый сервис будет использовать batch inference из `src/inference_ensemble_3fold.py`, нужно синхронизировать пути к весам.

В исходном скрипте пути могли быть жестко заданы как:

```text
outputs/exp_convnext_base_clean_lr1e4_fold1/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold2/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold3/best_model.pth
```

В продуктовой структуре корректные пути:

```text
models/weights/fold1/best_model.pth
models/weights/fold2/best_model.pth
models/weights/fold3/best_model.pth
```

Для простого UI это исправление не требуется, потому что `app.py` берет веса из `ROOM_MODEL_WEIGHTS_DIR`. Для batch inference рекомендуется доработать `src/inference_ensemble_3fold.py`, чтобы он тоже читал переменную `ROOM_MODEL_WEIGHTS_DIR`.

Минимальная ручная правка в `src/inference_ensemble_3fold.py`:

```python
weights_dir = os.getenv("ROOM_MODEL_WEIGHTS_DIR", "models/weights")
fold_weight_paths = [
    os.path.join(weights_dir, "fold1", "best_model.pth"),
    os.path.join(weights_dir, "fold2", "best_model.pth"),
    os.path.join(weights_dir, "fold3", "best_model.pth"),
]
```

## 7. Создание продуктового README

Создать корневой `README.md`:

```powershell
@'
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
cd .\prectice2curse\avito_room_type_classification_prod_service
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:ROOM_MODEL_WEIGHTS_DIR=".\prectice2curse\avito_room_type_classification_prod_service\models\weights"
python app\app.py
```

Открыть:

```text
http://localhost:7860
```

## Docker

```powershell
cd .\prectice2curse\avito_room_type_classification_prod_service
docker compose up --build
```

Открыть:

```text
http://localhost:7860
```

## Лучший результат проекта

OOF Macro F1: `0.7565167133953731`.

## Главный итоговый артефакт

```text
artifacts/stacking/submission.csv
```
'@ | Set-Content "$prod\README.md" -Encoding UTF8
```

## 8. Создание checklist для приемки

```powershell
@'
# Migration Checklist

## Файлы

- [ ] Есть `app/app.py`
- [ ] Есть `models/weights/fold1/best_model.pth`
- [ ] Есть `models/weights/fold2/best_model.pth`
- [ ] Есть `models/weights/fold3/best_model.pth`
- [ ] Есть `artifacts/stacking/submission.csv`
- [ ] Есть `artifacts/stacking/meta_info.json`
- [ ] Есть `Dockerfile`
- [ ] Есть `docker-compose.yml`
- [ ] Есть `README.md`
- [ ] Есть документы в `docs/`

## Локальный запуск

- [ ] Создается виртуальное окружение
- [ ] Устанавливаются зависимости из `requirements.txt`
- [ ] `python app\app.py` стартует без ошибки
- [ ] UI открывается на `http://localhost:7860`
- [ ] Загружается изображение
- [ ] Возвращается топ-5 классов

## Docker

- [ ] `docker compose up --build` собирает образ
- [ ] Контейнер стартует
- [ ] UI открывается на `http://localhost:7860`
- [ ] Веса подключены через volume `./models/weights:/models/weights:ro`

## Демонстрация заказчику

- [ ] Подготовлены 3-5 тестовых фото помещений
- [ ] Показан одиночный inference через UI
- [ ] Показан итоговый `submission.csv`
- [ ] Показан отчет `docs/FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md`
'@ | Set-Content "$prod\MIGRATION_CHECKLIST.md" -Encoding UTF8
```

## 9. Проверка после миграции

### 9.1 Проверка структуры

```powershell
cd .\prectice2curse\avito_room_type_classification_prod_service

Test-Path .\app\app.py
Test-Path .\models\weights\fold1\best_model.pth
Test-Path .\models\weights\fold2\best_model.pth
Test-Path .\models\weights\fold3\best_model.pth
Test-Path .\artifacts\stacking\submission.csv
Test-Path .\Dockerfile
Test-Path .\docker-compose.yml
Test-Path .\README.md
```

Все значения должны быть `True`.

### 9.2 Проверка веса файлов

```powershell
Get-ChildItem .\models\weights -Recurse -Filter best_model.pth | Select-Object FullName,Length
```

Ожидаемо: три файла примерно по `350 МБ`.

### 9.3 Проверка Python-синтаксиса

```powershell
python -m py_compile .\app\app.py
```

### 9.4 Локальный запуск

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:ROOM_MODEL_WEIGHTS_DIR="$PWD\models\weights"
python app\app.py
```

Открыть:

```text
http://localhost:7860
```

### 9.5 Docker-запуск

```powershell
docker compose up --build
```

Открыть:

```text
http://localhost:7860
```

## 10. Что показывать заказчику и экспертам

Для демонстрации подготовить:

1. Открытый UI на `http://localhost:7860`.
2. Несколько фотографий помещений:
   - кухня;
   - спальня;
   - ванная;
   - коридор;
   - сложный класс, например кабинет или универсальная комната.
3. Показ топ-5 вероятностей и confidence.
4. Открытый файл:

```text
artifacts/stacking/submission.csv
```

5. Открытый отчет:

```text
docs/FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md
```

Ключевые тезисы для демонстрации:

- проект прошел две фазы развития;
- baseline достиг `Macro F1 = 0.6093`;
- лучший pipeline достиг `OOF Macro F1 = 0.7565`;
- использованы open-source модели и локальный inference;
- есть готовый Docker-запуск;
- сервис не отправляет фотографии во внешние API.

## 11. Важные продуктовые решения

### 11.1 Почему копируем веса внутрь prod service

Для продуктового релиза сервис не должен зависеть от соседней папки `avito_solution_bundle`. Поэтому веса переносятся в:

```text
models/weights
```

### 11.2 Почему artifacts отделены от app

`artifacts/stacking` хранит итоговые результаты проекта, но не участвует в одиночном UI inference. Это позволяет:

- показывать финальный submission;
- хранить meta-information;
- не смешивать runtime-код и отчетные данные.

### 11.3 Почему Docker не включает веса внутрь image

Веса занимают около `1 ГБ` суммарно. Лучше подключать их через volume:

```text
./models/weights:/models/weights:ro
```

Так образ остается легче, а веса можно обновлять без пересборки image.

### 11.4 Почему Gradio оставлен основным UI

Gradio быстрее всего демонстрирует ценность модели:

- загрузка изображения;
- кнопка предсказания;
- топ-5 классов;
- минимум инфраструктуры.

Для production API следующим шагом можно добавить FastAPI.

## 12. Рекомендуемое развитие после миграции

После успешной миграции в `prod_service` рекомендуется:

1. Добавить FastAPI backend:
   - `GET /health`;
   - `GET /classes`;
   - `POST /predict`;
   - `GET /metrics`.
2. Добавить отдельную папку `tests/`.
3. Добавить smoke-test для загрузки модели.
4. Добавить batch inference endpoint.
5. Добавить Grad-CAM или heatmap-интерпретацию.
6. Подключить MLflow или Weights & Biases для новых экспериментов.
7. Оформить Hugging Face model card, если публикация весов разрешена.

## 13. Однокомандный сценарий миграции

Ниже полный PowerShell-сценарий. Его можно сохранить как `migrate_to_prod.ps1` и выполнить из `.\prectice2curse`.

```powershell
$root = ".\prectice2curse"
$prod = Join-Path $root "avito_room_type_classification_prod_service"
$ui = Join-Path $root "avito_room_type_classification_const\micro_interface"
$bundle = Join-Path $root "avito_solution_bundle"

New-Item -ItemType Directory -Force $prod | Out-Null
New-Item -ItemType Directory -Force "$prod\app","$prod\src","$prod\scripts","$prod\configs","$prod\models\weights","$prod\artifacts\stacking","$prod\artifacts\reports","$prod\docs" | Out-Null

Copy-Item "$ui\app.py" "$prod\app\app.py" -Force
Copy-Item "$ui\requirements.txt" "$prod\app\requirements.txt" -Force
Copy-Item "$ui\README.md" "$prod\app\README.md" -Force
Copy-Item "$ui\UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md" "$prod\app\UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md" -Force
Copy-Item "$ui\.dockerignore" "$prod\.dockerignore" -Force

Copy-Item "$bundle\src\*" "$prod\src" -Recurse -Force
Copy-Item "$bundle\scripts\*" "$prod\scripts" -Recurse -Force
Copy-Item "$bundle\configs\*" "$prod\configs" -Recurse -Force
Copy-Item "$bundle\weights\*" "$prod\models\weights" -Recurse -Force
Copy-Item "$bundle\stacking\*" "$prod\artifacts\stacking" -Recurse -Force

Copy-Item "$root\FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md" "$prod\docs\FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md" -Force
Copy-Item "$root\avito_room_type_classification_const\PROJECT_DESCRIPTION.md" "$prod\docs\PROJECT_DESCRIPTION.md" -Force
Copy-Item "$root\avito_room_type_classification_const\EXPERIMENT_REPORT.md" "$prod\docs\EXPERIMENT_REPORT.md" -Force
Copy-Item "$root\avito_solution_bundle\PROJECT_DESCRIPTION.md" "$prod\docs\BUNDLE_PROJECT_DESCRIPTION.md" -Force
Copy-Item "$root\avito_room_type_classification_av\README.md" "$prod\docs\AV_PHASE1_README.md" -Force
Copy-Item "$root\avito_room_type_classification_av\conception\concept_CV_flat_rooms.md" "$prod\docs\concept_CV_flat_rooms.md" -Force

Copy-Item "$prod\app\requirements.txt" "$prod\requirements.txt" -Force

@'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV ROOM_MODEL_WEIGHTS_DIR=/models/weights

WORKDIR /app

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/app.py ./app.py

EXPOSE 7860

CMD ["python", "app.py"]
'@ | Set-Content "$prod\Dockerfile" -Encoding UTF8

@'
services:
  room-classifier-ui:
    build: .
    container_name: avito-room-classifier-ui
    ports:
      - "7860:7860"
    environment:
      ROOM_MODEL_WEIGHTS_DIR: /models/weights
      FORCE_CPU: "1"
    volumes:
      - ./models/weights:/models/weights:ro
'@ | Set-Content "$prod\docker-compose.yml" -Encoding UTF8

@'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
outputs/
*.log
artifacts/stacking/oof_meta.csv
artifacts/stacking/submission.csv
models/weights/
'@ | Set-Content "$prod\.dockerignore" -Encoding UTF8

@'
# Avito Room Type Classification: Production Service

Продуктовая сборка сервиса распознавания типа комнаты по фотографии.

## Запуск локально

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:ROOM_MODEL_WEIGHTS_DIR="$PWD\models\weights"
python app\app.py
```

## Docker

```powershell
docker compose up --build
```

Открыть: http://localhost:7860

## Лучший результат

OOF Macro F1: 0.7565167133953731

## Итоговый submission

`artifacts/stacking/submission.csv`
'@ | Set-Content "$prod\README.md" -Encoding UTF8

Write-Host "Migration completed: $prod"
```

## 14. Финальный критерий успешной миграции

Миграция считается успешной, если:

- продуктовая папка не зависит от соседних проектных директорий для UI-запуска;
- веса лежат в `models/weights`;
- `docker compose up --build` поднимает UI;
- `http://localhost:7860` открывается;
- пользователь может загрузить изображение и получить предсказание;
- итоговый `submission.csv` доступен в `artifacts/stacking`;
- отчетные документы доступны в `docs`.

