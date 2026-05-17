# Описание проекта Avito Room Type Classification

## Что делает проект

Проект решает задачу классификации фотографий недвижимости по типу помещения. На вход подается изображение из объявления, а модель должна определить один из 20 классов: кухня, кухня-гостиная, универсальная комната, гостиная, спальня, кабинет, детская, ванная, туалет, санузел, коридор/прихожая, гардеробная, балкон/лоджия, вид из окна, дом снаружи, подъезд, другое, предметы интерьера, без мебели и дополнительный класс `19`.

Проект нужен для автоматической разметки и проверки изображений в объявлениях недвижимости. Такой классификатор может использоваться для улучшения качества карточек объявлений, фильтрации фотографий, аналитики по объектам недвижимости и подготовки сабмита для соревнования/практического курса.

Основная метрика качества - `Macro F1`. Она усредняет F1-score по классам без учета их размера, поэтому проект оптимизируется не только под частые классы, но и под редкие/сложные категории.

## Какие результаты дает проект

Результатом работы являются:

- обученные checkpoint-файлы моделей `best_model.pth` в папках `outputs/<experiment_name>/`;
- отчеты по обучению: `history.csv` и `classification_report.txt`;
- графики и статистика EDA в `outputs/eda/`;
- итоговые файлы предсказаний `submission.csv` в формате:

```csv
image_id_ext,Predicted
12345,11
12346,7
```

В текущей директории уже лежат готовые финальные артефакты:

- `outputs/ensemble_convnext_3fold_tta/submission.csv` - сабмит 3-fold ConvNeXt ensemble с TTA;
- `outputs/ensemble_convnext_3fold_stacking_tta/submission.csv` - сабмит stacking-ансамбля;
- `outputs/ensemble_convnext_3fold_stacking_tta/oof_meta.csv` - out-of-fold вероятности для meta-model;
- `outputs/ensemble_convnext_3fold_stacking_tta/meta_model.pkl` - обученная LogisticRegression meta-model;
- `outputs/ensemble_convnext_3fold_stacking_tta/meta_info.json` - параметры stacking-запуска.

По отчетам проекта достигнуты следующие ориентировочные offline-результаты:

- baseline EfficientNet-B0 после исправления путей: `Macro F1 ~= 0.5999`;
- лучший single-model ConvNeXt на validation: `Macro F1 ~= 0.7228`;
- 3-fold ConvNeXt CV: среднее `Macro F1 ~= 0.7559`, std `0.0135`;
- OOF stacking поверх 3 fold-моделей: `Macro F1 ~= 0.7565`;
- финальный тестовый артефакт содержит около `48003` строк предсказаний.

Важно: эти значения являются offline-валидацией по локальным разбиениям. Результат на public/private leaderboard может отличаться.

## Реализованный функционал

### 1. Подготовка и очистка данных

Скрипты в `scripts/` собирают расширенные train-наборы:

- `build_full_train.py` - объединяет базовый train с внешними CSV, нормализует колонки, скачивает внешние изображения, удаляет дубли;
- `build_full_train_clean.py` - фильтрует внешние данные по доменам, убирая источники с сильным domain shift, и ограничивает прирост примеров по классам;
- `build_full_train_class_focus.py` - делает более строгую очистку для сложных классов;
- `run_convnext_3fold.py` - строит stratified 3-fold split и запускает обучение ConvNeXt по фолдам.

Внешние данные очищаются, потому что сырые изображения из каталогов/e-commerce ухудшали качество из-за отличия от реальных фото объявлений.

### 2. Dataset и аугментации

Файл `src/dataset.py` содержит:

- `RoomDataset` для train/val/test-режимов;
- загрузку изображений по `image_id_ext`;
- fallback на черную картинку при ошибке чтения;
- опциональный preload изображений в RAM;
- train-аугментации через Albumentations: RandomResizedCrop, HorizontalFlip, ColorJitter, GaussNoise, Rotate, Normalize;
- validation/test-преобразования: Resize, CenterCrop, Normalize;
- weighted sampler для борьбы с дисбалансом классов.

### 3. Модель

Файл `src/model.py` реализует универсальный классификатор `RoomClassifier` на базе `timm`:

- backbone создается через `timm.create_model`;
- поддерживаются разные архитектуры, например `efficientnet_b0`, `convnext_base`, `maxvit`, `eva02`;
- голова модели: `Dropout + Linear`;
- число классов по конфигам - `20`.

### 4. Обучение

Файл `src/train.py` реализует полный train-loop:

- чтение YAML-конфига;
- выбор `cuda`, если доступна GPU, иначе `cpu`;
- загрузка train/val CSV и изображений;
- фильтрация по `ratio`;
- CrossEntropyLoss с label smoothing;
- опциональный weighted loss;
- AdamW optimizer;
- warmup + cosine learning rate scheduler;
- early stopping;
- расчет `Macro F1`;
- сохранение лучшей модели;
- сохранение `history.csv` и `classification_report.txt`.

Быстрый запуск baseline вынесен в `run_baseline.py`.

### 5. Инференс

В проекте есть два основных режима инференса:

- `src/inference.py` - инференс одной модели по checkpoint-файлу;
- `src/inference_ensemble_3fold.py` - инференс 3 fold-моделей ConvNeXt с усреднением logits и TTA.

Поддерживаемые TTA-режимы:

- `none` - исходное изображение;
- `hflip` - горизонтальный flip;
- `scale_up` - небольшой zoom-in;
- `scale_down` - небольшой zoom-out.

### 6. Stacking ensemble

Скрипт `scripts/stacking_convnext_3fold.py` реализует meta-ensemble:

- собирает OOF-вероятности с fold-валидаций;
- обучает `LogisticRegression` как meta-model;
- усредняет test-вероятности fold-моделей с TTA;
- применяет meta-model;
- сохраняет `submission.csv`, `oof_meta.csv`, `meta_model.pkl`, `meta_info.json`.

### 7. EDA и сравнение экспериментов

- `src/eda.py` строит распределение классов, распределение `ratio`, примеры изображений по классам и summary-статистику;
- `src/compare_experiments.py` собирает результаты из `outputs/*/history.csv`, строит сводную таблицу и learning curves.

## Структура проекта

```text
avito_room_type_classification_const/
|-- configs/                         # YAML-конфиги экспериментов
|   |-- baseline.yaml                 # baseline EfficientNet-B0
|   |-- exp2_ratio_filter.yaml        # эксперимент с фильтрацией по ratio
|   |-- exp3_weighted_loss.yaml       # weighted loss
|   |-- exp4_maxvit.yaml              # MaxViT-эксперимент
|   |-- exp_convnext_base_*.yaml      # ConvNeXt-эксперименты
|   |-- exp_eva02_small_clean.yaml    # EVA02-эксперимент
|   `-- exp_full_train*.yaml          # эксперименты с расширенными данными
|
|-- scripts/                          # вспомогательные пайплайны
|   |-- build_full_train.py
|   |-- build_full_train_clean.py
|   |-- build_full_train_class_focus.py
|   |-- run_convnext_3fold.py
|   `-- stacking_convnext_3fold.py
|
|-- src/                              # основной код
|   |-- dataset.py                    # Dataset, transforms, sampler
|   |-- model.py                      # timm-based classifier
|   |-- train.py                      # обучение
|   |-- inference.py                  # инференс одной модели
|   |-- inference_ensemble_3fold.py   # 3-fold ensemble inference
|   |-- eda.py                        # анализ данных
|   |-- compare_experiments.py        # сравнение экспериментов
|   `-- __init__.py
|
|-- outputs/                          # готовые результаты и сабмиты
|   |-- ensemble_convnext_3fold_tta/
|   `-- ensemble_convnext_3fold_stacking_tta/
|-- micro_interface/                  # Gradio UI для распознавания комнаты по одному фото
|
|-- venv/                             # локальное виртуальное окружение
|-- README.md                         # краткий отчет проекта
|-- EXPERIMENT_REPORT.md              # подробный handover-отчет
|-- requirements.txt                  # зависимости
|-- run_baseline.py                   # запуск baseline
`-- PROJECT_DESCRIPTION.md            # этот документ
```

## Важные замечания по данным и путям

В текущей папке проекта нет полных датасетов и checkpoint-файлов fold-моделей, кроме готовых CSV-результатов в `outputs/`. Для полного переобучения и повторного инференса нужны исходные CSV, изображения и веса моделей.

Конфиги используют относительные пути вроде:

- `../train_df.csv`;
- `../val_df.csv`;
- `../test_df.csv`;
- `../train_images/train_images`;
- `../val_images/val_images`;
- `../test_images/test_images`;
- `../train_images_full_clean/train_images_full_clean`.

Это означает, что при запуске из корня проекта данные должны находиться уровнем выше директории `avito_room_type_classification_const`, либо пути в YAML-конфигах нужно поправить под фактическое расположение данных.

Также часть скриптов подготовки данных исторически ссылается на папку `baseline/data` и на родительскую директорию через `Path(__file__).resolve().parents[2]`. Если проект запускается именно из `avito_room_type_classification_const`, перед использованием этих скриптов нужно проверить и при необходимости заменить в них пути `ROOT / "baseline"` на текущий корень проекта.

## Проверка переносимого пакета `avito_solution_bundle`

Дополнительно проверена соседняя директория:

```text
.\prectice2curse\avito_solution_bundle
```

Это не тренировочный проект, а переносимый пакет для инференса и хранения финального решения. Он содержит код модели, конфиги трех фолдов, веса обученных моделей и готовые stacking-артефакты.

Фактическая структура bundle:

```text
avito_solution_bundle/
|-- configs/
|   |-- exp_convnext_base_clean_lr1e4_fold1.yaml
|   |-- exp_convnext_base_clean_lr1e4_fold2.yaml
|   `-- exp_convnext_base_clean_lr1e4_fold3.yaml
|-- src/
|   |-- dataset.py
|   |-- model.py
|   `-- inference_ensemble_3fold.py
|-- scripts/
|   `-- stacking_convnext_3fold.py
|-- weights/
|   |-- fold1/best_model.pth
|   |-- fold2/best_model.pth
|   `-- fold3/best_model.pth
|-- stacking/
|   |-- submission.csv
|   |-- oof_meta.csv
|   |-- meta_model.pkl
|   `-- meta_info.json
|-- BUNDLE_README.txt
|-- PROJECT_DESCRIPTION.md
`-- requirements.txt
```

Что уже есть в bundle:

- три checkpoint-файла `ConvNeXt Base`: `weights/fold1/best_model.pth`, `weights/fold2/best_model.pth`, `weights/fold3/best_model.pth`;
- каждый checkpoint занимает примерно `350 МБ`;
- готовый итоговый файл `stacking/submission.csv` на `48003` строки;
- готовая stacking-модель `stacking/meta_model.pkl`;
- OOF-таблица `stacking/oof_meta.csv`;
- параметры запуска `stacking/meta_info.json`, где указано `oof_macro_f1_meta = 0.7565167133953731`.

Что отсутствует в bundle:

- `test_df.csv`;
- папка `test_images/test_images`;
- train/val split-файлы для повторной сборки OOF;
- папка `outputs/exp_convnext_base_clean_lr1e4_fold*/best_model.pth`.

Поэтому готовый файл `stacking/submission.csv` можно использовать сразу, но новый инференс требует добавить тестовый датасет и поправить пути к весам.

### Состояние inference-скриптов в bundle

В `src/inference_ensemble_3fold.py` реализован простой 3-fold ensemble inference:

- читает `test_csv` и `test_images_dir` из YAML-конфига;
- загружает 3 модели `convnext_base`;
- применяет TTA-режимы `none`, `hflip`, `scale_up`, `scale_down`;
- усредняет logits по TTA и фолдам;
- сохраняет `submission.csv`.

Но в этом скрипте пути к весам заданы жестко:

```text
outputs/exp_convnext_base_clean_lr1e4_fold1/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold2/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold3/best_model.pth
```

В bundle таких файлов нет: веса лежат в `weights/fold1..3`. Поэтому перед запуском простого ensemble inference нужно выбрать один из вариантов:

1. Создать ожидаемую структуру `outputs/...` и скопировать туда веса.
2. Или заменить список `fold_weight_paths` в `src/inference_ensemble_3fold.py` на:

```python
fold_weight_paths = [
    "weights/fold1/best_model.pth",
    "weights/fold2/best_model.pth",
    "weights/fold3/best_model.pth",
]
```

В `scripts/stacking_convnext_3fold.py` реализован полный stacking-пайплайн:

- загружает fold-конфиги;
- берет путь к весам из `output_dir` каждого конфига;
- собирает OOF-вероятности на validation split каждого фолда;
- обучает `LogisticRegression`;
- делает test inference с TTA;
- сохраняет `submission.csv`, `oof_meta.csv`, `meta_model.pkl`, `meta_info.json`.

Здесь есть похожее ограничение: в `configs/*.yaml` поле `output_dir` указывает на `outputs/exp_convnext_base_clean_lr1e4_fold*`, а фактические веса в bundle лежат в `weights/fold*`. Для запуска stacking из bundle нужно либо исправить `output_dir` в YAML:

```yaml
output_dir: weights/fold1
```

для первого фолда, аналогично `weights/fold2` и `weights/fold3`, либо разложить веса в ожидаемые папки `outputs/exp_convnext_base_clean_lr1e4_fold*/`.

### Рекомендуемый запуск готового bundle

Если нужно просто получить уже готовый результат, используйте:

```text
.\prectice2curse\avito_solution_bundle\stacking\submission.csv
```

Если нужно заново сделать инференс по тестовым изображениям:

1. Перейти в bundle:

```powershell
cd .\prectice2curse\avito_solution_bundle
```

2. Активировать окружение или создать новое:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

3. Установить зависимости с подходящим PyTorch под CPU или CUDA, затем:

```powershell
pip install -r requirements.txt
```

4. Добавить данные. По текущим конфигам они ожидаются на уровень выше bundle:

```text
.\prectice2curse\test_df.csv
.\prectice2curse\test_images\test_images\*.jpg
```

Если данные лежат в другом месте, нужно поправить `test_csv` и `test_images_dir` в `configs/exp_convnext_base_clean_lr1e4_fold*.yaml`.

5. Поправить пути к весам одним из способов:

```powershell
New-Item -ItemType Directory -Force outputs\exp_convnext_base_clean_lr1e4_fold1,outputs\exp_convnext_base_clean_lr1e4_fold2,outputs\exp_convnext_base_clean_lr1e4_fold3
Copy-Item weights\fold1\best_model.pth outputs\exp_convnext_base_clean_lr1e4_fold1\best_model.pth
Copy-Item weights\fold2\best_model.pth outputs\exp_convnext_base_clean_lr1e4_fold2\best_model.pth
Copy-Item weights\fold3\best_model.pth outputs\exp_convnext_base_clean_lr1e4_fold3\best_model.pth
```

6. Запустить простой ensemble inference:

```powershell
python src/inference_ensemble_3fold.py --tta none hflip scale_up scale_down
```

Результат будет сохранен в:

```text
outputs/ensemble_convnext_3fold_tta/submission.csv
```

### Когда использовать `avito_room_type_classification_const`, а когда `avito_solution_bundle`

`avito_room_type_classification_const` удобен как исследовательский и тренировочный проект: там есть train-loop, EDA, сравнение экспериментов, подготовка расширенных данных, конфиги разных моделей и история экспериментов.

`avito_solution_bundle` удобен как переносимый inference-пакет: там уже есть веса трех моделей, готовый stacking-сабмит и минимальный код, нужный для повторного получения предсказаний. При переносе на новый компьютер главный риск - не код, а пути к данным и весам.

## UI/UX-сервис для демонстрации модели

В проект добавлен демонстрационный веб-интерфейс:

```text
micro_interface/
```

Назначение сервиса: пользователь загружает фотографию комнаты, сервис применяет лучший доступный pipeline проекта - `3-fold ConvNeXt Base ensemble` из `avito_solution_bundle` - и показывает предсказанный тип помещения, confidence и топ-5 альтернатив.

Сервис реализован на `Gradio`, контейнеризуется через `Docker` и не требует `test_df.csv` для одиночного изображения. По умолчанию веса ищутся в:

```text
.\prectice2curse\avito_solution_bundle\weights
```

Локальный запуск:

```powershell
cd .\prectice2curse\avito_room_type_classification_const\micro_interface
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Docker-запуск:

```powershell
cd .\prectice2curse\avito_room_type_classification_const\micro_interface
docker compose up --build
```

После запуска интерфейс доступен на:

```text
http://localhost:7860
```

Подробные требования к UI/UX, Docker, MLOps, лицензиям и фазам развития проекта вынесены в корневой документ:

```text
.\prectice2curse\UI_UX_ROOM_CLASSIFIER_SERVICE_GUIDE.md
```

## Приложение. Инструкция по запуску проекта

### 1. Перейти в директорию проекта

```powershell
cd .\prectice2curse\avito_room_type_classification_const
```

### 2. Подготовить Python

Рекомендуется Python `3.10` или `3.11`. В проекте уже есть папка `venv`, но для воспроизводимого запуска лучше создать окружение заново:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Если PowerShell запрещает активацию окружения, временно разрешите выполнение скриптов для текущего процесса:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

CPU-вариант:

```powershell
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

GPU-вариант для CUDA 11.8:

```powershell
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

GPU-вариант для CUDA 12.8:

```powershell
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Проверка GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 4. Подготовить данные

Для baseline нужны:

```text
train_df.csv
val_df.csv
test_df.csv
train_images/train_images/*.jpg
val_images/val_images/*.jpg
test_images/test_images/*.jpg
```

С учетом текущих конфигов эти файлы и папки должны лежать на уровень выше проекта:

```text
.\prectice2curse\
|-- train_df.csv
|-- val_df.csv
|-- test_df.csv
|-- train_images\train_images\
|-- val_images\val_images\
|-- test_images\test_images\
`-- avito_room_type_classification_const\
```

Альтернативный вариант - положить данные внутрь проекта и изменить пути в `configs/*.yaml`.

Для экспериментов с расширенными данными дополнительно нужны:

```text
train_images_full/train_images_full/
train_images_full_clean/train_images_full_clean/
train_images_full_class_focus/train_images_full_class_focus/
data/train_df_full.csv
data/train_df_full_clean.csv
data/train_df_full_class_focus.csv
```

Если расширенные датасеты еще не собраны, запустите скрипты подготовки после проверки путей:

```powershell
python scripts/build_full_train.py
python scripts/build_full_train_clean.py
python scripts/build_full_train_class_focus.py
```

### 5. Запустить baseline

```powershell
python run_baseline.py
```

Или напрямую:

```powershell
python src/train.py --config configs/baseline.yaml
```

После обучения появятся:

```text
outputs/baseline_efficientnet_b0/best_model.pth
outputs/baseline_efficientnet_b0/history.csv
outputs/baseline_efficientnet_b0/classification_report.txt
```

### 6. Запустить лучший single-model ConvNeXt

```powershell
python src/train.py --config configs/exp_convnext_base_class_focus_lr1e4.yaml
```

Для этого должны существовать class-focused CSV и изображения, указанные в конфиге.

### 7. Запустить 3-fold ConvNeXt CV

Перед запуском проверьте в `scripts/run_convnext_3fold.py` переменную `PYTHON`. В текущем файле она указывает на `.venv\Scripts\python.exe`, а в проекте окружение называется `venv`. При необходимости замените:

```python
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
```

Затем:

```powershell
python scripts/run_convnext_3fold.py
```

Скрипт создаст fold-разбиения, конфиги и обучит 3 модели:

```text
outputs/exp_convnext_base_clean_lr1e4_fold1/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold2/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold3/best_model.pth
```

### 8. Запустить инференс одной модели

```powershell
python src/inference.py --config configs/baseline.yaml --weights outputs/baseline_efficientnet_b0/best_model.pth
```

Результат:

```text
outputs/baseline_efficientnet_b0/submission.csv
```

### 9. Запустить 3-fold ensemble inference

Для запуска нужны три checkpoint-файла:

```text
outputs/exp_convnext_base_clean_lr1e4_fold1/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold2/best_model.pth
outputs/exp_convnext_base_clean_lr1e4_fold3/best_model.pth
```

Инференс без TTA:

```powershell
python src/inference_ensemble_3fold.py --tta none
```

Инференс с TTA:

```powershell
python src/inference_ensemble_3fold.py --tta none hflip scale_up scale_down
```

Результат:

```text
outputs/ensemble_convnext_3fold_tta/submission.csv
```

### 10. Запустить stacking ensemble

```powershell
python scripts/stacking_convnext_3fold.py --tta none hflip scale_up scale_down
```

Результаты:

```text
outputs/ensemble_convnext_3fold_stacking_tta/submission.csv
outputs/ensemble_convnext_3fold_stacking_tta/oof_meta.csv
outputs/ensemble_convnext_3fold_stacking_tta/meta_model.pkl
outputs/ensemble_convnext_3fold_stacking_tta/meta_info.json
```

### 11. Запустить EDA

Перед запуском убедитесь, что пути в `src/eda.py` соответствуют фактическому расположению данных. Затем:

```powershell
python src/eda.py
```

Результаты сохраняются в:

```text
outputs/eda/
```

### 12. Сравнить эксперименты

```powershell
python src/compare_experiments.py
```

Результаты:

```text
outputs/experiment_summary.csv
outputs/learning_curves.png
```

## Рекомендуемый порядок полного воспроизведения

1. Настроить окружение и установить зависимости.
2. Проверить доступность CUDA, если планируется обучение ConvNeXt.
3. Разложить исходные CSV и изображения по ожидаемым путям или поправить YAML-конфиги.
4. Запустить `python run_baseline.py` для sanity-check.
5. Собрать расширенные train-наборы, если нужны эксперименты `full_train`.
6. Обучить лучший single-model ConvNeXt.
7. Обучить 3-fold ConvNeXt через `scripts/run_convnext_3fold.py`.
8. Выполнить `src/inference_ensemble_3fold.py` или `scripts/stacking_convnext_3fold.py`.
9. Использовать полученный `submission.csv` как итоговый файл предсказаний.
