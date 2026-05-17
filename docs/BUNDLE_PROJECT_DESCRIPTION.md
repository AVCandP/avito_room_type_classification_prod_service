# Описание проекта avito_solution_bundle

## Что делает проект

`avito_solution_bundle` - это переносимый пакет для инференса ML-решения задачи классификации изображений помещений Avito. Проект принимает изображения объявлений и для каждого изображения предсказывает один из 20 числовых классов типа/категории помещения.

Основная идея решения: использовать ансамбль из трех нейросетевых моделей `ConvNeXt Base`, обученных на разных фолдах, усилить предсказания test-time augmentation и затем применить stacking-мета-модель на вероятностях классов. Итоговый результат сохраняется в формате CSV с колонками `image_id_ext` и `Predicted`.

## Для чего он нужен

Проект нужен для автоматической классификации фотографий недвижимости или помещений в пайплайне Avito/хакатонной задачи. Он позволяет:

- запускать готовую обученную модель без повторного обучения;
- получать предсказания классов для тестового набора изображений;
- воспроизводить ансамблевый инференс по трем фолдам;
- использовать уже подготовленную stacking-модель для улучшения качества предсказаний;
- хранить веса, конфиги, код инференса и итоговые артефакты в одном переносимом bundle.

## Какие результаты дает

Проект формирует файл `submission.csv`, где каждой картинке сопоставлен предсказанный класс:

```csv
image_id_ext,Predicted
15822317371,7
15948569108,2
16030687230,10
```

В текущем bundle уже лежит готовый результат:

- `stacking/submission.csv` - итоговые предсказания для 48 003 тестовых изображений;
- `stacking/oof_meta.csv` - out-of-fold вероятности, истинные классы и предсказания мета-модели для 4 612 объектов;
- `stacking/meta_model.pkl` - обученная stacking-модель `LogisticRegression`;
- `stacking/meta_info.json` - параметры stacking-запуска.

Согласно `stacking/meta_info.json`, качество мета-модели на OOF-данных:

```text
OOF macro F1: 0.7565167133953731
```

## Структура проекта

```text
avito_solution_bundle/
├── BUNDLE_README.txt
├── PROJECT_DESCRIPTION.md
├── requirements.txt
├── configs/
│   ├── exp_convnext_base_clean_lr1e4_fold1.yaml
│   ├── exp_convnext_base_clean_lr1e4_fold2.yaml
│   └── exp_convnext_base_clean_lr1e4_fold3.yaml
├── scripts/
│   └── stacking_convnext_3fold.py
├── src/
│   ├── dataset.py
│   ├── inference_ensemble_3fold.py
│   └── model.py
├── stacking/
│   ├── meta_info.json
│   ├── meta_model.pkl
│   ├── oof_meta.csv
│   └── submission.csv
├── weights/
│   ├── fold1/best_model.pth
│   ├── fold2/best_model.pth
│   └── fold3/best_model.pth
└── venv/
```

### `src/`

Основной код модели и инференса.

- `src/model.py` - класс `RoomClassifier`. Это PyTorch-модель на базе `timm.create_model`: backbone без классификационной головы, затем `Dropout` и `Linear` на 20 классов. Для текущих конфигов используется `convnext_base`.
- `src/dataset.py` - датасет `RoomDataset`, загрузка изображений по `image_id_ext`, преобразования через Albumentations, нормализация ImageNet, режимы `train`, `val`, `test`, опциональный preload изображений в RAM и weighted sampler для дисбаланса классов.
- `src/inference_ensemble_3fold.py` - скрипт простого ансамблевого инференса: загружает три fold-модели, усредняет logits, поддерживает TTA-режимы `none`, `hflip`, `scale_up`, `scale_down`, сохраняет `submission.csv`.

### `scripts/`

- `scripts/stacking_convnext_3fold.py` - полный stacking-пайплайн:
  1. загружает три fold-конфига и веса моделей;
  2. собирает OOF-вероятности на validation split каждого фолда;
  3. обучает multinomial `LogisticRegression` как мета-модель;
  4. запускает инференс на тесте с TTA;
  5. усредняет вероятности трех моделей;
  6. применяет мета-модель;
  7. сохраняет `submission.csv`, `oof_meta.csv`, `meta_model.pkl`, `meta_info.json`.

### `configs/`

YAML-конфиги трех фолдов. В них заданы:

- имя эксперимента;
- пути к train/val/test CSV;
- пути к директориям изображений;
- число классов: `20`;
- модель: `convnext_base`;
- размер изображения: `224`;
- batch size: `8`;
- параметры обучения, использованные при подготовке весов: `lr=0.0001`, `weight_decay=0.0001`, `label_smoothing=0.05`, `num_epochs=12`, `seed=42`;
- директория с весами конкретного фолда.

Важный момент: в bundle датасет не включен. Пути в конфигах рассчитаны на соседние директории с файлами `test_df.csv`, `train_images_full_clean` и `test_images`. При переносе проекта на другой компьютер эти пути нужно проверить и при необходимости поправить.

### `weights/`

Готовые веса трех fold-моделей:

- `weights/fold1/best_model.pth`;
- `weights/fold2/best_model.pth`;
- `weights/fold3/best_model.pth`.

Каждый файл весов занимает примерно 350 МБ. Это обученные PyTorch checkpoint-файлы для `ConvNeXt Base`.

### `stacking/`

Готовые результаты stacking-решения:

- `submission.csv` - финальный файл отправки/предсказаний;
- `oof_meta.csv` - вероятности `p0`...`p19`, target и `pred_meta` для OOF-оценки;
- `meta_model.pkl` - сериализованная scikit-learn LogisticRegression;
- `meta_info.json` - параметры запуска и OOF macro F1.

### `requirements.txt`

Список зависимостей для запуска проекта. Основные библиотеки:

- `torch`, `torchvision`;
- `timm`;
- `albumentations`, `opencv-python`, `Pillow`;
- `numpy`, `pandas`;
- `scikit-learn`;
- `PyYAML`, `tqdm`;
- `matplotlib`, `seaborn` для анализа/визуализации.

## Реализованный функционал

1. Загрузка изображений по идентификатору `image_id_ext`.
2. Преобразование изображений к формату модели: resize, center crop, ImageNet-нормализация, tensor conversion.
3. Training-аугментации в коде датасета: random resized crop, horizontal flip, color jitter, noise, rotation.
4. Классификационная модель `RoomClassifier` поверх backbone из библиотеки `timm`.
5. Инференс одной fold-модели внутри ансамбля.
6. Ансамблирование трех fold-моделей.
7. Test-time augmentation:
   - исходное изображение;
   - горизонтальный flip;
   - легкий zoom-in;
   - легкий zoom-out.
8. Усреднение вероятностей или logits между TTA и fold-моделями.
9. Stacking через multinomial LogisticRegression.
10. Расчет OOF macro F1 для мета-модели.
11. Сохранение финального `submission.csv`.
12. Сохранение диагностических и воспроизводимых артефактов stacking-запуска.

## Как запустить

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Простой ансамблевый инференс без stacking:

```bash
python src/inference_ensemble_3fold.py --tta none hflip scale_up scale_down
```

Полный stacking-инференс:

```bash
python scripts/stacking_convnext_3fold.py --tta none hflip scale_up scale_down
```

Перед запуском нужно убедиться, что:

- доступны `test_df.csv` и директория с тестовыми изображениями;
- пути в `configs/*.yaml` соответствуют текущему расположению данных;
- пути к весам соответствуют фактическому расположению `best_model.pth`;
- установлен подходящий PyTorch под CPU или CUDA.

## Краткий вывод

Проект представляет собой готовое ML-решение для классификации изображений помещений на 20 классов. Он нужен, чтобы по набору изображений автоматически получить CSV-файл с предсказанными классами. Внутри используются три обученные ConvNeXt-модели, TTA и stacking-мета-модель. Главный практический результат проекта - `stacking/submission.csv` с предсказаниями для 48 003 изображений; качество мета-модели на OOF-валидации составляет macro F1 около `0.7565`.
