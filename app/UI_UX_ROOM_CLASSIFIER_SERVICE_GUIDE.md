# Инструкция и требования к UI/UX-сервису распознавания типа комнаты

## Назначение документа

Документ описывает требования к реализации демонстрационного UI/UX-сервиса для проекта Avito Room Type Classification. Сервис должен позволять пользователю загрузить фотографию помещения, выполнить локальный инференс лучшей обученной модели и получить предсказание типа комнаты.

Реализация интерфейса размещена в:

```text
E:\Python\MIFI\prectice2curse\avito_room_type_classification_const\micro_interface
```

Лучшие веса модели и готовые результаты находятся в:

```text
E:\Python\MIFI\prectice2curse\avito_solution_bundle
```

## Краткое описание решения

Сервис решает задачу многоклассовой классификации фотографий недвижимости. Пользователь загружает изображение, модель возвращает один из 19 классов:

| ID | Класс |
|---:|---|
| 0 | кухня / столовая |
| 1 | кухня-гостиная |
| 2 | универсальная комната |
| 3 | гостиная |
| 4 | спальня |
| 5 | кабинет |
| 6 | детская |
| 7 | ванная комната |
| 8 | туалет |
| 9 | совмещенный санузел |
| 10 | коридор / прихожая |
| 11 | гардеробная / кладовая / постирочная |
| 12 | балкон / лоджия |
| 13 | вид из окна / с балкона |
| 14 | дом снаружи / двор |
| 15 | подъезд / лестничная площадка |
| 16 | другое |
| 17 | предметы интерьера / бытовая техника |
| 18 | комната без мебели |

Практическая ценность: автоматическая проверка и категоризация фотографий в объявлениях недвижимости, помощь модераторам, улучшение качества карточек и подготовка аналитики по изображениям.

## Фазы развития проекта

### Фаза 1: baseline и первичная концепция

Проект начинался в директории:

```text
E:\Python\MIFI\prectice2curse\avito_room_type_classification_av
```

На первой фазе были выполнены:

- анализ постановки задачи и бизнес-контекста;
- EDA по train/val/test данным;
- baseline на EfficientNet-B0;
- базовый inference pipeline;
- первые эксперименты с фильтрацией `ratio`, weighted loss и MaxViT;
- анализ слабых классов через classification report и confusion matrix.

Результат первой фазы:

```text
Best Val Macro F1: 0.6093
```

Слабые классы первой итерации:

- `5 - кабинет`;
- `2 - универсальная комната`;
- `17 - предметы интерьера / бытовая техника`;
- частично `11 - гардеробная / кладовая / постирочная`.

### Фаза 2: расширение данных, ConvNeXt, TTA и stacking

Вторая фаза расположена в:

```text
E:\Python\MIFI\prectice2curse\avito_room_type_classification_const
E:\Python\MIFI\prectice2curse\avito_solution_bundle
```

Главное отличие второй фазы: команда вручную добирала данные для проблемных классов и более аккуратно фильтровала внешние источники.

Ручной добор данных:

- Артем: `118` изображений класса `кабинет` и `89` изображений класса `другое`;
- Владимир: `170` изображений класса `кухня-гостиная` и `70` изображений класса `предметы интерьера / бытовая техника`;
- Константин: `180` изображений класса `универсальная комната` и `60` изображений класса `предметы интерьера / бытовая техника`.

После этого проект перешел к более сильному pipeline:

- `ConvNeXt Base` через `timm`;
- 3-fold cross-validation;
- test-time augmentation: `none`, `hflip`, `scale_up`, `scale_down`;
- ансамблирование fold-моделей;
- stacking meta-model на `LogisticRegression`;
- сохранение переносимого bundle с весами.

Лучший результат второй фазы:

```text
OOF Macro F1: 0.7565167133953731
```

Итоговый готовый сабмит:

```text
E:\Python\MIFI\prectice2curse\avito_solution_bundle\stacking\submission.csv
```

## Используемая модель

Для UI/UX-сервиса используется лучшая доступная модель проекта:

```text
3-fold ConvNeXt Base ensemble
```

Файлы весов:

```text
E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights\fold1\best_model.pth
E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights\fold2\best_model.pth
E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights\fold3\best_model.pth
```

Особенность: тренировочный pipeline второй фазы работал с 20 выходами, где дополнительный класс `19` дублировал/уточнял класс "комната без мебели". UI должен показывать пользователю 19 классов задания. Поэтому в интерфейсе вероятности классов `18` и `19` объединяются в итоговый класс:

```text
18 - комната без мебели
```

## Требования к моделям и лицензиям

Обязательные требования:

- модель должна быть open-source;
- лицензия должна быть открытой, например MIT, Apache 2.0 или совместимая permissive-лицензия;
- модель должна запускаться локально без обращения к внешним API;
- веса должны храниться локально или в контролируемом артефакт-хранилище;
- inference должен работать без отправки пользовательских фото во внешние сервисы.

Текущий стек соответствует этим требованиям:

| Компонент | Назначение | Лицензирование / статус |
|---|---|---|
| PyTorch | runtime модели | open-source |
| torchvision | preprocessing и tensor transforms | open-source |
| timm | ConvNeXt Base backbone | open-source, Apache 2.0 |
| ConvNeXt Base | архитектура модели | open-source-архитектура, локальный запуск |
| Gradio | демонстрационный UI | open-source |
| scikit-learn | LogisticRegression stacking в основном pipeline | open-source |
| Docker | контейнеризация | локальный запуск |

Рекомендуемые сервисы для дальнейшей промышленной доработки:

- MLflow или Weights & Biases для логирования экспериментов;
- Hugging Face Hub для публикации модели, model card и датасета при наличии прав на данные;
- Streamlit или Gradio для демонстрации;
- FastAPI для production API, если нужен отдельный backend и интеграция с другими сервисами.

## UX-концепция

Сервис должен быть простым демонстрационным инструментом, а не аналитическим комбайном. Главный сценарий:

1. Пользователь открывает веб-страницу.
2. Загружает изображение комнаты.
3. Нажимает кнопку распознавания.
4. Видит предсказанный тип комнаты, уверенность и топ-5 альтернатив.
5. При необходимости загружает другое фото.

### Целевая аудитория

- члены команды на защите проекта;
- преподаватель/эксперт, проверяющий работоспособность модели;
- потенциальный бизнес-пользователь, которому нужно быстро понять, что делает модель;
- разработчик, который будет переносить прототип в production.

### Основные UX-требования

- первый экран сразу содержит загрузку изображения и результат, без лендинга;
- интерфейс не требует знания Python или структуры проекта;
- названия классов показываются на русском языке;
- пользователю виден не только `ID`, но и смысл класса;
- рядом с предсказанием отображается confidence;
- показывается топ-5 классов, чтобы продемонстрировать неопределенность модели;
- ошибки загрузки весов или отсутствия файла объясняются понятным текстом;
- интерфейс должен работать локально и не отправлять изображения во внешние сервисы.

### Рекомендуемый layout

Экран делится на две рабочие области:

- слева: загрузка изображения;
- справа: кнопка запуска, топ-5 классов, технические детали инференса.

Технические детали должны быть видны, но не доминировать:

- модель: `3-fold ConvNeXt Base ensemble`;
- TTA: включено/выключено;
- устройство: `cpu` или `cuda`;
- время обработки;
- путь к весам.

### UX-состояния

Интерфейс должен покрывать состояния:

- `initial`: изображение еще не загружено;
- `ready`: изображение загружено, можно запускать инференс;
- `processing`: модель считает предсказание;
- `success`: показаны топ-классы и детали;
- `error`: отсутствуют веса, неподдерживаемый файл, ошибка модели.

## Реализованный micro-interface

В проект добавлен сервис:

```text
E:\Python\MIFI\prectice2curse\avito_room_type_classification_const\micro_interface
```

Состав:

```text
micro_interface/
|-- app.py              # Gradio UI и inference logic
|-- requirements.txt    # зависимости сервиса
|-- Dockerfile          # Docker-сборка
|-- docker-compose.yml  # запуск с volume для весов
|-- .dockerignore
`-- README.md
```

### Что делает `app.py`

- загружает три fold-модели `ConvNeXt Base`;
- берет веса из `avito_solution_bundle\weights`;
- выполняет preprocessing: resize, center crop, ImageNet normalization;
- поддерживает horizontal flip TTA;
- усредняет logits по TTA и fold-моделям;
- применяет softmax;
- объединяет классы `18` и `19` в единый пользовательский класс `18`;
- выводит топ-5 классов в Gradio UI.

## Зависимости UI-сервиса

Файл:

```text
avito_room_type_classification_const\micro_interface\requirements.txt
```

Минимальный набор:

```text
torch==2.9.1
torchvision==0.24.1
timm==1.0.22
gradio==5.49.1
Pillow==11.3.0
numpy==2.3.4
```

Для CPU-запуска используется PyTorch CPU wheel. Для GPU-запуска нужно заменить установку PyTorch на CUDA-вариант, соответствующий видеокарте и драйверу.

## Локальный запуск UI

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_const\micro_interface
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Открыть в браузере:

```text
http://localhost:7860
```

Если PowerShell блокирует активацию:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\venv\Scripts\Activate.ps1
```

## Настройка путей к весам

По умолчанию сервис ищет веса здесь:

```text
E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights
```

Можно переопределить папку:

```powershell
$env:ROOM_MODEL_WEIGHTS_DIR="E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights"
python app.py
```

Или задать три конкретных файла:

```powershell
$env:ROOM_MODEL_WEIGHTS="E:\models\fold1.pth;E:\models\fold2.pth;E:\models\fold3.pth"
python app.py
```

Для принудительного CPU:

```powershell
$env:FORCE_CPU="1"
python app.py
```

## Docker-запуск

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_const\micro_interface
docker compose up --build
```

Открыть:

```text
http://localhost:7860
```

В `docker-compose.yml` веса подключаются как read-only volume:

```yaml
volumes:
  - ../../avito_solution_bundle/weights:/models/weights:ro
```

Переменная окружения внутри контейнера:

```yaml
ROOM_MODEL_WEIGHTS_DIR: /models/weights
```

## Production-вариант с FastAPI

FastAPI не обязателен для демонстрации, но рекомендуется для production-интеграции. Возможная схема:

```text
[Browser / Streamlit / Gradio]
        |
        v
[FastAPI Gateway]
        |
        v
[Inference Service: PyTorch + ConvNeXt ensemble]
        |
        v
[Response: class_id, class_name, probability, top_k]
```

Минимальные endpoints:

| Endpoint | Метод | Назначение |
|---|---|---|
| `/health` | GET | проверка статуса сервиса и загрузки модели |
| `/classes` | GET | список 19 классов |
| `/predict` | POST | загрузка изображения и получение предсказания |
| `/metrics` | GET | latency, count, device, version |

Формат ответа `/predict`:

```json
{
  "class_id": 4,
  "class_name": "спальня",
  "confidence": 0.82,
  "top_k": [
    {"class_id": 4, "class_name": "спальня", "probability": 0.82},
    {"class_id": 3, "class_name": "гостиная", "probability": 0.08}
  ],
  "model": "3-fold ConvNeXt Base ensemble",
  "device": "cuda",
  "latency_ms": 340
}
```

## Логирование и MLOps

Для дальнейшего развития рекомендуется:

### MLflow

Логировать:

- `experiment_name`;
- `model_name`;
- `fold`;
- `img_size`;
- `batch_size`;
- `lr`;
- `weight_decay`;
- `label_smoothing`;
- `Macro F1`;
- `classification_report.txt`;
- `confusion_matrix.png`;
- `best_model.pth`;
- `submission.csv`.

### Weights & Biases

Использовать для:

- мониторинга train/val loss;
- графиков Macro F1;
- сравнения архитектур;
- хранения артефактов;
- визуализации ошибок на примерах изображений.

### Hugging Face

Использовать для:

- публикации model card;
- хранения open-source inference code;
- публикации синтетического/обезличенного датасета, если это разрешено правилами данных;
- демонстрации Spaces на Gradio, если данные и веса можно публиковать.

## Требования к безопасности и приватности

- изображения пользователей обрабатываются локально;
- сервис не должен логировать исходные изображения без явного включения debug-режима;
- Docker volume с весами должен подключаться read-only;
- внешние API не используются в inference path;
- при публикации в сеть нужно добавить ограничение размера файла и MIME-type проверку;
- в production нужно добавить rate limiting и очистку временных файлов.

## Критерии готовности UI/UX-сервиса

Сервис считается готовым, если:

- запускается локально командой `python app.py`;
- запускается через `docker compose up --build`;
- открывается по адресу `http://localhost:7860`;
- принимает JPG/PNG изображение;
- возвращает один из 19 классов;
- показывает топ-5 вероятностей;
- использует лучшие веса из `avito_solution_bundle`;
- не требует тестового CSV для одиночного изображения;
- корректно сообщает об ошибке, если веса не найдены;
- документация описывает фазы развития проекта и лучший результат.

## Итоговая рекомендация

Для защиты и демонстрации использовать `micro_interface` как основной UI. Он показывает практическую ценность проекта: пользователь загружает фотографию комнаты и получает понятное предсказание на русском языке. В отчете и презентации обязательно подчеркнуть, что текущий интерфейс построен на лучшей второй итерации модели, а первая итерация важна как исследовательская фаза, где были выявлены слабые классы и обоснована необходимость ручного добора данных.

