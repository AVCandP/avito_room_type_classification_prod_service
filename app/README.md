# Micro Interface: Room Type Classifier

Мини-сервис для демонстрации лучшей модели проекта: загрузка фотографии комнаты, локальный инференс ансамбля `ConvNeXt Base` и вывод топ-5 вероятных типов помещения.

## Модель

Сервис использует веса из переносимого пакета:

```text
E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights
```

По умолчанию ожидаются файлы:

```text
weights/fold1/best_model.pth
weights/fold2/best_model.pth
weights/fold3/best_model.pth
```

В UI показываются 19 классов задания. Так как обучающий pipeline второй итерации работал с 20 выходами, дополнительный класс `19` объединяется с классом `18 - комната без мебели`.

## Локальный запуск

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_const\micro_interface
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Откройте:

```text
http://localhost:7860
```

Если веса лежат в другом месте:

```powershell
$env:ROOM_MODEL_WEIGHTS_DIR="E:\Python\MIFI\prectice2curse\avito_solution_bundle\weights"
python app.py
```

Можно также передать конкретные файлы через `ROOM_MODEL_WEIGHTS`, разделяя пути точкой с запятой:

```powershell
$env:ROOM_MODEL_WEIGHTS="E:\models\fold1.pth;E:\models\fold2.pth;E:\models\fold3.pth"
python app.py
```

## Docker

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_const\micro_interface
docker compose up --build
```

Сервис будет доступен на:

```text
http://localhost:7860
```

В `docker-compose.yml` веса подключаются read-only из `../../avito_solution_bundle/weights`.

## Ограничения

- CPU-инференс ConvNeXt Base может быть медленным.
- Для ускорения используйте CUDA-образ и GPU runtime Docker.
- Загруженные изображения не отправляются во внешние сервисы.

