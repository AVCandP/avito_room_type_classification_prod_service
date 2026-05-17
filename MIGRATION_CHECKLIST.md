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
