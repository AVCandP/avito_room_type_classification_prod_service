# Публикация product service в Git

Эта инструкция описывает публикацию готового проекта:

```text
E:\Python\MIFI\prectice2curse\avito_room_type_classification_prod_service
```

## Важно про веса модели

В проекте есть три файла весов:

```text
models/weights/fold1/best_model.pth
models/weights/fold2/best_model.pth
models/weights/fold3/best_model.pth
```

Каждый файл занимает около `350 МБ`. Обычный GitHub/GitLab push без Git LFS для таких файлов не подходит. Поэтому в проект добавлен `.gitattributes`, который отправляет `.pth`, `.pkl` и другие бинарные ML-артефакты в Git LFS.

## Подготовленные git-файлы

```text
.gitignore       # исключает venv, cache, локальные env-файлы, временные данные
.gitattributes   # включает Git LFS для весов и бинарных ML-артефактов
.env.example     # пример переменных окружения
```

## Вариант A. Публикация с весами через Git LFS

Рекомендуемый вариант для полноценной демонстрации проекта.

### 1. Перейти в папку проекта

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_prod_service
```

### 2. Инициализировать git

```powershell
git init
git branch -M main
```

### 3. Установить и включить Git LFS

Если Git LFS еще не установлен, установить его с официального сайта:

```text
https://git-lfs.com/
```

Затем:

```powershell
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "*.onnx"
git lfs track "*.pkl"
git lfs track "*.joblib"
```

Команды `git lfs track` должны подтвердить или обновить `.gitattributes`.

### 4. Проверить, что веса попадут в LFS

```powershell
git check-attr filter -- models/weights/fold1/best_model.pth
git check-attr filter -- models/weights/fold2/best_model.pth
git check-attr filter -- models/weights/fold3/best_model.pth
```

Ожидаемый результат:

```text
filter: lfs
```

### 5. Добавить файлы

```powershell
git add .
git status --short
```

Проверить LFS-файлы:

```powershell
git lfs ls-files
```

В списке должны быть:

```text
models/weights/fold1/best_model.pth
models/weights/fold2/best_model.pth
models/weights/fold3/best_model.pth
artifacts/stacking/meta_model.pkl
```

### 6. Создать коммит

```powershell
git commit -m "Initial production service release"
```

### 7. Создать удаленный репозиторий

Создать пустой репозиторий на GitHub/GitLab/другой Git-платформе, например:

```text
avito-room-type-classification-prod-service
```

Не добавлять README/license/gitignore на стороне платформы, чтобы не получить конфликт с локальным репозиторием.

### 8. Добавить remote и отправить

Заменить URL на свой:

```powershell
git remote add origin https://github.com/<OWNER>/avito-room-type-classification-prod-service.git
git push -u origin main
```

Если используется SSH:

```powershell
git remote add origin git@github.com:<OWNER>/avito-room-type-classification-prod-service.git
git push -u origin main
```

## Вариант B. Публикация без весов

Если Git LFS недоступен или квота LFS ограничена, веса можно не публиковать в Git, а передать отдельно через Release/облако/артефакт-хранилище.

Тогда нужно добавить в `.gitignore`:

```text
models/weights/
```

И выполнить:

```powershell
git rm --cached -r models/weights
git add .gitignore
git commit -m "Exclude model weights from git"
```

В README обязательно указать, куда положить веса:

```text
models/weights/fold1/best_model.pth
models/weights/fold2/best_model.pth
models/weights/fold3/best_model.pth
```

## Проверка перед push

```powershell
git status
git lfs ls-files
```

Проверить, что в репозиторий не попали:

```text
venv/
.venv/
__pycache__/
.env
data/
```

Проверить, что в репозитории есть:

```text
app/app.py
src/model.py
src/inference_ensemble_3fold.py
models/weights/fold1/best_model.pth      # через LFS
models/weights/fold2/best_model.pth      # через LFS
models/weights/fold3/best_model.pth      # через LFS
artifacts/stacking/submission.csv
Dockerfile
docker-compose.yml
README.md
docs/FINAL_PROJECT_REPORT_AVITO_ROOM_CLASSIFICATION.md
```

## Проверка после clone

На другой машине:

```powershell
git clone https://github.com/<OWNER>/avito-room-type-classification-prod-service.git
cd avito-room-type-classification-prod-service
git lfs pull
```

Проверить веса:

```powershell
Get-ChildItem models\weights -Recurse -Filter best_model.pth | Select-Object FullName,Length
```

Запуск через Docker:

```powershell
docker compose up --build
```

Открыть:

```text
http://localhost:7860
```

## Быстрая последовательность команд

```powershell
cd E:\Python\MIFI\prectice2curse\avito_room_type_classification_prod_service
git init
git branch -M main
git lfs install
git lfs track "*.pth" "*.pt" "*.ckpt" "*.onnx" "*.pkl" "*.joblib"
git add .
git lfs ls-files
git commit -m "Initial production service release"
git remote add origin https://github.com/<OWNER>/avito-room-type-classification-prod-service.git
git push -u origin main
```

