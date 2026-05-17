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
