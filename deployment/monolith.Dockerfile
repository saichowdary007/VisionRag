FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install python-json-logger

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]


