FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for both services
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt ./
# Install root requirements; allow failure on optional heavy deps like byaldi during retriever builds
RUN pip install -r requirements.txt || true

# Copy API service requirements
COPY api/requirements.txt ./api/
RUN pip install -r api/requirements.txt

# Copy all source code
COPY . .

# Expose both ports
EXPOSE 8080 8081

# Create a startup script
RUN echo '#!/bin/bash\nif [ "$SERVICE" = "api" ]; then\n    uvicorn api.main:app --host 0.0.0.0 --port 8080\nelse\n    PYTHONPATH=/app uvicorn server:app --host 0.0.0.0 --port 8081\nfi' > /app/start.sh && chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]

