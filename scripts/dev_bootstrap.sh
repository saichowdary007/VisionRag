#!/usr/bin/env bash
set -euo pipefail

echo "[VisionRag] Building and starting services..."
docker compose up -d --build

echo "[VisionRag] Waiting a few seconds for services to settle..."
sleep 3

echo "[VisionRag] Health checks:"
echo "- API:" && curl -sS http://localhost:8080/healthz || true
echo
echo "- Retriever:" && curl -sS http://localhost:8081/healthz || true
echo
echo "If LM Studio is running on your host, models:"
curl -sS http://localhost:1234/v1/models || true
echo
echo "Open the web UI: http://localhost:5173"

