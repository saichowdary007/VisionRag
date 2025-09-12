Testing Guide

Prereqs
- Install and run Ollama on host
- Pull the model:
  - `ollama serve`
  - `ollama pull qwen2.5-vl:7b`

Start Stack
```bash
cd deployment
docker compose up --build
```

Connectivity
- From host: `curl http://localhost:8080/healthz`
- From backend container: `docker compose exec backend sh -lc 'curl -sS http://host.docker.internal:11434/api/tags || curl -sS http://host.docker.internal:11434'`

Ingestion
1) Upload a small PDF (≤5 pages) via UI (Upload button).
2) Expect a toast/alert: "Ingested N pages from <filename>".
3) Stop backend: `docker compose stop backend`; upload again → UI should display an error, HTTP status 4xx/5xx.

Model Query
1) Ask a visual question that references the uploaded PDF.
2) Expect a real answer within configured timeout (default 120s).
3) Stop Ollama (kill `ollama serve`) and ask again → UI should surface error, no mock answer.

Linux Host Connectivity
- Verify: `docker compose exec backend sh -lc 'curl -sS http://host.docker.internal:11434 | head -c 200 | cat'`

Logs
- Retriever logs include ingest start/complete and per-page errors.
- API logs include VLM call timing and errors.

Env Timeouts
- Next.js ingest: `INGEST_TIMEOUT_MS` (default 60000)
- Next.js chat: `QUERY_TIMEOUT_MS` (default 120000)
- API→Ollama: `VLM_TIMEOUT_SEC` (default 180)


