Vision RAG Service

Overview

- Universal RAG stack optimized for performance across all architectures.
- Hybrid retrieval (FAISS + SQLite FTS5) with single-stage reranking.
- Streamed ingestion to keep peak memory low during PDF processing.
- FastAPI service for query, with Prometheus metrics and Docker Compose.
- Async processing for improved LLM response times.

Quick Start (Local)

1) Install prerequisites: Homebrew, Python 3.11, Docker.
2) Run setup: `bash ./setup-rag.sh`
3) Put PDFs in `rag-service/documents/` then build indexes:
   `python build_indexes.py --memory-limit 12GB --quantization 4bit`
4) Start the API:
   `docker compose up -d`
5) UI: Open `http://localhost:8001/` for chat + upload.
6) API: `POST http://localhost:8001/query` with `{ "query": "...", "k": 5 }`.
7) Chat API: `POST http://localhost:8001/chat` with `{ "message": "...", "k": 5 }`.

Key Paths

- API: `rag-service/main.py`
- Frontend: `rag-service/static/index.html`
- Config: `config/config.yaml`
- Index builder: `build_indexes.py`
- Requirements: `rag-service/requirements.txt`
- Docker Compose: `docker-compose.yml`

Notes

- The offline index builder creates `vectors.faiss` and `documents.db`.
- The runtime service memory-maps FAISS and uses SQLite FTS5.
- Automatic device detection (MPS/CUDA/CPU) with optimal performance.
- Reranker: `BAAI/bge-reranker-base` with lazy load on first query.
- Async processing for improved LLM response times.
- The UI triggers `/ingest` then `/admin/reindex` to rebuild indexes.
