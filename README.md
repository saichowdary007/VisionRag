M1 Pro Optimized Multimodal RAG (16GB RAM)

Overview

- Apple Silicon–optimized RAG stack targeting M1 Pro with 16GB RAM.
- Hybrid retrieval (FAISS + SQLite FTS5) with single-stage reranking.
- Streamed ingestion to keep peak memory low during PDF processing.
- FastAPI service for query, with Prometheus metrics and Docker Compose.

Quick Start (Local)

1) Install prerequisites: Homebrew, Python 3.11, Docker.
2) Run setup: `bash ./setup-m1-rag.sh`
3) Put PDFs in `rag-service/documents/` then build indexes:
   `python build_indexes.py --memory-limit 12GB --quantization 4bit`
4) Start the API:
   `docker-compose -f docker-compose-m1.yml up -d`
5) UI: Open `http://localhost:8001/` for chat + upload.
6) API: `POST http://localhost:8001/query` with `{ "query": "...", "k": 5 }`.
7) Chat API: `POST http://localhost:8001/chat` with `{ "message": "...", "k": 5 }`.

Key Paths

- API: `rag-service/main.py`
- Frontend: `rag-service/static/index.html`
- Config: `config/config.yaml`
- Index builder: `build_indexes.py`
- Requirements: `rag-service/requirements-m1.txt`
- Docker Compose: `docker-compose-m1.yml`

Notes

- The offline index builder creates `vectors.faiss` and `documents.db`.
- The runtime service memory-maps FAISS and uses SQLite FTS5.
- Models run with MPS if available; CPU fallback is automatic.
- Reranker: `BAAI/bge-reranker-base` with lazy load on first query.
- The UI triggers `/ingest` then `/admin/reindex` to rebuild indexes.
