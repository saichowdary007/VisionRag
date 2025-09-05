Vision RAG Service

Overview

- Universal RAG stack optimized for performance across all architectures.
- Hybrid retrieval (FAISS + SQLite FTS5 + optional Binary Hamming) with cross-encoder reranking.
- Multi-query RAG-Fusion (optional): expand the user query via LLM, retrieve per sub-query, fuse via RRF.
- Explanatory, structured generation with persona, numbered steps, bold key terms, and graceful failure mode.
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

New (v1.0) Features

- Hybrid Search: Dense FAISS + SQLite FTS5 BM25 with Reciprocal Rank Fusion and optional binary pre-filter.
- Cross-Encoder Re-ranking: `BAAI/bge-reranker-v2-m3` for high-precision top-K.
- RAG-Fusion: LLM-based query expansion and RRF fusion across sub-queries.
- Explanatory Prompts: Persona + structured output (Summary, Steps, Notes, Sources) and graceful failure.
- Visual Retrieval Enhancements:
  - Region-aware diagram labels: during ingestion, text near extracted image regions is indexed as dedicated label chunks to capture short annotations (e.g., voltages, pin names).
  - Cross-modal fusion: text queries are matched against the image index (SigLIP) and fused with text results.
  - Optional CLIP vision reranker: reorders image-page candidates using text–image similarity.

UI

- Chat page includes toggles for Multi-Query (RAG-Fusion), number of expansions, and Top K.
- Streaming responses with inline citations list.
- Persistent chat sessions: the UI generates a `session_id` in `localStorage` and includes it with each chat request so the model sees prior turns.

Metrics

- `query_latency_seconds`: end-to-end query latency.
- `query_expansion_seconds`: latency of LLM-based query expansion.
- `query_fusion_seconds`: latency of RRF fusion across sub-queries.

Performance tips

- If responses feel slow, try:
  - Disable CLIP reranker: `ENABLE_VISION_RERANKER=false` (saves image I/O + model scoring).
  - Lower expansions: `QUERY_EXPANSIONS=2` or disable query fusion for one-off queries.
  - Reduce candidates: `RERANK_CANDIDATES=30`, `SEARCH_K_MAX=80`, `IMAGE_SEARCH_TOPK=40`.
  - Ensure LM Studio streaming is enabled; the server falls back to non-streaming automatically if no tokens arrive.

Configuration

- `ENABLE_QUERY_FUSION` (bool, default false): enable multi-query RAG-Fusion.
- `QUERY_EXPANSIONS` (int, default 3): number of sub-queries to generate.
- `QUERY_FUSION_RRF_K` (int, default 60): RRF hyperparameter for fusion across queries.
- `QUERY_EXPANSION_TEMPERATURE` (float, default 0.2): LLM temperature for query expansion.
- Generation uses `LM_STUDIO_URL` and `LM_STUDIO_MODEL`; prompts are structured by default and can be overridden via `LM_STUDIO_SYSTEM_PROMPT`.

Retrieval Tuning

- `RERANK_CANDIDATES` (int, default 50): number of fused candidates sent to cross-encoder reranker.
- `SEARCH_K_MAX` (int, default 100): cap for dense/lexical candidate pool before fusion.
- `SEARCH_K_MULTIPLIER` (int, default 3): scales `top_k` to compute candidate pool size before fusion.
- Vision reranker (optional):
  - `ENABLE_VISION_RERANKER` (bool, default false)
  - `VISION_RERANKER_MODEL` (default `openai/clip-vit-base-patch32`)
  - `VISION_RERANK_WEIGHT` (float 0–1, default 0.7)

Notes

- The offline index builder creates `vectors.faiss` and `documents.db`.
- The runtime service memory-maps FAISS and uses SQLite FTS5.
- Automatic device detection (MPS/CUDA/CPU) with optimal performance.
- Reranker: `BAAI/bge-reranker-base` with lazy load on first query.
- Async processing for improved LLM response times.
- The UI triggers `/ingest` then `/admin/reindex` to rebuild indexes.
