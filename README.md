VisionRag — ColPali Vision RAG
================================

Minimal vision-RAG stack that retrieves page images with ColPali/Byaldi and answers with a VLM (LM Studio, OpenAI‑compatible).

Stack
- Retrieval: ColPali page‑image indexing and search (late‑interaction, MaxSim)
- Generation: LM Studio VLM for grounded answers
- Runtime: 3 services (api, retriever, web). Optional `vlm` service commented.
- UI: Tiny Vite+React single‑page chat that streams tokens and shows page thumbnails

Quick Start (Docker Compose)
1) Copy env: `cp .env.example .env`
2) Build + run: `make up` (or `docker compose up -d --build`)
3) Open the web UI: http://localhost:5173
4) Ingest a PDF (URL or file) from the UI, then ask a question

CLI examples
- Ingest:
  ```bash
  curl -X POST http://localhost:8080/ingest \
    -H "Content-Type: application/json" \
    -d '{"pdf_url":"https://arxiv.org/pdf/2403.05530.pdf"}'
  ```
- Ask (streams plain text tokens):
  ```bash
  curl -N -X POST http://localhost:8080/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"What are the payment terms?","k":5,"m":3}'
  ```

Services
- api: FastAPI server exposing `/ingest`, `/ask`, `/healthz`. Serves page images under `/pages`.
- retriever: Byaldi/ColPali wrapper with `/index` and `/search`.
- web: Minimal Vite+React single‑page chat UI.
- vlm (optional): LM Studio (headless) exposing `/v1/chat/completions`.

Environment (.env)
- `DATA_DIR=/data`
- Retriever (Byaldi/ColPali):
  - `RETRIEVER_MODEL=vidore/colpali-v1.2`
  - `RETRIEVER_INDEX_DIR=/data/index`
  - `RETRIEVER_DEVICE=cpu` (use `cuda` when available)
- API bases:
  - `API_PUBLIC_BASE=http://localhost:8080`
  - `API_INTERNAL_BASE`: VLM inside Docker → `http://api:8080`; VLM on host → `http://host.docker.internal:8080`
- VLM (OpenAI-compatible):
  - `VLM_BASE_URL=http://host.docker.internal:1234` (or `http://vlm:1234` if containerized)
  - `VLM_API_KEY=lm-studio`
  - `VLM_MODEL=google/gemma-3-12b-it`
  - `VLM_MAX_IMAGES=5`
- Hybrid / ranking:
  - `TOP_K=5`
  - `HYBRID_ALPHA=0.2` (0 disables BM25)

Notes
- First line of the `/ask` stream emits a JSON meta line with the images used by the VLM so the UI can show thumbnails.
- If the retriever returns no hits, the API streams `No relevant pages found.` and ends.
- If the VLM service is not running, the API will stream a brief notice and still return the retrieved image list so you can inspect context.

Known-good .env (VLM on host)
```
DATA_DIR=/data
RETRIEVER_MODEL=vidore/colpali-v1.2
RETRIEVER_INDEX_DIR=/data/index
RETRIEVER_DEVICE=cpu
RETRIEVER_BASE_URL=http://retriever:8081
API_PUBLIC_BASE=http://localhost:8080
API_INTERNAL_BASE=http://host.docker.internal:8080
WEB_ORIGIN=http://localhost:5173

VLM_BASE_URL=http://host.docker.internal:1234
VLM_API_KEY=lm-studio
VLM_MODEL=google/gemma-3-12b-it
VLM_MAX_IMAGES=5
TOP_K=5
HYBRID_ALPHA=0.2
```

Development
- Run only the API locally: `pip install -r requirements.txt && make dev`, then open `http://localhost:8080/docs`.
- Run the web locally: `cd web && npm install && npm run dev`.

Health checks
- API: `curl http://localhost:8080/healthz`
- Retriever: `curl http://localhost:8081/healthz`
- LM Studio: `curl http://localhost:1234/v1/models` (or via `host.docker.internal` when called from a container)

Search behavior
- Dense retrieval with ColPali via Byaldi.
- Optional hybrid fusion with BM25 over page texts (PyMuPDF/PDFium extraction).
- `/ask` streams tokens and returns source thumbnails and heatmap overlays.
