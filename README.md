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
- `LMSTUDIO_BASE_URL=http://vlm:1234` and `LMSTUDIO_API_KEY=lm-studio`
- `LMSTUDIO_MODEL=google/gemma-3-12b-it`
- `RETRIEVER_BASE_URL=http://retriever:8081`
- `API_INTERNAL_BASE=http://api:8080` (for internal image URLs to LM Studio)
- `API_PUBLIC_BASE=http://localhost:8080` (for the browser)
- `WEB_ORIGIN=http://localhost:5173`

Notes
- First line of the `/ask` stream emits a JSON meta line with the images used by the VLM so the UI can show thumbnails.
- If the retriever returns no hits, the API streams `No relevant pages found.` and ends.
- If the VLM service is not running, the API will stream a brief notice and still return the retrieved image list so you can inspect context.

Development
- Run only the API locally: `make dev` then open `http://localhost:8080/docs`.
- Run the web locally: `cd web && npm install && npm run dev`.
