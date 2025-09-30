# VisionRag

VisionRag is a multimodal retrieval-augmented generation (RAG) prototype that pairs the ColPali vision retriever with a Qwen-VL model hosted behind an OpenAI-compatible API (Ollama by default). The repository ships two FastAPI services:

- **Ingestion service** – converts uploaded PDFs into page images, generates ColPali multi-vector embeddings, and stores them in Milvus Lite.
- **API service** – offers `/upload` and `/ask` endpoints to add new PDFs and query them with reranked retrieval plus a vision-language model.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)
- An OpenAI-compatible vision model endpoint. The provided configuration targets an [Ollama](https://ollama.com/) server exposing the `qwen2.5-vl:7b` model on `http://localhost:11434/v1`.

> **Important:** The Ollama (or alternative) server is **not** started by the Docker Compose stack. Start it manually before running the services so that the `/ask` endpoint can obtain answers.

Example Ollama commands:

```bash
ollama serve
ollama run qwen2.5-vl:7b
```

## Quick start with Docker Compose

1. Copy the sample environment file and customise if needed:
   ```bash
   cp .env.example .env
   ```
2. Build and start the services:
   ```bash
   docker compose -f deployment/docker-compose.yml up --build
   ```
3. Visit `http://localhost:8000` to access the API service's index page.
4. Upload PDFs via the UI or API, then send questions to `/ask`.

The compose file maps the project `documents/` and `pages/` folders for persistence and stores Milvus Lite data under `./storage`.

## Running without Docker

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```
3. Copy `.env.example` to `.env` and adjust configuration variables.
4. Start the ingestion service to index PDFs:
   ```bash
   python services/ingestion_service.py
   ```
5. Start the API service in another terminal:
   ```bash
   uvicorn services.api_service:app --reload --host 0.0.0.0 --port 8000
   ```

## Environment variables

The application reads configuration from environment variables (or the defaults shown below). Update `.env` as needed:

| Variable | Description | Default |
| --- | --- | --- |
| `MILVUS_URI` | Path or URI for Milvus Lite database. | `./storage/milvus.db` |
| `COLLECTION_NAME` | Milvus collection name. | `rag_vision_collection` |
| `IMAGE_DPI` | DPI used when rasterising PDF pages. | `200` |
| `RETRIEVER_MODEL_ID` | Hugging Face identifier for the ColPali model. | `vidore/colpali-v1.3` |
| `TOP_K_CANDIDATES` | Number of candidate pages retrieved before reranking. | `50` |
| `TOP_K_FINAL` | Number of pages passed to the LLM prompt. | `3` |
| `QA_API_BASE` | Base URL for OpenAI-compatible completions API. | `http://localhost:11434/v1` |
| `QA_API_KEY` | API key/token for the completions API. | `ollama` |
| `QA_MODEL_ID` | Model name to request from the completions API. | `qwen2.5-vl:7b` |

If you run an alternative LLM backend (e.g., vLLM or OpenAI), update `QA_API_BASE`, `QA_API_KEY`, and `QA_MODEL_ID` accordingly.

## Troubleshooting

- **`/ask` returns 500 errors:** Ensure the Ollama (or alternative) server is running and reachable at the configured `QA_API_BASE`.
- **Milvus connection errors:** Confirm the application has write access to the `storage/` directory and that the path matches `MILVUS_URI`.
- **PDF conversion failures:** The ingestion image installs `poppler-utils`. When running locally, install Poppler and ensure `pdftoppm` is on your `PATH`.

