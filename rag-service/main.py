from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import orjson
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_service.config import get_config
from rag_service.embeddings.text_embedder import M1OptimizedTextEmbedder
from rag_service.search.vector_store import VectorStore
from rag_service.search.lexical_store import LexicalStore
from rag_service.search.hybrid_search import HybridSearcher
from rag_service.rerank.bge import BGEReranker
from rag_service.monitoring.perf import M1PerformanceMonitor, QUERY_LATENCY
from prometheus_client import Histogram
from rag_service.admin.reindex import start_reindex, get_status
from rapidfuzz import fuzz
from rag_service.generation.lm_studio import generate_with_lm_studio, generate_with_lm_studio_async
from rag_service.search.binary_vector_store import BinaryVectorStore
from rag_service.cache.redis_cache import get_client as get_redis_client


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: object) -> bytes:
        return orjson.dumps(content)


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True


cfg = get_config()
app = FastAPI(title="M1-Optimized RAG Service", default_response_class=ORJSONResponse)
monitor = M1PerformanceMonitor()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_text_embedder: Optional[M1OptimizedTextEmbedder] = None
_vector_store = VectorStore(cfg.faiss_index_path)
_lexical_store = LexicalStore(cfg.db_path)
_reranker: Optional[BGEReranker] = None
_binary_store: Optional[BinaryVectorStore] = None
_redis_client = get_redis_client(cfg.redis_url)


def get_searcher(rerank: bool) -> HybridSearcher:
    global _reranker
    global _text_embedder
    global _binary_store
    if _text_embedder is None:
        _text_embedder = M1OptimizedTextEmbedder(cfg.text_embedding_model, device=cfg.device)
    if rerank and _reranker is None:
        _reranker = BGEReranker(cfg.reranker_model, device=cfg.device)
    if _binary_store is None and Path(cfg.faiss_binary_index_path).exists():
        _binary_store = BinaryVectorStore(cfg.faiss_binary_index_path, cfg.faiss_binary_ids_path)
    return HybridSearcher(
        _text_embedder,
        _vector_store,
        _lexical_store,
        _reranker if rerank else None,
        _binary_store,
        _redis_client,
        cache_ttl_seconds=cfg.cache_ttl_seconds,
    )


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": f"UI not found. Ensure {index_path} exists."}

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    m = monitor.get_memory_usage()
    return {"status": "ok", "device": cfg.device, **m}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(monitor.metrics_response(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/query")
def query(req: QueryRequest):
    with QUERY_LATENCY.time():
        searcher = get_searcher(req.rerank)
        hits = searcher.search(req.query, top_k=req.k)
        return {
            "results": [
                {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "page": h.page, "score": h.score, "text": h.text}
                for h in hits
            ]
        }


class ChatRequest(BaseModel):
    message: str
    k: int = 5
    rerank: bool = True
    use_llm: bool = True


@app.post("/chat")
async def chat(req: ChatRequest):
    start_time = time.time()
    with QUERY_LATENCY.time():
        searcher = get_searcher(req.rerank)

        # Cache check (optional)
        cache_start = time.time()
        cache_key = None
        if _redis_client is not None:
            cache_key = f"chat:v1:k{req.k}:{req.message}"
            from rag_service.cache.redis_cache import get_json
            cached = get_json(_redis_client, cache_key)
            if cached:
                cache_time = time.time() - cache_start
                print(f"Cache hit: {cache_time:.3f}s")
                return cached
        cache_time = time.time() - cache_start

        # Search timing
        search_start = time.time()
        hits = searcher.search(req.message, top_k=req.k)
        search_time = time.time() - search_start
        # If LM Studio configured, prefer LLM synthesis; otherwise fall back to extractive
        citations = []
        for h in hits[: min(len(hits), 3)]:
            snippet = h.text.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "…"
            citations.append({
                "doc_id": h.doc_id,
                "page": h.page,
                "chunk_id": h.chunk_id,
                "score": h.score,
                "snippet": snippet,
            })

        if hits and cfg.lm_studio_url and (cfg.generator == "lm_studio" or req.use_llm):
            try:
                # LLM timing
                llm_start = time.time()
                text, _raw = await generate_with_lm_studio_async(
                    url=cfg.lm_studio_url,
                    model=cfg.lm_studio_model,
                    user_query=req.message,
                    chunks=hits[: max(8, req.k)],
                    temperature=cfg.lm_studio_temperature,
                    max_tokens=cfg.lm_studio_max_tokens,
                    system_prompt=(cfg.lm_studio_system_prompt or None),
                )
                llm_time = time.time() - llm_start
                payload = {"response": text, "citations": citations}
                if _redis_client is not None and cache_key:
                    from rag_service.cache.redis_cache import set_json
                    set_json(_redis_client, cache_key, payload, ttl_seconds=cfg.cache_ttl_seconds)
                total_time = time.time() - start_time
                print(f"Performance: Cache={cache_time:.3f}s, Search={search_time:.3f}s, LLM={llm_time:.3f}s, Total={total_time:.3f}s")
                return payload
            except Exception as e:
                llm_time = time.time() - time.time()  # Will be 0 since exception occurred
                print(f"LLM call failed after {time.time() - start_time:.3f}s: {e}")
                # Fall through to extractive answer on failure
                pass

        # Simple extractive reply fallback
        fallback_start = time.time()
        if hits:
            top = hits[0]
            lines = [ln.strip() for ln in top.text.split("\n") if ln.strip()]
            scored = sorted(lines, key=lambda ln: fuzz.partial_ratio(req.message.lower(), ln.lower()), reverse=True)
            best = "\n".join(scored[:3]) if scored else top.text[:500]
            response_text = best
        else:
            response_text = "I couldn't find relevant content. Try rephrasing or reindexing."
        fallback_time = time.time() - fallback_start

        payload = {"response": response_text, "citations": citations}
        if _redis_client is not None and cache_key:
            from rag_service.cache.redis_cache import set_json
            set_json(_redis_client, cache_key, payload, ttl_seconds=cfg.cache_ttl_seconds)

        total_time = time.time() - start_time
        print(f"Performance: Cache={cache_time:.3f}s, Search={search_time:.3f}s, Fallback={fallback_time:.3f}s, Total={total_time:.3f}s")
        return payload


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Save file to documents directory; index build is offline via build_indexes.py
    docs_dir = Path(cfg.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / file.filename
    with out.open("wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": str(out)}


@app.post("/admin/reindex")
def reindex():
    st = start_reindex()
    return {"status": "started" if st.running else "idle", "pid": st.pid, "log": st.log_path}


@app.get("/admin/status")
def reindex_status():
    st = get_status()
    return {"running": st.running, "pid": st.pid, "log": st.log_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
