from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import orjson
from fastapi import FastAPI, UploadFile, File, Form
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_service.config import get_config
from rag_service.embeddings.text_embedder import OptimizedTextEmbedder
from rag_service.embeddings.image_embedder import SiglipImageEmbedder, SiglipTextEmbedder
from rag_service.search.vector_store import VectorStore
from rag_service.search.lexical_store import LexicalStore
from rag_service.search.hybrid_search import HybridSearcher
from rag_service.rerank.bge import BGEReranker
from rag_service.monitoring.perf import PerformanceMonitor, QUERY_LATENCY, LM_STUDIO_UP, LM_STUDIO_LATENCY
from prometheus_client import Histogram
from rag_service.admin.reindex import start_reindex, get_status
from rapidfuzz import fuzz
from rag_service.generation.lm_studio import generate_with_lm_studio, generate_with_lm_studio_async
from rag_service.search.hybrid_search import reciprocal_rank_fusion
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
app = FastAPI(title="Vision RAG Service", default_response_class=ORJSONResponse)
monitor = PerformanceMonitor()
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


_text_embedder: Optional[OptimizedTextEmbedder] = None
_image_embedder: Optional[SiglipImageEmbedder] = None
_siglip_text: Optional[SiglipTextEmbedder] = None
_vector_store = VectorStore(cfg.faiss_index_path)
_lexical_store = LexicalStore(cfg.db_path)
_reranker: Optional[BGEReranker] = None
_binary_store: Optional[BinaryVectorStore] = None
_image_vector_store: Optional[VectorStore] = None
_redis_client = get_redis_client(cfg.redis_url)


def get_searcher(rerank: bool) -> HybridSearcher:
    global _reranker
    global _text_embedder
    global _binary_store
    if _text_embedder is None:
        _text_embedder = OptimizedTextEmbedder(cfg.text_embedding_model, device=cfg.device)
    if rerank and _reranker is None:
        _reranker = BGEReranker(cfg.reranker_model, device=cfg.device)
    if _binary_store is None and Path(cfg.faiss_binary_index_path).exists():
        _binary_store = BinaryVectorStore(cfg.faiss_binary_index_path, cfg.faiss_binary_ids_path)
    if cfg.enable_vision and _image_vector_store is None and Path(cfg.faiss_image_index_path).exists():
        # Lazy-load image index if configured
        globals()['_image_vector_store'] = VectorStore(cfg.faiss_image_index_path)
    return HybridSearcher(
        _text_embedder,
        _vector_store,
        _lexical_store,
        _reranker if rerank else None,
        _binary_store,
        _redis_client,
        cache_ttl_seconds=cfg.cache_ttl_seconds,
        binary_first=cfg.use_binary_primary,
        binary_candidates=cfg.binary_filter_candidates,
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


@app.get("/health/lm")
async def health_lm():
    if not cfg.lm_studio_url:
        return {"configured": False, "up": False}
    import httpx
    t0 = time.time()
    up = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(cfg.lm_studio_url, headers={"Accept": "application/json"})
        up = r.status_code < 500
        LM_STUDIO_UP.set(1 if up else 0)
    except Exception:
        LM_STUDIO_UP.set(0)
        up = False
    dt = time.time() - t0
    if up:
        LM_STUDIO_LATENCY.observe(dt)
    return {"configured": True, "up": up, "latency_s": dt}


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
                    timeout_s=cfg.lm_studio_timeout,
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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    searcher = get_searcher(req.rerank)
    hits = searcher.search(req.message, top_k=req.k)
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

    if not hits or not cfg.lm_studio_url:
        async def _gen_fallback():
            yield orjson.dumps({"token": "", "done": True, "citations": citations}) + b"\n"
        return StreamingResponse(_gen_fallback(), media_type="application/json")

    from rag_service.generation.lm_studio import generate_with_lm_studio_stream

    async def token_gen():
        start = time.time()
        first = True
        try:
            async for tok in generate_with_lm_studio_stream(
                url=cfg.lm_studio_url,
                model=cfg.lm_studio_model,
                user_query=req.message,
                chunks=hits[: max(8, req.k)],
                temperature=cfg.lm_studio_temperature,
                max_tokens=cfg.lm_studio_max_tokens,
                system_prompt=(cfg.lm_studio_system_prompt or None),
                timeout_s=cfg.lm_studio_timeout,
            ):
                if first:
                    LM_STUDIO_LATENCY.observe(time.time() - start)
                    LM_STUDIO_UP.set(1)
                    first = False
                yield (orjson.dumps({"token": tok}) + b"\n")
        except Exception:
            LM_STUDIO_UP.set(0)
        yield (orjson.dumps({"done": True, "citations": citations}) + b"\n")

    return StreamingResponse(token_gen(), media_type="application/json")


@app.post("/query/multimodal")
async def multimodal_query(
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    k: int = Form(default=5),
):
    global _image_embedder, _image_vector_store, _siglip_text
    if not text and not image:
        return JSONResponse({"error": "Provide text or image"}, status_code=400)

    if text and not image:
        searcher = get_searcher(rerank=True)
        hits = searcher.search(text, top_k=k)
        # Optional cross-modal fusion
        if cfg.enable_vision and cfg.enable_cross_modal:
            if _siglip_text is None:
                _siglip_text = SiglipTextEmbedder(cfg.siglip_model, device=cfg.device)
            if _image_vector_store is None and Path(cfg.faiss_image_index_path).exists():
                _image_vector_store = VectorStore(cfg.faiss_image_index_path)
            image_rank: list[tuple[str, float]] = []
            if _image_vector_store is not None:
                tvec = _siglip_text.embed_texts([text]).astype("float32")
                iscores, iidxs = _image_vector_store.search(tvec, top_k=max(50, k * 5))
                if iidxs.size and iidxs[0].size:
                    rowids = [int(i) for i in iidxs[0].tolist() if int(i) >= 0]
                    image_meta = _lexical_store.get_images_by_rowid(rowids)
                    for (meta, sc) in zip(image_meta, iscores[0].tolist()):
                        _img_id, doc_id, page, _ = meta
                        if not doc_id or page < 0:
                            continue
                        rows = _lexical_store.get_chunks_by_doc_page(doc_id, int(page), limit=2)
                        for (_d, cid, _pg, _txt) in rows:
                            image_rank.append((cid, float(sc)))
            text_rank = [(h.chunk_id, float(h.score)) for h in hits]
            fused = reciprocal_rank_fusion([text_rank, image_rank])[:k]
            chunk_ids = [cid for cid, _ in fused]
            rows = _lexical_store.get_chunks(chunk_ids)
            row_map = {cid: (doc, page, txt) for (doc, cid, page, txt) in rows}
            results = []
            for cid, score in fused:
                if cid in row_map:
                    doc, page, txt = row_map[cid]
                    results.append({"doc_id": doc, "chunk_id": cid, "page": page, "text": txt, "score": score})
            return {"results": results}
        # Default text-only results
        return {
            "results": [
                {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "page": h.page, "score": h.score, "text": h.text}
                for h in hits
            ]
        }

    # Visual search path (image provided)
    if cfg.enable_vision and _image_vector_store is None and Path(cfg.faiss_image_index_path).exists():
        _image_vector_store = VectorStore(cfg.faiss_image_index_path)
    if _image_embedder is None and cfg.enable_vision:
        _image_embedder = SiglipImageEmbedder(cfg.siglip_model, device=cfg.device)
    if not cfg.enable_vision or _image_vector_store is None or _image_embedder is None:
        return JSONResponse({"error": "Vision search not available"}, status_code=400)

    import io
    from PIL import Image
    data = await image.read()
    im = Image.open(io.BytesIO(data)).convert("RGB")
    # Embed transiently via temp path API
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
        im.save(tf.name)
        qvec = _image_embedder.embed_paths([tf.name])
    # Search image index for nearest images
    scores, idxs = _image_vector_store.search(qvec.astype("float32"), top_k=max(50, k * 5))
    items = []
    if idxs.size and idxs[0].size:
        rowids = [int(i) for i in idxs[0].tolist() if int(i) >= 0]
        image_meta = _lexical_store.get_images_by_rowid(rowids)
        seen_pages = set()
        for (img_id, doc_id, page, _path), sc in zip(image_meta, scores[0].tolist()):
            if not doc_id or page < 0:
                continue
            if (doc_id, page) in seen_pages:
                continue
            seen_pages.add((doc_id, page))
            # Fetch a few chunks from that page
            rows = _lexical_store.get_chunks_by_doc_page(doc_id, int(page), limit=2)
            for (d, cid, pg, txt) in rows:
                items.append({"doc_id": d, "chunk_id": cid, "page": pg, "text": txt, "score": float(sc)})
            if len(items) >= k:
                break
    return {"results": items[:k]}


@app.get("/index/analyze")
def analyze_index():
    # Basic index statistics
    cur = _lexical_store.conn.cursor()
    cur.execute("SELECT COUNT(1) FROM chunks")
    total_chunks = int(cur.fetchone()[0])
    total_images = _lexical_store.count_images()
    return {
        "total_chunks": total_chunks,
        "total_images": total_images,
    }


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
