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
from rag_service.monitoring.perf import PerformanceMonitor, QUERY_LATENCY, LM_STUDIO_UP, LM_STUDIO_LATENCY, EXPANSION_LATENCY, FUSION_LATENCY
from prometheus_client import Histogram
from rag_service.admin.reindex import start_reindex, get_status
from rapidfuzz import fuzz
from rag_service.generation.lm_studio import generate_with_lm_studio, generate_with_lm_studio_async
from rag_service.search.hybrid_search import reciprocal_rank_fusion
from rag_service.search.binary_vector_store import BinaryVectorStore
from rag_service.cache.redis_cache import get_client as get_redis_client, get_json as redis_get_json, set_json as redis_set_json
from rag_service.types import SearchResult
from rag_service.rerank.vision_clip import ClipVisionReranker
from rag_service.search.query_transform import expand_queries_lmstudio


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: object) -> bytes:
        return orjson.dumps(content)


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    multi_query: bool = False
    expansions: int | None = None


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
_clip_reranker: Optional[ClipVisionReranker] = None


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
        rerank_candidates=cfg.rerank_candidates,
        search_k_max=cfg.search_k_max,
        search_k_multiplier=cfg.search_k_multiplier,
    )


def _multi_query_fused_search(
    base_query: str,
    k: int,
    rerank: bool,
    use_fusion: bool,
) -> list[dict]:
    """Execute single- or multi-query search with RRF fusion across queries. Returns list of materialized results dicts."""
    searcher = get_searcher(rerank)

    # Decide if we should do multi-query fusion
    do_fusion = use_fusion or cfg.enable_query_fusion
    if not do_fusion or not cfg.lm_studio_url:
        hits = searcher.search(base_query, top_k=k)
        return [
            {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "page": h.page, "score": h.score, "text": h.text}
            for h in hits
        ]

    # Determine expansion count and measure expansion latency
    n = max(1, int(cfg.query_expansions))
    t0 = time.time()
    queries = expand_queries_lmstudio(
        url=cfg.lm_studio_url,
        model=cfg.lm_studio_model,
        query=base_query,
        n=n,
        temperature=cfg.query_expansion_temperature,
        timeout_s=min(60, cfg.lm_studio_timeout),
    )
    EXPANSION_LATENCY.observe(max(0.0, time.time() - t0))

    # Execute searches per query and collect rankings
    rankings: list[list[tuple[str, float]]] = []
    for q in queries:
        res = searcher.search(q, top_k=max(k, 10))
        rankings.append([(r.chunk_id, float(r.score)) for r in res])

    # Fuse via RRF across queries with a larger candidate pool first
    t1 = time.time()
    fused_candidates = max(50, k * 3)
    fused = reciprocal_rank_fusion(rankings, k=cfg.query_fusion_rrf_k)[:fused_candidates]
    FUSION_LATENCY.observe(max(0.0, time.time() - t1))

    # Optional: Cross-modal fusion with image index using SigLIP text->image retrieval
    if cfg.enable_vision and cfg.enable_cross_modal:
        try:
            global _siglip_text, _image_vector_store
            if _siglip_text is None:
                _siglip_text = SiglipTextEmbedder(cfg.siglip_model, device=cfg.device)
            if _image_vector_store is None and Path(cfg.faiss_image_index_path).exists():
                _image_vector_store = VectorStore(cfg.faiss_image_index_path)
            image_rank: list[tuple[str, float]] = []
            if _image_vector_store is not None and _siglip_text is not None:
                tvec = _siglip_text.embed_texts([base_query]).astype("float32")
                img_topk = max(10, min(cfg.image_search_topk, max(50, k * 5)))
                iscores, iidxs = _image_vector_store.search(tvec, top_k=img_topk)
                if iidxs.size and iidxs[0].size:
                    rowids = [int(i) for i in iidxs[0].tolist() if int(i) >= 0]
                    image_meta = _lexical_store.get_images_by_rowid(rowids)
                    # Optional vision reranking with CLIP
                    if cfg.enable_vision_reranker:
                        try:
                            global _clip_reranker
                            if _clip_reranker is None:
                                _clip_reranker = ClipVisionReranker(cfg.vision_reranker_model, device=cfg.device)
                            # Limit number of images to rerank for speed
                            lim = max(1, min(int(cfg.vision_rerank_max_images), len(image_meta)))
                            paths = [m[3] for m in image_meta[:lim]]
                            clip_scores = _clip_reranker.score(base_query, paths)
                        except Exception:
                            clip_scores = [0.0] * len(image_meta)
                    else:
                        clip_scores = [0.0] * len(image_meta)
                    seen_pages: set[tuple[str, int]] = set()
                    for idx, (meta, sc) in enumerate(zip(image_meta, iscores[0].tolist())):
                        _img_id, doc_id, page, _path = meta
                        if not doc_id or page < 0:
                            continue
                        key = (doc_id, int(page))
                        if key in seen_pages:
                            continue
                        seen_pages.add(key)
                        # Combine similarity with CLIP score (for idx beyond lim, fallback to 0 clip score)
                        cs = float(clip_scores[idx]) if idx < len(clip_scores) else 0.0
                        comb = float(sc) * (1.0 - float(cfg.vision_rerank_weight)) + cs * float(cfg.vision_rerank_weight)
                        rows = _lexical_store.get_chunks_by_doc_page(doc_id, int(page), limit=3)
                        for (_d, cid, _pg, _txt) in rows:
                            image_rank.append((cid, comb))
            if image_rank:
                # Query routing: upweight image ranks for diagram-like queries
                def _is_diagram_query(q: str) -> bool:
                    ql = q.lower()
                    keys = ["pin", "diagram", "schematic", "connector", "sensor", "wiring", "pwm", "phase"]
                    return any(k in ql for k in keys)
                weight = 2 if _is_diagram_query(base_query) else 1
                lists = [fused] + [image_rank] * max(1, weight)
                fused = reciprocal_rank_fusion(lists, k=cfg.query_fusion_rrf_k)[:k]
            else:
                fused = fused[:k]
        except Exception:
            fused = fused[:k]
    else:
        fused = fused[:k]

    cids = [cid for cid, _ in fused]
    rows = _lexical_store.get_chunks(cids)
    row_map = {cid: (doc, page, txt) for (doc, cid, page, txt) in rows}
    results: list[dict] = []
    for cid, score in fused:
        if cid in row_map:
            doc, page, txt = row_map[cid]
            results.append({"doc_id": doc, "chunk_id": cid, "page": page, "text": txt, "score": score})
    return results


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
        # Allow request-level override for expansions
        if req.expansions is not None:
            try:
                cfg.query_expansions = int(req.expansions)
            except Exception:
                pass
        results = _multi_query_fused_search(
            base_query=req.query,
            k=req.k,
            rerank=req.rerank,
            use_fusion=req.multi_query,
        )
        return {"results": results}


class ChatRequest(BaseModel):
    message: str
    k: int = 5
    rerank: bool = True
    use_llm: bool = True
    multi_query: bool = False
    expansions: int | None = None
    session_id: str | None = None
    reset_session: bool = False

# In-memory fallback session store when Redis is unavailable
_session_store: dict[str, list[dict]] = {}


def _history_key(session_id: str) -> str:
    return f"chat:history:v1:{session_id}"


def _get_history(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    if _redis_client is not None:
        hist = redis_get_json(_redis_client, _history_key(session_id))
        return hist or []
    return _session_store.get(session_id, [])


def _set_history(session_id: str, messages: list[dict]) -> None:
    if not session_id:
        return
    # Trim to last 20 entries
    messages = messages[-20:]
    if _redis_client is not None:
        redis_set_json(_redis_client, _history_key(session_id), messages, ttl_seconds=cfg.cache_ttl_seconds)
    else:
        _session_store[session_id] = messages


def _append_history(session_id: str | None, role: str, content: str) -> None:
    if not session_id:
        return
    hist = _get_history(session_id)
    hist.append({"role": role, "content": content})
    _set_history(session_id, hist)


def _clear_history(session_id: str | None) -> None:
    if not session_id:
        return
    if _redis_client is not None:
        try:
            # Overwrite with empty list
            redis_set_json(_redis_client, _history_key(session_id), [], ttl_seconds=cfg.cache_ttl_seconds)
        except Exception:
            pass
    else:
        _session_store.pop(session_id, None)


# --- Targeted context augmentation for pin queries ---
import re as _re


def _augment_pin_context(query: str, hits: list) -> list:
    """If query asks about a specific pin (e.g., 'pin 26'),
    pull additional chunks from the same page to give the LLM the adjacent labels from the diagram.
    Returns an augmented list of hit-like objects.
    """
    m = _re.search(r"\bpin\s*(\d+)\b", query, flags=_re.I)
    if not m or not hits:
        return hits
    try:
        top = hits[0]
        rows = _lexical_store.get_chunks_by_doc_page(top.doc_id, int(top.page), limit=12)
        extra = []
        seen = {getattr(h, 'chunk_id', '') for h in hits}
        for (doc_id, cid, page, text) in rows:
            if cid in seen:
                continue
            extra.append(type("_SR", (), {"doc_id": doc_id, "chunk_id": cid, "page": page, "text": text, "score": getattr(top, 'score', 0.0)}))
        # Prepend page-local extras so LLM sees the full band (labels near pins)
        augmented = extra + hits
        # Cap to a reasonable size
        return augmented[:max(12, len(hits))]
    except Exception:
        return hits


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat/reset")
def chat_reset(req: ResetRequest):
    if not req.session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    _clear_history(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    start_time = time.time()
    with QUERY_LATENCY.time():
        # Session handling
        if req.reset_session and req.session_id:
            _clear_history(req.session_id)
        history_messages: list[dict] = _get_history(req.session_id)

        # Optional override for expansions
        if req.expansions is not None:
            try:
                cfg.query_expansions = int(req.expansions)
            except Exception:
                pass

        # Cache check (optional)
        cache_start = time.time()
        cache_key = None
        if _redis_client is not None:
            sid = req.session_id or ""
            cache_key = f"chat:v2:k{req.k}:mq{int(req.multi_query or cfg.enable_query_fusion)}:sid:{sid}:{req.message}"
            from rag_service.cache.redis_cache import get_json
            cached = get_json(_redis_client, cache_key)
            if cached:
                cache_time = time.time() - cache_start
                print(f"Cache hit: {cache_time:.3f}s")
                return cached
        cache_time = time.time() - cache_start

        # Search timing
        search_start = time.time()
        results = _multi_query_fused_search(
            base_query=req.message,
            k=req.k,
            rerank=req.rerank,
            use_fusion=req.multi_query,
        )
        # Coerce to simple struct-like objects for downstream code
        hits = [
            type("_SR", (), {"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "page": r["page"], "text": r["text"], "score": r["score"]})
            for r in results
        ]
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
                # Augment context for pin queries to include adjacent labels from same page
                hits = _augment_pin_context(req.message, hits)
                text, _raw = await generate_with_lm_studio_async(
                    url=cfg.lm_studio_url,
                    model=cfg.lm_studio_model,
                    user_query=req.message,
                    chunks=hits[: max(8, req.k)],
                    temperature=cfg.lm_studio_temperature,
                    max_tokens=cfg.lm_studio_max_tokens,
                    system_prompt=(cfg.lm_studio_system_prompt or None),
                    timeout_s=cfg.lm_studio_timeout,
                    history_messages=history_messages[-12:],
                )
                llm_time = time.time() - llm_start
                # Update session history after successful generation
                if req.session_id:
                    _append_history(req.session_id, "user", req.message)
                    _append_history(req.session_id, "assistant", text)
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
    # Optional override for expansions
    if req.expansions is not None:
        try:
            cfg.query_expansions = int(req.expansions)
        except Exception:
            pass
    # Prepare session history; append user message before streaming
    history_messages: list[dict] = _get_history(req.session_id)
    if req.session_id:
        _append_history(req.session_id, "user", req.message)
    results = _multi_query_fused_search(
        base_query=req.message,
        k=req.k,
        rerank=req.rerank,
        use_fusion=req.multi_query,
    )
    hits = [
        type("_SR", (), {"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "page": r["page"], "text": r["text"], "score": r["score"]})
        for r in results
    ]
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
        full_text_parts: list[str] = []
        try:
            # Augment context for pin queries to include page-adjacent labels
            aug_hits = _augment_pin_context(req.message, hits)
            async for tok in generate_with_lm_studio_stream(
                url=cfg.lm_studio_url,
                model=cfg.lm_studio_model,
                user_query=req.message,
                chunks=aug_hits[: max(8, req.k)],
                temperature=cfg.lm_studio_temperature,
                max_tokens=cfg.lm_studio_max_tokens,
                system_prompt=(cfg.lm_studio_system_prompt or None),
                timeout_s=cfg.lm_studio_timeout,
                history_messages=history_messages[-12:],
            ):
                if first:
                    LM_STUDIO_LATENCY.observe(time.time() - start)
                    LM_STUDIO_UP.set(1)
                    first = False
                full_text_parts.append(tok)
                yield (orjson.dumps({"token": tok}) + b"\n")
        except Exception:
            LM_STUDIO_UP.set(0)
        # Fallback to non-streaming if nothing was yielded
        if not full_text_parts:
            try:
                text, _raw = await generate_with_lm_studio_async(
                    url=cfg.lm_studio_url,
                    model=cfg.lm_studio_model,
                    user_query=req.message,
                    chunks=hits[: max(8, req.k)],
                    temperature=cfg.lm_studio_temperature,
                    max_tokens=cfg.lm_studio_max_tokens,
                    system_prompt=(cfg.lm_studio_system_prompt or None),
                    timeout_s=cfg.lm_studio_timeout,
                    history_messages=history_messages[-12:],
                )
                full_text_parts.append(text)
                yield (orjson.dumps({"token": text}) + b"\n")
            except Exception:
                pass
        # After stream ends, persist assistant message to chat history
        try:
            if req.session_id:
                _append_history(req.session_id, "assistant", "".join(full_text_parts))
        except Exception:
            pass
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
