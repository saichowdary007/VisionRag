from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .byaldi_wrapper import ByaldiRetriever
from .bm25 import BM25Index
from .heatmaps import generate_placeholder_heatmap


DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Retriever settings
RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "vidore/colpali-v1.2")
RETRIEVER_DEVICE = os.getenv("RETRIEVER_DEVICE", "cpu")
RETRIEVER_INDEX_DIR = Path(os.getenv("RETRIEVER_INDEX_DIR", str(DATA_DIR / "index")))
RETRIEVER_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Hybrid
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0"))

# Heatmaps
HEATMAPS_DIR = Path(os.getenv("HEATMAPS_DIR", str(DATA_DIR / "heatmaps")))
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)


# Initialize retriever and BM25
retriever = ByaldiRetriever(str(RETRIEVER_INDEX_DIR), model_name=RETRIEVER_MODEL, device=RETRIEVER_DEVICE)
bm25 = BM25Index.open(RETRIEVER_INDEX_DIR)

app = FastAPI(title="Retriever Service", version="2.0.0")


class IndexRequest(BaseModel):
    doc_id: str
    images: List[str]
    texts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_id: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]]
    total: int


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "retriever",
        "model": RETRIEVER_MODEL,
        "device": RETRIEVER_DEVICE,
        "index_dir": str(RETRIEVER_INDEX_DIR),
        "hybrid_alpha": HYBRID_ALPHA,
    }


@app.post("/index")
def index_document(request: IndexRequest):
    try:
        count = retriever.index_document(request.doc_id, request.images, request.texts)
        # Update BM25 texts
        if request.texts:
            items = []
            for i, _ in enumerate(request.images):
                page_num = i + 1
                page_id = f"{request.doc_id}:{page_num}"
                txt = request.texts[i] if i < len(request.texts) else None
                items.append((page_id, txt))
            bm25.bulk_upsert(items)
        return {"status": "success", "doc_id": request.doc_id, "images_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


def _rrf(scores: Dict[str, float], C: int = 60) -> Dict[str, float]:
    # Input: id -> rank (1-based). Convert to RRF score 1/(C+rank)
    out: Dict[str, float] = {}
    for i, (_id, rank) in enumerate(sorted(scores.items(), key=lambda x: x[1])):
        out[_id] = 1.0 / (C + float(rank))
    return out


@app.post("/search")
def search_documents(request: SearchRequest) -> SearchResponse:
    try:
        # ColPali
        dense_hits = retriever.search(request.query, k=request.k, doc_id=request.doc_id)
        dense_ids = [h.page_id for h in dense_hits]
        dense_rank = {pid: (i + 1) for i, pid in enumerate(dense_ids)}
        dense_rrf = _rrf(dense_rank) if dense_rank else {}

        # BM25 hybrid
        bm25_hits = bm25.search(request.query, k=request.k * 2, filter_prefix=request.doc_id)
        bm25_ids = [pid for pid, _ in bm25_hits]
        bm25_rank = {pid: (i + 1) for i, pid in enumerate(bm25_ids)}
        bm25_rrf = _rrf(bm25_rank) if bm25_rank else {}

        alpha = HYBRID_ALPHA
        fused: Dict[str, float] = {}
        all_ids = set(dense_ids) | set(bm25_ids)
        for pid in all_ids:
            fused[pid] = (1 - alpha) * dense_rrf.get(pid, 0.0) + alpha * bm25_rrf.get(pid, 0.0)

        # Build hit map from dense metadata if available, else fallback to BM25 store via index dir
        hit_meta: Dict[str, Dict[str, Any]] = {}
        for h in dense_hits:
            hit_meta[h.page_id] = {
                "doc_id": h.doc_id,
                "page": h.page_num,
                "image_path": h.image_path,
                "dense_score": h.score,
            }
        # For BM25-only entries, attempt to recover metadata from stored texts file name convention
        for pid in bm25_ids:
            if pid in hit_meta:
                continue
            # Try to infer paths under /data/pages/<doc>/<page>.png (convention)
            if ":" in pid:
                d_id, p = pid.split(":", 1)
                try:
                    pnum = int(p)
                except Exception:
                    pnum = 1
                # Assume pages stored as /data/pages/<doc_id>/<page:04d>.png, but fallback to any png
                img_dir = DATA_DIR / "pages" / d_id
                # Use the zero-padded convention first
                paths = [img_dir / f"{pnum:04d}.png", img_dir / f"{pnum}.png"]
                image_path = None
                for cand in paths:
                    if cand.exists():
                        image_path = str(cand.resolve())
                        break
                hit_meta[pid] = {
                    "doc_id": d_id,
                    "page": pnum,
                    "image_path": image_path or "",
                    "dense_score": 0.0,
                }

        # Order by fused score
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: request.k]
        results: List[Dict[str, Any]] = []
        for pid, fused_score in ordered:
            meta = hit_meta.get(pid, {})
            image_path = meta.get("image_path", "")
            # Heatmap path (placeholder deterministic overlay)
            if image_path:
                hm_out = HEATMAPS_DIR / meta.get("doc_id", "") / f"{meta.get('page', 1):04d}.png"
                heatmap_path = generate_placeholder_heatmap(request.query, image_path, str(hm_out)) or ""
            else:
                heatmap_path = ""
            results.append(
                {
                    "page_id": pid,
                    "doc_id": meta.get("doc_id", ""),
                    "page": int(meta.get("page", 1)),
                    "image_path": image_path,
                    "heatmap_path": heatmap_path,
                    "score": float(fused_score),
                }
            )
        return SearchResponse(hits=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


