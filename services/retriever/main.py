from __future__ import annotations

import os
import time
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

# Optional heavy dependency; allow running in mock mode without torch
try:
    import torch  # type: ignore
except Exception:  # torch is optional
    torch = None  # type: ignore

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus:19530")

# Mode: 'mock' (default) or 'milvus-lite'
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "mock").strip().lower()

# Keep optional heavy deps None to support mock mode by default
AutoProcessor = None
AutoModel = None
MilvusClient = None
DataType = None


# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "vidore/colpali-v1.3")
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "./milvus_data/milvus.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "colpali_multivector_collection")
DEVICE = os.getenv("DEVICE", "cpu")
DATA_DIR = os.getenv("DATA_DIR", "/data")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
# Some vision-text models require text tokens even for image-only forward passes.
# Provide a minimal prompt to ensure input_ids are present when encoding images.
INGEST_TEXT_PROMPT = os.getenv("INGEST_TEXT_PROMPT", " ")
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "./local_data")

# Milvus-lite simple embedding dimension
EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))

app = FastAPI(title="Retriever (Milvus)", version="0.1.0")

# ---------------- In-memory registry for mock mode ----------------
_PAGES: Dict[str, str] = {}  # page_id -> image path
_ORDER: List[str] = []       # insertion order for simple ranking


def _register_page(page_id: str, img: Image.Image) -> str:
    out_path = _image_path_for_page_id(page_id)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
    except Exception:
        # Fallback to a local writable directory if /data is not writable
        doc_id, page = _parse_page_id(page_id)
        out_path = os.path.join(LOCAL_DATA_DIR, "pages", doc_id, f"{page:04d}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            img.save(out_path)
        except Exception:
            # Silent best-effort; still register in memory
            pass
    _PAGES[page_id] = out_path
    if page_id not in _ORDER:
        _ORDER.append(page_id)
    return out_path


class IngestItem(BaseModel):
    page_id: str
    image_b64: str


class IngestPayload(BaseModel):
    pages: List[IngestItem]


@app.post("/ingest")
def ingest(body: Dict[str, Any]):
    """Accepts either:
    - { pages: [{ page_id, image_b64 }] }
    - { doc_id: str, images: List[str] }  # absolute paths (compat)
    Registers pages in-memory and saves copies under DATA_DIR for /page_image.
    """
    try:
        pages_added = 0

        # Case 1: pages with base64 images
        if isinstance(body, dict) and isinstance(body.get("pages"), list):
            for p in body["pages"]:
                try:
                    page_id = p.get("page_id")
                    b64 = p.get("image_b64") or p.get("image")
                    if not page_id or not b64:
                        continue
                    img = _pil_from_b64(b64)
                    _register_page(page_id, img)
                    pages_added += 1
                except Exception:
                    continue
            return {"status": "ingestion complete", "pages_added": pages_added}

        # Case 2: compatibility: doc_id + images (absolute file paths)
        doc_id = body.get("doc_id") if isinstance(body, dict) else None
        images = body.get("images") if isinstance(body, dict) else None
        if doc_id and isinstance(images, list):
            for idx, path in enumerate(images, start=1):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                page_id = f"{doc_id}:{idx}"
                _register_page(page_id, img)
                pages_added += 1
            return {"status": "ingestion complete", "pages_added": pages_added}

        return {"status": "no-op", "pages_added": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

# ------------------------------------------------------------------

# --- Pydantic Models ---
class SearchQuery(BaseModel):
    # Support both new (text/top_k) and legacy (query/k) fields; include optional doc_id
    text: Optional[str] = None
    query: Optional[str] = None
    top_k: Optional[int] = None
    k: Optional[int] = None
    doc_id: Optional[str] = None


# --- Helpers ---
def _pil_from_b64(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def _get_embeddings(proc: Any, mdl: Any, inputs: Dict[str, Any]):
    """Return token-level embeddings for either text or image inputs.

    Supports common HF vision-text models like CLIP/SigLIP where token-level
    outputs live under text_model_output/vision_model_output, and falls back
    to top-level last_hidden_state when available.
    """
    if torch is None:
        raise RuntimeError("Torch not available for embeddings")
    @torch.no_grad()
    def _forward():
        outputs = mdl(**inputs)
        tokens = None
        # Prefer explicit branches when we know the modality
        if "pixel_values" in inputs and hasattr(outputs, "vision_model_output"):
            vout = getattr(outputs, "vision_model_output")
            if hasattr(vout, "last_hidden_state"):
                return vout.last_hidden_state
        if tokens is None and "input_ids" in inputs and hasattr(outputs, "text_model_output"):
            tout = getattr(outputs, "text_model_output")
            if hasattr(tout, "last_hidden_state"):
                return tout.last_hidden_state
        # Generic fallback
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        try:
            return outputs[0]
        except Exception:
            raise RuntimeError("Model output does not provide token-level representations")
    tokens = _forward()
    return tokens.squeeze(0)


def _l2_normalize(t):
    if torch is None:
        raise RuntimeError("Torch not available for normalization")
    return torch.nn.functional.normalize(t, p=2, dim=-1)


def _maxsim_score(q_tokens, d_tokens) -> float:
    if torch is None:
        return 0.0
    q_norm, d_norm = _l2_normalize(q_tokens), _l2_normalize(d_tokens)
    sim = torch.matmul(q_norm, d_norm.transpose(0, 1))
    max_sims = sim.max(dim=1).values
    return float(max_sims.sum().detach().cpu().item())


def _parse_page_id(page_id: str) -> tuple[str, int]:
    if ":" in page_id:
        d, p = page_id.split(":", 1)
        try:
            return d, int(p)
        except Exception:
            return d, 1
    return page_id, 1


def _image_path_for_page_id(page_id: str) -> str:
    doc_id, page = _parse_page_id(page_id)
    bases = [DATA_DIR, LOCAL_DATA_DIR]
    # Prefer zero-padded naming
    candidates = []
    for base in bases:
        candidates.append(os.path.join(base, "pages", doc_id, f"{page:04d}.png"))
        candidates.append(os.path.join(base, "pages", doc_id, f"{page}.png"))
    for c in candidates:
        if os.path.exists(c):
            return c
    # Default to first candidate path (may not exist yet)
    return candidates[0]


# --- Lazy singletons (to avoid import costs unless the service runs) ---
_processor = None
_model = None
_client = None
_dim = None


def _ensure_model():
    global _processor, _model
    if _processor is None or _model is None:
        if AutoProcessor is None or AutoModel is None or torch is None:
            raise RuntimeError("ML stack not available in this environment")
        # Keep disabled to avoid heavy downloads in typical dev
        raise RuntimeError("Model loading disabled (mock mode)")
    return _processor, _model


def _connect_milvus_with_retries(retries=10, delay=2.0):
    """Connect to Milvus with retries to handle startup timing issues."""
    global MilvusClient, DataType
    if MilvusClient is None or DataType is None:
        try:
            from pymilvus import MilvusClient as _MC, DataType as _DT  # type: ignore
            MilvusClient = _MC
            DataType = _DT
        except Exception as e:
            raise RuntimeError(f"pymilvus not available in this environment: {e}")

    for i in range(retries):
        try:
            client = MilvusClient(uri=MILVUS_URI)
            # Test the connection by checking if we can list collections
            client.list_collections()
            return client
        except Exception as e:
            if i == retries - 1:
                raise RuntimeError(f"Failed to connect to Milvus after {retries} attempts: {e}")
            print(f"Milvus connection attempt {i+1}/{retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def _ensure_milvus(dimension: int):
    global _client
    if MilvusClient is None or DataType is None:
        _connect_milvus_with_retries(retries=1, delay=0)
    if _client is None:
        _client = _connect_milvus_with_retries()
        # Create collection if needed
        if not _client.has_collection(collection_name=COLLECTION_NAME):
            schema = _client.create_schema(auto_id=True, description="ColPali multi-vector store")
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="page_id", datatype=DataType.VARCHAR, max_length=1024)
            index_params = _client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="IP")
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params=index_params,
                consistency_level="Eventually",
            )
    return _client


# ----------------- Milvus-lite simple embeddings (no ML) -----------------
def _img_to_vec(img: Image.Image, dim: int = EMBED_DIM) -> np.ndarray:
    # dim must be a square number (e.g., 64 = 8x8)
    side = int(np.sqrt(dim))
    if side * side != dim:
        side = int(np.sqrt(64))
    img = img.convert("L").resize((side, side))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    vec = arr.flatten()
    # L2 normalize
    n = np.linalg.norm(vec) + 1e-8
    return (vec / n).astype(np.float32)


def _text_to_vec(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    # Simple hashing trick into dim buckets
    import hashlib
    buckets = np.zeros((dim,), dtype=np.float32)
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
        buckets[h % dim] += 1.0
    if buckets.sum() > 0:
        buckets /= np.linalg.norm(buckets) + 1e-8
    return buckets


@app.get("/healthz")
def healthz():
    """Simple health check."""
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "milvus": MILVUS_URI,
        "collection": COLLECTION_NAME,
        "mode": RETRIEVER_MODE,
        "milvus_connected": False,  # Real connection disabled in mock mode
        "pages_indexed": len(_ORDER),
        "message": "Health check working"
    }


@app.post("/search")
def search(q: SearchQuery):
    """Search endpoint.

    Modes:
    - mock: deterministic ranking over ingested pages (no external deps)
    - milvus-lite: use Milvus with simple numpy embeddings (no ML)
    """
    try:
        top_k = int(q.top_k or q.k or int(os.getenv("TOP_K", "5")))
        text = (q.text or q.query or "").strip()

        if RETRIEVER_MODE == "milvus-lite":
            # Try Milvus vector search with simple text embedding
            try:
                client = _ensure_milvus(EMBED_DIM)
                qvec = _text_to_vec(text, EMBED_DIM).tolist()
                res = client.search(
                    collection_name=COLLECTION_NAME,
                    data=[qvec],
                    anns_field="vector",
                    limit=max(1, top_k),
                    output_fields=["page_id"],
                )
                # pymilvus client returns list per query
                out = []
                for hit in (res[0] if isinstance(res, list) else []):
                    out.append({"page_id": hit["entity"]["page_id"], "score": float(hit["distance"])})
                return {"hits": out}
            except Exception:
                # Fall through to mock ranking
                pass

        # mock ranking
        def score_for(pid: str) -> float:
            if not text:
                return float(len(_ORDER) - _ORDER.index(pid)) if pid in _ORDER else 0.0
            doc_id, page_num = _parse_page_id(pid)
            s = 0.0
            if doc_id and doc_id in text:
                s += 1.0
            tokens = set(filter(None, [t.lower() for t in text.split()]))
            if str(page_num) in tokens:
                s += 0.2
            if doc_id:
                dl = doc_id.lower()
                for t in tokens:
                    if t in dl:
                        s += 0.1
            if pid in _ORDER:
                s += 0.01 * (len(_ORDER) - _ORDER.index(pid))
            return s

        candidates = list(_PAGES.keys())
        if not candidates:
            # Discover existing assets on disk if present
            base_dirs = [os.path.join(DATA_DIR, "pages"), os.path.join(LOCAL_DATA_DIR, "pages")]
            for base in base_dirs:
                for root, _dirs, files in os.walk(base):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            parts = os.path.normpath(root).split(os.sep)
                            if len(parts) >= 1:
                                doc_id = parts[-1]
                                page = os.path.splitext(f)[0]
                                pid = f"{doc_id}:{page}"
                                _PAGES[pid] = os.path.join(root, f)
                                if pid not in _ORDER:
                                    _ORDER.append(pid)
            candidates = list(_PAGES.keys())

        ranked = sorted(candidates, key=score_for, reverse=True)
        hits = [{"page_id": pid, "score": float(score_for(pid))} for pid in ranked[: max(1, top_k)]]
        return {"hits": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.get("/page_image/{page_id}")
def get_page_image(page_id: str):
    """Return the stored page image as base64. Images are persisted during ingest/index.

    The page_id format is "{doc_id}:{page_num}".
    """
    try:
        path = _image_path_for_page_id(page_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Image not found for page_id={page_id}")
        img = Image.open(path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"page_id": page_id, "image_b64": b64}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {e}")


# --- Compatibility endpoints with existing API ---

class IndexRequest(BaseModel):
    doc_id: str
    images: List[str]
    texts: Optional[List[str]] = None


@app.post("/index")
def index_pages(req: IndexRequest):
    """Compatibility with existing API's indexing contract.

    Expects absolute file paths under a shared DATA_DIR volume. Derives page_id
    as "{doc_id}:{page_num}" where page_num is based on list order (1-based).
    """
    try:
        count = 0
        if RETRIEVER_MODE == "milvus-lite":
            try:
                client = _ensure_milvus(EMBED_DIM)
            except Exception:
                client = None
        else:
            client = None

        for idx, path in enumerate(req.images or [], start=1):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            page_id = f"{req.doc_id}:{idx}"
            _register_page(page_id, img)

            # If Milvus-lite enabled and connected, insert vector
            if client is not None:
                try:
                    vec = _img_to_vec(img, EMBED_DIM).tolist()
                    entities = [{"vector": vec, "page_id": page_id}]
                    client.insert(collection_name=COLLECTION_NAME, data=entities)
                except Exception:
                    # ignore vector store failures
                    pass
            count += 1

        return {"status": "success", "doc_id": req.doc_id, "images_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.retriever.main:app", host="0.0.0.0", port=8081, reload=False)
