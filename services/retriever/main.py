from __future__ import annotations

import os
import base64
from io import BytesIO
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

try:
    from transformers import AutoProcessor, AutoModel
except Exception as _e:  # pragma: no cover - only used in new service container
    AutoProcessor = None  # type: ignore
    AutoModel = None  # type: ignore

try:
    from pymilvus import MilvusClient, DataType
except Exception as _e:  # pragma: no cover
    MilvusClient = None  # type: ignore
    DataType = None  # type: ignore


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

app = FastAPI(title="Retriever (Milvus)", version="0.1.0")


# --- Pydantic Models ---
class IngestItem(BaseModel):
    page_id: str
    image_b64: str


class IngestRequest(BaseModel):
    pages: List[IngestItem]


class SearchQuery(BaseModel):
    # Support both new (text/top_k) and legacy (query/k) fields; include optional doc_id
    text: str | None = None
    query: str | None = None
    top_k: int | None = None
    k: int | None = None
    doc_id: str | None = None


# --- Helpers ---
def _pil_from_b64(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


@torch.no_grad()
def _get_embeddings(proc: Any, mdl: Any, inputs: Dict[str, Any]) -> torch.Tensor:
    """Return token-level embeddings for either text or image inputs.

    Supports common HF vision-text models like CLIP/SigLIP where token-level
    outputs live under text_model_output/vision_model_output, and falls back
    to top-level last_hidden_state when available.
    """
    outputs = mdl(**inputs)
    tokens = None
    # Prefer explicit branches when we know the modality
    if "pixel_values" in inputs and hasattr(outputs, "vision_model_output"):
        vout = getattr(outputs, "vision_model_output")
        if hasattr(vout, "last_hidden_state"):
            tokens = vout.last_hidden_state
    if tokens is None and "input_ids" in inputs and hasattr(outputs, "text_model_output"):
        tout = getattr(outputs, "text_model_output")
        if hasattr(tout, "last_hidden_state"):
            tokens = tout.last_hidden_state
    # Generic fallback
    if tokens is None and hasattr(outputs, "last_hidden_state"):
        tokens = outputs.last_hidden_state
    if tokens is None:
        # Last resort: try tuple-like access
        try:
            tokens = outputs[0]
        except Exception:
            raise RuntimeError("Model output does not provide token-level representations")
    return tokens.squeeze(0)


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, p=2, dim=-1)


def _maxsim_score(q_tokens: torch.Tensor, d_tokens: torch.Tensor) -> float:
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
    # Prefer zero-padded naming
    candidates = [
        os.path.join(DATA_DIR, "pages", doc_id, f"{page:04d}.png"),
        os.path.join(DATA_DIR, "pages", doc_id, f"{page}.png"),
    ]
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
        if AutoProcessor is None or AutoModel is None:
            raise RuntimeError("transformers not available in this environment")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
    return _processor, _model


def _ensure_milvus(dimension: int):
    global _client
    if MilvusClient is None:
        raise RuntimeError("pymilvus not available in this environment")
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
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


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "milvus": MILVUS_URI,
        "collection": COLLECTION_NAME,
    }


@app.post("/ingest")
def ingest(body: IngestRequest):
    try:
        proc, mdl = _ensure_model()
        # Compute one forward pass to discover embedding dimension
        if not body.pages:
            return {"status": "no-op", "pages_added": 0}
        sample_img = _pil_from_b64(body.pages[0].image_b64)
        # Include a minimal text prompt to satisfy models that require input_ids
        sample_inputs = proc(images=sample_img, text=INGEST_TEXT_PROMPT, return_tensors="pt").to(DEVICE)
        sample_tokens = _get_embeddings(proc, mdl, sample_inputs)
        dimension = int(sample_tokens.shape[-1])
        client = _ensure_milvus(dimension)
        # Ingest all pages (also persist images under DATA_DIR for API serving)
        total = 0
        for item in body.pages:
            img = _pil_from_b64(item.image_b64)
            # Persist image
            out_path = _image_path_for_page_id(item.page_id)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img.save(out_path)
            # Always include a minimal text prompt to ensure input_ids exist
            inputs = proc(images=img, text=INGEST_TEXT_PROMPT, return_tensors="pt").to(DEVICE)
            d_tokens = _get_embeddings(proc, mdl, inputs)  # [T, D]
            entities = [
                {"vector": vec.detach().cpu().numpy().tolist(), "page_id": item.page_id}
                for vec in d_tokens
            ]
            if entities:
                client.insert(collection_name=COLLECTION_NAME, data=entities)
                total += 1
        return {"status": "ingestion complete", "pages_added": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@app.post("/search")
def search(q: SearchQuery):
    try:
        proc, mdl = _ensure_model()
        # Resolve query text and top_k
        text = q.text or q.query or ""
        top_k = q.top_k or q.k or TOP_K_DEFAULT
        # Encode query and build probe vector (mean)
        inputs = proc(text=text, return_tensors="pt").to(DEVICE)
        q_tokens = _get_embeddings(proc, mdl, inputs)  # [Tq, D]
        probe = q_tokens.mean(dim=0, keepdim=True).detach().cpu().numpy().tolist()
        # Ensure Milvus client with correct dim
        client = _ensure_milvus(int(q_tokens.shape[-1]))
        # ANN search to get candidate pool
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=probe,
            limit=max(1, top_k) * 20,
            output_fields=["page_id"],
        )
        # Extract unique candidate page_ids
        candidate_pages = set()
        try:
            for res_list in res:
                for hit in res_list:
                    ent = hit.get("entity") or {}
                    pid = ent.get("page_id")
                    if pid:
                        candidate_pages.add(pid)
        except Exception:
            pass

        # Rerank with MaxSim per page by fetching all token vectors
        scored_pages: List[Dict[str, Any]] = []
        for pid in candidate_pages:
            rows = client.query(
                collection_name=COLLECTION_NAME,
                filter=f"page_id == '{pid}'",
                output_fields=["vector"],
            )
            if not rows:
                continue
            try:
                d_tokens = torch.tensor([r["vector"] for r in rows], device=DEVICE, dtype=torch.float32)
            except Exception:
                # Fallback if rows are dicts nested differently
                vectors = []
                for r in rows:
                    v = r.get("vector") if isinstance(r, dict) else None
                    if v is not None:
                        vectors.append(v)
                if not vectors:
                    continue
                d_tokens = torch.tensor(vectors, device=DEVICE, dtype=torch.float32)
            score = _maxsim_score(q_tokens, d_tokens)
            scored_pages.append({"page_id": pid, "score": float(score)})

        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        # Optional filter by doc_id if provided
        if q.doc_id:
            scored_pages = [h for h in scored_pages if h.get("page_id", "").startswith(f"{q.doc_id}:")]
        return {"hits": scored_pages[: max(1, top_k)]}
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
    texts: List[str] | None = None


@app.post("/index")
def index_pages(req: IndexRequest):
    """Compatibility with existing API's indexing contract.

    Expects absolute file paths under a shared DATA_DIR volume. Derives page_id
    as "{doc_id}:{page_num}" where page_num is based on list order (1-based).
    """
    try:
        proc, mdl = _ensure_model()
        if not req.images:
            return {"status": "success", "doc_id": req.doc_id, "images_indexed": 0}
        # Discover dimension
        first_img = Image.open(req.images[0]).convert("RGB")
        tokens = _get_embeddings(proc, mdl, proc(images=first_img, return_tensors="pt").to(DEVICE))
        client = _ensure_milvus(int(tokens.shape[-1]))
        count = 0
        for idx, path in enumerate(req.images, start=1):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            page_id = f"{req.doc_id}:{idx}"
            # Ensure persisted copy exists under DATA_DIR for serving
            out_path = _image_path_for_page_id(page_id)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                img.save(out_path)
            except Exception:
                pass
            d_tokens = _get_embeddings(proc, mdl, proc(images=img, return_tensors="pt").to(DEVICE))
            entities = [
                {"vector": vec.detach().cpu().numpy().tolist(), "page_id": page_id}
                for vec in d_tokens
            ]
            if entities:
                client.insert(collection_name=COLLECTION_NAME, data=entities)
                count += 1
        return {"status": "success", "doc_id": req.doc_id, "images_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.retriever.main:app", host="0.0.0.0", port=8081, reload=False)

