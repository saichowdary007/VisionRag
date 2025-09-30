"""VLM RAG API (Corrected): ColPali multi-vector + Milvus + Ollama (OpenAI-compatible)"""
import os
import base64
from pathlib import Path
from typing import Dict, Any, List

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from pdf2image import convert_from_path
import openai

# --- Environment / Config ---
MILVUS_URI = os.getenv("MILVUS_URI", "./storage/milvus.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_vision_collection")
QA_API_BASE = os.getenv("QA_API_BASE", "http://host.docker.internal:11434/v1")
QA_API_KEY = os.getenv("QA_API_KEY", "ollama")
QA_MODEL_ID = os.getenv("QA_MODEL_ID", "qwen2.5-vl:7b")
RETRIEVER_MODEL_ID = os.getenv("RETRIEVER_MODEL_ID", "vidore/colpali-v1.3")
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "50"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))
IMAGE_DPI = int(os.getenv("IMAGE_DPI", "200"))
DIMENSION = 128  # ColPali outputs 128-dim patch vectors

# Device selection (Apple MPS -> CUDA -> CPU)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# --- Directories ---
ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"
PAGES_DIR = ROOT / "pages"
DOCUMENTS_DIR.mkdir(exist_ok=True)
PAGES_DIR.mkdir(exist_ok=True)

# --- FastAPI ---
app = FastAPI(title="VLM RAG API", version="1.0.0")
app.mount("/pages", StaticFiles(directory=PAGES_DIR), name="pages")

@app.get("/")
async def serve_index() -> HTMLResponse:
    index_path = ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(index_path.read_text(), media_type="text/html")

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# --- Milvus (new MilvusClient API) ---
from pymilvus import MilvusClient, DataType
_milvus: MilvusClient | None = None

def milvus() -> MilvusClient:
    global _milvus
    if _milvus is None:
        _milvus = MilvusClient(uri=MILVUS_URI)
    return _milvus

def ensure_collection():
    client = milvus()
    if client.has_collection(collection_name=COLLECTION_NAME):
        return
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field("page_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
    index_params = MilvusClient.prepare_index_params()
    # Inner Product (dot) is standard for late interaction scoring
    index_params.add_index(field_name="vector", metric_type="IP")
    milvus().create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)

# --- ColPali retriever ---
from colpali_engine.models import ColPali, ColPaliProcessor
_model: ColPali | None = None
_proc: ColPaliProcessor | None = None

def retriever():
    global _model, _proc
    if _model is None:
        _model = ColPali.from_pretrained(RETRIEVER_MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE).eval()
        _proc = ColPaliProcessor.from_pretrained(RETRIEVER_MODEL_ID)
    return _model, _proc

# --- OpenAI-compatible client (Ollama / vLLM) ---
_openai = None

def oa():
    global _openai
    if _openai is None:
        _openai = openai.OpenAI(base_url=QA_API_BASE, api_key=QA_API_KEY)
    return _openai

class QueryRequest(BaseModel):
    query: str

# --- Embedding helpers ---
@torch.no_grad()
def embed_page(image: Image.Image):
    model, proc = retriever()
    inputs = proc.process_images([image]).to(DEVICE)
    # Output: (1, 1024, 128) -> return (1024, 128)
    return _to_cpu(model(**inputs))[0]

@torch.no_grad()
def embed_query(text: str):
    model, proc = retriever()
    q_inputs = proc.process_queries([text]).to(DEVICE)
    # Output: (1, Q, 128)
    return _to_cpu(model(**q_inputs))

def _to_cpu(t):
    import torch as _t
    return t.detach().to("cpu").numpy()

# --- Upload: PDF -> pages -> multi-vector insert ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    ensure_collection()
    pdf_path = DOCUMENTS_DIR / file.filename
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    try:
        images = convert_from_path(pdf_path, dpi=IMAGE_DPI)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    doc_id = Path(file.filename).stem
    total_vectors = 0
    for i, img in enumerate(images):
        page_name = f"{doc_id}_page_{i+1}.png"
        page_path = PAGES_DIR / page_name
        img.save(page_path, "PNG")
        vecs = embed_page(img)  # (1024, 128)
        entities = [
            {"vector": vecs[j].tolist(), "page_path": str(page_path), "doc_id": doc_id}
            for j in range(vecs.shape[0])
        ]
        milvus().insert(collection_name=COLLECTION_NAME, data=entities)
        total_vectors += vecs.shape[0]

    return {"filename": file.filename, "pages": len(images), "vectors_inserted": total_vectors}

# --- Ask: retrieve + MaxSim rerank + VLM answer ---
@app.post("/ask")
async def ask_question(req: QueryRequest) -> Dict[str, Any]:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    ensure_collection()
    # 1) Encode query (1, Q, 128)
    q_vecs = embed_query(q)

    # 2) First-stage retrieval: mean-pooled probe -> patch hits
    import numpy as _np
    probe = _np.mean(q_vecs, axis=1)  # (1, 128)
    hits = milvus().search(
        collection_name=COLLECTION_NAME,
        data=probe,
        limit=TOP_K_CANDIDATES * 1024,
        output_fields=["page_path", "doc_id"],
    )

    # Collect candidate unique pages
    candidate_pages: List[str] = []
    seen = set()
    for h in hits:
        page_path = h["entity"]["page_path"]
        if page_path not in seen:
            seen.add(page_path)
            candidate_pages.append(page_path)
        if len(candidate_pages) >= TOP_K_CANDIDATES:
            break

    if not candidate_pages:
        return {"answer": "No relevant documents found.", "sources": []}

    # 3) Rerank with MaxSim using stored patch vectors
    from colpali_engine.models import ColPaliProcessor  # for score function
    _, proc = retriever()
    scores: Dict[str, float] = {}
    for p in candidate_pages:
        rows = milvus().query(
            collection_name=COLLECTION_NAME,
            filter=f"page_path == '{p}'",
            output_fields=["vector"],
        ) or []
        if not rows:
            continue
        import numpy as _np
        import torch as _t
        page_vecs = _t.tensor(_np.array([r["vector"] for r in rows], dtype=_np.float32)).unsqueeze(0).to(DEVICE)  # (1, 1024, 128)
        q_t = _t.tensor(q_vecs, dtype=_t.float32).to(DEVICE)
        s = proc.score_multi_vector(q_t, page_vecs)  # scalar
        scores[p] = float(s.item())

    if not scores:
        return {"answer": "No valid page vectors found for reranking.", "sources": []}

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_FINAL]
    top_paths = [p for p, _ in top]

    # 4) Build VLM message with images (use base64 URLs for OpenAI-compatible payload)
    def img_to_b64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    content = [{"type": "text", "text": q}]
    for p in top_paths:
        b64 = img_to_b64(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    try:
        resp = oa().chat.completions.create(
            model=QA_MODEL_ID,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {e}")

    sources = [{
        "page": Path(p).name,
        "score": scores[p],
        "source_url": f"/pages/{Path(p).name}",
    } for p in top_paths]

    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
