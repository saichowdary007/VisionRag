from __future__ import annotations
import io
import json
import os
from pathlib import Path
from typing import List, Optional, Any
import requests
from fastapi import FastAPI, UploadFile, File, Body, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .settings import get_settings
from .schemas import IngestBody, IngestResponse, AskBody
from .pdf import render_pdf_to_images
from . import retriever_client
from .vlm_client import chat_vision
s = get_settings()
app = FastAPI(title="Vision-RAG API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.WEB_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Ensure data directories
DATA_DIR = Path(s.DATA_DIR)
PAGES_DIR = DATA_DIR / "pages"
HEATMAPS_DIR = DATA_DIR / "heatmaps"
DOCS_DIR = DATA_DIR / "docs"
PAGES_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
# Serve page images
app.mount("/pages", StaticFiles(directory=str(PAGES_DIR)), name="pages")
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAPS_DIR)), name="heatmaps")
@app.get("/healthz")
def healthz():
    # Try to fetch retriever and VLM health; do not fail hard.
    info: dict[str, Any] = {"ok": True}
    # Retriever
    try:
        r = requests.get(f"{s.RETRIEVER_BASE_URL.rstrip('/')}/healthz", timeout=5)
        info["retriever"] = r.json()
    except Exception as e:
        info["retriever"] = {"ok": False, "error": str(e)}
    # VLM models
    try:
        headers = {"Authorization": f"Bearer {s.VLM_API_KEY}"}
        base = s.VLM_BASE_URL.rstrip("/")
        url = base if base.endswith("/v1") else f"{base}/v1"
        r = requests.get(f"{url}/models", headers=headers, timeout=5)
        info["vlm"] = {"ok": r.ok, "status": r.status_code}
    except Exception as e:
        info["vlm"] = {"ok": False, "error": str(e)}
    return info
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    pdf_file: UploadFile | None = File(default=None),
    pdf_url: Optional[str] = Body(default=None),
    doc_id: Optional[str] = Body(default=None),
):
    content_type = request.headers.get("content-type", "")
    pdf_path: Optional[Path] = None
    if "multipart/form-data" in content_type:
        if not pdf_file:
            raise HTTPException(status_code=400, detail="Missing file upload 'pdf_file'")
        name = pdf_file.filename or "document.pdf"
        stem = doc_id or Path(name).stem
        pdf_path = DOCS_DIR / f"{stem}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(await pdf_file.read())
    else:
        # JSON body
        try:
            body = await request.json()
        except Exception:
            body = {}
        pdf_url = body.get("pdf_url") or pdf_url
        doc_id = body.get("doc_id") or doc_id
        if not pdf_url:
            raise HTTPException(status_code=400, detail="Provide 'pdf_url' or multipart 'pdf_file'")
        # Download
        r = requests.get(pdf_url, stream=True, timeout=120)
        r.raise_for_status()
        stem = doc_id or Path(pdf_url.split("?")[0]).stem
        pdf_path = DOCS_DIR / f"{stem}.pdf"
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    assert pdf_path is not None
    # Render pages
    did, images, texts = render_pdf_to_images(str(pdf_path), str(PAGES_DIR), max_dim=s.PAGE_MAX_DIM)
    # Index (pass page texts for hybrid BM25 fallback/search)
    retriever_client.index_pages(did, images, texts=texts)
    # Persist a tiny manifest
    try:
        manifest_dir = DATA_DIR / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / f"{did}.json").write_text(
            json.dumps({"doc_id": did, "pages": len(images)}, ensure_ascii=False)
        )
    except Exception:
        pass
    # Build public image URLs for response
    public_images = []
    for img_path in images:
        # Expecting: {PAGES_DIR}/{doc_id}/{file}
        p = Path(img_path)
        try:
            # find relative to PAGES_DIR
            rel = p.relative_to(PAGES_DIR)
        except Exception:
            rel = p.name
        public_images.append(f"{s.API_PUBLIC_BASE.rstrip('/')}/pages/{rel}")
    return IngestResponse(ok=True, doc_id=did, pages=len(images), images=public_images)
def _stream_vlm(question: str, vlm_urls: List[str], meta: Dict[str, Any]):
    # Yield a first line with meta (images, sources) for the UI to parse
    print(f"DEBUG: _stream_vlm called with {len(vlm_urls)} images", flush=True)
    yield json.dumps({"type": "meta", **meta}) + "\n"
    try:
        # Try streaming first, fallback to non-streaming if needed
        try:
            with chat_vision(question, vlm_urls, stream=True) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        try:
                            raw = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                    if not raw.startswith("data:"):
                        continue
                    data = raw[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    # OpenAI-style stream delta
                    try:
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        # Fallback: yield raw
                        pass
        except Exception:
            # Fallback to non-streaming request
            print("Falling back to non-streaming VLM request", flush=True)
            r = chat_vision(question, vlm_urls, stream=False)
            r.raise_for_status()
            result = r.json()
            try:
                content = result["choices"][0]["message"]["content"]
                yield content
            except (KeyError, IndexError):
                yield "VLM returned an unexpected response format.\n"
    except Exception as e:
        # Gracefully degrade when VLM is unavailable
        print(f"VLM Error: {str(e)}", flush=True)  # Debug logging
        yield "[VLM unavailable] "
        yield "The vision model service could not be reached.\n"
        yield "You can still inspect the retrieved pages above.\n"
@app.post("/ask")
def ask(body: AskBody):
    # Retrieve images
    hits = retriever_client.search(body.question, k=body.k, doc_id=body.doc_id)
    if not hits:
        def _no_hits():
            yield "No relevant pages found."
        return StreamingResponse(_no_hits(), media_type="text/plain")
    # Top-m
    # Cap image count sent to VLM
    m = min(max(1, int(body.m)), s.VLM_MAX_IMAGES)
    top = hits[: m]

    vlm_urls: List[str] = []
    sources: List[Dict[str, Any]] = []
    public_images: List[str] = []
    for h in top:
        ipath = h.get("image_path", "")
        p = Path(ipath)
        try:
            rel = p.relative_to(PAGES_DIR)
        except Exception:
            rel = p.name
        img_public = f"{s.API_PUBLIC_BASE.rstrip('/')}/pages/{rel}"
        public_images.append(img_public)
        vlm_urls.append(f"{s.API_INTERNAL_BASE.rstrip('/')}/pages/{rel}")
        # Heatmap URL if present
        hmpath = h.get("heatmap_path", "")
        heat_url = None
        if hmpath:
            hp = Path(hmpath)
            try:
                hrel = hp.relative_to(HEATMAPS_DIR)
            except Exception:
                hrel = hp.name
            heat_url = f"{s.API_PUBLIC_BASE.rstrip('/')}/heatmaps/{hrel}"
        sources.append(
            {
                "doc_id": h.get("doc_id"),
                "page": h.get("page"),
                "score": h.get("score"),
                "image_url": img_public,
                "heatmap_url": heat_url,
            }
        )

    meta = {"images": public_images, "sources": sources}
    return StreamingResponse(_stream_vlm(body.question, vlm_urls, meta), media_type="text/plain")
