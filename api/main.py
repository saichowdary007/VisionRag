from __future__ import annotations
import io
import json
import os
from pathlib import Path
from typing import List, Optional
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
DOCS_DIR = DATA_DIR / "docs"
PAGES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
# Serve page images
app.mount("/pages", StaticFiles(directory=str(PAGES_DIR)), name="pages")
@app.get("/healthz")
def healthz():
    return {"ok": True}
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
    # Index (pass page texts for fallback search)
    retriever_client.index_pages(did, images, texts=texts)
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
def _stream_vlm(question: str, vlm_urls: List[str], public_urls: List[str]):
    # Yield a first line with used image URLs for the UI to optionally parse
    yield json.dumps({"type": "images", "images": public_urls}) + "\n"
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
    except Exception as e:
        # Gracefully degrade when VLM is unavailable
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
    top = hits[: max(1, int(body.m))]
    vlm_urls: List[str] = []
    public_urls: List[str] = []
    for h in top:
        p = Path(h.get("image_path"))
        try:
            rel = p.relative_to(PAGES_DIR)
        except Exception:
            rel = p.name
        vlm_urls.append(f"{s.API_INTERNAL_BASE.rstrip('/')}/pages/{rel}")
        public_urls.append(f"{s.API_PUBLIC_BASE.rstrip('/')}/pages/{rel}")
    return StreamingResponse(_stream_vlm(body.question, vlm_urls, public_urls), media_type="text/plain")
