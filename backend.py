from __future__ import annotations

import base64
import logging
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


# ----------------------------- Configuration -----------------------------
RETRIEVER_MODE: str = os.getenv("RETRIEVER_MODE", "mock").strip().lower()
TOP_K_DEFAULT: int = int(os.getenv("TOP_K", "5"))
MAX_IMAGES_DEFAULT: int = int(os.getenv("MAX_IMAGES", os.getenv("VLM_MAX_IMAGES", "3")))
DATA_DIR: str = os.getenv("DATA_DIR", "/data")
LOCAL_DATA_DIR: str = os.getenv("LOCAL_DATA_DIR", "./local_data")

VLM_BASE_URL: str = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
VLM_MODEL: str = os.getenv("VLM_MODEL", "qwen2.5-vl:7b")
VLM_TIMEOUT_SEC: int = int(os.getenv("VLM_TIMEOUT_SEC", "180"))


# ------------------------------- App/Logging ------------------------------
app = FastAPI(title="VisionRAG Monolith Backend", version="0.1.0")
logger = logging.getLogger("visionrag-monolith")
if not logger.handlers:
    try:
        from pythonjsonlogger import jsonlogger  # type: ignore

        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception:
        logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


# ------------------------------- Data Store -------------------------------
_PAGES: Dict[str, str] = {}  # page_id -> image file path
_ORDER: List[str] = []       # insertion order


# --------------------------------- Models ---------------------------------
class SearchQuery(BaseModel):
    text: Optional[str] = None
    query: Optional[str] = None
    top_k: Optional[int] = None
    k: Optional[int] = None
    doc_id: Optional[str] = None
    page: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    max_images: Optional[int] = None


# ------------------------------ Helper Utils ------------------------------
def _parse_page_id(page_id: str) -> Tuple[str, int]:
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
    candidates = []
    for base in bases:
        candidates.append(os.path.join(base, "pages", doc_id, f"{page:04d}.png"))
        candidates.append(os.path.join(base, "pages", doc_id, f"{page}.png"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _pil_from_b64(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def _register_page(page_id: str, img: Image.Image) -> str:
    out_path = _image_path_for_page_id(page_id)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
    except Exception:
        # Fallback to local writable directory
        doc_id, page = _parse_page_id(page_id)
        out_path = os.path.join(LOCAL_DATA_DIR, "pages", doc_id, f"{page:04d}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            img.save(out_path)
        except Exception:
            pass
    _PAGES[page_id] = out_path
    if page_id not in _ORDER:
        _ORDER.append(page_id)
    return out_path


# ------------------------------ VLM Client --------------------------------
def _pil_to_b64_jpeg(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _vlm_url() -> str:
    return VLM_BASE_URL if VLM_BASE_URL.endswith("/v1") else f"{VLM_BASE_URL}/v1"


def vision_chat(prompt: str, images: Iterable[Image.Image], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    images_b64 = [_pil_to_b64_jpeg(im) for im in images]
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{_vlm_url()}/chat/completions", json=payload, timeout=VLM_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


# ------------------------------- Endpoints --------------------------------
@app.get("/healthz")
def healthz():
    info: Dict[str, Any] = {
        "ok": True,
        "mode": RETRIEVER_MODE,
        "top_k_default": TOP_K_DEFAULT,
        "max_images_default": MAX_IMAGES_DEFAULT,
        "vlm_base_url": VLM_BASE_URL,
        "vlm_model": VLM_MODEL,
        "pages_indexed": len(_ORDER),
    }
    try:
        r = requests.get(f"{_vlm_url()}/models", timeout=2)
        info["vlm_connectivity"] = r.ok
    except Exception as e:
        info["vlm_connectivity"] = False
        info["vlm_error"] = str(e)
    return info


@app.post("/ingest")
def ingest(body: Dict[str, Any]):
    try:
        pages_added = 0
        logger.info(
            "ingest.start payload_pages=%d",
            len(body.get("pages", [])) if isinstance(body, dict) and isinstance(body.get("pages"), list) else 0,
        )

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
                except Exception as e:
                    logger.error("ingest.page.error page_id=%s error=%s", str(p.get("page_id")), str(e))
                    continue
            logger.info("ingest.complete pages_added=%d", pages_added)
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
            logger.info("ingest.complete pages_added=%d", pages_added)
            return {"status": "ingestion complete", "pages_added": pages_added}

        logger.info("ingest.noop")
        return {"status": "no-op", "pages_added": 0}
    except Exception as e:
        logger.error("ingest.error error=%s", str(e))
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@app.get("/page_image/{page_id}")
def get_page_image(page_id: str):
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


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


@app.post("/search")
def search(q: SearchQuery):
    try:
        top_k = int(q.top_k or q.k or TOP_K_DEFAULT)
        text = (q.text or q.query or "").strip()
        text_lower = text.lower()

        def score_for(pid: str) -> float:
            if not text:
                return float(len(_ORDER) - _ORDER.index(pid)) if pid in _ORDER else 0.0
            doc_id, page_num = _parse_page_id(pid)
            s = 0.0
            if doc_id:
                dl = doc_id.lower()
                toks = _tokenize(text)
                if dl in toks or dl in text_lower:
                    s += 1.0
                for t in toks:
                    if t and t in dl:
                        s += 0.1
            if str(page_num) in _tokenize(text):
                s += 0.2
            if pid in _ORDER:
                s += 0.01 * (len(_ORDER) - _ORDER.index(pid))
            return s

        candidates = list(_PAGES.keys())
        if q.doc_id:
            want = q.doc_id.lower()
            candidates = [pid for pid in candidates if _parse_page_id(pid)[0].lower() == want]
        if q.page is not None:
            try:
                pnum = int(q.page)
                candidates = [pid for pid in candidates if _parse_page_id(pid)[1] == pnum]
            except Exception:
                pass
        ranked = sorted(candidates, key=score_for, reverse=True)
        hits = [{"page_id": pid, "score": float(score_for(pid))} for pid in ranked[: max(1, top_k)]]
        return {"hits": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


def _get_page_image_b64(page_id: str) -> str:
    try:
        path = _image_path_for_page_id(page_id)
        if not os.path.exists(path):
            return ""
        img = Image.open(path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


@app.post("/query")
def query(req: QueryRequest):
    try:
        top_k = req.top_k or TOP_K_DEFAULT
        max_images = req.max_images or MAX_IMAGES_DEFAULT

        # 1) Retrieve candidate hits
        try:
            hits: List[Dict[str, Any]] = search(
                SearchQuery(text=req.question, top_k=max(1, int(top_k)))
            )["hits"]
        except Exception:
            hits = []

        # 2) Fetch top page images
        images: List[Image.Image] = []
        for hit in hits[: max_images]:
            pid = hit.get("page_id")
            if not pid:
                continue
            b64 = _get_page_image_b64(pid)
            if b64:
                try:
                    images.append(_pil_from_b64(b64))
                except Exception:
                    continue

        # 3) Ask VLM
        start = time.time()
        try:
            logger.info("vlm.call.start question_len=%d images=%d", len(req.question or ""), len(images))
            answer = vision_chat(req.question, images, temperature=0.1, max_tokens=1024)
            elapsed = time.time() - start
            logger.info("vlm.call.success elapsed_ms=%d", int(elapsed * 1000))
        except Exception as e:
            elapsed = time.time() - start
            logger.error("vlm.call.error elapsed_ms=%d error=%s", int(elapsed * 1000), str(e))
            raise HTTPException(status_code=502, detail=f"VLM call failed: {e}")

        return {"answer": answer, "hits": hits[: max_images]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8080, reload=False)


