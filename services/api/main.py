from __future__ import annotations

import os
import base64
from io import BytesIO
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from libs.clients.vlm_client import vision_chat


# --- Configuration ---
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8081").rstrip("/")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
MAX_IMAGES_DEFAULT = int(os.getenv("MAX_IMAGES", os.getenv("VLM_MAX_IMAGES", "3")))
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5-vl:7b")

app = FastAPI(title="RAG Vision API Gateway", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    max_images: Optional[int] = None


def _pil_from_b64(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def _get_page_image_b64(page_id: str) -> str:
    url = f"{RETRIEVER_URL}/page_image/{page_id}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Page image not found for {page_id}")
    r.raise_for_status()
    data = r.json()
    return data.get("image_b64") or data.get("image") or ""


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "retriever_url": RETRIEVER_URL,
        "vlm_base_url": VLM_BASE_URL,
        "vlm_model": VLM_MODEL,
        "top_k_default": TOP_K_DEFAULT,
        "max_images_default": MAX_IMAGES_DEFAULT,
    }


@app.post("/query")
def query(req: QueryRequest):
    try:
        top_k = req.top_k or TOP_K_DEFAULT
        max_images = req.max_images or MAX_IMAGES_DEFAULT

        # 1) Retrieve candidate hits from retriever
        sr = requests.post(
            f"{RETRIEVER_URL}/search",
            json={"text": req.question, "top_k": int(max(1, top_k))},
            timeout=120,
        )
        sr.raise_for_status()
        hits: List[Dict[str, Any]] = sr.json().get("hits", [])

        # 2) Fetch top page images from retriever and build PIL list
        images: List[Image.Image] = []
        for hit in hits[: max_images]:
            pid = hit.get("page_id")
            if not pid:
                continue
            try:
                b64 = _get_page_image_b64(pid)
                if b64:
                    images.append(_pil_from_b64(b64))
            except Exception:
                continue

        # 3) Ask the VLM with images context
        if not images:
            # Always attempt to answer, even without images
            images = []
        answer = vision_chat(req.question, images, temperature=0.1, max_tokens=1024)

        return {
            "answer": answer,
            "hits": hits[: max_images],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8080, reload=False)


