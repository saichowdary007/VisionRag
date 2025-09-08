from __future__ import annotations
import os
from typing import List, Optional, Dict, Any
import requests
from .settings import get_settings
def _base() -> str:
    return get_settings().RETRIEVER_BASE_URL.rstrip("/")
def index_pages(doc_id: str, image_paths: List[str], texts: Optional[List[str]] = None) -> Dict[str, Any]:
    url = f"{_base()}/index"
    payload: Dict[str, Any] = {"doc_id": doc_id, "images": image_paths}
    if texts is not None:
        payload["texts"] = texts
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()
def search(query: str, k: int = 5, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    url = f"{_base()}/search"
    payload = {"query": query, "k": int(k)}
    if doc_id:
        payload["doc_id"] = doc_id
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json() or {}
    return data.get("hits", [])
