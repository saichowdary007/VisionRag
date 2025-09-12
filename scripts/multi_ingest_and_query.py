#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import time
import base64
from io import BytesIO
from typing import List, Tuple

import requests
from PIL import Image


RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8080").rstrip("/")
API_URL = os.getenv("BACKEND_API_URL", os.getenv("API_URL", "http://localhost:8080")).rstrip("/")


def to_b64_jpeg(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def make_square(color: Tuple[int, int, int], size: int = 64) -> Image.Image:
    bg = (200, 200, 200)
    img = Image.new("RGB", (size, size), color=bg)
    # centered square
    m1, m2 = size // 4, size - size // 4
    for x in range(m1, m2):
        for y in range(m1, m2):
            img.putpixel((x, y), color)
    return img


def ensure_services() -> None:
    rh = requests.get(f"{RETRIEVER_URL}/healthz", timeout=5)
    print("retriever /healthz:", rh.status_code, rh.json())
    ah = requests.get(f"{API_URL}/healthz", timeout=5)
    print("api /healthz:", ah.status_code, ah.json())


def ingest_multi() -> None:
    # Create simple colored pages across multiple docs
    docs = [
        ("redDoc", [(1, (255, 0, 0))]),
        ("greenDoc", [(1, (0, 200, 0)), (2, (255, 215, 0))]),  # page 2 is yellow
        ("blueDoc", [(1, (0, 128, 255))]),
    ]
    pages = []
    for doc_id, pagespec in docs:
        for page_idx, rgb in pagespec:
            img = make_square(rgb)
            page_id = f"{doc_id}:{page_idx}"
            pages.append({"page_id": page_id, "image_b64": to_b64_jpeg(img)})
    # First run may download CLIP model; allow generous timeout
    r = requests.post(f"{RETRIEVER_URL}/ingest", json={"pages": pages}, timeout=180)
    r.raise_for_status()
    print("ingest_multi:", r.status_code, r.json())


def do_search(text: str, top_k: int = 5):
    r = requests.post(f"{RETRIEVER_URL}/search", json={"text": text, "top_k": top_k}, timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"search [{text!r}]:", json.dumps(data, indent=2))
    return data.get("hits", [])


def ask_api(question: str, top_k: int = 5, max_images: int = 1, timeout: int = 60):
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": top_k, "max_images": max_images},
        timeout=timeout,
    )
    print(f"api /query [{question!r}]:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


def main():
    ensure_services()
    ingest_multi()
    time.sleep(0.5)

    # Focused searches to test ranking
    do_search("greenDoc", top_k=5)
    do_search("Which doc has green square?", top_k=5)
    do_search("greenDoc 2", top_k=5)
    do_search("blueDoc", top_k=5)

    # API queries that guide retrieval via doc/page tokens
    ask_api("In redDoc, what color is the square?", top_k=5, max_images=1)
    ask_api("In greenDoc, what color is the square?", top_k=5, max_images=1)
    ask_api("In greenDoc page 2, what color is the square?", top_k=5, max_images=1)
    ask_api("In blueDoc, what color is the square?", top_k=5, max_images=1)


if __name__ == "__main__":
    main()
