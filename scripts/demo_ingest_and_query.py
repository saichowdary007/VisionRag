#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import base64
import json
from io import BytesIO
from typing import Optional

import requests
from PIL import Image


RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8080").rstrip("/")
API_URL = os.getenv("BACKEND_API_URL", os.getenv("API_URL", "http://localhost:8080")).rstrip("/")


def to_b64_jpeg(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ensure_services():
    try:
        rh = requests.get(f"{RETRIEVER_URL}/healthz", timeout=3)
        print("retriever /healthz:", rh.status_code, rh.json())
    except Exception as e:
        print("retriever not reachable at", RETRIEVER_URL, e)
        sys.exit(1)
    try:
        ah = requests.get(f"{API_URL}/healthz", timeout=3)
        print("api /healthz:", ah.status_code, ah.json())
    except Exception as e:
        print("api not reachable at", API_URL, e)
        print("continuing: API is optional for retriever demo...")


def demo_ingest(doc_id: str = "demoDoc") -> str:
    # Generate a simple gray image with a distinct color block
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (255, 0, 0))

    b64 = to_b64_jpeg(img)
    page_id = f"{doc_id}:1"
    payload = {"pages": [{"page_id": page_id, "image_b64": b64}]}
    # First run may download CLIP model; allow generous timeout
    r = requests.post(f"{RETRIEVER_URL}/ingest", json=payload, timeout=180)
    r.raise_for_status()
    print("ingest:", r.status_code, r.json())
    return page_id


def demo_search(query: str, top_k: int = 3):
    r = requests.post(f"{RETRIEVER_URL}/search", json={"text": query, "top_k": top_k}, timeout=10)
    r.raise_for_status()
    data = r.json()
    print("search:", json.dumps(data, indent=2))
    return data.get("hits", [])


def demo_query_api(question: str, top_k: int = 3, max_images: int = 1):
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k, "max_images": max_images},
            timeout=20,
        )
        print("api /query:", r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print("API query failed:", e)


def main():
    ensure_services()
    page_id = demo_ingest("demoDoc")
    time.sleep(0.2)
    demo_search("demoDoc", top_k=3)
    demo_query_api("What is in demoDoc?", top_k=3, max_images=1)
    # fetch the image to prove it exists
    try:
        r = requests.get(f"{RETRIEVER_URL}/page_image/{page_id}", timeout=10)
        print("page_image:", r.status_code, "b64_len=", len(r.json().get("image_b64", "")))
    except Exception as e:
        print("page_image fetch failed:", e)


if __name__ == "__main__":
    main()
