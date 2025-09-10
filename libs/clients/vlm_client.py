from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Iterable

import requests
from PIL import Image


BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
MODEL = os.getenv("VLM_MODEL", "qwen2.5-vl:7b")


def _pil_to_b64_jpeg(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def vision_chat(prompt: str, images: Iterable[Image.Image], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    content = [{"type": "text", "text": prompt}]
    for im in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64_jpeg(im)}"},
        })

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = BASE_URL if BASE_URL.endswith("/v1") else f"{BASE_URL}/v1"
    r = requests.post(f"{url}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Fallback safe string
        return str(data)


