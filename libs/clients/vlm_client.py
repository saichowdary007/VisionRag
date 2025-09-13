from __future__ import annotations

"""Client helper to chat with a vision-language model server (Ollama / LM Studio).

Uses Ollama's native API format. Images are sent as base64 data URIs.
"""

from io import BytesIO
import base64
import os
from typing import List

import requests
import json
from PIL import Image

__all__ = ["vision_chat", "pil_to_b64"]

BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:latest")
TIMEOUT = int(os.getenv("VLM_TIMEOUT", "180"))  # seconds


def pil_to_b64(img: Image.Image) -> str:
    """Convert PIL Image to base64-encoded JPEG string."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def vision_chat(prompt: str, images: List[Image.Image], *, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Send chat completion request with embedded images and return assistant reply."""
    # Combine prompt and images into a single content string for Ollama
    content_parts = [prompt]

    for img in images:
        b64_data = pil_to_b64(img)
        content_parts.append(f"[Image: data:image/jpeg;base64,{b64_data}]")

    content = " ".join(content_parts)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    url = f"{BASE_URL}/api/chat"
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()

    # Ollama returns streaming response, collect all content
    full_content = ""
    lines = resp.text.strip().split('\n')
    for line in lines:
        if line.strip():
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    if content:  # Only add non-empty content
                        full_content += content
                if data.get("done", False):
                    break  # Stop when done
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue

    return full_content.strip() if full_content.strip() else "I apologize, but I couldn't generate a response. Please try again."


