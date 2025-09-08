from __future__ import annotations
import os
from typing import List
import requests
from .settings import get_settings
def chat_vision(question: str, image_urls: List[str], stream: bool = True) -> requests.Response:
    s = get_settings()
    headers = {
        "Authorization": f"Bearer {s.LMSTUDIO_API_KEY}",
        "Content-Type": "application/json",
    }
    content = []
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": question})
    body = {
        "model": s.LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Answer strictly using the provided page images; cite page filenames when relevant.",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "stream": stream,
    }
    return requests.post(
        f"{s.LMSTUDIO_BASE_URL.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json=body,
        stream=stream,
        timeout=None if stream else 120,
    )
