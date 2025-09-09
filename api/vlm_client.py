from __future__ import annotations
import os
import base64
import mimetypes
from io import BytesIO
from typing import List
import requests
from PIL import Image
from .settings import get_settings

def _image_url_to_base64(image_url: str) -> str:
    """Fetch image from URL, resize for VLM, and convert to base64 data URL format."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        # Open image with PIL and resize for VLM efficiency
        img = Image.open(BytesIO(response.content))

        # Resize to VLM_SQUARE x VLM_SQUARE for better performance
        s = get_settings()
        max_size = (s.VLM_SQUARE, s.VLM_SQUARE)
        img.thumbnail(max_size, Image.LANCZOS)

        # Save as PNG with compression
        output_buffer = BytesIO()
        img.save(output_buffer, format='PNG', optimize=True, quality=85)
        compressed_image_data = output_buffer.getvalue()

        # Convert to base64
        image_data = base64.b64encode(compressed_image_data).decode('utf-8')

        # Return as data URL
        return f"data:image/png;base64,{image_data}"
    except Exception as e:
        print(f"Image processing error for {image_url}: {str(e)}", flush=True)
        # If image processing fails, return empty data URL as fallback
        return f"data:image/png;base64,"

def chat_vision(question: str, image_urls: List[str], stream: bool = False) -> requests.Response:
    s = get_settings()
    headers = {
        "Authorization": f"Bearer {s.VLM_API_KEY}",
        "Content-Type": "application/json",
    }
    content = []
    for url in image_urls:
        # Convert HTTP URL to base64 data URL for VLM compatibility
        base64_url = _image_url_to_base64(url)
        content.append({"type": "image_url", "image_url": {"url": base64_url}})
    content.append({"type": "text", "text": question})
    body = {
        "model": s.VLM_MODEL,
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
    # Allow base URL to be configured with or without trailing /v1
    base = s.VLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    # Set timeout based on streaming
    timeout = 120 if stream else 120
    return requests.post(
        url,
        headers=headers,
        json=body,
        stream=stream,
        timeout=timeout,
    )
