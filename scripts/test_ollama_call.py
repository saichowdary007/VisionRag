from __future__ import annotations

import os
from PIL import Image

from libs.clients.vlm_client import vision_chat


def main():
    # Configure via environment if needed
    # VLM_BASE_URL (default http://localhost:11434/v1)
    # VLM_MODEL (default qwen2.5-vl:7b)
    # VLM_TIMEOUT_SEC (default 180)

    # Use a sample image if available
    sample_path = os.path.join("local_data", "pages", "docB", "0001.png")
    if not os.path.exists(sample_path):
        raise SystemExit(f"Sample image not found at {sample_path}. Ingest a PDF first or update the path.")

    img = Image.open(sample_path).convert("RGB")
    question = "What is shown in this page?"

    print("Calling VLM...\n")
    answer = vision_chat(question, [img])
    print("Answer:\n", answer)


if __name__ == "__main__":
    main()


