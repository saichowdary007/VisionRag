from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


class SiglipImageEmbedder:
    def __init__(self, model_name: str, device: str = "mps", batch_size: int = 16):
        self.device = self._select_device(device)
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, preferred: str) -> str:
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.inference_mode()
    def embed_paths(self, image_paths: List[str]) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # SigLIP puts pooled image features in last_hidden_state/pooled_output depending on model
            if hasattr(outputs, "pooler_output"):
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embs.append(pooled.detach().cpu().numpy().astype(np.float32))
            if self.device == "mps":
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass
        if not embs:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(embs)

