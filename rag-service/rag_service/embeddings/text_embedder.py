from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class OptimizedTextEmbedder:
    def __init__(self, model_name: str, device: str = "mps", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._select_device(device)
        self.model = self._load_model()

    def _select_device(self, preferred: str) -> str:
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> SentenceTransformer:
        # Some community models (e.g., nomic-ai/nomic-embed-text-v1.5) require
        # trust_remote_code=True to load custom modules.
        model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )
        return model

    @torch.inference_mode()
    def embed(self, texts: List[str]) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False, device=self.device, normalize_embeddings=True)
            embs.append(vecs)
            # Avoid torch.mps.empty_cache() per batch; it increases latency significantly
        if not embs:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(embs).astype(np.float32)
