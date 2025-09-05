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


class SiglipTextEmbedder:
    """Text tower of SigLIP/CLIP-like models to produce text embeddings
    compatible with image embeddings for cross-modal retrieval.
    """

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
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            except TypeError:
                # Fallback for processors using 'text' vs 'inputs'
                inputs = self.processor(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = None
            # Try CLIP-like API first
            if hasattr(self.model, "get_text_features"):
                try:
                    feats = self.model.get_text_features(**inputs)
                except Exception:
                    outputs = self.model(**inputs)
                    feats = getattr(outputs, "last_hidden_state", None)
                    if feats is not None:
                        feats = feats[:, 0]
            else:
                outputs = self.model(**inputs)
                # Heuristic: use CLS token or pooled output
                feats = getattr(outputs, "pooler_output", None)
                if feats is None:
                    feats = getattr(outputs, "text_embeds", None)
                if feats is None:
                    feats = getattr(outputs, "last_hidden_state", None)
                    if feats is not None:
                        feats = feats[:, 0]
            if feats is None:
                # As a last resort, compute mean over sequence
                outputs = outputs or self.model(**inputs)
                last = outputs.last_hidden_state
                feats = last.mean(dim=1)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embs.append(feats.detach().cpu().numpy().astype(np.float32))
            if self.device == "mps":
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass
        if not embs:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(embs)
