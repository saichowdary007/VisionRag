from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


@dataclass
class ClipVisionReranker:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "mps"

    def __post_init__(self):
        self.device = self._select_device(self.device)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, preferred: str) -> str:
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.inference_mode()
    def score(self, query: str, image_paths: List[str]) -> List[float]:
        if not image_paths:
            return []
        imgs: List[Image.Image] = []
        for p in image_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), color=(255, 255, 255)))
        inputs = self.processor(text=[query], images=imgs, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # logits_per_image: (n_images, n_texts=1)
        scores = out.logits_per_image.squeeze(-1).detach().float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return [float(s) for s in scores]

