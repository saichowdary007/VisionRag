from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class BGEReranker:
    model_name: str
    device: str = "mps"

    def __post_init__(self):
        self.device = self._select_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, preferred: str) -> str:
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.inference_mode()
    def rerank(self, query: str, items: List[Tuple[str, str]], top_k: int) -> List[Tuple[str, float]]:
        # items: list of (chunk_id, text)
        scores: List[Tuple[str, float]] = []
        batch_size = 32
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            pairs = [[query, text] for _, text in batch]
            toks = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            logits = self.model(**toks).logits.squeeze(-1)
            s = logits.detach().float().cpu().tolist()
            for (cid, _), sc in zip(batch, s):
                scores.append((cid, float(sc)))
            # Avoid torch.mps.empty_cache() on every batch; it's expensive and can stall
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

