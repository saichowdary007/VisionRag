from __future__ import annotations

from typing import Dict, List, Tuple


class Reranker:
    """Optional CrossEncoder reranker with safe fallback.

    - Uses sentence-transformers CrossEncoder if available
    - If unavailable or disabled, returns empty scores
    - Designed for reranking short candidate sets (<=100)
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu", enabled: bool = False):
        self.enabled = enabled
        self.device = device
        self._ce = None
        self._loaded = False
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _ensure_loaded(self):
        if self._loaded or not self.enabled:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._ce = CrossEncoder(self.model_name, device=self.device)
        except Exception:
            self._ce = None
            self.enabled = False
        finally:
            self._loaded = True

    def rerank(self, query: str, candidates: Dict[str, str], batch_size: int = 32) -> Dict[str, float]:
        """Return id -> score for provided candidates.

        Inputs:
          - query: user query string
          - candidates: mapping of candidate_id -> candidate_text
        Output scores are higher-is-better.
        """
        if not candidates:
            return {}
        self._ensure_loaded()
        if not self.enabled or self._ce is None:
            return {cid: 0.0 for cid in candidates.keys()}

        pairs: List[Tuple[str, str]] = [(query, txt or "") for txt in candidates.values()]
        try:
            scores: List[float] = self._ce.predict(pairs, batch_size=batch_size, convert_to_numpy=True).tolist()  # type: ignore
        except Exception:
            # Fallback to zeros on any runtime error
            return {cid: 0.0 for cid in candidates.keys()}
        return {cid: float(s) for cid, s in zip(candidates.keys(), scores)}


