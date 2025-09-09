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
        self._ce = None  # sentence-transformers CrossEncoder
        self._flag = None  # FlagEmbedding reranker (BGEM3 or generic)
        self._strategy = None  # "st" | "flag_m3" | "flag_generic"
        self._loaded = False
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _ensure_loaded(self):
        if self._loaded or not self.enabled:
            return
        name = (self.model_name or "").lower()
        # Prefer specialized implementations when model hints suggest it
        if "bge-reranker-v2-m3" in name:
            # BGEM3 reranker
            try:
                from FlagEmbedding import BGEM3FlagReranker  # type: ignore

                use_fp16 = self.device != "cpu"
                self._flag = BGEM3FlagReranker(self.model_name, use_fp16=use_fp16)
                self._strategy = "flag_m3"
                self._loaded = True
                return
            except Exception:
                # Fall through to generic loaders
                self._flag = None
        if "bge-reranker" in name and self._flag is None:
            # Generic BGE reranker (large/base)
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                use_fp16 = self.device != "cpu"
                self._flag = FlagReranker(self.model_name, use_fp16=use_fp16)
                self._strategy = "flag_generic"
                self._loaded = True
                return
            except Exception:
                self._flag = None

        # Default: sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._ce = CrossEncoder(self.model_name, device=self.device)
            self._strategy = "st"
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
        if not self.enabled:
            return {cid: 0.0 for cid in candidates.keys()}

        cand_ids = list(candidates.keys())
        cand_texts = [candidates[cid] or "" for cid in cand_ids]

        if self._strategy == "flag_m3" and self._flag is not None:
            try:
                # BGEM3FlagReranker: compute_score(query, docs)
                scores = self._flag.compute_score(query, cand_texts)  # type: ignore[attr-defined]
                # Some versions return a single float if only one doc is provided
                if not isinstance(scores, list):
                    scores = [float(scores)]
                return {cid: float(s) for cid, s in zip(cand_ids, scores)}
            except Exception:
                pass

        if self._strategy == "flag_generic" and self._flag is not None:
            try:
                # FlagReranker API variants
                try:
                    scores = self._flag.compute_score(query, cand_texts)  # type: ignore[attr-defined]
                except TypeError:
                    pairs = [(query, t) for t in cand_texts]
                    scores = self._flag.compute_score(pairs)  # type: ignore[attr-defined]
                if not isinstance(scores, list):
                    scores = [float(scores)]
                return {cid: float(s) for cid, s in zip(cand_ids, scores)}
            except Exception:
                pass

        if self._strategy == "st" and self._ce is not None:
            pairs: List[Tuple[str, str]] = [(query, t) for t in cand_texts]
            try:
                scores: List[float] = self._ce.predict(pairs, batch_size=batch_size, convert_to_numpy=True).tolist()  # type: ignore
                return {cid: float(s) for cid, s in zip(cand_ids, scores)}
            except Exception:
                pass

        # Fallback: zeros
        return {cid: 0.0 for cid in cand_ids}


