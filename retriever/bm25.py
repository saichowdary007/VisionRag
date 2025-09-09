from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from rank_bm25 import BM25Okapi as _ExtBM25
except Exception:  # pragma: no cover - provide minimal built-in BM25
    _ExtBM25 = None

class _MiniBM25:
    """Tiny BM25Okapi-like scorer used when rank_bm25 is not installed.

    Not optimized; sufficient for tests and small corpora.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / self.N if self.N else 0.0
        # document frequencies
        self.df: Dict[str, int] = {}
        for doc in corpus_tokens:
            seen = set(doc)
            for t in seen:
                self.df[t] = self.df.get(t, 0) + 1
        # term frequencies per doc
        self.tf: List[Dict[str, int]] = []
        for doc in corpus_tokens:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)

    def _idf(self, term: str) -> float:
        import math
        n_t = self.df.get(term, 0)
        return math.log(1 + (self.N - n_t + 0.5) / (n_t + 0.5)) if self.N else 0.0

    def get_scores(self, query_tokens: List[str]):
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            dl = sum(tf.values())
            denom_base = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            s = 0.0
            for q in query_tokens:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                idf = self._idf(q)
                s += idf * (f * (self.k1 + 1)) / (f + denom_base)
            scores[i] = s
        return scores


def _default_tokenize(text: str) -> List[str]:
    return [t for t in (text or "").lower().split() if t]


@dataclass
class BM25Index:
    """Lightweight BM25 wrapper over per-page texts.

    Persists texts as JSON mapping {page_id -> text} and builds an in-process
    BM25Okapi index. Designed for small/medium corpora and deterministic tests.
    """

    base_dir: Path
    texts_file: Path
    _texts: Dict[str, str]
    _bm25: BM25Okapi | None

    @classmethod
    def open(cls, base_dir: str | Path) -> "BM25Index":
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        texts_file = base / "bm25_texts.json"
        texts: Dict[str, str] = {}
        if texts_file.exists():
            try:
                texts = json.loads(texts_file.read_text())
            except Exception:
                texts = {}
        idx = cls(base, texts_file, texts, None)
        idx._rebuild()
        return idx

    def _rebuild(self):
        corpus = [v for v in self._texts.values()]
        tokenized = [_default_tokenize(t) for t in corpus]
        if not tokenized:
            self._bm25 = None
        elif _ExtBM25 is not None:
            self._bm25 = _ExtBM25(tokenized)
        else:
            self._bm25 = _MiniBM25(tokenized)

    def upsert(self, page_id: str, text: str | None):
        if text is None:
            return
        self._texts[page_id] = text
        self.texts_file.write_text(json.dumps(self._texts))
        self._rebuild()

    def bulk_upsert(self, items: List[Tuple[str, str | None]]):
        changed = False
        for pid, txt in items:
            if txt is None:
                continue
            if self._texts.get(pid) != txt:
                self._texts[pid] = txt
                changed = True
        if changed:
            self.texts_file.write_text(json.dumps(self._texts))
            self._rebuild()

    def search(self, query: str, k: int = 5, filter_prefix: str | None = None) -> List[Tuple[str, float]]:
        if not self._bm25:
            return []
        tokens = _default_tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ids = list(self._texts.keys())
        results: List[Tuple[str, float]] = []
        for pid, score in zip(ids, scores):
            if filter_prefix and not pid.startswith(filter_prefix + ":"):
                continue
            results.append((pid, float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # Lightweight getters to expose stored texts for reranking or analytics
    def get_text(self, page_id: str) -> str | None:
        return self._texts.get(page_id)

    def get_texts(self, page_ids: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for pid in page_ids:
            txt = self._texts.get(pid)
            if txt is not None:
                out[pid] = txt
        return out
