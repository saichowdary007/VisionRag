from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    # bits: (n, d) uint8 {0,1} -> (n, d//8) uint8
    return np.packbits(bits, axis=1)


def float_to_binary(vecs: np.ndarray) -> np.ndarray:
    """Binarize float vectors by sign to 0/1 bits and pack to bytes.

    vecs: (n, d) float32 -> (n, d//8) uint8
    """
    bits = (vecs > 0).astype(np.uint8)
    return _pack_bits(bits)


@dataclass
class BinaryVectorStore:
    index_path: str
    ids_path: str

    def __post_init__(self):
        p = Path(self.index_path)
        if p.exists():
            try:
                self.index = faiss.read_index_binary(self.index_path)
            except Exception:
                # Fallback if binary reader not available in this build
                self.index = None
        else:
            self.index = None
        # Load ID map if present
        ids_p = Path(self.ids_path)
        self.ids = None
        if ids_p.exists():
            try:
                self.ids = np.load(str(ids_p))
            except Exception:
                self.ids = None

    def has_data(self) -> bool:
        return self.index is not None and self.ids is not None and len(self.ids) > 0

    def search(self, query_vecs: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_data():
            return np.array([]), np.array([[]])
        qbin = float_to_binary(query_vecs)
        D, I = self.index.search(qbin, top_k)
        # Map internal indices to rowids via self.ids
        if I.size:
            mapped = np.vectorize(lambda idx: self.ids[idx] if 0 <= idx < len(self.ids) else -1)(I)
            return D, mapped
        return D, I

