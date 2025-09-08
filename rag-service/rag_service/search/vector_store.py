from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


@dataclass
class VectorStore:
    index_path: str

    def __post_init__(self):
        p = Path(self.index_path)
        if p.exists():
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            try:
                # For IVF
                if hasattr(self.index, "nprobe"):
                    self.index.nprobe = 16
            except Exception:
                pass
        else:
            # Create an empty flat index; will be replaced after build
            self.index = None

    def search(self, query_vecs: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            return np.array([]), np.array([[]])
        scores, idxs = self.index.search(query_vecs, top_k)
        return scores, idxs

