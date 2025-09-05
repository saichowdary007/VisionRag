from __future__ import annotations

"""
Quick recall@K check between binary and full-precision indexes.
Usage (inside container):
  python -m scripts.validate_quantization --k 20 --n 50
"""

import argparse
import sqlite3
from pathlib import Path
from typing import List

import faiss
import numpy as np


def load_texts(db_path: Path, n: int) -> List[str]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT text FROM chunks ORDER BY RANDOM() LIMIT ?", (n,))
    return [r[0] for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/app/data/documents.db")
    ap.add_argument("--full", default="/app/data/vectors.faiss")
    ap.add_argument("--binary", default="/app/data/vectors.bin")
    ap.add_argument("--ids", default="/app/data/vectors.bin.ids.npy")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    full = faiss.read_index(args.full)
    b = faiss.read_index_binary(args.binary)
    ids = np.load(args.ids)

    # naive query set: use random texts -> random vectors via PCA? Not available.
    # We just measure nearest-neighbor agreement using existing vectors: sample rows and use their vectors.
    # This requires that the full index is an IDMap2 with vectors retrievable; if not, we cannot fetch raw vectors.
    # Fallback: use random binary codes as queries to compare internal consistency.
    print("Note: This quick script assumes availability of comparable queries. Use with caution.")

    nq = min(args.n, full.ntotal)
    # Create queries by selecting random db ids, then searching both indexes and comparing recall@k
    rng = np.random.default_rng(42)
    sel = rng.choice(full.ntotal, size=nq, replace=False)
    # We cannot directly read vectors back from index; so use the selected vectors as queries by extracting via reconstruct if supported
    can_reconstruct = hasattr(full, "reconstruct")
    if not can_reconstruct:
        print("Full index does not support reconstruct(); recall check skipped.")
        return 0
    queries = np.vstack([full.reconstruct(int(i)) for i in sel]).astype(np.float32)
    # Binary pack
    qbits = np.packbits((queries > 0).astype(np.uint8), axis=1)

    Df, If = full.search(queries, args.k)
    Db, Ib = b.search(qbits, args.k)
    Ib_mapped = np.vectorize(lambda idx: ids[idx] if 0 <= idx < len(ids) else -1)(Ib)

    def recall_at_k(a, b):
        inter = len(set(a) & set(b))
        return inter / max(1, len(set(a)))

    recalls = []
    for i in range(nq):
        recalls.append(recall_at_k(If[i].tolist(), Ib_mapped[i].tolist()))
    r20 = float(np.mean(recalls))
    print({"recall_at_k": r20, "k": args.k, "nq": nq})
    if r20 < 0.5:
        print("Warning: Low recall; consider tuning binary quantization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

