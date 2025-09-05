from __future__ import annotations

import sqlite3
import orjson
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss

from ..config import get_config
from ..ingestion.streaming_processor import StreamingDocumentProcessor
from ..embeddings.text_embedder import M1OptimizedTextEmbedder
from ..search.binary_vector_store import float_to_binary


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            doc_id TEXT,
            chunk_id TEXT PRIMARY KEY,
            page INTEGER,
            text TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            page UNINDEXED,
            tokenize='porter unicode61 remove_diacritics 2'
        )
        """
    )
    conn.commit()


def _chunk_meta(c) -> Dict[str, Any]:
    meta = {}
    for key in ("section", "subsection", "figure_id", "table_id"):
        val = getattr(c, key, None)
        if val:
            meta[key] = val
    return meta


def insert_chunks(conn: sqlite3.Connection, chunks) -> List[int]:
    cur = conn.cursor()
    ids: List[int] = []
    for c in chunks:
        cur.execute(
            "INSERT OR REPLACE INTO chunks(doc_id, chunk_id, page, text, meta) VALUES (?, ?, ?, ?, ?)",
            (c.doc_id, c.chunk_id, c.page, c.text, orjson.dumps(_chunk_meta(c)).decode()),
        )
        cur.execute(
            "INSERT INTO chunks_fts(text, chunk_id, doc_id, page) VALUES (?, ?, ?, ?)",
            (c.text, c.chunk_id, c.doc_id, c.page),
        )
        cur.execute("SELECT rowid FROM chunks WHERE chunk_id=?", (c.chunk_id,))
        rid = cur.fetchone()[0]
        ids.append(rid)
    conn.commit()
    return ids


def build_faiss_index(embeddings: np.ndarray, ids: np.ndarray, out_path: Path) -> None:
    assert embeddings.shape[0] == ids.shape[0]
    n, d = embeddings.shape
    # For small corpora, use a simple flat index to avoid training failures
    if n < 1000:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
        index.add_with_ids(embeddings, ids)
        faiss.write_index(index, str(out_path))
        return
    nlist = min(1024, max(1, n // 32))
    opq = faiss.OPQMatrix(d, min(64, d))
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexPreTransform(opq, ivf)
    ntrain = min(100_000, n)
    sample_idx = np.random.choice(n, ntrain, replace=False)
    train_vecs = embeddings[sample_idx]
    index.train(train_vecs)
    id_map = faiss.IndexIDMap2(index)
    id_map.add_with_ids(embeddings, ids)
    faiss.write_index(id_map, str(out_path))


def main() -> int:
    cfg = get_config()
    docs_dir = Path(cfg.docs_dir)
    db_path = Path(cfg.db_path)
    index_path = Path(cfg.faiss_index_path)
    bin_index_path = Path(cfg.faiss_binary_index_path)
    bin_ids_path = Path(cfg.faiss_binary_ids_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    bin_index_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA mmap_size = 268435456")
    conn.execute("PRAGMA cache_size = -50000")
    ensure_schema(conn)

    processor = StreamingDocumentProcessor(max_memory_mb=4000)
    embedder = M1OptimizedTextEmbedder(cfg.text_embedding_model, device=cfg.device, batch_size=16)

    pdfs = sorted(list(docs_dir.glob("**/*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {docs_dir}")
        return 0

    all_texts: List[str] = []
    all_rowids: List[int] = []

    for pdf in pdfs:
        print(f"Processing {pdf}...")
        chunks = list(processor.process_pdf_streaming(str(pdf)))
        rowids = insert_chunks(conn, chunks)
        all_rowids.extend(rowids)
        all_texts.extend([c.text for c in chunks])

    if not all_texts:
        print("No text extracted from PDFs; aborting index build.")
        return 0

    embeddings = embedder.embed(all_texts)
    ids = np.array(all_rowids, dtype=np.int64)
    print(f"Embeddings shape: {embeddings.shape}; IDs: {ids.shape}")

    print("Building FAISS index (IVF + OPQ or Flat for small corpora)...")
    build_faiss_index(embeddings, ids, index_path)
    print(f"Wrote index to {index_path}")

    # Build binary index
    print("Building binary index (Hamming distance) ...")
    bin_codes = float_to_binary(embeddings)
    d_bits = bin_codes.shape[1] * 8
    bindex = faiss.IndexBinaryFlat(d_bits)
    bindex.add(bin_codes)
    faiss.write_index_binary(bindex, str(bin_index_path))
    np.save(str(bin_ids_path), ids)
    print(f"Wrote binary index to {bin_index_path} and ids to {bin_ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
