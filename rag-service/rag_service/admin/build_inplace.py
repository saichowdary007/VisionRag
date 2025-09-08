from __future__ import annotations

import sqlite3
import orjson
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss

from ..config import get_config
from ..ingestion.streaming_processor import StreamingDocumentProcessor
from ..embeddings.text_embedder import OptimizedTextEmbedder
from ..embeddings.image_embedder import SiglipImageEmbedder
from ..search.binary_vector_store import float_to_binary


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            doc_id TEXT,
            chunk_id TEXT PRIMARY KEY,
            page INTEGER,
            text TEXT,
            meta TEXT
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page INTEGER,
            path TEXT
        )
        """
    )
    conn.commit()


def _chunk_meta(c) -> Dict[str, Any]:
    meta = {}
    for key in ("section", "subsection", "figure_id", "table_id", "image_ids", "bbox", "diagram_label"):
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


def insert_images(conn: sqlite3.Connection, images: List[tuple[str, str, int, str, str]]) -> List[int]:
    if not images:
        return []
    cur = conn.cursor()
    for rec in images:
        cur.execute(
            "INSERT OR REPLACE INTO images(image_id, doc_id, page, path, bbox) VALUES (?, ?, ?, ?, ?)",
            rec,
        )
    rowids: List[int] = []
    for image_id, *_ in images:
        cur.execute("SELECT rowid FROM images WHERE image_id=?", (image_id,))
        rowids.append(cur.fetchone()[0])
    conn.commit()
    return rowids


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
    image_index_path = Path(cfg.faiss_image_index_path)
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
    embedder = OptimizedTextEmbedder(cfg.text_embedding_model, device=cfg.device, batch_size=16)
    img_embedder = SiglipImageEmbedder(cfg.siglip_model, device=cfg.device, batch_size=16) if cfg.enable_vision else None

    pdfs = sorted(list(docs_dir.glob("**/*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {docs_dir}")
        return 0

    all_texts: List[str] = []
    all_rowids: List[int] = []
    all_image_paths: List[str] = []
    all_image_rowids: List[int] = []

    for pdf in pdfs:
        print(f"Processing {pdf}...")
        chunks = list(processor.process_pdf_streaming(str(pdf)))
        rowids = insert_chunks(conn, chunks)
        all_rowids.extend(rowids)
        all_texts.extend([c.text for c in chunks])
        image_records = processor.extract_document_images(str(pdf))
        if image_records:
            i_rowids = insert_images(conn, image_records)
            all_image_paths.extend([r[3] for r in image_records])
            all_image_rowids.extend(i_rowids)

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

    # Optional vision index
    if cfg.enable_vision and all_image_paths and img_embedder is not None:
        print(f"Embedding {len(all_image_paths)} images with {cfg.siglip_model} ...")
        img_embs = img_embedder.embed_paths(all_image_paths)
        if img_embs.shape[0] == len(all_image_rowids):
            print("Building image FAISS index ...")
            build_faiss_index(img_embs, np.array(all_image_rowids, dtype=np.int64), image_index_path)
            print(f"Wrote image index to {image_index_path}")
        else:
            print("Warning: image embeddings count mismatch; skipping image index build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
