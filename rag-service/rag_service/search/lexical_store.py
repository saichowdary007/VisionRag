from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class LexicalStore:
    db_path: str

    def __post_init__(self):
        # Ensure parent directory exists (named volumes may start empty)
        db_parent = Path(self.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
        self.conn.execute("PRAGMA cache_size = -50000")  # 50MB
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self.conn.cursor()
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
                path TEXT,
                bbox TEXT
            )
            """
        )
        self.conn.commit()
        # Try to add meta column in case of existing DBs without it
        try:
            cur.execute("ALTER TABLE chunks ADD COLUMN meta TEXT")
            self.conn.commit()
        except Exception:
            pass
        # Try to add bbox column to images if missing
        try:
            cur.execute("ALTER TABLE images ADD COLUMN bbox TEXT")
            self.conn.commit()
        except Exception:
            pass

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        cur = self.conn.cursor()
        # Some SQLite builds error on parameterized MATCH and certain punctuation.
        # Sanitize to basic tokens and escape quotes.
        sanitized = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in query)
        q = sanitized.replace("'", "''").strip()
        limit = max(1, min(int(top_k), 1000))
        sql = (
            "SELECT chunk_id, bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH '" + q + "' "
            f"ORDER BY score LIMIT {limit}"
        )
        cur.execute(sql)
        rows = cur.fetchall()
        # bm25 lower is better; invert to higher is better
        return [(cid, float(-score)) for cid, score in rows]

    def get_chunks(self, chunk_ids: List[str]) -> List[Tuple[str, str, int, str]]:
        if not chunk_ids:
            return []
        qmarks = ",".join(["?"] * len(chunk_ids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT doc_id, chunk_id, page, text FROM chunks WHERE chunk_id IN ({qmarks})", chunk_ids)
        return cur.fetchall()

    def get_chunk_ids_by_rowid(self, rowids: List[int]) -> List[str]:
        if not rowids:
            return []
        qmarks = ",".join(["?"] * len(rowids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT chunk_id FROM chunks WHERE rowid IN ({qmarks})", rowids)
        return [r[0] for r in cur.fetchall()]

    # --- Multimodal helpers ---
    def get_chunks_by_doc_page(self, doc_id: str, page: int, limit: int = 5) -> List[Tuple[str, str, int, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT doc_id, chunk_id, page, text FROM chunks WHERE doc_id=? AND page=? LIMIT ?",
            (doc_id, int(page), int(limit)),
        )
        return cur.fetchall()

    def get_images_by_rowid(self, rowids: List[int]) -> List[Tuple[str, str, int, str]]:
        """Return list of (image_id, doc_id, page, path) for given rowids, preserving order length."""
        if not rowids:
            return []
        qmarks = ",".join(["?"] * len(rowids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT rowid, image_id, doc_id, page, path FROM images WHERE rowid IN ({qmarks})", rowids)
        rows = cur.fetchall()
        # Map rowid->record and normalize to given order
        by_id = {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}
        return [by_id.get(rid, ("", "", -1, "")) for rid in rowids]

    def count_images(self) -> int:
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT COUNT(1) FROM images")
            return int(cur.fetchone()[0])
        except Exception:
            return 0
