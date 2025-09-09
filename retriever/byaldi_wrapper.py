from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Hit:
    page_id: str
    score: float
    image_path: str
    doc_id: str
    page_num: int


class LazyIndex:
    """Lazy wrapper over Byaldi index with JSON fallback.

    - Uses a single collection under `index_dir` rather than per-doc indices.
    - Falls back to a file-backed list when `byaldi` is not available or disabled.
    """

    def __init__(self, index_dir: str, model_name: str, device: str = "cpu"):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.device = device
        self._index = None
        self._use_fallback = os.getenv("BYALDI_USE_FALLBACK", "0") == "1"
        self._store_file = self.index_dir / "fallback_index.jsonl"
        self._store: List[Dict[str, Any]] = []
        self._index_name = "pages"

    def _ensure_index(self):
        if self._index is not None or self._use_fallback:
            return
        try:
            from byaldi import RAGMultiModalModel
        except Exception as e:
            print(f"Byaldi import failed: {e}; using fallback")
            self._use_fallback = True
            self._load_fallback()
            return
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            # Use index_root under our chosen directory
            # If existing index present, load it; else create a model and set index_name for incremental adds
            existing = (self.index_dir / self._index_name)
            if existing.exists():
                self._index = RAGMultiModalModel.from_index(
                    self._index_name, device=self.device, index_root=str(self.index_dir)
                )
            else:
                self._index = RAGMultiModalModel.from_pretrained(
                    self.model_name, device=self.device, index_root=str(self.index_dir)
                )
                # Prime an empty index by setting index_name so add_to_index can export
                self._index.index_name = self._index_name
        except Exception as e:
            print(f"Failed to load ColPali model: {e}; using fallback")
            self._use_fallback = True
            self._load_fallback()

    def _load_fallback(self):
        self._store = []
        if self._store_file.exists():
            try:
                with open(self._store_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        self._store.append(json.loads(line))
            except Exception:
                self._store = []

    def _append_fallback(self, rec: Dict[str, Any]):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self._store_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._store.append(rec)

    def index_pages(
        self,
        doc_id: str,
        images: List[str],
        texts: Optional[List[str]] = None,
    ) -> int:
        """Upsert page images (and optional texts) to the index.

        Returns number of pages indexed.
        """
        self._ensure_index()
        count = 0
        if self._use_fallback:
            for i, path in enumerate(images):
                page_num = i
                page_id = f"{doc_id}:{page_num+1}"
                self._append_fallback(
                    {
                        "page_id": page_id,
                        "doc_id": doc_id,
                        "page_num": page_num + 1,
                        "image_path": path,
                        "text": (texts[i] if texts and i < len(texts) else ""),
                    }
                )
                count += 1
        else:
            # Byaldi: incrementally add each page as a separate document with metadata
            try:
                for i, path in enumerate(images):
                    page_num = i + 1
                    metadata = {"doc": doc_id, "page": page_num, "image_path": path}
                    # Let Byaldi assign new doc_ids automatically
                    self._index.add_to_index(
                        path,
                        store_collection_with_index=False,
                        doc_id=None,
                        metadata=metadata,  # type: ignore[arg-type]
                    )
                    count += 1
            except Exception as e:
                print(f"Byaldi index failed: {e}; switching to fallback")
                self._use_fallback = True
                self._load_fallback()
                # Recurse as fallback
                return self.index_pages(doc_id, images, texts)
        return count

    def search(self, query: str, k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
        self._ensure_index()
        hits: List[Hit] = []
        if self._use_fallback:
            # filename/text heuristic fallback
            q = query.lower()
            cands = [r for r in self._store if (not doc_id or r.get("doc_id") == doc_id)]
            scored = []
            for r in cands:
                s = 0.0
                if q in (r.get("image_path", "").lower()):
                    s = max(s, 1.0)
                if q in (r.get("doc_id", "").lower()):
                    s = max(s, 0.8)
                if q in (r.get("text", "").lower()):
                    s = max(s, 0.9)
                if s > 0:
                    scored.append((r, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            for r, s in scored[:k]:
                hits.append(
                    Hit(
                        page_id=r.get("page_id", ""),
                        score=float(s),
                        image_path=r.get("image_path", ""),
                        doc_id=r.get("doc_id", ""),
                        page_num=int(r.get("page_num", 1)),
                    )
                )
            return hits
        # True Byaldi
        try:
            filter_meta = {"doc": doc_id} if doc_id else None
            results = self._index.search(query, k=k, filter_metadata=filter_meta)
            for res in results:
                score = float(getattr(res, "score", 0.0))
                meta = getattr(res, "metadata", {}) or {}
                d_str = str(meta.get("doc", ""))
                pnum = int(meta.get("page", getattr(res, "page_num", 1)) or 1)
                # Reconstruct image path from convention or metadata
                img_path = str(meta.get("image_path", ""))
                hits.append(
                    Hit(
                        page_id=f"{d_str}:{pnum}",
                        score=score,
                        image_path=img_path,
                        doc_id=d_str,
                        page_num=pnum,
                    )
                )
        except Exception as e:
            print(f"Byaldi search failed: {e}")
        return hits[:k]


class ByaldiRetriever:
    def __init__(self, index_dir: str, model_name: str, device: str = "cpu"):
        self.index_dir = Path(index_dir)
        self.index = LazyIndex(index_dir, model_name=model_name, device=device)

    def index_document(self, doc_id: str, images: List[str], texts: Optional[List[str]] = None) -> int:
        return self.index.index_pages(doc_id, images, texts)

    def search(self, query: str, k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
        return self.index.search(query, k=k, doc_id=doc_id)

