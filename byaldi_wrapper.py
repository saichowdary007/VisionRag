from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import hashlib
import json


class LazyIndex:
    """Lazy wrapper for Byaldi index."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self._index = None
        # Allow forcing fallback (useful for tests/CPU-only envs)
        self._use_fallback = os.getenv("BYALDI_USE_FALLBACK", "0") == "1"
        self._documents = []
        self._index_file = self.base_path / "index.json"

    def _ensure_index(self):
        """Ensure the Byaldi index is loaded."""
        if self._index is not None:
            return

        try:
            from byaldi import RAGMultiModalModel
        except ImportError:
            # No byaldi → fallback
            self._use_fallback = True
            return self._load_fallback_index()

        # Check if CUDA is available
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            # Set environment variable to force CPU usage
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Try to load existing index
        if (self.base_path / ".byaldi").exists() and not self._use_fallback:
            try:
                self._index = RAGMultiModalModel.from_index(str(self.base_path))
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                self._use_fallback = True
                self._load_fallback_index()
        elif not self._use_fallback:
            try:
                self._index = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device=device)
            except Exception as e:
                print(f"Failed to create ColPali index, using fallback: {e}")
                self._use_fallback = True
                self._documents = []

    def _load_fallback_index(self):
        """Load documents from fallback JSON index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    data = json.load(f)
                    self._documents = data.get('documents', [])
            except Exception as e:
                print(f"Failed to load fallback index: {e}")
                self._documents = []
        else:
            self._documents = []

    def _save_fallback_index(self):
        """Save documents to fallback JSON index."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            with open(self._index_file, 'w') as f:
                json.dump({'documents': self._documents}, f)
        except Exception as e:
            print(f"Failed to save fallback index: {e}")

    def add_images(self, image_paths: List[str], doc_ids: Optional[List[str]] = None, texts: Optional[List[str]] = None):
        """Add images to the index."""
        self._ensure_index()

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(image_paths))]

        if self._use_fallback:
            # Fallback implementation: just store file paths
            for i, image_path in enumerate(image_paths):
                doc_id = doc_ids[i] if i < len(doc_ids) else f"doc_{i}"
                self._documents.append({
                    'doc_id': doc_id,
                    'image_path': image_path,
                    'text': (texts[i] if (texts and i < len(texts)) else ""),
                    'id': len(self._documents)
                })
            self._save_fallback_index()
        else:
            # Use ColPali
            for i, image_path in enumerate(image_paths):
                doc_id = doc_ids[i] if i < len(doc_ids) else f"doc_{i}"
                self._index.add_to_index(image_path, store_collection_with_index=True, doc_id=i)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the index."""
        self._ensure_index()

        if self._use_fallback:
            # Fallback implementation: simple text matching
            query_lower = query.lower()
            results = []
            for doc in self._documents:
                # Simple scoring based on filename matching
                score = 0.0
                if query_lower in doc['image_path'].lower():
                    score = 1.0
                elif query_lower in doc['doc_id'].lower():
                    score = 0.8
                elif query_lower in (doc.get('text','').lower()):
                    score = 0.9

                if score > 0:
                    results.append({
                        "doc_id": doc['doc_id'],
                        "image_path": doc['image_path'],
                        "score": score,
                        "page_num": 0
                    })

            # Sort by score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
        else:
            # Use ColPali
            try:
                results = self._index.search(query, k=k)
                return [
                    {
                        "doc_id": result.doc_id if hasattr(result, 'doc_id') else "",
                        "image_path": result.image_path if hasattr(result, 'image_path') else "",
                        "score": float(result.score) if hasattr(result, 'score') else 0.0,
                        "page_num": result.page_num if hasattr(result, 'page_num') else 0
                    }
                    for result in results
                ]
            except Exception as e:
                print(f"Search failed: {e}")
                return []


class ByaldiRetriever:
    """Retriever using Byaldi/ColPali for document search."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.indices: Dict[str, LazyIndex] = {}

    def get_index(self, doc_id: str) -> LazyIndex:
        """Get or create index for a document."""
        if doc_id not in self.indices:
            index_path = self.data_dir / doc_id
            index_path.mkdir(parents=True, exist_ok=True)
            self.indices[doc_id] = LazyIndex(str(index_path))

        return self.indices[doc_id]

    def index_document(self, doc_id: str, image_paths: List[str], texts: Optional[List[str]] = None):
        """Index a document with its page images (and optional per-page texts)."""
        index = self.get_index(doc_id)
        index.add_images(image_paths, texts=texts)

    def search(self, query: str, doc_id: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if doc_id:
            # Search specific document
            index = self.get_index(doc_id)
            return index.search(query, k=k)
        else:
            # Search all documents
            all_results = []
            for doc_id_key in self.indices.keys():
                index = self.get_index(doc_id_key)
                results = index.search(query, k=k)
                all_results.extend(results)

            # Sort by score and return top k
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:k]

    def list_documents(self) -> List[str]:
        """List all indexed documents."""
        return list(self.indices.keys())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document index."""
        if doc_id in self.indices:
            del self.indices[doc_id]
            # TODO: Also delete the actual index files
            return True
        return False
