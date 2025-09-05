from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from ..embeddings.text_embedder import OptimizedTextEmbedder
from ..search.vector_store import VectorStore
from ..search.binary_vector_store import BinaryVectorStore
from ..search.lexical_store import LexicalStore
from ..types import SearchResult
from ..cache.redis_cache import get_json, set_json


def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    # rankings: list of [(id, score)] with higher=better
    # Convert to ranks, then sum 1/(k + rank)
    score_map: dict[str, float] = {}
    for r in rankings:
        # Sort by score desc
        r_sorted = sorted(r, key=lambda x: x[1], reverse=True)
        for rank, (cid, _) in enumerate(r_sorted, start=1):
            score_map[cid] = score_map.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


@dataclass
class HybridSearcher:
    text_embedder: OptimizedTextEmbedder
    vector_store: VectorStore
    lexical_store: LexicalStore
    reranker: object | None = None
    binary_store: Optional[BinaryVectorStore] = None
    redis_client: object | None = None
    cache_ttl_seconds: int = 3600
    binary_first: bool = False
    binary_candidates: int = 1000

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        # Cache check
        cache_key = f"search:v1:k{top_k}:{query}"
        cached = get_json(self.redis_client, cache_key)
        if cached:
            return [SearchResult(**r) for r in cached]

        # Stage 1: Vector + lexical (optimized for speed)
        qvec = self.text_embedder.embed([query]).astype(np.float32)
        # Optimized candidate counts for faster search while maintaining quality
        search_k = min(100, top_k * 3)  # Adaptive candidate count based on requested top_k

        # Optional: binary-first filtering to narrow candidates
        candidate_cids: set[str] | None = None
        if self.binary_first and self.binary_store and self.binary_store.has_data():
            try:
                bD, bI = self.binary_store.search(qvec, top_k=max(self.binary_candidates, search_k))
                if bI.size and bI[0].size:
                    rowids = [int(idx) for idx in bI[0].tolist() if int(idx) >= 0]
                    cids = set(self.lexical_store.get_chunk_ids_by_rowid(rowids))
                    candidate_cids = cids
            except Exception:
                candidate_cids = None

        vs_scores, vs_idxs = self.vector_store.search(qvec, top_k=search_k)
        vector_rank: List[Tuple[str, float]] = []
        if vs_idxs.size and vs_idxs[0].size:
            rowids = [int(idx) for idx in vs_idxs[0].tolist()]
            scrs = vs_scores[0].tolist()
            cids = self.lexical_store.get_chunk_ids_by_rowid(rowids)
            vector_rank = list(zip(cids, scrs))
            if candidate_cids is not None:
                vector_rank = [(cid, s) for cid, s in vector_rank if cid in candidate_cids]

        binary_rank: List[Tuple[str, float]] = []
        if self.binary_store and self.binary_store.has_data():
            bD, bI = self.binary_store.search(qvec, top_k=search_k)
            if bI.size and bI[0].size:
                rowids = [int(idx) for idx in bI[0].tolist() if int(idx) >= 0]
                scores = [float(-d) for d in bD[0].tolist()]
                cids = self.lexical_store.get_chunk_ids_by_rowid(rowids)
                binary_rank = list(zip(cids, scores))

        lexical_rank = self.lexical_store.search(query, top_k=search_k)
        if candidate_cids is not None:
            lexical_rank = [(cid, s) for cid, s in lexical_rank if cid in candidate_cids]

        # Optimized fusion with reduced candidate pool for faster reranking
        fusion_candidates = min(25, top_k * 2)  # Fewer candidates for fusion
        fused = reciprocal_rank_fusion([vector_rank, lexical_rank, binary_rank])[:fusion_candidates]

        # Stage 2: Rerank (optimized)
        chunk_map = {cid: text for _, cid, _, text in self.lexical_store.get_chunks([cid for cid, _ in fused])}
        items = [(cid, chunk_map.get(cid, "")) for cid, _ in fused]
        if self.reranker:
            reranked = self.reranker.rerank(query, items, top_k=top_k)
        else:
            reranked = fused[:top_k]

        # Materialize results
        # Avoid duplicate DB reads: unique CIDs
        unique_cids = list({cid for cid, _ in reranked})
        rows = self.lexical_store.get_chunks(unique_cids)
        row_map = {cid: (doc_id, page, text) for (doc_id, cid, page, text) in rows}
        results: List[SearchResult] = []
        for cid, score in reranked:
            if cid in row_map:
                doc_id, page, text = row_map[cid]
                results.append(SearchResult(doc_id=doc_id, chunk_id=cid, page=page, text=text, score=score))
        # Cache set
        try:
            set_json(self.redis_client, cache_key, [r.__dict__ for r in results], ttl_seconds=self.cache_ttl_seconds)
        except Exception:
            pass
        return results
