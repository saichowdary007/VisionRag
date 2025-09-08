from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from byaldi_wrapper import ByaldiRetriever


# Configuration (support both DATA_DIR and legacy BYALDI_DATA)
DATA_DIR = Path(os.getenv("DATA_DIR") or os.getenv("BYALDI_DATA") or "/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize retriever
retriever = ByaldiRetriever(str(DATA_DIR))

app = FastAPI(title="Retriever Service", version="1.0.0")


# Pydantic models
class IndexRequest(BaseModel):
    doc_id: str
    images: List[str]
    texts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_id: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]]
    total: int


@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    return {"status": "ok", "service": "retriever"}


@app.post("/index")
def index_document(request: IndexRequest):
    """Index a document with its page images."""
    try:
        retriever.index_document(request.doc_id, request.images, request.texts)
        return {
            "status": "success",
            "doc_id": request.doc_id,
            "images_indexed": len(request.images)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/search")
def search_documents(request: SearchRequest) -> SearchResponse:
    """Search for relevant documents."""
    try:
        hits = retriever.search(request.query, request.doc_id, request.k)
        return SearchResponse(hits=hits, total=len(hits))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    try:
        docs = retriever.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document from the index."""
    try:
        success = retriever.delete_document(doc_id)
        if success:
            return {"status": "success", "doc_id": doc_id}
        else:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/documents/{doc_id}/stats")
def get_document_stats(doc_id: str):
    """Get statistics for a specific document."""
    try:
        # This would need to be implemented in the ByaldiRetriever
        # For now, return basic info
        if doc_id in retriever.list_documents():
            return {"doc_id": doc_id, "status": "indexed"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
