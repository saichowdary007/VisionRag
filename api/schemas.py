from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field
class IngestBody(BaseModel):
    pdf_url: Optional[HttpUrl] = Field(None, description="URL to a PDF to ingest")
    doc_id: Optional[str] = Field(None, description="Optional document id to use for storage")
class IngestResponse(BaseModel):
    ok: bool
    doc_id: str
    pages: int
    images: List[str]
class AskBody(BaseModel):
    question: str
    k: int = 5
    m: int = 3
    doc_id: Optional[str] = None
class RetrieverHit(BaseModel):
    image_path: str
    score: float
class SearchResponse(BaseModel):
    hits: List[RetrieverHit]
