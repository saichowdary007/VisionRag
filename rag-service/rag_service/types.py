from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    page: int
    text: str
    image_paths: Optional[List[str]] = None
    image_ids: Optional[List[str]] = None
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] in PDF coords
    # Optional richer metadata
    section: Optional[str] = None
    subsection: Optional[str] = None
    figure_id: Optional[str] = None
    table_id: Optional[str] = None


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    page: int
    text: str
    score: float
