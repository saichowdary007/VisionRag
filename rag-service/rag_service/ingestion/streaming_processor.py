from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Tuple, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import io
import re
import shutil

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore

from ..types import Chunk


@dataclass
class StreamingDocumentProcessor:
    max_memory_mb: int = 4000

    def __post_init__(self):
        self.max_memory = self.max_memory_mb * 1024 * 1024
        self.temp_dir = Path(tempfile.mkdtemp(prefix="rag_ingest_"))

    def process_pdf_streaming(self, pdf_path: str) -> Iterator[Chunk]:
        p = Path(pdf_path)
        doc_id = p.stem
        doc = fitz.open(pdf_path)

        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                # Extract text blocks with positions
                blocks = self._extract_text_blocks(page)
                # Extract images with bboxes and save
                images = self._extract_page_images_with_bboxes(page, doc_id, page_num)
                # OCR fallback for page-level text
                page_text = "\n".join(b["text"] for b in blocks) if blocks else ""
                if len(page_text.strip()) < 20 and pytesseract is not None and shutil.which("tesseract"):
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        page_text = pytesseract.image_to_string(img) or page_text
                    except Exception:
                        pass
                    if page_text:
                        blocks = [{"bbox": [0, 0, 0, 0], "text": page_text}]
                # Create chunks from blocks and associate nearest images (by bbox centroid distance)
                for cid, chunk in enumerate(self._create_page_chunks(blocks, images, doc_id, page_num)):
                    chunk.chunk_id = f"{doc_id}_{page_num}_{cid}"
                    yield chunk
                # Cleanup
                del blocks, images
                gc.collect()
        finally:
            doc.close()

    def _extract_text_blocks(self, page: fitz.Page) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        try:
            for b in page.get_text("blocks"):
                # PyMuPDF returns (x0, y0, x1, y1, text, block_no, block_type, ...)
                if len(b) >= 5 and isinstance(b[4], str) and b[4].strip():
                    x0, y0, x1, y1, text = b[:5]
                    blocks.append({"bbox": [float(x0), float(y0), float(x1), float(y1)], "text": text})
        except Exception:
            pass
        return blocks

    def _extract_page_images_with_bboxes(self, page: fitz.Page, doc_id: str, page_num: int) -> List[Dict[str, Any]]:
        # Map xref -> bbox via raw dict
        xref_boxes: List[Tuple[int, List[float]]] = []
        try:
            raw = page.get_text("rawdict")
            for block in raw.get("blocks", []):
                if block.get("type") == 1:  # image block
                    bbox = [float(v) for v in block.get("bbox", [0, 0, 0, 0])]
                    img = block.get("image", {})
                    xref = int(img.get("xref", -1))
                    if xref > 0:
                        xref_boxes.append((xref, bbox))
        except Exception:
            pass

        results: List[Dict[str, Any]] = []
        for i, (xref, bbox) in enumerate(xref_boxes):
            pix = None
            try:
                pix = fitz.Pixmap(page.parent, xref)
                out_path = self.temp_dir / f"{doc_id}_p{page_num}_{i}.png"
                pix.save(out_path)
                image_id = f"{doc_id}_p{page_num}_{i}"
                results.append({
                    "image_id": image_id,
                    "path": str(out_path),
                    "bbox": bbox,
                })
            except Exception:
                continue
            finally:
                pix = None
        return results

    def _create_page_chunks(self, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], doc_id: str, page_num: int) -> List[Chunk]:
        # Merge consecutive text blocks until target char count, track bbox union
        target = 1200
        overlap_ratio = 0.25
        chunks: List[Chunk] = []
        buf_text: List[str] = []
        buf_boxes: List[List[float]] = []
        cur_len = 0

        def pack_chunk() -> None:
            nonlocal buf_text, buf_boxes, chunks
            if not buf_text:
                return
            bbox = self._union_bbox(buf_boxes) if buf_boxes else None
            ch = Chunk(doc_id=doc_id, chunk_id="", page=page_num, text="\n".join(buf_text), image_paths=[], image_ids=[], bbox=bbox)
            chunks.append(ch)

        for b in blocks:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            b_len = len(text)
            if cur_len + b_len + (1 if buf_text else 0) > target and buf_text:
                pack_chunk()
                # Overlap: keep tail portion of previous chunk
                overlap_chars = int(target * overlap_ratio)
                tail = " ".join("\n".join(buf_text)[-overlap_chars:].split(" ")[:100])
                buf_text = [tail] if tail else []
                buf_boxes = [b.get("bbox")] if b.get("bbox") else []
                cur_len = len(tail)
            buf_text.append(text)
            if b.get("bbox"):
                buf_boxes.append(b["bbox"])
            cur_len += b_len + 1

        if buf_text:
            pack_chunk()

        # Associate images to nearest chunk by centroid distance
        for im in images:
            ib = im.get("bbox")
            if not ib:
                continue
            ic = self._bbox_center(ib)
            nearest_idx = None
            nearest_d = 1e12
            for idx, ch in enumerate(chunks):
                if not ch.bbox:
                    continue
                cb = ch.bbox
                cc = self._bbox_center(cb)
                d = (ic[0] - cc[0]) ** 2 + (ic[1] - cc[1]) ** 2
                if d < nearest_d:
                    nearest_d = d
                    nearest_idx = idx
            if nearest_idx is not None:
                ch = chunks[nearest_idx]
                if ch.image_paths is None:
                    ch.image_paths = []
                if ch.image_ids is None:
                    ch.image_ids = []
                ch.image_paths.append(im["path"])
                ch.image_ids.append(im["image_id"])

        return chunks

    def _union_bbox(self, boxes: List[List[float]]) -> List[float]:
        xs0 = [b[0] for b in boxes if b]
        ys0 = [b[1] for b in boxes if b]
        xs1 = [b[2] for b in boxes if b]
        ys1 = [b[3] for b in boxes if b]
        if not xs0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))]

    def _bbox_center(self, b: List[float]) -> Tuple[float, float]:
        x0, y0, x1, y1 = b
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    # Compatibility helper retained (no longer used)
    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    # New: extract all document images with bboxes for indexing
    def extract_document_images(self, pdf_path: str) -> List[Tuple[str, str, int, str, str]]:
        """Return list of (image_id, doc_id, page, path, bbox_json) for entire PDF."""
        p = Path(pdf_path)
        doc_id = p.stem
        out: List[Tuple[str, str, int, str, str]] = []
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                images = self._extract_page_images_with_bboxes(page, doc_id, page_num)
                import json as _json
                for im in images:
                    bbox_json = _json.dumps(im.get("bbox", [0, 0, 0, 0]))
                    out.append((im["image_id"], doc_id, page_num, im["path"], bbox_json))
        finally:
            doc.close()
        return out
