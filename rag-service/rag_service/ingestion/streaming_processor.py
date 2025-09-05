from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator, List

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
                # Extract text blocks and images; OCR fallback if needed
                text_blocks = page.get_text("text") or ""
                if len(text_blocks.strip()) < 20 and pytesseract is not None and shutil.which("tesseract"):
                    # Render page to image and OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # upscale x2 for better OCR
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        text_blocks = pytesseract.image_to_string(img) or text_blocks
                    except Exception:
                        pass
                images = self._extract_page_images(page, doc_id, page_num)
                # OCR the extracted images and append to text for better recall of diagrams
                if images and pytesseract is not None and shutil.which("tesseract"):
                    ocr_texts: List[str] = []
                    for pth in images:
                        try:
                            with Image.open(pth) as im:
                                txt = pytesseract.image_to_string(im)
                                if txt and txt.strip():
                                    ocr_texts.append(txt.strip())
                        except Exception:
                            pass
                    if ocr_texts:
                        text_blocks = (text_blocks + "\n" + "\n".join(ocr_texts)).strip()
                # Create chunks
                for cid, chunk in enumerate(self._create_page_chunks(text_blocks, images, doc_id, page_num)):
                    chunk.chunk_id = f"{doc_id}_{page_num}_{cid}"
                    yield chunk
                # Cleanup
                del text_blocks, images
                gc.collect()
        finally:
            doc.close()

    def _extract_page_images(self, page: fitz.Page, doc_id: str, page_num: int) -> List[str]:
        paths: List[str] = []
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = None
            try:
                pix = fitz.Pixmap(page.parent, xref)
                out_path = self.temp_dir / f"{doc_id}_p{page_num}_{i}.png"
                pix.save(out_path)
                paths.append(str(out_path))
            except Exception:
                pass
            finally:
                pix = None  # release
        return paths

    def _create_page_chunks(self, text: str, image_paths: List[str], doc_id: str, page_num: int) -> List[Chunk]:
        # Sentence-aware chunking with overlap
        sentences = self._split_sentences(text)
        target = 1200  # target characters per chunk
        overlap_ratio = 0.25
        chunks: List[Chunk] = []
        buf: List[str] = []
        cur_len = 0
        for sent in sentences:
            if not sent:
                continue
            if cur_len + len(sent) + (1 if buf else 0) > target and buf:
                chunks.append(Chunk(doc_id=doc_id, chunk_id="", page=page_num, text="\n".join(buf), image_paths=image_paths[:]))
                # Overlap last portion
                overlap_chars = int(target * overlap_ratio)
                tail = " ".join("\n".join(buf)[-overlap_chars:].split(" ")[:100])
                buf = [tail] if tail else []
                cur_len = len(tail)
            buf.append(sent)
            cur_len += len(sent) + 1
        if buf:
            chunks.append(Chunk(doc_id=doc_id, chunk_id="", page=page_num, text="\n".join(buf), image_paths=image_paths[:]))
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        # Naive sentence split honoring punctuation
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]
