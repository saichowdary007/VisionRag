from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import pypdfium2 as pdfium
def _resize_long_side(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    if w >= h:
        new_w = max_dim
        new_h = int(h * (max_dim / float(w)))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / float(h)))
    return img.resize((new_w, new_h), Image.LANCZOS)
def render_pdf_to_images(
    pdf_path: str, out_dir: str, max_dim: int = 1024
) -> Tuple[str, List[str], List[str]]:
    """
    Render a PDF into page images.
    - Writes to `{out_dir}/{doc_id}/{page:04d}.png`
    - Resizes to keep aspect ratio with long side at `max_dim`.
    Returns (doc_id, [absolute image paths]).
    """
    pdf_path = str(pdf_path)
    out_dir = str(out_dir)
    p = Path(pdf_path)
    doc_id = p.stem
    dest_dir = Path(out_dir) / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    images: List[str] = []
    texts: List[str] = []
    for i in range(len(pdf)):
        page = pdf[i]
        # Render at a moderate scale for readability
        pil = page.render(scale=2).to_pil()
        pil = pil.convert("RGB")
        pil = _resize_long_side(pil, max_dim=max_dim)
        out_path = dest_dir / f"{i+1:04d}.png"
        pil.save(out_path, format="PNG")
        images.append(str(out_path.resolve()))
        # Try to extract page text using PyPDF2 as a fallback text source
        try:
            # PyPDF2 reads from the original pdf_path
            from PyPDF2 import PdfReader
            # Open once per loop (small PDFs). For large PDFs, this could be optimized.
            rd = PdfReader(pdf_path)
            pg = rd.pages[i]
            txt = (pg.extract_text() or "").strip()
        except Exception:
            txt = ""
        texts.append(txt)
    return doc_id, images, texts
