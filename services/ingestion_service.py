"""Ingestion service: PDFs → page images → ColPali multi‑vector embeddings → Milvus
- Uses ColPali (vidore/colpali-v1.3) multi‑vector outputs (1024×128 per page)
- Stores **one row per patch** with page metadata for late‑interaction MaxSim retrieval
- MilvusClient schema: FLOAT_VECTOR(128) + VARCHAR meta, metric=IP (dot)
"""

import os
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image
from pdf2image import convert_from_path

from pymilvus import MilvusClient, DataType
from colpali_engine.models import ColPali, ColPaliProcessor

# --- Environment (aligned with llm.txt / compose) ---
MILVUS_URI = os.getenv("MILVUS_URI", "./storage/milvus.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_vision_collection")
RETRIEVER_MODEL_ID = os.getenv("RETRIEVER_MODEL_ID", "vidore/colpali-v1.3")
IMAGE_DPI = int(os.getenv("IMAGE_DPI", "200"))
DIMENSION = 128  # ColPali patch vector dim

# --- Directories ---
ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"
PAGES_DIR = ROOT / "pages"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Device selection (Apple MPS → CUDA → CPU) ---
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# --- Lazy singletons ---
_milvus: MilvusClient | None = None
_model: ColPali | None = None
_proc: ColPaliProcessor | None = None


def milvus() -> MilvusClient:
    global _milvus
    if _milvus is None:
        _milvus = MilvusClient(uri=MILVUS_URI)
    return _milvus


def ensure_collection() -> None:
    client = milvus()
    if client.has_collection(collection_name=COLLECTION_NAME):
        return
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field("page_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)

    index_params = MilvusClient.prepare_index_params()
    # Inner Product (IP) is standard for ColBERT/ColPali late‑interaction scoring
    index_params.add_index(field_name="vector", metric_type="IP")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def retriever() -> tuple[ColPali, ColPaliProcessor]:
    global _model, _proc
    if _model is None:
        _model = ColPali.from_pretrained(
            RETRIEVER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        ).eval()
        _proc = ColPaliProcessor.from_pretrained(RETRIEVER_MODEL_ID)
    return _model, _proc


@torch.no_grad()
def embed_page(image: Image.Image):
    """Return (1024, 128) patch vectors for a single page image."""
    model, proc = retriever()
    inputs = proc.process_images([image]).to(DEVICE)
    # Model returns shape: (1, 1024, 128)
    return model(**inputs).detach().to("cpu").numpy()[0]


def process_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Convert PDF → images, embed each page, insert to Milvus (multi‑vector)."""
    print(f"Processing {pdf_path}")

    # Convert PDF to images (requires poppler-utils available in PATH)
    images: List[Image.Image]
    try:
        images = convert_from_path(pdf_path, dpi=IMAGE_DPI)
    except Exception as e:
        msg = f"Failed to convert {pdf_path.name}: {e}"
        print(msg)
        return {"pages": 0, "error": str(e)}

    ensure_collection()

    doc_id = pdf_path.stem
    total_vectors = 0

    for i, img in enumerate(images):
        page_num = i + 1
        page_name = f"{doc_id}_page_{page_num}.png"
        page_path = PAGES_DIR / page_name
        img.save(page_path, "PNG")

        # Generate multi‑vector embeddings: (1024, 128)
        vecs = embed_page(img)

        # Insert **one row per patch** with page metadata
        entities = [
            {"vector": vecs[j].tolist(), "page_path": str(page_path), "doc_id": doc_id}
            for j in range(vecs.shape[0])
        ]
        milvus().insert(collection_name=COLLECTION_NAME, data=entities)
        total_vectors += vecs.shape[0]
        print(f"  Indexed page {page_num}: {vecs.shape[0]} vectors → {page_name}")

    print(f"Successfully processed {pdf_path.name}: {len(images)} pages, {total_vectors} vectors")
    return {"pages": len(images), "vectors": total_vectors}


def process_all_pdfs() -> Dict[str, Any]:
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in documents directory.")
        return {"documents": 0, "total_pages": 0, "vectors": 0, "collection": COLLECTION_NAME}

    print(f"Found {len(pdf_files)} PDF file(s)")
    total_pages = 0
    total_vecs = 0
    processed_docs = 0

    for pdf in pdf_files:
        res = process_pdf(pdf)
        if "error" in res:
            print(f"  Error: {res['error']}")
            continue
        total_pages += res.get("pages", 0)
        total_vecs += res.get("vectors", 0)
        processed_docs += 1

    return {
        "documents": processed_docs,
        "total_pages": total_pages,
        "vectors": total_vecs,
        "collection": COLLECTION_NAME,
    }


if __name__ == "__main__":
    print(f"Loading retriever '{RETRIEVER_MODEL_ID}' on device '{DEVICE}'…")
    retriever()  # warm‑load
    print("Connecting to Milvus…")
    ensure_collection()
    print("Starting ingestion…")
    result = process_all_pdfs()
    print(f"Ingestion complete: {result}")
