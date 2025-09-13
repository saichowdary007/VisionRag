import os
import sys
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from PIL import Image
import base64
from io import BytesIO

# Add project root to Python path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set default Milvus URI before importing pymilvus to avoid connection errors
if not os.getenv("MILVUS_URI"):
    os.environ["MILVUS_URI"] = "http://localhost:19530"

from libs.clients.vlm_client import vision_chat
from pymilvus import MilvusClient, DataType
from colpali_engine.models import ColPali, ColPaliProcessor

# --- Configuration ---
# Retriever Model (same as ingestion)
RETRIEVER_MODEL_ID = os.getenv("RETRIEVER_MODEL_ID", "vidore/colpali-v1.3")

# Enhanced GPU detection with fallback
def detect_device():
    """Detect the best available device for model loading."""
    # Priority: CUDA -> MPS (Apple Silicon) -> CPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CUDA available: {gpu_count} GPUs detected")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
        return "mps"
    else:
        print("Using CPU (no GPU acceleration available)")
        return "cpu"

# Device selection with auto-detection
device_env = os.getenv("DEVICE", "").lower()
if device_env == "auto":
    DEVICE = detect_device()
else:
    DEVICE = device_env or detect_device()

print(f"Selected device: {DEVICE}")

# Milvus Vector DB (disabled for Apple Silicon compatibility)
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_vision_collection")
USE_MILVUS = os.getenv("USE_MILVUS", "false").lower() == "true"  # Set to "true" to enable Milvus
INDEX_ON_INGEST = os.getenv("INDEX_ON_INGEST", "false").lower() == "true"

# QA Model (served by Ollama/vLLM)
QA_API_BASE = os.getenv("QA_API_BASE", "http://localhost:11434/v1")
QA_API_KEY = os.getenv("QA_API_KEY", "ollama")
QA_MODEL_ID = os.getenv("QA_MODEL_ID", "qwen3:4b-instruct-2507-q4_K_M")

# Retrieval Parameters
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "50"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))

# --- Initialize Application and Models ---
app = FastAPI(title="RAG Vision API", version="1.0.0")

# Initialize Milvus Client (only if enabled)
milvus_client = None
if USE_MILVUS:
    try:
        milvus_client = MilvusClient(uri=MILVUS_URI)
        print("Milvus client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Milvus client: {e}")
        print("Running without vector database - retrieval will be limited")
else:
    print("Milvus disabled - running with limited functionality")

# Ensure collection exists when using Milvus
def ensure_collection():
    if not (USE_MILVUS and milvus_client):
        return
    try:
        if milvus_client.has_collection(collection_name=COLLECTION_NAME):
            return
        schema = MilvusClient.create_schema(auto_id=True, description="ColPali multi-vector store")
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
        schema.add_field(field_name="page_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type="IP")
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
            consistency_level="Eventually",
        )
        print(f"Created Milvus collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Failed to ensure collection: {e}")

retriever_model = None
retriever_processor = None

def init_retriever():
    global retriever_model, retriever_processor
    if not USE_MILVUS:
        return
    try:
        print(f"Loading retriever model '{RETRIEVER_MODEL_ID}' on device '{DEVICE}'...")

        # Device-specific optimizations
        loading_kwargs = {
            "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
            "trust_remote_code": True,
        }

        # Data type and device mapping optimization
        if DEVICE == "cuda":
            loading_kwargs.update({
                "torch_dtype": torch.bfloat16,  # Best for CUDA GPUs
                "device_map": "auto",  # Automatic device placement
            })
        elif DEVICE == "mps":
            loading_kwargs.update({
                "torch_dtype": torch.float16,  # MPS works well with float16 on M1/M2
                "device_map": {"": DEVICE},  # Explicit MPS mapping
            })
        else:  # CPU
            loading_kwargs.update({
                "torch_dtype": torch.float32,  # More stable on CPU
                "device_map": {"": DEVICE},
            })

        # Load model with optimizations
        model = ColPali.from_pretrained(
            RETRIEVER_MODEL_ID,
            **loading_kwargs
        ).eval()

        # Ensure model is on the correct device
        if hasattr(model, 'device') and str(model.device) != DEVICE:
            if DEVICE in ["cuda", "mps"]:
                model = model.to(DEVICE)
            elif DEVICE == "auto":
                # Let transformers handle device placement
                pass

        # Load processor with fast processing
        processor = ColPaliProcessor.from_pretrained(
            RETRIEVER_MODEL_ID,
            use_fast=True  # Enable fast image processing
        )

        ensure_collection()
        retriever_model = model
        retriever_processor = processor
        print(f"✅ Retriever model loaded successfully on {DEVICE}")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")

        # Device information
        if DEVICE == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print(f"   Multi-GPU setup: {gpu_count} GPUs available")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"     GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   Single GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        elif DEVICE == "mps" and torch.backends.mps.is_available():
            print(f"   Apple Silicon GPU: MPS (Metal Performance Shaders)")
            # Get MPS memory info if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                print(f"   System Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
            except ImportError:
                print("   MPS acceleration enabled")

    except Exception as e:
        print(f"❌ Failed to load retriever model: {e}")
        if "CUDA" in str(e):
            print("💡 CUDA error - trying with CPU fallback...")
            # Fallback to CPU if GPU fails
            try:
                model = ColPali.from_pretrained(
                    RETRIEVER_MODEL_ID,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                ).eval()
                processor = ColPaliProcessor.from_pretrained(RETRIEVER_MODEL_ID, use_fast=True)
                retriever_model = model
                retriever_processor = processor
                print("✅ Model loaded on CPU as fallback")
            except Exception as e2:
                print(f"❌ CPU fallback also failed: {e2}")
                retriever_model = None
                retriever_processor = None
        else:
            retriever_model = None
            retriever_processor = None

@app.on_event("startup")
def _startup():
    # Load retriever in a background thread to avoid blocking server start
    try:
        import threading
        threading.Thread(target=init_retriever, daemon=True).start()
    except Exception as e:
        print(f"Failed to start retriever init thread: {e}")

# QA Model Client is initialized via vision_chat function

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

# --- Ingestion Models ---
class IngestItem(BaseModel):
    page_id: str = Field(..., description="Document and page identifier, e.g., 'doc123:1'")
    image_b64: str = Field(..., description="Base64-encoded JPEG of the rendered page")

class IngestRequest(BaseModel):
    pages: List[IngestItem]

class IngestResponse(BaseModel):
    status: str
    pages_added: int

class SearchRequest(BaseModel):
    text: str
    top_k: int = 5

class SearchResponse(BaseModel):
    hits: List[dict]

# --- Core Query Logic ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    query_text = request.query

    # Check if models are available
    if not milvus_client or not USE_MILVUS or not retriever_model or not retriever_processor:
        # Fallback: Use QA model directly without document retrieval
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question directly since document retrieval is not available."
            },
            {
                "role": "user",
                "content": query_text
            }
        ]

        try:
            # Use vision_chat function with empty image list for text-only queries
            answer = vision_chat(query_text, [])
            note = "[Note: Running without vector database"
            if not retriever_model:
                note += " and retriever model"
            note += "] "
            return QueryResponse(
                answer=f"{note}{answer}",
                sources=[]
            )
        except Exception as e:
            return QueryResponse(
                answer=f"Failed to get answer from QA model: {str(e)}",
                sources=[]
            )

    # 1. Encode the user's query to multi-vector representation
    with torch.no_grad():
        query_inputs = retriever_processor.process_queries([query_text]).to(DEVICE)
        query_embeddings = retriever_model(**query_inputs)  # [1, num_tokens, 128]

    # 2. Phase 1: Candidate Retrieval from Milvus
    # Search for candidate vectors using the mean of query embeddings as a probe
    query_probe = torch.mean(query_embeddings, dim=1).cpu().float().numpy()[0].tolist()

    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_probe],
        limit=min(max(TOP_K_CANDIDATES * 1024, 1024), 16384),  # Cap at Milvus limit of 16384
        output_fields=["page_path", "doc_id"],
    )

    # Extract unique pages from search results
    candidate_pages = set()
    for hit in search_res[0]:  # search_res is a list with one result set
        if 'entity' in hit:
            page_path = hit['entity'].get('page_path')
            if page_path:
                candidate_pages.add(page_path)

    # 3. Phase 2: Rerank with MaxSim Late Interaction
    page_scores = {}
    for page_path in candidate_pages:
        # Retrieve all 1024 vectors for this page
        page_vectors_res = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f"page_path == '{page_path}'",
            output_fields=["vector"],
        )

        if not page_vectors_res:
            continue

        # [1, 1024, 128]
        page_vectors = torch.tensor([item["vector"] for item in page_vectors_res], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # MaxSim score: sum over query tokens of max dot-product with page patches
        with torch.no_grad():
            # query_embeddings: [1, T, 128]
            q = query_embeddings[0]  # [T, 128]
            d = page_vectors[0].T    # [128, 1024]
            sims = torch.matmul(q, d)  # [T, 1024]
            score = sims.max(dim=1).values.sum()
            page_scores[page_path] = float(score.item())

    # Sort pages by score in descending order
    sorted_pages = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)
    top_pages_paths = [page_path for page_path, _ in sorted_pages[:TOP_K_FINAL]]

    # 4. Phase 3: Generate Answer with QA Model
    if not top_pages_paths:
        return QueryResponse(
            answer="I could not find any relevant documents to answer your question.",
            sources=[]
        )

    # Load top pages as images and call the VLM
    images: List[Image.Image] = []
    for page_path in top_pages_paths:
        try:
            images.append(Image.open(page_path).convert("RGB"))
        except Exception:
            pass
    try:
        answer = vision_chat(query_text, images)
    except Exception as e:
        return QueryResponse(answer=f"Failed to query VLM: {e}", sources=[])

    # Prepare sources information
    sources = [
        {"path": path, "score": page_scores[path]}
        for path in top_pages_paths
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pages(payload: IngestRequest):
    """Accepts page images, saves them, and indexes in Milvus when enabled.

    To avoid blocking uploads, inline indexing is controlled by INDEX_ON_INGEST (default false).
    Use /reindex to index saved pages after the retriever is ready.
    """
    pages_dir = os.getenv("PAGE_IMAGE_DIR", "./pages")
    os.makedirs(pages_dir, exist_ok=True)

    ensure_collection()

    pages_added = 0
    for item in payload.pages:
        try:
            # Parse page id like "docId:pageNum"
            doc_id = item.page_id
            page_num = pages_added + 1
            if ":" in item.page_id:
                parts = item.page_id.split(":", 1)
                doc_id = parts[0] or "doc"
                try:
                    page_num = int(parts[1])
                except Exception:
                    pass

            # Decode
            raw = base64.b64decode(item.image_b64)
            img = Image.open(BytesIO(raw)).convert("RGB")

            # Save image to disk (used later by QA model)
            filename = f"{doc_id}_page_{page_num}.png"
            page_path = os.path.join(pages_dir, filename)
            img.save(page_path, format="PNG")

            # Index into Milvus if enabled and requested
            if INDEX_ON_INGEST and USE_MILVUS and milvus_client and retriever_model and retriever_processor:
                with torch.no_grad():
                    inputs = retriever_processor.process_images([img]).to(DEVICE)
                    # [1, 1024, 128]
                    page_embeddings = retriever_model(**inputs).cpu().float().numpy()[0]

                entities = [
                    {
                        "vector": vec.tolist(),
                        "page_path": page_path,
                        "doc_id": doc_id,
                    }
                    for vec in page_embeddings
                ]
                try:
                    milvus_client.insert(collection_name=COLLECTION_NAME, data=entities)
                except Exception as e:
                    print(f"Milvus insert failed for {page_path}: {e}")

            pages_added += 1
        except Exception as e:
            print(f"Failed to process page {item.page_id}: {e}")

    return IngestResponse(status="ok", pages_added=pages_added)

@app.get("/healthz")
def health_check():
    return {
        "status": "ok",
        "model": RETRIEVER_MODEL_ID,
        "use_milvus": USE_MILVUS,
        "retriever_loaded": bool(retriever_model is not None and retriever_processor is not None),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not (USE_MILVUS and milvus_client and retriever_model and retriever_processor):
        return SearchResponse(hits=[])

    with torch.no_grad():
        q = retriever_processor.process_queries([req.text]).to(DEVICE)
        q_embed = retriever_model(**q)  # [1, T, 128]
        probe = torch.mean(q_embed, dim=1).cpu().float().numpy()[0].tolist()

    res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[probe],
        limit=max(req.top_k * 1024, 1024),
        output_fields=["page_path", "doc_id"],
    )

    pages = set()
    for hit in res[0]:
        if 'entity' in hit:
            p = hit['entity'].get('page_path')
            if p:
                pages.add(p)

    scored = []
    for p in pages:
        vecs = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f"page_path == '{p}'",
            output_fields=["vector"],
        )
        if not vecs:
            continue
        dv = torch.tensor([x["vector"] for x in vecs], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            sims = torch.matmul(q_embed[0], dv[0].T)
            score = sims.max(dim=1).values.sum().item()
        scored.append({"page_path": p, "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return SearchResponse(hits=scored[: req.top_k])


@app.post("/reindex", response_model=IngestResponse)
def reindex_pages():
    """Index existing PNG pages under PAGE_IMAGE_DIR into Milvus."""
    if not (USE_MILVUS and milvus_client and retriever_model and retriever_processor):
        return IngestResponse(status="skipped", pages_added=0)

    pages_dir = os.getenv("PAGE_IMAGE_DIR", "./pages")
    ensure_collection()

    added = 0
    try:
        for name in os.listdir(pages_dir):
            if not name.lower().endswith(".png"):
                continue
            page_path = os.path.join(pages_dir, name)
            # Skip if already indexed
            try:
                exists = milvus_client.query(
                    collection_name=COLLECTION_NAME,
                    filter=f"page_path == '{page_path}'",
                    output_fields=["page_path"],
                )
                if exists:
                    continue
            except Exception:
                pass

            # Parse doc id from filename pattern: <doc>_page_<n>.png
            base = os.path.basename(name)
            doc_id = base
            if "_page_" in base:
                doc_id = base.split("_page_")[0]

            try:
                img = Image.open(page_path).convert("RGB")
            except Exception:
                continue
            with torch.no_grad():
                inputs = retriever_processor.process_images([img]).to(DEVICE)
                embs = retriever_model(**inputs).cpu().float().numpy()[0]
            entities = [
                {"vector": v.tolist(), "page_path": page_path, "doc_id": doc_id}
                for v in embs
            ]
            try:
                milvus_client.insert(collection_name=COLLECTION_NAME, data=entities)
                added += 1
            except Exception as e:
                print(f"Reindex insert failed for {page_path}: {e}")
    except Exception as e:
        print(f"Reindex failed: {e}")
    return IngestResponse(status="ok", pages_added=added)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
