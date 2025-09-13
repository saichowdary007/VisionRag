import os
import torch
from PIL import Image
from pdf2image import convert_from_path
from pymilvus import MilvusClient, DataType
from colpali_engine.models import ColPali, ColPaliProcessor
from tqdm import tqdm

# --- Configuration ---
# Models & Processing
RETRIEVER_MODEL_ID = os.getenv("RETRIEVER_MODEL_ID", "vidore/colpali-v1.3")
DEVICE = os.getenv("DEVICE", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DPI = int(os.getenv("IMAGE_DPI", "200"))

# Paths
SOURCE_DOCS_DIR = os.getenv("SOURCE_DOCS_DIR", "./documents")
PAGE_IMAGE_DIR = os.getenv("PAGE_IMAGE_DIR", "./pages")
os.makedirs(PAGE_IMAGE_DIR, exist_ok=True)
os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)

# Milvus Vector DB
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_vision_collection")
DIMENSION = int(os.getenv("DIMENSION", "128"))

# --- Milvus Client and Collection Setup ---
client = MilvusClient(uri=MILVUS_URI)

def setup_milvus_collection():
    """Creates the Milvus collection with the required schema if it doesn't exist."""
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field(field_name="page_path", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="IP")  # Inner Product is required for MaxSim

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print("Collection created successfully.")

# --- Model Loading ---
print(f"Loading retriever model '{RETRIEVER_MODEL_ID}' on device '{DEVICE}'...")
model = ColPali.from_pretrained(
    RETRIEVER_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
).eval()
processor = ColPaliProcessor.from_pretrained(RETRIEVER_MODEL_ID)

# --- Core Ingestion Logic ---
def process_and_index_documents():
    """Main function to convert PDFs to images, generate embeddings, and index them in Milvus."""
    setup_milvus_collection()
    
    pdf_files = [f for f in os.listdir(SOURCE_DOCS_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {SOURCE_DOCS_DIR}")
        return
    
    for pdf_file in pdf_files:
        doc_id = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(SOURCE_DOCS_DIR, pdf_file)
        print(f"\nProcessing document: {pdf_file}")
        
        # 1. Convert PDF to Images
        try:
            images = convert_from_path(pdf_path, dpi=IMAGE_DPI)
        except Exception as e:
            print(f"Could not process PDF {pdf_file}: {e}")
            continue
        
        # 2. Process each page
        for i, image in enumerate(tqdm(images, desc=f"Embedding pages for {doc_id}")):
            page_num = i + 1
            page_path = os.path.join(PAGE_IMAGE_DIR, f"{doc_id}_page_{page_num}.png")
            image.save(page_path, "PNG")
            
            # 3. Generate Multi-Vector Embeddings
            with torch.no_grad():
                inputs = processor.process_images([image]).to(DEVICE)
                # Output is a tensor of shape (1 page, 1024 patches, 128 dims)
                page_embeddings = model(**inputs).cpu().numpy()[0]  # Remove batch dimension
                
            # 4. Insert into Milvus
            # Each of the 1024 patch vectors is a separate entry
            entities = [
                {
                    "vector": vector.tolist(),
                    "page_path": page_path,
                    "doc_id": doc_id
                }
                for vector in page_embeddings
            ]
            
            client.insert(collection_name=COLLECTION_NAME, data=entities)
    
    print("\nIngestion complete. All documents have been processed and indexed.")

if __name__ == "__main__":
    process_and_index_documents()