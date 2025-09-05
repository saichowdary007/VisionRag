#!/bin/bash
set -euo pipefail

echo "Setting up M1-Optimized RAG System..."

if [[ $(uname -m) != "arm64" ]]; then
  echo "Warning: This setup is optimized for Apple Silicon (arm64)."
fi

# Ensure Homebrew & Python 3.11
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew required. Install from https://brew.sh"; exit 1;
fi

brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
brew list cmake >/dev/null 2>&1 || brew install cmake

PY=python3.11
$PY -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements-m1.txt

echo "Creating folders..."
mkdir -p rag-service/documents config rag-service/rag_service/__pycache__
mkdir -p data

if [[ ! -f config/config.yaml ]]; then
  cat > config/config.yaml <<'YAML'
device: mps
max_memory_gb: 8
db_path: ./rag-service/data/documents.db
faiss_index_path: ./rag-service/data/vectors.faiss
docs_dir: ./rag-service/documents
prometheus_port: 9108
text_embedding_model: nomic-ai/nomic-embed-text-v1.5
reranker_model: BAAI/bge-reranker-base
siglip_model: google/siglip-base-patch16-224
YAML
fi

echo "You can now add PDFs to rag-service/documents and build indexes:"
echo "  source venv/bin/activate && python build_indexes.py --memory-limit 12GB"
echo "Then start services:"
echo "  docker-compose -f docker-compose-m1.yml up -d"
echo "API: http://localhost:8001  |  Prometheus: http://localhost:9090"

