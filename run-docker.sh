#!/bin/bash

# VLM RAG Application Docker Runner
# This script helps you run the entire application stack in Docker

set -e

echo "🚀 VLM RAG Application Docker Setup"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating default .env file..."
    cat > .env << EOF
# VLM Configuration
VLM_BASE_URL=http://host.docker.internal:11434/v1
VLM_MODEL=qwen2.5-vl:7b

# Retriever Configuration
MODEL_ID=vidore/colpali-v1.3
MILVUS_URI=./milvus_data/milvus.db
COLLECTION_NAME=colpali_multivector_collection
TOP_K=5
MAX_IMAGES=3
EOF
    echo "✅ Created .env file with default configuration"
fi

# Create milvus_data directory if it doesn't exist
if [ ! -d "milvus_data" ]; then
    echo "📁 Creating milvus_data directory..."
    mkdir -p milvus_data
fi

echo "🏗️  Building and starting all services..."
echo "This may take several minutes on first run..."
echo ""

cd deployment

# Use docker compose (newer version) if available, otherwise docker-compose
if docker compose version &> /dev/null; then
    docker compose up --build -d
else
    docker-compose up --build -d
fi

echo ""
echo "🎉 Application is starting up!"
echo ""
echo "📊 Service Status:"
echo "  - Frontend:    http://localhost:3000"
echo "  - Backend API: http://localhost:8080"
echo "  - Retriever:   http://localhost:8081"
echo ""
echo "🔍 To check service logs:"
echo "  cd deployment && docker compose logs -f [service-name]"
echo ""
echo "🛑 To stop all services:"
echo "  cd deployment && docker compose down"
echo ""
echo "⏳ Please wait for all services to be healthy before accessing the frontend."
echo "   You can check status with: docker compose ps"
