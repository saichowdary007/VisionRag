# RAG Vision Pipeline

A production-grade RAG (Retrieval-Augmented Generation) vision pipeline built with ColPali for multi-vector document retrieval and vision-language models for question answering. Features a modern Next.js frontend with real-time streaming responses.

## Features

### 🎨 User Interface
- **Modern Chat Design**: Clean, ChatGPT-inspired interface
- **Message Bubbles**: Distinct styling for user and AI messages
- **Responsive Layout**: Optimized for mobile and desktop
- **Dark/Light Theme**: Toggle between themes with system preference detection
- **Typing Indicators**: Animated typing indicator during AI responses
- **Message Timestamps**: Formatted timestamps for each message

### 🚀 Functionality
- **Real-time Streaming**: Server-sent events for streaming AI responses
- **Message Persistence**: Local storage for chat history
- **Error Handling**: Comprehensive error states and user feedback
- **Input Validation**: Message sanitization and validation
- **Auto-scroll**: Automatic scrolling to new messages
- **Clear History**: Option to clear all messages

### 🛠 Technical Implementation
- **Next.js 14+**: App Router with TypeScript
- **Custom Hooks**: Reusable logic for chat and theme management
- **Component Architecture**: Modular, maintainable component structure
- **Performance Optimized**: Memoization and efficient re-renders
- **Accessibility**: ARIA labels and keyboard navigation support

## Project Structure

```
src/
├── app/
│   ├── api/
│   │   ├── chat/route.ts       # Chat API endpoint with streaming
│   │   ├── health/route.ts     # Health check endpoint
│   │   ├── ingest/route.ts     # Document ingestion endpoint
│   │   └── reindex/route.ts    # Reindexing endpoint
│   ├── globals.css             # Global styles and theme variables
│   ├── layout.tsx              # Root layout component
│   └── page.tsx                # Main page component
├── components/
│   ├── chat/
│   │   ├── ChatContainer.tsx   # Main chat container
│   │   ├── ChatHeader.tsx      # Header with controls
│   │   ├── MessageBubble.tsx   # Individual message component
│   │   └── MessageInput.tsx    # Message input with auto-resize
│   └── ui/
│       ├── Button.tsx          # Reusable button component
│       └── TypingIndicator.tsx # Animated typing indicator
├── hooks/
│   ├── useChat.ts              # Chat state management
│   ├── useIngest.ts            # Document ingestion hook
│   └── useTheme.ts             # Theme management
├── lib/
│   └── utils.ts                # Utility functions
└── types/
    └── chat.ts                 # TypeScript interfaces

services/
├── api_service.py              # Consolidated API service with ColPali and VLM
└── ingestion_service.py        # Document processing and vector indexing

libs/
├── clients/
│   └── vlm_client.py           # Vision-Language Model client
└── storage/
    └── milvus_store.py         # Milvus vector database operations

deployment/
├── docker-compose.yml          # Multi-service Docker deployment
├── Dockerfile.api              # API service container
├── Dockerfile.ingestion        # Ingestion service container
└── frontend.Dockerfile         # Frontend container
```

## Getting Started

### Prerequisites
- Node.js 18+ (for local development)
- npm or yarn (for local development)
- Docker and Docker Compose (for containerized deployment)

### Local Development

1. **Clone and install dependencies:**
   ```bash
   npm install
   ```

2. **Start Docker services (Milvus, API, Ingestion):**
   ```bash
   cd deployment
   docker compose up -d
   ```

3. **Configure environment (optional):**
   Create a `.env.local` file in the root directory:
   ```bash
   # When running Next.js locally, connect to Docker services
   BACKEND_API_URL=http://localhost:8000
   ```

4. **Run the development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Backend Services

The backend consists of two main services:

**API Service** (`services/api_service.py`):
- Combined ColPali retrieval and VLM chat functionality
- FastAPI endpoints for chat, health checks, and reindexing
- Runs on port 8000

**Ingestion Service** (`services/ingestion_service.py`):
- Document processing and vector indexing
- PDF to image conversion and ColPali embedding generation
- One-time execution for document ingestion

To run the API service locally:
```bash
pip install -r requirements.txt
python services/api_service.py
```

To run document ingestion:
```bash
pip install -r requirements.txt
python services/ingestion_service.py
```

### Docker Deployment

To run the entire application stack (frontend, API service, ingestion service, and Milvus) in Docker:

1. **Ensure Docker and Docker Compose are installed**

2. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   # VLM Configuration
   VLM_BASE_URL=http://host.docker.internal:11434/v1
   VLM_MODEL=qwen2.5-vl:7b

   # API Service Configuration
   RETRIEVER_MODEL_ID=vidore/colpali-v1.3
   MILVUS_URI=http://milvus:19530
   COLLECTION_NAME=rag_vision_collection
   TOP_K=5
   MAX_IMAGES=3

   # Ingestion Service Configuration
   SOURCE_DOCS_DIR=/app/documents
   PAGE_IMAGE_DIR=/app/pages
   DIMENSION=128
   ```

3. **Build and run the entire stack:**
   ```bash
   cd deployment
   docker compose up --build
   ```

4. **Run document ingestion (one-time):**
   ```bash
   cd deployment
   docker compose --profile ingestion run --rm ingestion_service
   ```

5. **Access the application:**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - API Service: [http://localhost:8000](http://localhost:8000)

### Services Architecture

- **Frontend**: Next.js application with document upload and chat interface
- **API Service**: Consolidated FastAPI service combining retrieval and chat functionality
- **Ingestion Service**: Document processing service for PDF conversion and vector indexing
- **Milvus**: Vector database for document embeddings and similarity search
- **VLM**: External Ollama service for vision-language model inference

### Ollama (Vision Model) Setup

Ensure Ollama is running on the host and the Qwen model is available before using chat:

```bash
ollama serve
ollama pull qwen2.5-vl:7b
```

Notes:
- The first response may be slow due to model warm-up.
- The API container is configured to reach the host via `host.docker.internal` and includes an `extra_hosts` entry for Linux.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBED_DIM` | Dimension for lite embeddings (square number) | `64` |
| `MILVUS_URI` | Milvus database URI | `http://milvus:19530` |
| `RETRIEVER_MODEL_ID` | ColPali model for document retrieval | `vidore/colpali-v1.3` |
| `TOP_K` | Number of top results to retrieve | `5` |
| `MAX_IMAGES` | Maximum images to include in responses | `3` |
| `VLM_BASE_URL` | URL for the VLM service | `http://host.docker.internal:11434/v1` |
| `VLM_TIMEOUT_SEC` | Timeout for server->Ollama calls (seconds) | `180` |
| `VLM_MODEL` | Vision-language model to use | `qwen2.5-vl:7b` |

## API Endpoints

The application provides several API endpoints for chat, document ingestion, health monitoring, and system management:

### Frontend API Endpoints

- **`GET /api/health`** - Health check endpoint that proxies to the backend service
- **`POST /api/chat`** - Chat endpoint with streaming responses for RAG queries
- **`POST /api/ingest`** - Document ingestion endpoint for uploading and processing PDFs
- **`POST /api/reindex`** - Reindexing endpoint to rebuild vector embeddings

### Backend API Endpoints (Direct)

- **`GET /healthz`** - System health check including Milvus connectivity
- **`POST /chat`** - Direct chat endpoint with vision-language model integration
- **`POST /ingest`** - Direct document ingestion with image processing
- **`POST /reindex`** - Force reindexing of all documents in the system

## API Integration

### Current Implementation
The frontend API endpoints proxy to the consolidated FastAPI backend service and will surface real errors if the backend or VLM is unavailable.

### OpenAI Integration
To integrate with OpenAI's API:

1. **Install OpenAI SDK:**
   ```bash
   npm install openai
   ```

2. **Add environment variables:**
   ```bash
   # .env.local
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Uncomment the OpenAI code** in `/api/chat/route.ts` and comment out the mock implementation.

### Custom AI Integration
The API endpoint can be easily modified to work with any AI service:
- Replace the mock response generation with your AI service calls
- Maintain the streaming response format for real-time updates
- Update error handling as needed

## Customization

### Styling
- **Colors**: Modify CSS variables in `globals.css`
- **Components**: Update Tailwind classes in component files
- **Themes**: Extend theme configuration in `tailwind.config.ts`

### Functionality
- **Message Format**: Update interfaces in `types/chat.ts`
- **Storage**: Modify persistence logic in `useChat.ts`
- **AI Responses**: Customize API logic in `api/chat/route.ts`

## Performance Considerations

- **Component Memoization**: Critical components use React.memo
- **Efficient Re-renders**: State updates are optimized to prevent unnecessary renders
- **Lazy Loading**: Components load only when needed
- **Memory Management**: Chat history is managed efficiently with cleanup

## Browser Support

- **Modern Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **Mobile**: iOS Safari, Chrome Mobile
- **Features**: Server-sent events, localStorage, CSS Grid/Flexbox

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Deployment

### Vercel (Recommended)
```bash
npm run build
# Deploy to Vercel
```

### Other Platforms
The app can be deployed to any platform that supports Next.js:
- Netlify
- Railway
- AWS Amplify
- Docker containers

## Troubleshooting

### Common Issues

1. **Streaming not working**: Check browser support for Server-Sent Events
2. **Theme not persisting**: Verify localStorage is available
3. **Messages not saving**: Check browser storage permissions

### Docker Issues

1. **Build failures**: Ensure you have enough disk space and Docker Desktop is running
2. **Service not starting**: Check logs with `docker compose logs [service-name]`
3. **Port conflicts**: If ports 3000, 8080, or 8081 are in use, stop conflicting services
4. **VLM connection issues**: Ensure Ollama is running locally and accessible at `http://host.docker.internal:11434`. From inside the `api` container, you should be able to `curl http://host.docker.internal:11434`.
5. **Memory issues**: Increase Docker Desktop memory allocation if builds fail (Milvus needs ≥8GB RAM)
6. **Apple Silicon**: All services use `platform: linux/arm64` for compatibility

### Milvus Connection Issues

**Error: "Fail connecting to server on milvus:19530"**

**Symptoms:**
- 500 errors when uploading documents
- API service fails to start
- Connection timeout errors

**Solutions:**

1. **Verify Milvus is running:**
   ```bash
   docker compose ps
   # Should show: milvus-standalone with "0.0.0.0:19530->19530/tcp"
   ```

2. **Check Milvus health:**
   ```bash
   docker compose logs milvus
   # Wait for: "Milvus standalone is ready"
   ```

3. **Test connectivity:**
   ```bash
   # From host (when Next.js runs locally):
   nc -zv 127.0.0.1 19530

   # From API service container:
   docker compose exec api python -c "from pymilvus import MilvusClient; print(MilvusClient('http://milvus:19530').list_collections())"
   ```

4. **Check health endpoint:**
   ```bash
   curl http://localhost:8000/health
   # Should show: "milvus_connected": true
   ```

5. **Common fixes:**
   - Ensure Docker has ≥8GB RAM allocated
   - Wait 2-3 minutes after starting services (Milvus takes time to initialize)
   - Restart services: `docker compose restart`
   - Clear volumes if corrupted: `docker compose down -v`

### Debug Mode
Enable debug logging by adding to your environment:
```bash
NODE_ENV=development
```

### Docker Commands

```bash
# View service status
docker compose ps

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f frontend

# Stop all services
docker compose down

# Rebuild specific service
docker compose up --build frontend

# Clean up
docker compose down --volumes --remove-orphans
```

## Future Enhancements

- [ ] File upload support
- [ ] Message search functionality
- [ ] Export chat history
- [ ] Multiple conversation threads
- [ ] Voice input/output
- [ ] Message reactions
- [ ] User authentication
- [ ] Real-time collaboration
### Testing Locally

After starting Ollama and pulling the model, you can test the VLM integration:

```bash
export VLM_BASE_URL=http://localhost:11434/v1
export VLM_MODEL=qwen2.5-vl:7b
python3 -c "from libs.clients.vlm_client import vision_chat; print('VLM connection test:', vision_chat('Hello'))"
```

If running inside Docker, ensure `VLM_BASE_URL` is set to `http://host.docker.internal:11434/v1` in the `api` service.

Notes:
- The ingestion service processes documents and creates vector embeddings in Milvus
- The API service handles both retrieval and chat functionality
- The system will fall back gracefully if the VLM is unavailable
