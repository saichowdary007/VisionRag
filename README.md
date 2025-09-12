# ChatGPT-Style Chat Application

A modern, responsive chat interface built with Next.js 14+, TypeScript, and Tailwind CSS. This application mimics the ChatGPT user experience with real-time message streaming, dark/light theme support, and persistent message history.

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
│   ├── api/chat/route.ts      # Chat API endpoint with streaming
│   ├── globals.css            # Global styles and theme variables
│   ├── layout.tsx             # Root layout component
│   └── page.tsx               # Main page component
├── components/
│   ├── chat/
│   │   ├── ChatContainer.tsx  # Main chat container
│   │   ├── ChatHeader.tsx     # Header with controls
│   │   ├── MessageBubble.tsx  # Individual message component
│   │   └── MessageInput.tsx   # Message input with auto-resize
│   └── ui/
│       ├── Button.tsx         # Reusable button component
│       └── TypingIndicator.tsx # Animated typing indicator
├── hooks/
│   ├── useChat.ts             # Chat state management
│   └── useTheme.ts            # Theme management
├── lib/
│   └── utils.ts               # Utility functions
└── types/
    └── chat.ts                # TypeScript interfaces
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

2. **Start Docker services (Milvus, retriever, API):**
   ```bash
   # Start all services in background
   ./run-docker.sh

   # Or manually:
   cd deployment
   docker compose up -d
   ```

3. **Configure environment (optional):**
   Create a `.env.local` file in the root directory:
   ```bash
   # When running Next.js locally, connect to Docker services
   RETRIEVER_URL=http://localhost:8080
   ```

4. **Run the development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Monolith Backend (Single File)

You can run the entire backend (retriever + API gateway) from a single FastAPI app:

Python (host):
```bash
pip install -r services/api/requirements.txt python-json-logger
uvicorn backend:app --host 0.0.0.0 --port 8080
```

Docker (single image):
```bash
cd deployment
docker build -f monolith.Dockerfile -t visionrag-monolith ..
docker run --rm -p 8080:8080 \
  -e VLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e VLM_MODEL=qwen2.5-vl:7b \
  -e TOP_K=5 -e MAX_IMAGES=3 \
  -v $(pwd)/../local_data:/app/local_data \
  --add-host host.docker.internal:host-gateway \
  visionrag-monolith
```

Endpoints:
- Health: `GET /healthz`
- Ingest: `POST /ingest` { pages: [{ page_id, image_b64 }] }
- Search: `POST /search` { text, top_k }
- Query: `POST /query` { question, top_k, max_images }

Front-end can keep using `/api/chat` and `/api/ingest` as-is when running Next.js; or directly call the monolith endpoints if you prefer.

### Docker Deployment

To run the entire application stack (frontend, backend API, and retriever) in Docker:

1. **Ensure Docker and Docker Compose are installed**

2. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   # VLM Configuration
   VLM_BASE_URL=http://host.docker.internal:11434/v1
   VLM_MODEL=qwen2.5-vl:7b

   # Retriever Configuration (for Docker services)
   MODEL_ID=vidore/colpali-v1.3
   MILVUS_URI=http://milvus:19530  # Use service name when in Docker network
   COLLECTION_NAME=colpali_multivector_collection
   TOP_K=5
   MAX_IMAGES=3

   # For local development (when Next.js runs outside Docker)
   RETRIEVER_URL=http://localhost:8080
   ```

3. **Build and run the entire stack:**
   ```bash
   # Option 1: Use the convenience script (recommended)
   ./run-docker.sh

   # Option 2: Manual commands
   cd deployment
   docker compose up --build
   ```

4. **Access the application:**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Backend API: [http://localhost:8080](http://localhost:8080)

### Services Architecture

- **Frontend**: Next.js application with PDF upload and chat interface
- **API Gateway**: FastAPI service that orchestrates queries and responses
- **Retriever**: FastAPI service with two modes:
  - `mock` (default): in-memory + filesystem store, deterministic ranking; no external deps
  - `milvus-lite`: simple NumPy embeddings + Milvus vector search; no heavy ML
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
| `RETRIEVER_MODE` | Retriever mode: `mock` or `milvus-lite` | `mock` |
| `EMBED_DIM` | Dimension for lite embeddings (square number) | `64` |
| `MILVUS_URI` | Milvus database URI | `http://milvus:19530` |
| `MODEL_ID` | (Reserved for future ML mode) | `vidore/colpali-v1.3` |
| `TOP_K` | Number of top results to retrieve | `5` |
| `MAX_IMAGES` | Maximum images to include in responses | `3` |
| `VLM_BASE_URL` | URL for the VLM service | `http://host.docker.internal:11434/v1` |
| `VLM_TIMEOUT_SEC` | Timeout for server->Ollama calls (seconds) | `180` |
| `VLM_MODEL` | Vision-language model to use | `qwen2.5-vl:7b` |

## API Integration

### Current Implementation
The chat endpoint now proxies to the FastAPI backend and will surface real errors if the backend or VLM is unavailable. No mock fallbacks are used in production paths.

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
- Retriever service fails to start
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

   # From retriever container:
   docker compose exec backend python -c "from pymilvus import MilvusClient; print(MilvusClient('http://milvus:19530').list_collections())"
   ```

4. **Check health endpoint:**
   ```bash
   curl http://localhost:8080/healthz
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
### Local Demo (No Docker)

If you just want to exercise the retriever and API locally without Docker:

- Start retriever in one terminal:
  - `uvicorn services.retriever.main:app --host 0.0.0.0 --port 8081`
- Start API gateway in another terminal:
  - `uvicorn services.api.main:app --host 0.0.0.0 --port 8080`
- Run the demo script to ingest a generated image, search it, and query the API:
  - `python3 scripts/demo_ingest_and_query.py`

### Testing the VLM call manually

After starting Ollama and pulling the model, you can run a quick test against the VLM endpoint directly from the host:

```bash
export VLM_BASE_URL=http://localhost:11434/v1
export VLM_MODEL=qwen2.5-vl:7b
python3 scripts/test_ollama_call.py
```

If running inside Docker, ensure `VLM_BASE_URL` is set to `http://host.docker.internal:11434/v1` in the `api` service.

Notes:
- The retriever runs in a mock mode by default (no Milvus/ML required). It stores assets under `./local_data` when `/data` is not writable.
- The API will fall back to a safe mock answer if the VLM is unavailable.
5. **Switch retriever mode (optional):**
   - Default is `mock`. For Milvus-backed lite retrieval, set `RETRIEVER_MODE=milvus-lite` in your `.env` and re-run compose. Ensure Milvus is healthy first.
