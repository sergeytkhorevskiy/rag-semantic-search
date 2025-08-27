# 🔎 RAG Semantic Search

A comprehensive RAG (Retrieval-Augmented Generation) system with hybrid search, client-side reranking and advanced features for semantic document search and question answering.

## ✨ Features

- **🔍 Hybrid Search**: Combines vector similarity, BM25 lexical search, and MMR diversity
- **🎯 Client-Side Reranking**: Advanced reranking using CrossEncoder models
- **🐳 Docker Support**: Complete containerization with development and production configs
- **📊 Multiple Search Modes**: Vector, BM25, and hybrid search with configurable parameters
- **🎨 Modern UI**: Gradio interface with highlighting and real-time search
- **⚡ Performance Optimized**: Caching, fallback mechanisms, and resource management
- **🔒 Production Ready**: Security hardening, rate limiting, and monitoring
- **📈 Evaluation Framework**: MRR/nDCG metrics for search quality assessment

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rag-semantic-search

# Create environment file
cp env_config.txt .env
# Edit .env and add your OPENAI_API_KEY

# Add documents to index
cp your-documents.pdf data/raw/

# Build and run with Docker
make build
make run

# Access the application
# API: http://localhost:8000
# UI: http://localhost:7860
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env_config.txt .env
# Edit .env and add your OPENAI_API_KEY

# Index documents
python scripts/ingest.py

# Start services
uvicorn app.server:app --reload --port 8000  # API server
python ui/gradio_app.py                       # Gradio UI
```

## 📁 Project Structure

```
rag-semantic-search/
├── app/
│   └── server.py              # FastAPI backend server
├── data/
│   ├── raw/                   # Source documents (PDF, HTML, MD)
│   ├── processed/             # Processed document chunks
│   └── eval/                  # Evaluation datasets
├── index/                     # FAISS index and chunks storage
├── scripts/
│   ├── ingest.py              # Document ingestion pipeline
│   └── evaluate.py            # Search evaluation
├── src/
│   ├── ingest/                # Document processing modules
│   ├── retriever.py           # Hybrid search implementation
│   ├── rerank.py              # Reranking with CrossEncoder
│   ├── rag.py                 # RAG answer generation
│   └── types.py               # Data type definitions
├── ui/
│   └── gradio_app.py          # Gradio web interface
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Development Docker setup
├── docker-compose.dev.yml     # Development with hot reload
├── docker-compose.prod.yml    # Production configuration
├── nginx.conf                 # Reverse proxy configuration
├── Makefile                   # Docker management commands
└── README-Docker.md          # Docker documentation
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# OpenAI Configuration (Required for RAG)
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini

# Embedding Model
EMBED_MODEL=intfloat/multilingual-e5-base

# Index Configuration
INDEX_PATH=./index/faiss.index
CHUNKS_PATH=./index/chunks.jsonl

# Search Configuration
SEARCH_MODE=hybrid              # vector, bm25, hybrid
HYBRID_ALPHA=0.65              # Weight for vector vs lexical (0-1)
FETCH_K=64                     # Number of candidates to fetch
LEXICAL_FALLBACK=true          # Use BM25 as fallback

# Advanced Features
RE_RANK=false                  # Server-side reranking
USE_MMR=false                  # Maximum Marginal Relevance
```

### Search Modes

1. **Vector Search** (`SEARCH_MODE=vector`): Pure semantic similarity
2. **BM25 Search** (`SEARCH_MODE=bm25`): Traditional keyword-based search
3. **Hybrid Search** (`SEARCH_MODE=hybrid`): Combines both with configurable weights

## 🐳 Docker Deployment

### Development Mode
```bash
# Hot reload with source mounting
make dev

# With Redis caching
make dev-cache
```

### Production Mode
```bash
# Basic production
make prod

# With Nginx reverse proxy
make prod-nginx

# Scale to multiple instances
make scale
```

### Docker Commands
```bash
# Show all available commands
make help

# Build and run
make build && make run

# View logs
make logs

# Shell access
make shell

# Health check
make health

# Cleanup
make clean
```

## 🔍 Search Features

### Hybrid Search Algorithm
The system implements a sophisticated hybrid search combining:

1. **Vector Similarity**: Semantic embeddings using multilingual models
2. **Lexical Search**: BM25 algorithm for keyword matching
3. **Diversity**: MMR (Maximum Marginal Relevance) for result variety
4. **Fallback**: Automatic fallback to lexical search when vector search fails

### Client-Side Reranking
Advanced reranking using CrossEncoder models:
- Re-ranks search results based on query-document relevance
- Configurable reranking models
- Fallback to simple scoring if model fails

### Search Parameters
- `top_k`: Number of results to return
- `mode`: Search mode (vector/bm25/hybrid)
- `alpha`: Hybrid search weight (0-1)
- `mmr`: Enable diversity in results
- `fetch_k`: Number of candidates to fetch before reranking

## 🎨 User Interface

### Gradio Web Interface
- **Search Tab**: Direct document search with highlighting
- **Ask Tab**: RAG-powered question answering
- **Advanced Options**: 
  - Client-side reranking toggle
  - MMR diversity control
  - Search mode selection
- **Real-time Results**: Instant search with highlighted snippets

### API Endpoints
- `GET /search` - Document search
- `POST /ask` - RAG question answering
- `GET /health` - Health check
- `GET /docs` - API documentation

## 📊 Evaluation Framework

### Setup Evaluation Data
Create `data/eval/qa.jsonl`:
```jsonl
{"question": "How to change charging mode settings?", "relevant_doc_ids": ["data/raw/inverter_manual.pdf"]}
{"question": "How to export tasks from Asana?", "relevant_doc_ids": ["data/raw/asana_help.html"]}
```

### Run Evaluation
```bash
# Generate search results
python scripts/evaluate.py

# View metrics
# - Mean Reciprocal Rank (MRR)
# - Normalized Discounted Cumulative Gain (nDCG)
```

## 🔒 Production Features

### Security
- Rate limiting (30 req/s search, 10 req/s API)
- Read-only volumes in production
- Resource limits and monitoring
- HTTPS support with Nginx

### Monitoring
- Health checks with automatic restart
- Comprehensive logging
- Performance metrics
- Error handling and fallbacks

### Scaling
- Horizontal scaling with multiple instances
- Load balancing with Nginx
- Redis caching support
- Resource optimization

## 🛠️ Development Workflow

### Adding New Documents
```bash
# Copy documents to data/raw/
cp new-document.pdf data/raw/

# Rebuild index
make ingest  # Docker
# or python scripts/ingest.py  # Local
```

### Testing Changes
```bash
# Development mode with hot reload
make dev

# View logs
make logs

# Health check
make health
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```bash
   # Add to .env file
   OPENAI_API_KEY=your_key_here
   ```

2. **Index Not Found**
   ```bash
   # Rebuild index
   make ingest
   ```

3. **Port Already in Use**
   ```bash
   # Stop conflicting services
   make stop
   ```

4. **Memory Issues**
   ```bash
   # Use smaller model
   EMBED_MODEL=intfloat/multilingual-e5-small
   ```

### Debug Mode
```bash
# Run with debug output
docker-compose up

# Check logs
make logs
```

## 📈 Performance Optimization

### Model Selection
- **Fast**: `intfloat/multilingual-e5-small`
- **Balanced**: `intfloat/multilingual-e5-base`
- **Accurate**: `intfloat/multilingual-e5-large`

### Resource Tuning
```yaml
# docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

### Caching
- Enable Redis for result caching
- Use `make dev-cache` for development
- Configure cache TTL in production