# Production RAG Pipeline

Retrieval-Augmented Generation system with configurable chunking strategies, multiple embedding backends, vector database integration, and local LLM support. Built for production deployment with enterprise document processing.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Documents  │ ──► │   Ingest    │ ──► │   Vector    │ ──► │  Query      │
│  (PDF/TXT/  │     │  (chunk +   │     │   DB        │     │  Engine     │
│   MD/HTML)  │     │  embed)     │     │ (Chroma/    │     │ (RAG + LLM) │
└─────────────┘     └─────────────┘     │  Memory)    │     └─────────────┘
                                        └─────────────┘           │
                                                                  ▼
                                                           ┌─────────────┐
                                                           │    LLM      │
                                                           │ (llama.cpp, │
                                                           │  OpenAI,    │
                                                           │  any API)   │
                                                           └─────────────┘
```

## Features

- **Multiple document formats** — PDF, TXT, MD, HTML ingestion
- **Two chunking strategies** — Fixed-size with overlap, or semantic (paragraph-aware)
- **Vector database backends** — ChromaDB (persistent) or in-memory
- **Flexible LLM backends** — llama.cpp server (native + OpenAI API mode), OpenAI
- **Streaming responses** — Token-by-token output for interactive UX
- **Source citations** — Track which documents contributed to each answer
- **Embedding backends** — Local (sentence-transformers) or OpenAI API

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Quick Start (Ingest + Query)

```bash
# Ingest docs and query in one command
python cli.py rag sample_docs "What is the system architecture?" --server http://localhost:8080

# Stream the response
python cli.py rag ./my-docs "How does the API work?" --server http://localhost:8080 --stream

# Show source citations
python cli.py rag ./docs "What are the requirements?" --server http://localhost:8080 --sources
```

### 3. Separate Ingest and Query

```bash
# Ingest once (persists to ./rag_data by default)
python cli.py ingest ./docs

# Query anytime (reads from persisted store)
python cli.py query "What is the architecture?" --server http://localhost:8080

# Stream response
python cli.py query "How does the API work?" --server http://localhost:8080 --stream

# Show sources
python cli.py query "What are the security requirements?" --sources
```

### 3. Query

```bash
# Basic query (local llama.cpp server)
python cli.py query "What is the system architecture?" --server http://localhost:8080

# Stream the response
python cli.py query "How does the API work?" --server http://localhost:8080 --stream

# Show source documents
python cli.py query "What are the security requirements?" --server http://localhost:8080 --sources

# Connect to remote server
python cli.py query "Tell me about the deployment" --server http://192.168.1.100:11434 --stream
```

## Python API

```python
from rag.ingest import DocumentIngestor, LocalEmbeddingBackend, ChromaVectorStore, ChunkingStrategy
from rag.query import QueryEngine, LocalServerBackend

# Setup components
embedding = LocalEmbeddingBackend(model_name="all-MiniLM-L6-v2")
vector_store = ChromaVectorStore(persist_dir="./data")
chunk_strategy = ChunkingStrategy(chunk_size=1024, chunk_overlap=256)
llm = LocalServerBackend(server_url="http://localhost:8080")

# Ingest
ingestor = DocumentIngestor(embedding, vector_store, chunk_strategy)
chunks = ingestor.ingest_directory("./documents")
print(f"Ingested {chunks} chunks")

# Query
engine = QueryEngine(
    embedding_backend=embedding,
    vector_store=vector_store,
    llm_backend=llm,
    top_k=5,
)

# Get answer
answer = engine.query("What is the revenue model?")
print(answer)

# Get answer with sources
result = engine.query_with_sources("What is the revenue model?")
print(result["answer"])
for src in result["sources"]:
    print(f"  Source: {src['source']} (score: {src['score']:.3f})")
```

## Chunking Strategies

| Strategy | Best For | Parameters |
|----------|----------|------------|
| `ChunkingStrategy` | General purpose, code docs | `chunk_size`, `chunk_overlap`, `separator` |
| `SemanticChunkingStrategy` | Natural language, reports | `max_chunk_size`, `min_chunk_size` |

## Vector Store Backends

| Backend | Persistence | Best For |
|---------|-------------|----------|
| `MemoryVectorStore` | No (in-memory) | Development, testing, single-session |
| `ChromaVectorStore` | Yes (disk) | Production, multi-session, shared data |

## LLM Backends

| Backend | Setup |
|---------|-------|
| `LocalServerBackend` | Point to any llama.cpp server (auto-detects native/OpenAI API) |
| `OpenAIBackend` | Set `OPENAI_API_KEY` env var or pass `api_key` |

## Embedding Backends

| Backend | Setup |
|---------|-------|
| `LocalEmbeddingBackend` | `pip install sentence-transformers` (downloads model on first use) |
| `OpenAIEmbeddingBackend` | Set `OPENAI_API_KEY` env var |

## Performance Tips

- **Chunk size**: Smaller chunks (512) = more precise retrieval. Larger (2048) = better context, slightly slower
- **top_k**: Start with 5. Increase to 10 if answers lack detail. Decrease to 3 if answers are noisy
- **similarity_threshold**: Default 0.3 works for most cases. Increase (0.5+) for stricter matching
- **Flash attention**: Enable on your llama.cpp server for faster generation with long RAG context prompts

## Sample Documents

The `sample_docs/` directory includes example documents for testing:
- Technical documentation
- API reference
- Architecture overview

## License

MIT
