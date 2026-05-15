# System Architecture — RAG Pipeline v2

## Overview

This document describes the architecture of our production RAG (Retrieval-Augmented Generation) pipeline for enterprise document processing and Q&A.

## Components

### 1. Document Ingestion Layer

The ingestion layer handles document loading, preprocessing, and chunking:

- **Supports formats**: PDF, TXT, Markdown, HTML
- **Chunking strategies**: Fixed-size (1024 chars, 256 overlap) or semantic (paragraph-aware)
- **Embedding**: Local sentence-transformers or OpenAI text-embedding-3-small
- **Output**: Chunks with metadata (source, chunk_index, page, custom fields)

### 2. Vector Storage Layer

Two backends are available:

- **MemoryVectorStore**: In-memory, fast, no persistence. Good for development.
- **ChromaVectorStore**: Persistent ChromaDB with HNSW index. Production-ready.

The vector store stores embeddings alongside chunk metadata for source tracking.

### 3. Retrieval Layer

The retriever handles query processing:

- Embeds the user query using the same embedding model as ingestion
- Performs cosine similarity search against stored chunks
- Filters by similarity threshold (default: 0.3)
- Optional BM25 reranking for improved relevance
- Assembles top-k results into context with source attribution

### 4. Query Engine

The query engine orchestrates the RAG pipeline:

1. Receives user question
2. Embeds question and retrieves relevant chunks
3. Assembles context from retrieved chunks
4. Constructs prompt with system template, context, and question
5. Sends to LLM backend (local llama.cpp or OpenAI)
6. Returns answer, optionally with source citations

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| chunk_size | 1024 | Characters per chunk |
| chunk_overlap | 256 | Overlap between chunks |
| top_k | 5 | Number of chunks to retrieve |
| similarity_threshold | 0.3 | Minimum similarity score |
| max_context_length | 4096 | Max characters in assembled context |
| temperature | 0.7 | LLM generation temperature |

## Performance

- Ingestion: ~50-200 documents/minute (depends on embedding backend)
- Query latency: 200-500ms retrieval + LLM generation time
- Memory: ~1MB per 1000 chunks (embeddings only)
