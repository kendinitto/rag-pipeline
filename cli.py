#!/usr/bin/env python3
"""
RAG Pipeline CLI — Ingest documents and query with local LLM.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import (
    DocumentIngestor,
    ChunkingStrategy,
    SemanticChunkingStrategy,
    LocalEmbeddingBackend,
    MemoryVectorStore,
    ChromaVectorStore,
)
from rag.query import QueryEngine, LocalServerBackend


DEFAULT_PERSIST_DIR = "./rag_data"


def setup_backends(args):
    """Setup embedding and vector store backends."""
    embedding = LocalEmbeddingBackend(model_name=getattr(args, "embedding_model", "all-MiniLM-L6-v2"))

    store_choice = getattr(args, "store", "chroma")
    persist_dir = getattr(args, "persist_dir", DEFAULT_PERSIST_DIR)

    if store_choice == "chroma":
        vector_store = ChromaVectorStore(persist_dir=persist_dir)
    else:
        vector_store = MemoryVectorStore()

    return embedding, vector_store


def run_ingest(args):
    """Run document ingestion."""
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {args.path} does not exist")
        return

    embedding, vector_store = setup_backends(args)

    chunk_strategy = (
        SemanticChunkingStrategy(max_chunk_size=args.chunk_size * 2)
        if args.semantic
        else ChunkingStrategy(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    )

    ingestor = DocumentIngestor(
        embedding_backend=embedding,
        vector_store=vector_store,
        chunk_strategy=chunk_strategy,
    )

    if path.is_dir():
        print(f"Ingesting directory: {args.path}")
        total = ingestor.ingest_directory(str(path))
    else:
        print(f"Ingesting file: {args.path}")
        total = ingestor.ingest_file(str(path))

    print(f"Ingested {total} chunks from {len(ingestor.ingested_files)} file(s)")
    if args.store == "chroma":
        print(f"Stored in ChromaDB at: {args.persist_dir or DEFAULT_PERSIST_DIR}")


def run_query(args):
    """Run RAG query."""
    embedding, vector_store = setup_backends(args)

    llm = LocalServerBackend(
        server_url=args.server,
        api_mode=args.api_mode,
    )

    engine = QueryEngine(
        embedding_backend=embedding,
        vector_store=vector_store,
        llm_backend=llm,
        top_k=args.top_k,
    )

    if args.sources:
        result = engine.query_with_sources(args.question)
        print(f"\nQuestion: {result['question']}\n")
        print(f"Answer: {result['answer']}\n")
        print(f"Sources ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source']} (score: {src['score']:.3f})")
            print(f"      {src['text_preview']}")
    elif args.stream:
        print(f"\nQuery: {args.question}\n")
        for token in engine.query(args.question, stream=True):
            print(token, end="", flush=True)
        print()
    else:
        print(f"\nQuery: {args.question}\n")
        answer = engine.query(args.question, stream=False)
        print(answer)


def run_rag(args):
    """Ingest + query in one command."""
    embedding, vector_store = setup_backends(args)

    # Ingest
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {args.path} does not exist")
        return

    chunk_strategy = (
        SemanticChunkingStrategy(max_chunk_size=args.chunk_size * 2)
        if args.semantic
        else ChunkingStrategy(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    )

    ingestor = DocumentIngestor(
        embedding_backend=embedding,
        vector_store=vector_store,
        chunk_strategy=chunk_strategy,
    )

    if path.is_dir():
        print(f"Ingesting directory: {args.path}")
        total = ingestor.ingest_directory(str(path))
    else:
        print(f"Ingesting file: {args.path}")
        total = ingestor.ingest_file(str(path))

    print(f"Ingested {total} chunks from {len(ingestor.ingested_files)} file(s)\n")

    # Query
    llm = LocalServerBackend(
        server_url=args.server,
        api_mode=args.api_mode,
    )

    engine = QueryEngine(
        embedding_backend=embedding,
        vector_store=vector_store,
        llm_backend=llm,
        top_k=args.top_k,
    )

    print(f"Query: {args.question}\n")
    if args.stream:
        for token in engine.query(args.question, stream=True):
            print(token, end="", flush=True)
        print()
    elif args.sources:
        result = engine.query_with_sources(args.question)
        print(f"Answer: {result['answer']}\n")
        print(f"Sources ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source']} (score: {src['score']:.3f})")
            print(f"      {src['text_preview']}")
    else:
        answer = engine.query(args.question, stream=False)
        print(answer)


def main():
    parser = argparse.ArgumentParser(
        description="Production RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start: ingest + query in one command
  python cli.py rag sample_docs "What is the architecture?" --server http://localhost:8080

  # Ingest documents
  python cli.py ingest ./docs

  # Query (persistent store)
  python cli.py query "How does the API work?" --server http://localhost:8080 --stream

  # Show sources
  python cli.py query "What are the requirements?" --sources
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument("path", help="File or directory to ingest")
    ingest_parser.add_argument("--store", choices=["memory", "chroma"], default="chroma", help="Vector store backend (default: chroma)")
    ingest_parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help=f"ChromaDB persistence directory (default: {DEFAULT_PERSIST_DIR})")
    ingest_parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in characters")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=256, help="Chunk overlap in characters")
    ingest_parser.add_argument("--semantic", action="store_true", help="Use semantic chunking strategy")
    ingest_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--server", default="http://localhost:8080", help="LLM server URL")
    query_parser.add_argument("--store", choices=["memory", "chroma"], default="chroma", help="Vector store backend (default: chroma)")
    query_parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help=f"ChromaDB persistence directory (default: {DEFAULT_PERSIST_DIR})")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.add_argument("--stream", action="store_true", help="Stream the response")
    query_parser.add_argument("--api-mode", choices=["auto", "native", "openai"], default="auto")
    query_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    query_parser.add_argument("--sources", action="store_true", help="Show source documents")

    # RAG command (ingest + query)
    rag_parser = subparsers.add_parser("rag", help="Ingest documents + query in one command")
    rag_parser.add_argument("path", help="File or directory to ingest")
    rag_parser.add_argument("question", help="Question to ask")
    rag_parser.add_argument("--server", default="http://localhost:8080", help="LLM server URL")
    rag_parser.add_argument("--store", choices=["memory", "chroma"], default="chroma", help="Vector store backend (default: chroma)")
    rag_parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help=f"ChromaDB persistence directory (default: {DEFAULT_PERSIST_DIR})")
    rag_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    rag_parser.add_argument("--stream", action="store_true", help="Stream the response")
    rag_parser.add_argument("--api-mode", choices=["auto", "native", "openai"], default="auto")
    rag_parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in characters")
    rag_parser.add_argument("--chunk-overlap", type=int, default=256, help="Chunk overlap in characters")
    rag_parser.add_argument("--semantic", action="store_true", help="Use semantic chunking strategy")
    rag_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    rag_parser.add_argument("--sources", action="store_true", help="Show source documents")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "rag":
        run_rag(args)


if __name__ == "__main__":
    main()
