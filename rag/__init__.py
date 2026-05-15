"""
Production RAG Pipeline
========================
Retrieval-Augmented Generation system with configurable chunking,
multiple embedding backends, and vector database integration.
"""

from rag.ingest import DocumentIngestor
from rag.retrieval import Retriever
from rag.query import QueryEngine


__version__ = "0.1.0"

__all__ = ["DocumentIngestor", "Retriever", "QueryEngine"]
