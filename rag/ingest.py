"""
Document Ingestion Module
==========================
Handles document loading, chunking, embedding, and vector storage.
Supports PDF, TXT, MD, and HTML sources with configurable chunking strategies.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    page: Optional[int] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page": self.page,
            "metadata": self.metadata or {},
        }


class ChunkingStrategy:
    """Text chunking with overlap."""

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        separator: str = "\n\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, text: str, source: str) -> list[Chunk]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        # Try splitting by separator first
        segments = text.split(self.separator)
        chunks = []
        current = ""
        chunk_index = 0

        for segment in segments:
            if len(current) + len(segment) + len(self.separator) <= self.chunk_size:
                current = current + self.separator + segment if current else segment
            else:
                if current:
                    chunks.append(current)
                current = segment

        if current:
            chunks.append(current)

        # Split oversized chunks by character
        final_chunks = []
        for chunk_text in chunks:
            if len(chunk_text) <= self.chunk_size:
                final_chunks.append(chunk_text)
            else:
                for i in range(0, len(chunk_text), self.chunk_size - self.chunk_overlap):
                    final_chunks.append(chunk_text[i:i + self.chunk_size])

        total = len(final_chunks)
        return [
            Chunk(
                text=chunk_text,
                source=source,
                chunk_index=i,
                total_chunks=total,
            )
            for i, chunk_text in enumerate(final_chunks)
        ]


class SemanticChunkingStrategy:
    """Chunk by semantic boundaries (paragraphs/sections)."""

    def __init__(
        self,
        max_chunk_size: int = 2048,
        min_chunk_size: int = 256,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, source: str) -> list[Chunk]:
        """Split text by semantic boundaries with size constraints."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        chunk_index = 0

        for para in paragraphs:
            if len(para) <= self.min_chunk_size:
                if len(current) + len(para) + 2 <= self.max_chunk_size:
                    current = current + "\n\n" + para if current else para
                else:
                    if current:
                        chunks.append(current)
                    current = para
            else:
                if current:
                    chunks.append(current)
                    current = ""
                # Split large paragraphs by sentences
                sentences = para.replace(". ", ".|").split("|")
                sentence_buf = ""
                for sent in sentences:
                    if len(sentence_buf) + len(sent) + 1 <= self.max_chunk_size:
                        sentence_buf = sentence_buf + " " + sent if sentence_buf else sent
                    else:
                        if sentence_buf:
                            chunks.append(sentence_buf)
                        sentence_buf = sent
                if sentence_buf:
                    chunks.append(sentence_buf)

        if current:
            chunks.append(current)

        total = len(chunks)
        return [
            Chunk(
                text=chunk_text,
                source=source,
                chunk_index=i,
                total_chunks=total,
            )
            for i, chunk_text in enumerate(chunks)
        ]


class EmbeddingBackend:
    """Abstract embedding backend interface."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]


class LocalEmbeddingBackend(EmbeddingBackend):
    """Local embedding using sentence-transformers (optional) or simple fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers for local embeddings: "
                    "pip install sentence-transformers"
                )

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        import torch
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.tolist()


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI API embeddings."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def encode(self, texts: list[str]) -> list[list[float]]:
        import urllib.request
        import json as j

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = j.dumps({"input": batch, "model": self.model}).encode()
            req = urllib.request.Request(url, data=payload, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = j.loads(resp.read())
                    for item in data["data"]:
                        all_embeddings.append(item["embedding"])
            except Exception as e:
                print(f"Embedding API error: {e}")
                raise

        return all_embeddings


class VectorStore:
    """Abstract vector store interface."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store backend."""

    def __init__(self, collection_name: str = "rag_docs", persist_dir: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._db = None
        self._collection = None

    def _init_db(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir) if self.persist_dir else chromadb.Client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._db = client

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._init_db()
        ids = [
            f"{c.source}::{c.chunk_index}"
            for c in chunks
        ]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                **{k: v for k, v in {"page": c.page}.items() if v is not None},
                **(c.metadata or {}),
            }
            for c in chunks
        ]

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            self._collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        self._init_db()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })

        return output

    def clear(self) -> None:
        if self._db and self._collection:
            self._db.delete_collection(self.collection_name)
            self._collection = None
            self._init_db()


class MemoryVectorStore(VectorStore):
    """In-memory vector store for development/testing."""

    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        if not self.embeddings:
            return []

        import numpy as np

        query = np.array(embedding)
        stored = np.array(self.embeddings)

        # Cosine similarity
        norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(query)
        similarities = np.dot(stored, query) / (norms + 1e-10)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "chunk": self.chunks[i].text,
                "metadata": {
                    "source": self.chunks[i].source,
                    "chunk_index": self.chunks[i].chunk_index,
                },
                "score": float(similarities[i]),
                "distance": 1 - float(similarities[i]),
            }
            for i in top_indices
        ]

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = []


class DocumentIngestor:
    """Main document ingestion pipeline."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_store: VectorStore,
        chunk_strategy: Optional[ChunkingStrategy] = None,
    ):
        self.embedding_backend = embedding_backend
        self.vector_store = vector_store
        self.chunk_strategy = chunk_strategy or ChunkingStrategy()
        self.ingested_files: list[str] = []

    def ingest_file(self, filepath: str) -> int:
        """Ingest a single file. Returns number of chunks created."""
        path = Path(filepath)
        text = self._load_file(path)
        chunks = self.chunk_strategy.chunk(text, str(path))
        embeddings = self.embedding_backend.encode([c.text for c in chunks])
        self.vector_store.add(chunks, embeddings)
        self.ingested_files.append(filepath)
        return len(chunks)

    def ingest_directory(self, dirpath: str, extensions: Optional[list[str]] = None) -> int:
        """Ingest all documents in a directory."""
        if extensions is None:
            extensions = [".txt", ".md", ".pdf", ".html"]

        path = Path(dirpath)
        total_chunks = 0

        for ext in extensions:
            for filepath in path.glob(f"**/*{ext}"):
                chunks = self.ingest_file(str(filepath))
                total_chunks += chunks

        return total_chunks

    def _load_file(self, path: Path) -> str:
        """Load text from a file."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix == ".html":
            return self._load_html(path)
        else:
            return path.read_text(encoding="utf-8", errors="replace")

    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += f"\n--- Page {page.page_number} ---\n{extracted}\n"
            return text
        except ImportError:
            raise ImportError("Install pdfplumber for PDF support: pip install pdfplumber")

    def _load_html(self, path: Path) -> str:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup
            html = path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            return " ".join(line.strip() for line in text.splitlines() if line.strip())
        except ImportError:
            # Fallback: raw text
            return path.read_text(encoding="utf-8", errors="replace")
