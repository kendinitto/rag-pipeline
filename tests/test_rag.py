"""Tests for RAG Pipeline."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import ChunkingStrategy, SemanticChunkingStrategy, MemoryVectorStore, Chunk
from rag.retrieval import Retriever


class TestChunkingStrategy(unittest.TestCase):
    def test_basic_chunking(self):
        strategy = ChunkingStrategy(chunk_size=100, chunk_overlap=20)
        text = " ".join([f"Word{i}" for i in range(50)])
        chunks = strategy.chunk(text, "test.txt")
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 120)  # Allow some tolerance

    def test_chunk_metadata(self):
        strategy = ChunkingStrategy(chunk_size=100)
        text = "Hello world. This is a test document with some content."
        chunks = strategy.chunk(text, "test.txt")
        self.assertEqual(chunks[0].source, "test.txt")
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_empty_text(self):
        strategy = ChunkingStrategy()
        chunks = strategy.chunk("", "test.txt")
        self.assertEqual(len(chunks), 0)


class TestMemoryVectorStore(unittest.TestCase):
    def test_add_and_query(self):
        store = MemoryVectorStore()
        chunks = [
            Chunk(text="The sky is blue and the grass is green", source="doc1", chunk_index=0, total_chunks=1),
            Chunk(text="Machine learning models process data efficiently", source="doc2", chunk_index=0, total_chunks=1),
            Chunk(text="The ocean is vast and deep with many fish", source="doc3", chunk_index=0, total_chunks=1),
        ]
        # Dummy embeddings (random but distinct enough)
        embeddings = [
            [1.0, 0.1, 0.1, 0.0],
            [0.1, 1.0, 0.1, 0.0],
            [0.9, 0.0, 0.1, 0.0],
        ]
        store.add(chunks, embeddings)

        results = store.query([1.0, 0.1, 0.1, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("doc1", results[0]["metadata"]["source"])


class TestRetriever(unittest.TestCase):
    def test_assemble_context(self):
        retriever = Retriever(vector_store=MemoryVectorStore())
        results = [
            {"chunk": "First chunk of context.", "metadata": {"source": "a.txt", "chunk_index": 0}, "score": 0.9},
            {"chunk": "Second chunk with more details.", "metadata": {"source": "a.txt", "chunk_index": 1}, "score": 0.7},
        ]
        context = retriever.assemble_context(results)
        self.assertIn("First chunk", context)
        self.assertIn("Second chunk", context)
        self.assertIn("a.txt", context)


if __name__ == "__main__":
    unittest.main()
