"""
Retrieval Module
================
Handles chunk retrieval, reranking, and context assembly.
"""

from typing import Optional


class Retriever:
    """Document retrieval with configurable strategies."""

    def __init__(
        self,
        vector_store,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        use_reranking: bool = False,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_reranking = use_reranking

    def retrieve(self, query_embedding: list[float], query_text: str = "") -> list[dict]:
        """Retrieve relevant chunks for a query."""
        results = self.vector_store.query(query_embedding, top_k=self.top_k)

        # Filter by similarity threshold
        filtered = [
            r for r in results
            if r.get("score", 0) >= self.similarity_threshold
        ]

        # Rerank if enabled
        if self.use_reranking and query_text:
            filtered = self._rerank(filtered, query_text)

        return filtered

    def _rerank(self, results: list[dict], query: str) -> list[dict]:
        """Simple BM25-like reranking based on query term overlap."""
        query_terms = set(query.lower().split())

        for result in results:
            chunk_text = result["chunk"].lower()
            term_overlap = len(query_terms & set(chunk_text.split()))
            bm25_score = term_overlap / max(len(query_terms), 1)
            # Combine original score with BM25 score
            result["reranked_score"] = (
                result.get("score", 0) * 0.7 + bm25_score * 0.3
            )

        return sorted(results, key=lambda x: x.get("reranked_score", x.get("score", 0)), reverse=True)

    def assemble_context(self, results: list[dict], max_context_length: int = 4096) -> str:
        """Assemble retrieved chunks into context string."""
        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")

            header = f"[Source: {source} (chunk {metadata.get('chunk_index', '?')}), Score: {result.get('score', 0):.3f}]\n"
            part = header + chunk

            if total_length + len(part) > max_context_length:
                # Truncate to fit
                remaining = max_context_length - total_length
                if remaining > len(header) + 100:
                    context_parts.append(part[:remaining - len(header)])
                break

            context_parts.append(part)
            total_length += len(part)

        return "\n\n".join(context_parts)
