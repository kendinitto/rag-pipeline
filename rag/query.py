"""
Query Engine
============
Handles RAG queries with configurable LLM backends.
Supports local llama.cpp servers and OpenAI-compatible APIs.
"""

import json
import requests
from typing import Optional, Iterator


SYSTEM_PROMPT = """You are a helpful AI assistant. You are given a user question and some retrieved context from documents. Answer the question based on the context. If the context doesn't contain relevant information, say so honestly. Cite sources when possible.

Context:
{context}

Question: {question}

Answer:"""


class LLMBackend:
    """Abstract LLM backend interface."""

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        raise NotImplementedError

    def stream_generate(self, prompt: str, max_tokens: int = 4096) -> Iterator[str]:
        raise NotImplementedError


class LocalServerBackend(LLMBackend):
    """llama.cpp server backend."""

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_mode: str = "auto",
    ):
        self.server_url = server_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_mode = api_mode

    def _detect_mode(self) -> str:
        if self.api_mode != "auto":
            return self.api_mode
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            if r.status_code in (200, 503):
                return "native"
        except Exception:
            pass
        return "openai"

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        mode = self._detect_mode()
        tokens = []

        for token in self.stream_generate(prompt, max_tokens or self.max_tokens):
            tokens.append(token)

        return "".join(tokens)

    def stream_generate(self, prompt: str, max_tokens: Optional[int] = None) -> Iterator[str]:
        mode = self._detect_mode()
        n = max_tokens or self.max_tokens

        if mode == "native":
            yield from self._stream_native(prompt, n)
        else:
            yield from self._stream_openai(prompt, n)

    def _stream_native(self, prompt: str, max_tokens: int) -> Iterator[str]:
        """Stream from llama.cpp native /completion endpoint."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.server_url}/completion",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        content = data.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"\n[Error: {e}]"

    def _stream_openai(self, prompt: str, max_tokens: int) -> Iterator[str]:
        """Stream from OpenAI-compatible /v1/chat/completions endpoint."""
        payload = {
            "model": "llama",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                        continue
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices and choices[0].get("delta", {}).get("content"):
                            yield choices[0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"\n[Error: {e}]"


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
    ):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        tokens = []
        for token in self.stream_generate(prompt, max_tokens):
            tokens.append(token)
        return "".join(tokens)

    def stream_generate(self, prompt: str, max_tokens: int = 4096) -> Iterator[str]:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            with requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                stream=True,
                timeout=300,
            ) as resp:
                for line in resp.iter_lines():
                    if not line or b"[DONE]" in line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data: "):
                        continue
                    data = json.loads(line_str[6:])
                    choices = data.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        yield choices[0]["delta"]["content"]
        except Exception as e:
            yield f"\n[Error: {e}]"


class QueryEngine:
    """Main RAG query engine."""

    def __init__(
        self,
        embedding_backend,
        vector_store,
        llm_backend: LLMBackend,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        max_context_length: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        self.embedding_backend = embedding_backend
        self.llm_backend = llm_backend
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        from rag.retrieval import Retriever
        self.retriever = Retriever(
            vector_store=vector_store,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def query(self, question: str, stream: bool = False) -> str | Iterator[str]:
        """Execute a RAG query."""
        # 1. Embed the question
        query_embedding = self.embedding_backend.encode_single(question)

        # 2. Retrieve relevant chunks
        results = self.retriever.retrieve(query_embedding, query_text=question)

        if not results:
            if stream:
                return self._stream_no_results(question)
            return "No relevant documents found for this query."

        # 3. Assemble context
        context = self.retriever.assemble_context(results, self.max_context_length)

        # 4. Build prompt
        prompt = self.system_prompt.format(context=context, question=question)

        # 5. Generate answer
        if stream:
            return self.llm_backend.stream_generate(prompt)
        return self.llm_backend.generate(prompt)

    def query_with_sources(self, question: str) -> dict:
        """Execute query and return answer with source information."""
        query_embedding = self.embedding_backend.encode_single(question)
        results = self.retriever.retrieve(query_embedding, query_text=question)

        if not results:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "question": question,
            }

        context = self.retriever.assemble_context(results, self.max_context_length)
        prompt = self.system_prompt.format(context=context, question=question)
        answer = self.llm_backend.generate(prompt)

        sources = []
        for r in results:
            sources.append({
                "source": r.get("metadata", {}).get("source", "unknown"),
                "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
                "score": r.get("score", 0),
                "text_preview": r["chunk"][:200] + "...",
            })

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
        }

    def _stream_no_results(self, question: str) -> Iterator[str]:
        yield "No relevant documents found for this query."
