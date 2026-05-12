"""Minimal OpenAI-backed RAG pipeline for Lab 24.

This replaces the previous deterministic stand-in with a small, inspectable
retrieval + generation path:

1. load Markdown documents from ./docs
2. chunk them
3. embed chunks with OpenAI embeddings
4. retrieve top-k chunks by cosine similarity
5. answer with an OpenAI chat model using only retrieved context
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openai_utils import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, CostTracker, get_openai_client  # noqa: E402


DOCS_DIR = ROOT / "docs"
CACHE_PATH = ROOT / "phase-a" / "openai_embedding_cache.json"


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def load_markdown_docs(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(docs_dir.glob("**/*.md")):
        docs.append((str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")))
    if not docs:
        raise RuntimeError(f"No Markdown documents found under {docs_dir}")
    return docs


def chunk_text(text: str, max_words: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]
    chunks: list[str] = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + max_words]).strip()
        if piece:
            chunks.append(piece)
    return chunks


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_markdown_docs():
        for idx, chunk in enumerate(chunk_text(text), start=1):
            digest = hashlib.sha1(f"{source}:{idx}:{chunk}".encode("utf-8")).hexdigest()[:12]
            chunks.append(Chunk(chunk_id=digest, source=source, text=chunk))
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class OpenAIRAG:
    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 5,
    ) -> None:
        self.client = get_openai_client()
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.cost = CostTracker()
        self.chunks = build_chunks()
        self.embeddings = self._load_or_create_embeddings()

    def _load_or_create_embeddings(self) -> dict[str, list[float]]:
        cached: dict[str, list[float]] = {}
        if CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

        missing = [chunk for chunk in self.chunks if chunk.chunk_id not in cached]
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            response = self.client.embeddings.create(model=self.embedding_model, input=[c.text for c in batch])
            self.cost.add_usage(response.usage)
            for chunk, item in zip(batch, response.data, strict=True):
                cached[chunk.chunk_id] = item.embedding

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cached), encoding="utf-8")
        return cached

    def retrieve(self, question: str) -> list[Chunk]:
        response = self.client.embeddings.create(model=self.embedding_model, input=question)
        self.cost.add_usage(response.usage)
        q_vec = response.data[0].embedding
        ranked = sorted(
            ((cosine(q_vec, self.embeddings[chunk.chunk_id]), chunk) for chunk in self.chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        return [chunk for _, chunk in ranked[: self.top_k]]

    def answer(self, question: str) -> tuple[str, list[str]]:
        contexts = self.retrieve(question)
        context_text = "\n\n".join(f"[{idx}] {chunk.source}\n{chunk.text}" for idx, chunk in enumerate(contexts, 1))
        prompt = (
            "Answer the question using only the retrieved context. "
            "If the context is insufficient, say what evidence is missing. "
            "Keep the answer concise and grounded.\n\n"
            f"Question: {question}\n\nRetrieved context:\n{context_text}"
        )
        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a careful RAG assistant for a lab evaluation."},
                {"role": "user", "content": prompt},
            ],
        )
        self.cost.add_usage(response.usage)
        answer = response.choices[0].message.content or ""
        return answer.strip(), [chunk.text for chunk in contexts]


def answer_question(question: str, top_k: int = 5) -> tuple[str, list[str]]:
    return OpenAIRAG(top_k=top_k).answer(question)
