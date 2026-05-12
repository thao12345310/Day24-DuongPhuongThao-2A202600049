"""Input guardrails for Lab 24.

PII redaction combines VN-specific regex with Presidio NER (English).
Topic validation uses OpenAI embedding similarity.
Injection detection uses deterministic regex patterns.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path


VN_PII_PATTERNS: dict[str, str] = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"\b(?:\+84|0)(?:\d[\s.-]?){9,10}\b",
    "tax_code": r"\b\d{10}(?:-\d{3})?\b",
    "email": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    "address_vn": r"\b\d{1,4}\s+(?:Le Loi|Nguyen Trai|Tran Hung Dao|Lê Lợi|Nguyễn Trãi|Trần Hưng Đạo)\b",
}

EN_PII_PATTERNS: dict[str, str] = {
    "phone_us": r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
    "person_hint": r"\b(?:John Smith|Jane Doe|Nguyen Van A|Nguyễn Văn A|Ly Van Binh|Lý Văn Bình)\b",
    "street_us": r"\b\d{1,5}\s+(?:Main Street|Market Street|Broadway|Elm Street)\b",
}

INJECTION_PATTERNS: tuple[str, ...] = (
    r"\bignore (?:all )?(?:previous|system|developer) instructions\b",
    r"\bpretend you are\b",
    r"\bfrom now on you are\b",
    r"\bDAN\b",
    r"\bjailbreak\b",
    r"\bno restrictions\b",
    r"\bdecode this base64\b",
    r"\bexplain how to hack\b",
    r"\bhidden instruction\b",
    r"\bdisregard the policy\b",
)


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str
    latency_ms: float


class InputGuard:
    """PII redaction: VN/EN regex plus Presidio NER."""

    def __init__(self) -> None:
        self.analyzer = None
        self.anonymizer = None
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except Exception:
            pass

    def _apply_regexes(self, text: str) -> tuple[str, list[str]]:
        labels: list[str] = []
        out = text
        for name, pattern in {**VN_PII_PATTERNS, **EN_PII_PATTERNS}.items():
            if re.search(pattern, out, flags=re.IGNORECASE):
                labels.append(name)
                out = re.sub(pattern, f"[{name.upper()}]", out, flags=re.IGNORECASE)
        return out, labels

    def _apply_presidio(self, text: str) -> tuple[str, list[str]]:
        if not self.analyzer or not self.anonymizer or not text.strip():
            return text, []
        try:
            results = self.analyzer.analyze(text=text, language="en")
            labels = sorted({r.entity_type.lower() for r in results})
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results).text
            return anonymized, labels
        except Exception:
            return text, []

    def sanitize(self, text: str) -> tuple[str, float, list[str]]:
        start = time.perf_counter()
        scrubbed, regex_labels = self._apply_regexes(text or "")
        scrubbed, ner_labels = self._apply_presidio(scrubbed)
        latency_ms = (time.perf_counter() - start) * 1000
        return scrubbed, latency_ms, regex_labels + ner_labels

    async def sanitize_async(self, text: str) -> tuple[str, float, list[str]]:
        return await asyncio.to_thread(self.sanitize, text)


class TopicGuard:
    """Topic validator using OpenAI embedding similarity (Option 1)."""

    def __init__(self, allowed_topics: list[str] | None = None) -> None:
        self.allowed_topics = allowed_topics or [
            "rag evaluation",
            "ragas metrics",
            "llm judge",
            "guardrails",
            "prompt injection",
            "pii redaction",
            "latency benchmark",
            "slo",
            "blueprint",
            "retrieval",
            "faithfulness",
            "context precision",
            "context recall",
            "cohen kappa",
            "audit log",
            "evaluation",
            "document slo",
        ]
        self.embedding_model = os.getenv("LAB24_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        from openai_utils import get_openai_client

        self.client = get_openai_client()
        response = self.client.embeddings.create(model=self.embedding_model, input=self.allowed_topics)
        self.topic_vectors = [item.embedding for item in response.data]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if not na or not nb:
            return 0.0
        return dot / (na * nb)

    def check(self, text: str) -> tuple[bool, str]:
        lowered = (text or "").lower()
        if not lowered.strip():
            return False, "Please ask a question about RAG evaluation, judge calibration, or guardrails."
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        q_vec = response.data[0].embedding
        sims = [self._cosine(q_vec, topic_vec) for topic_vec in self.topic_vectors]
        max_sim = max(sims)
        best_topic = self.allowed_topics[sims.index(max_sim)]
        if max_sim > 0.60:
            return True, f"On topic: {best_topic} ({max_sim:.2f})"
        return False, (
            "This assistant is scoped to RAG evaluation and guardrails. "
            f"Closest allowed topic: {best_topic} ({max_sim:.2f})."
        )

    async def check_async(self, text: str) -> tuple[bool, str]:
        return await asyncio.to_thread(self.check, text)


class InjectionGuard:
    """Prompt-injection and jailbreak detector used in adversarial tests."""

    def check(self, text: str) -> tuple[bool, str]:
        lowered = text or ""
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                return False, f"Prompt injection detected: {pattern}"
        return True, "No injection pattern detected"

    async def check_async(self, text: str) -> tuple[bool, str]:
        return self.check(text)


def refuse_response() -> str:
    return (
        "I can't help with that request in this lab assistant. "
        "I can help with RAG evaluation, judge calibration, PII redaction, guardrails, and latency analysis."
    )
