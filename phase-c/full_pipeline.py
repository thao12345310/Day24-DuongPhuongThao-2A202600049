"""Full guardrail stack integration and latency benchmark."""

from __future__ import annotations

import asyncio
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from input_guard import InjectionGuard, InputGuard, TopicGuard, refuse_response
from output_guard import LlamaGuard3OutputGuard


ROOT = Path(__file__).resolve().parents[1]
AUDIT_LOG = ROOT / "phase-c" / "audit_log.jsonl"


input_guard = InputGuard()
topic_guard = TopicGuard()
injection_guard = InjectionGuard()
output_guard = LlamaGuard3OutputGuard()

_openai_rag = None


async def rag_pipeline_async(user_input: str) -> str:
    global _openai_rag
    if _openai_rag is None:
        sys.path.insert(0, str(ROOT / "phase-a"))
        from openai_rag import OpenAIRAG

        _openai_rag = OpenAIRAG()
    answer, _ = await asyncio.to_thread(_openai_rag.answer, user_input)
    return answer


async def audit_log(user_input: str, answer: str, timings: dict[str, float]) -> None:
    row = {"input": user_input[:200], "answer": answer[:200], "timings": timings}
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


async def guarded_pipeline(user_input: str) -> tuple[str, dict[str, float]]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    pii_task = asyncio.create_task(input_guard.sanitize_async(user_input))
    topic_task = asyncio.create_task(topic_guard.check_async(user_input))
    injection_task = asyncio.create_task(injection_guard.check_async(user_input))
    sanitized, _, _ = await pii_task
    topic_ok, topic_reason = await topic_task
    injection_ok, injection_reason = await injection_task
    timings["L1"] = (time.perf_counter() - t0) * 1000
    if not topic_ok or not injection_ok:
        timings["L2"] = 0.0
        timings["L3"] = 0.0
        return f"{refuse_response()} Reason: {topic_reason if not topic_ok else injection_reason}", timings

    t0 = time.perf_counter()
    answer = await rag_pipeline_async(sanitized)
    timings["L2"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    safe, label, _ = await output_guard.check_async(sanitized, answer)
    timings["L3"] = (time.perf_counter() - t0) * 1000
    if not safe:
        return f"{refuse_response()} Output guard: {label}", timings

    asyncio.create_task(audit_log(user_input, answer, timings))
    return answer, timings


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    weight = k - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def benchmark(n: int = 100) -> list[dict[str, float]]:
    queries = [
        "How should I measure RAG faithfulness and context precision?",
        "Explain guardrails for prompt injection in a RAG system.",
        "What does Cohen kappa mean for LLM judge calibration?",
        "How do I benchmark latency for PII redaction and Llama Guard?",
    ]
    rows: list[dict[str, float]] = []
    for idx in range(n):
        _, timings = await guarded_pipeline(queries[idx % len(queries)])
        rows.append({"request_id": idx + 1, **timings, "total": sum(timings.values())})
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for layer in ["L1", "L2", "L3", "total"]:
        vals = [row[layer] for row in rows]
        summary[layer] = {
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "mean": statistics.fmean(vals),
        }
    return summary


async def main() -> None:
    rows = await benchmark(100)
    out = ROOT / "phase-c" / "latency_benchmark.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "L1", "L2", "L3", "total"])
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (ROOT / "phase-c" / "latency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
