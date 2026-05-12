"""Run Phase A RAGAS evaluation with the OpenAI-backed RAG pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openai_utils import DEFAULT_CHAT_MODEL, CostTracker, extract_json, get_openai_client  # noqa: E402
from openai_rag import OpenAIRAG  # noqa: E402


TESTSET_PATH = ROOT / "phase-a" / "testset_v1.csv"
RESULTS_PATH = ROOT / "phase-a" / "ragas_results.csv"
SUMMARY_PATH = ROOT / "phase-a" / "ragas_summary.json"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def parse_contexts(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [raw]


def fallback_openai_metric_scores(client, model: str, row: dict, answer: str, contexts: list[str], cost: CostTracker) -> dict:
    """OpenAI rubric fallback when ragas is unavailable.

    It emits the same four required metric columns so the lab artifacts remain
    complete, but the summary notes clearly say it is an OpenAI rubric fallback.
    """

    prompt = f"""
Score this RAG answer from 0.0 to 1.0 on exactly these metrics:
- faithfulness: answer is supported by retrieved context
- answer_relevancy: answer directly addresses the question
- context_precision: retrieved contexts are useful and focused
- context_recall: retrieved contexts contain enough evidence for the ground truth

Return JSON only with numeric keys: faithfulness, answer_relevancy, context_precision, context_recall.

Question: {row['question']}
Ground truth: {row['ground_truth']}
Answer: {answer}
Retrieved contexts: {json.dumps(contexts, ensure_ascii=False)}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict RAGAS-style evaluator."},
            {"role": "user", "content": prompt},
        ],
    )
    cost.add_usage(response.usage)
    parsed = extract_json(response.choices[0].message.content or "{}")
    return {metric: max(0.0, min(1.0, float(parsed.get(metric, 0.0)))) for metric in METRICS}


def try_ragas_evaluate(records: list[dict], model: str):
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except Exception:
        return None

    dataset = Dataset.from_list(
        [
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
            }
            for record in records
        ]
    )
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatOpenAI(model=model, temperature=0),
        embeddings=OpenAIEmbeddings(),
    )
    return scores.to_pandas().to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Use a small number for live demo runs.")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-openai-rubric", action="store_true")
    args = parser.parse_args()

    test_rows = list(csv.DictReader(TESTSET_PATH.open(encoding="utf-8")))
    if args.limit:
        test_rows = test_rows[: args.limit]

    rag = OpenAIRAG(chat_model=args.model, top_k=args.top_k)
    records: list[dict] = []
    for row in test_rows:
        answer, contexts = rag.answer(row["question"])
        records.append(
            {
                "question_id": row.get("question_id", len(records) + 1),
                "question": row["question"],
                "evolution_type": row.get("evolution_type", ""),
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row["ground_truth"],
            }
        )

    metric_rows = None if args.force_openai_rubric else try_ragas_evaluate(records, args.model)
    eval_mode = "ragas"
    eval_cost = CostTracker()
    if metric_rows is None:
        eval_mode = "openai_rubric_fallback"
        client = get_openai_client()
        metric_rows = [
            fallback_openai_metric_scores(client, args.model, record, record["answer"], record["contexts"], eval_cost)
            for record in records
        ]

    output_rows: list[dict] = []
    for record, metric_row in zip(records, metric_rows, strict=True):
        normalized_metrics = {metric: round(float(metric_row.get(metric, 0.0)), 3) for metric in METRICS}
        avg = round(sum(normalized_metrics.values()) / len(METRICS), 3)
        output_rows.append(
            {
                **record,
                "contexts": json.dumps(record["contexts"], ensure_ascii=False),
                **normalized_metrics,
                "average": avg,
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "evolution_type",
                "answer",
                "contexts",
                "ground_truth",
                *METRICS,
                "average",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {metric: round(sum(float(row[metric]) for row in output_rows) / len(output_rows), 3) for metric in METRICS}
    usage = rag.cost
    usage.input_tokens += eval_cost.input_tokens
    usage.output_tokens += eval_cost.output_tokens
    summary.update(
        {
            "eval_mode": eval_mode,
            "rows": len(output_rows),
            "top_k": args.top_k,
            "chat_model": args.model,
            "embedding_model": rag.embedding_model,
            "total_eval_cost_usd": usage.as_dict(args.model)["estimated_cost_usd"],
            "cost_note": "Approximate token-cost log; cached corpus embeddings are counted only when newly created.",
            "token_usage": usage.as_dict(args.model),
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
