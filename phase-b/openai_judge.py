"""OpenAI LLM-as-Judge implementation for Phase B."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openai_utils import DEFAULT_CHAT_MODEL, CostTracker, extract_json, get_openai_client  # noqa: E402


RAGAS_RESULTS = ROOT / "phase-a" / "ragas_results.csv"
PAIRWISE_OUT = ROOT / "phase-b" / "pairwise_results.csv"
ABSOLUTE_OUT = ROOT / "phase-b" / "absolute_scores.csv"


def normalize_winner(value: str) -> str:
    winner = (value or "tie").strip().upper()
    if winner in {"A", "B"}:
        return winner
    return "tie"


def judge_json(client, model: str, prompt: str, cost: CostTracker) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an impartial evaluator. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    cost.add_usage(response.usage)
    try:
        return extract_json(response.choices[0].message.content or "{}")
    except Exception as exc:
        return {"winner": "tie", "reason": f"Parse error: {exc}"}


def pairwise_judge_with_swap(client, model: str, question: str, answer_a: str, answer_b: str, cost: CostTracker) -> dict:
    prompt = """
Compare two answers to the same question.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Rate based on factual accuracy, relevance, conciseness, and helpfulness.
Output JSON only: {{"winner": "A" or "B" or "tie", "reason": "..."}}
"""
    run1 = judge_json(client, model, prompt.format(question=question, answer_a=answer_a, answer_b=answer_b), cost)
    run2 = judge_json(client, model, prompt.format(question=question, answer_a=answer_b, answer_b=answer_a), cost)

    run1_winner = normalize_winner(str(run1.get("winner", "tie")))
    swapped_winner = normalize_winner(str(run2.get("winner", "tie")))
    if swapped_winner == "A":
        run2_winner = "B"
    elif swapped_winner == "B":
        run2_winner = "A"
    else:
        run2_winner = "tie"
    final = run1_winner if run1_winner == run2_winner else "tie"
    return {
        "run1_winner": run1_winner,
        "run1_reason": str(run1.get("reason", "")),
        "run2_winner": run2_winner,
        "run2_reason": str(run2.get("reason", "")),
        "winner_after_swap": final,
    }


def absolute_score(client, model: str, question: str, answer: str, cost: CostTracker) -> dict:
    prompt = f"""
Score the answer on 4 dimensions, each 1-5 scale:
1. Factual accuracy (1=many errors, 5=fully accurate)
2. Relevance (1=off-topic, 5=directly answers)
3. Conciseness (1=verbose, 5=appropriately brief)
4. Helpfulness (1=unclear, 5=actionable)

Question: {question}
Answer: {answer}

Output JSON only:
{{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int, "overall": float}}
"""
    parsed = judge_json(client, model, prompt, cost)
    dims = {}
    for key in ["accuracy", "relevance", "conciseness", "helpfulness"]:
        try:
            dims[key] = min(5, max(1, int(parsed.get(key, 3))))
        except Exception:
            dims[key] = 3
    dims["overall"] = round(sum(dims.values()) / 4, 2)
    return dims


def make_baseline_answer(row: dict) -> str:
    """Simple comparison answer for pairwise judging when only one RAG version exists."""

    return (
        "The answer should be evaluated with RAGAS and guardrails. "
        "More specific evidence is needed from the retrieved context."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args()

    rows = list(csv.DictReader(RAGAS_RESULTS.open(encoding="utf-8")))[: args.limit]
    client = get_openai_client()
    cost = CostTracker()

    pairwise_rows: list[dict] = []
    absolute_rows: list[dict] = []
    for row in rows:
        answer_a = row.get("answer", "")
        answer_b = make_baseline_answer(row)
        pairwise = pairwise_judge_with_swap(client, args.model, row["question"], answer_a, answer_b, cost)
        pairwise_rows.append(
            {
                "question_id": row.get("question_id", len(pairwise_rows) + 1),
                "question": row["question"],
                "answer_a": answer_a,
                "answer_b": answer_b,
                **pairwise,
            }
        )
        absolute_rows.append(
            {
                "question_id": row.get("question_id", len(absolute_rows) + 1),
                "question": row["question"],
                **absolute_score(client, args.model, row["question"], answer_a, cost),
            }
        )

    PAIRWISE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PAIRWISE_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "answer_a",
                "answer_b",
                "run1_winner",
                "run1_reason",
                "run2_winner",
                "run2_reason",
                "winner_after_swap",
            ],
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)

    with ABSOLUTE_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "question", "accuracy", "relevance", "conciseness", "helpfulness", "overall"],
        )
        writer.writeheader()
        writer.writerows(absolute_rows)

    print(json.dumps({"pairwise_rows": len(pairwise_rows), "absolute_rows": len(absolute_rows), **cost.as_dict(args.model)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
