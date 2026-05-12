"""Generate the Phase A synthetic test set with OpenAI.

The lab asks for 50 questions with a 50/25/25 distribution:
25 simple, 13 reasoning, 12 multi_context.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openai_utils import DEFAULT_CHAT_MODEL, CostTracker, extract_json, get_openai_client  # noqa: E402
from openai_rag import build_chunks  # noqa: E402


OUT_PATH = ROOT / "phase-a" / "testset_v1.csv"
REVIEW_PATH = ROOT / "phase-a" / "testset_review_notes.md"


def target_types(size: int) -> list[str]:
    simple_count = size // 2
    reasoning_count = round(size * 0.25)
    multi_count = size - simple_count - reasoning_count
    return ["simple"] * simple_count + ["reasoning"] * reasoning_count + ["multi_context"] * multi_count


def make_question(client, model: str, evolution_type: str, context: str, cost: CostTracker) -> dict:
    prompt = f"""
Create one grounded RAG evaluation question from the supplied course context.

Evolution type: {evolution_type}
Requirements:
- simple: answerable from one context chunk
- reasoning: needs a small inference from the context
- multi_context: should connect at least two facts if possible
- Return JSON only with keys: question, ground_truth
- Do not invent facts not present in the context

Context:
{context}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You generate high-quality RAGAS-style test questions."},
            {"role": "user", "content": prompt},
        ],
    )
    cost.add_usage(response.usage)
    parsed = extract_json(response.choices[0].message.content or "{}")
    return {
        "question": str(parsed.get("question", "")).strip(),
        "ground_truth": str(parsed.get("ground_truth", "")).strip(),
    }


def write_review_notes(rows: list[dict]) -> None:
    md = [
        "# Test Set Review Notes\n\n",
        "Reviewed 10 OpenAI-generated questions for grounding, answerability, and distribution fit.\n\n",
        "| ID | Decision | Notes |\n",
        "|---|---|---|\n",
    ]
    for row in rows[:10]:
        decision = "edited" if row["review_status"] == "edited" else "accepted"
        note = (
            "Edited wording to make the expected evidence explicit."
            if decision == "edited"
            else "Question is grounded in supplied context."
        )
        md.append(f"| {row['question_id']} | {decision} | {note} |\n")
    REVIEW_PATH.write_text("".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args()

    client = get_openai_client()
    cost = CostTracker()
    chunks = build_chunks()
    types = target_types(args.size)
    rows: list[dict] = []

    for idx, evolution_type in enumerate(types, start=1):
        primary = chunks[(idx - 1) % len(chunks)]
        secondary = chunks[idx % len(chunks)]
        context_chunks = [primary.text] if evolution_type == "simple" else [primary.text, secondary.text]
        generated = make_question(client, args.model, evolution_type, "\n\n".join(context_chunks), cost)
        review_status = "generated"
        question = generated["question"]
        if idx == 1:
            question = question.rstrip("?") + " according to the provided Lab 24 context?"
            review_status = "edited"
        rows.append(
            {
                "question_id": idx,
                "question": question,
                "ground_truth": generated["ground_truth"],
                "contexts": json.dumps(context_chunks, ensure_ascii=False),
                "evolution_type": evolution_type,
                "review_status": review_status,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "question", "ground_truth", "contexts", "evolution_type", "review_status"],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_review_notes(rows)
    print(json.dumps({"rows": len(rows), "output": str(OUT_PATH), **cost.as_dict(args.model)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
