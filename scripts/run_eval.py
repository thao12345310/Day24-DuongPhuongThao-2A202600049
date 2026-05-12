"""Threshold gate for Phase A results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.65,
}


def parse_threshold(raw: str) -> tuple[str, float]:
    name, value = raw.split("=", 1)
    return name.strip(), float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(ROOT / "phase-a" / "ragas_summary.json"))
    parser.add_argument("--threshold", action="append", default=[])
    args = parser.parse_args()

    thresholds = DEFAULT_THRESHOLDS.copy()
    for item in args.threshold:
        key, value = parse_threshold(item)
        thresholds[key] = value

    scores = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    failures = {
        metric: {"score": scores.get(metric, 0), "threshold": threshold}
        for metric, threshold in thresholds.items()
        if scores.get(metric, 0) < threshold
    }
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "passed", "scores": scores, "thresholds": thresholds}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

