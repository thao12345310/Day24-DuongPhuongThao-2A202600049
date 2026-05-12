"""Compute Cohen's kappa between human labels and LLM judge.

Run from any directory:
    python phase-b/kappa_analysis.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
PAIRWISE = ROOT / "phase-b" / "pairwise_results.csv"
HUMAN = ROOT / "phase-b" / "human_labels.csv"
REPORT = ROOT / "phase-b" / "kappa_result.json"

with PAIRWISE.open(newline="", encoding="utf-8") as f:
    judge_rows = list(csv.DictReader(f))[:10]
with HUMAN.open(newline="", encoding="utf-8") as f:
    human_rows = list(csv.DictReader(f))

judge = [r["winner_after_swap"] for r in judge_rows]
human = [r["human_winner"] for r in human_rows]
kappa = cohen_kappa_score(human, judge)
agreed = sum(h == j for h, j in zip(human, judge))

if kappa < 0:
    interpretation = "WORSE than chance — judge sai hệ thống"
elif kappa < 0.2:
    interpretation = "Slight agreement — không tin được"
elif kappa < 0.4:
    interpretation = "Fair agreement — vẫn yếu"
elif kappa < 0.6:
    interpretation = "Moderate agreement — có thể dùng cho monitoring"
elif kappa < 0.8:
    interpretation = "Substantial agreement — production-ready ✓"
else:
    interpretation = "Almost perfect agreement — hiếm gặp"

result = {
    "kappa": round(kappa, 4),
    "observed_agreement": round(agreed / len(human), 3),
    "sample_pairs": len(human),
    "agreed": agreed,
    "interpretation": interpretation,
}
REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Cohen's kappa: {kappa:.3f}")
print(f"Interpretation: {interpretation}")
print(f"Observed agreement: {agreed}/{len(human)} = {agreed/len(human):.1%}")

if kappa < 0.6:
    print("\nRoot cause analysis:")
    print("- Likely position bias: re-check judge_bias_report.md position-bias numbers.")
    print("- Likely length bias: if B wins when longer > 60%, cap answer length in prompt.")
    print("- Action: expand to 50+ calibration pairs; require 2nd annotator on disagreements.")
