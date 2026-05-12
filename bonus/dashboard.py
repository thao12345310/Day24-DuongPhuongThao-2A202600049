"""Streamlit dashboard for Lab 24 evaluation and guardrail artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


st.set_page_config(page_title="Lab 24 Eval Dashboard", layout="wide")
st.title("Lab 24 Evaluation and Guardrail Dashboard")

ragas_summary = json.loads((ROOT / "phase-a" / "ragas_summary.json").read_text(encoding="utf-8"))
latency_summary = json.loads((ROOT / "phase-c" / "latency_summary.json").read_text(encoding="utf-8"))
bonus_summary = json.loads((ROOT / "bonus" / "bonus_summary.json").read_text(encoding="utf-8"))

st.subheader("RAGAS Summary")
cols = st.columns(4)
for col, metric in zip(cols, ["faithfulness", "answer_relevancy", "context_precision", "context_recall"], strict=True):
    col.metric(metric.replace("_", " ").title(), ragas_summary[metric])

ragas = load_csv("phase-a/ragas_results.csv")
st.bar_chart(ragas[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

st.subheader("Judge and Calibration")
pairwise = load_csv("phase-b/pairwise_results.csv")
judge_counts = pairwise["winner_after_swap"].value_counts().rename_axis("winner").reset_index(name="count")
st.dataframe(judge_counts, use_container_width=True)

st.subheader("Guardrail Rates")
adv = load_csv("phase-c/adversarial_test_results.csv")
legit = load_csv("phase-c/legitimate_guardrail_results.csv")
prompt_guard = load_csv("bonus/prompt_guard_results.csv")
guard_cols = st.columns(3)
guard_cols[0].metric("Adversarial Detection", f"{adv['blocked'].mean():.0%}")
guard_cols[1].metric("Legitimate FP", f"{legit['blocked'].mean():.0%}")
guard_cols[2].metric("Prompt Guard Accuracy", f"{prompt_guard['correct'].mean():.0%}")

st.subheader("Latency P95")
lat_cols = st.columns(4)
for col, layer in zip(lat_cols, ["L1", "L2", "L3", "total"], strict=True):
    col.metric(layer, f"{latency_summary[layer]['p95']:.2f} ms")

st.subheader("Bonus Uncertainty Metrics")
selfcheck = load_csv("bonus/selfcheck_semantic_entropy.csv")
st.scatter_chart(selfcheck, x="consistency_score", y="semantic_entropy", color="flagged")

st.subheader("Bonus Summary")
st.json(bonus_summary)

