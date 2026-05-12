# Lab 24 — Full Evaluation & Guardrail System

## Overview

This repo implements the Lab 24 evaluation and guardrail system for a RAG assistant. Phase A builds a 50-question synthetic test set, runs a RAG evaluation with faithfulness, answer relevancy, context precision, and context recall, then clusters the lowest-scoring failures. Phase B adds an OpenAI LLM-as-Judge workflow with pairwise comparison, swap-and-average position-bias mitigation, absolute rubric scoring, and human calibration with Cohen's kappa. Phase C implements defense-in-depth guardrails for PII redaction, topic scope validation, prompt-injection detection, Llama Guard 3-compatible output screening, and end-to-end latency benchmarking. Phase D summarizes the system as a production blueprint with SLOs, an architecture diagram, incident playbooks, and monthly cost assumptions.

The repository now has two execution modes. The checked-in CSV/JSON/Markdown artifacts are deterministic so the project can be reviewed without spending API credits. The live lab path uses `OPENAI_API_KEY` through the official OpenAI SDK for test-set generation, retrieval/generation, RAGAS-compatible evaluation, and LLM judging.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
export GROQ_API_KEY=...  # optional, for Llama Guard 3 API mode
```

## Live OpenAI Runs

```bash
# Phase A.1: generate 50 OpenAI-backed synthetic questions
python phase-a/generate_testset_openai.py --size 50

# Phase A.2: run OpenAI-backed RAG + RAGAS when installed
python phase-a/run_ragas_openai.py

# Demo-friendly 5-question run
python phase-a/run_ragas_openai.py --limit 5

# Phase B.1-B.2: pairwise judge + absolute rubric scoring with OpenAI
python phase-b/openai_judge.py --limit 30

# CI/eval gate over phase-a/ragas_summary.json
python scripts/run_eval.py

# Phase C latency benchmark
python phase-c/full_pipeline.py
```

Run `python phase-c/full_pipeline.py` for the live full-stack demo (requires both OPENAI_API_KEY and GROQ_API_KEY).

## Results Summary

### Phase A (RAGAS)

- Test set: 50 questions, distribution 25 simple / 13 reasoning / 12 multi-context.
- Current checked-in summary: Faithfulness 0.773 | AR 0.808 | CP 0.698 | CR 0.743.
- Total eval cost: $0.00 (deterministic run). Live run via `python phase-a/run_ragas_openai.py` logs actual token usage and `total_eval_cost_usd` into `phase-a/ragas_summary.json` (estimated ~$0.15 for 50 questions with gpt-4o-mini).
- Observations: Context Precision (0.698) is the weakest metric. Multi-context questions underscore due to the retriever returning only one relevant chunk — see `phase-a/failure_analysis.md` Cluster C1.
- Identified 2 failure clusters in `phase-a/failure_analysis.md`.

### Phase B (LLM-Judge)

- Pairwise judging runs 30 questions with swap-and-average mitigation (2 runs per question, order swapped).
- Absolute rubric scoring across 4 dimensions: accuracy, relevance, conciseness, helpfulness.
- `phase-b/openai_judge.py` calls OpenAI for both pairwise and absolute rubric scoring.
- Human calibration: 10 labeled pairs in `phase-b/human_labels.csv`; Cohen's kappa = **0.630** (Substantial agreement — production-ready).
- Position bias: A wins when listed first 40% (mild, mitigated by swap). Length bias: B wins when longer 56.7%.
- Run `python phase-b/kappa_analysis.py` to recompute kappa and regenerate `phase-b/kappa_result.json`.

### Phase C (Guardrails)

- PII detection: 10 inputs (EN + VN mix), detection rate ≥ 80%; Latency P95 < 50 ms; edge cases: empty, very long, multilingual.
- Topic validator (Option 2 — LLM zero-shot / deterministic fallback): tested on 20 inputs (10 on-topic, 10 off-topic); accuracy ≥ 75%; graceful refusal message on off-topic.
- Adversarial defense: 20 attacks (DAN ×5, roleplay ×5, split ×3, encoding ×3, indirect ×4); detection rate 19/20 = **95%**; false positive on 10 legitimate queries = **0%**.
- Output guard (Llama Guard 3 via Groq API, local fallback): 10 unsafe / 10 safe outputs; detection 100% / FP 0%; Latency P95 measured.
- Full stack benchmark: **100 requests**, L1 P95 **0.44 ms** (target < 50 ms ✓), L3 P95 **0.17 ms** (target < 100 ms ✓), Total P95 **4.02 ms**.

### Phase D (Blueprint)

See `phase-d/blueprint.md`.

## Lessons Learned

Evaluation needs both metric-based scoring and calibrated judgment. RAGAS-style metrics surface grounding and retrieval failures, while pairwise judging helps compare versions but needs swap checks because position and length can move decisions.

Guardrails should be measured like product features. PII redaction, topic validation, injection detection, and output safety each need regression tests, false-positive tracking, and latency budgets. The full stack is only useful if it is fast enough and logs enough detail to debug incidents.

## Demo Video

https://drive.google.com/drive/folders/1D-Q3Ot-bSDBBHWD7MqSRBounV1ZygjBu?usp=sharing
