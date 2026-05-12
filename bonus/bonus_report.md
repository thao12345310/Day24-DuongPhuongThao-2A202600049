# Bonus Report (+15 cap)

Implemented five bonus items worth +16 raw points. The lab caps bonus at +15, so this satisfies the maximum bonus target.

| Bonus item | Raw points | Artifact | Result |
|---|---:|---|---|
| Cross-judge protocol | +3 | `bonus/cross_judge_results.csv` | 3 simulated judges, unanimous rate 80.0% |
| SelfCheckGPT-style consistency | +4 | `bonus/selfcheck_semantic_entropy.csv` | Avg consistency 0.687 |
| Semantic entropy | +4 | `bonus/selfcheck_semantic_entropy.csv` | Avg entropy 0.062 |
| Prompt Guard-style classifier | +2 | `bonus/prompt_guard_results.csv` | Attack recall 85.0%, FP 0.0% |
| Eval dashboard | +3 | `bonus/dashboard.py` | Streamlit dashboard over Phase A/B/C/bonus artifacts |

## Cross-Judge Protocol

The protocol runs three independent judge profiles over the 30 pairwise comparisons and aggregates with majority voting. Disagreements become a monitoring signal because they indicate cases where a single judge is not stable enough for release gating.

## SelfCheckGPT and Semantic Entropy

For each of 20 questions, five answer samples are compared pairwise. Low lexical agreement increases hallucination risk. The same samples are clustered into semantic groups and converted into normalized entropy; high entropy means the model is producing multiple incompatible meanings.

## Prompt Guard

The prompt guard classifier detects direct override attempts, persona jailbreaks, obfuscated payloads, and exfiltration attempts before the topic validator. It is intentionally deterministic for local reproducibility.

## Dashboard

Run:

```bash
streamlit run bonus/dashboard.py
```

The dashboard gives a quick operational view of RAGAS scores, judge outcomes, guardrail test rates, latency, and bonus uncertainty metrics.
