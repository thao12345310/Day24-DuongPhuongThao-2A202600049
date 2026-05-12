# Judge Bias Report

## Kappa Calibration Result

| Metric | Value |
|---|---:|
| Cohen's kappa | 0.630 |
| Observed agreement | 80% (8/10) |
| Sample pairs | 10 |
| Interpretation | Substantial agreement — production-ready ✓ |

Kappa of 0.63 falls in the 0.6–0.8 range (Substantial agreement). The judge is ready for
production monitoring with periodic human audits every 500–1000 queries.

## Position Bias

| Metric | Value |
|---|---:|
| A wins when listed first | 12/30 (40.0%) |
| Expected if neutral | about 50% |

The judge shows mild position preference but swap-and-average converts disagreements to `tie`.

## Length Bias

| Metric | Value |
|---|---:|
| B longer than A | 30/30 |
| B wins when longer | 17/30 (56.7%) |

Longer answers win too often when they add generic production phrasing. The mitigation is to cap answer length in the prompt, include conciseness in the rubric, and keep pairwise swap checks.

## Calibration Summary

Human labels disagree on two of ten sampled cases, mostly where the judge preferred extra detail but the human preferred concise grounded evidence.
