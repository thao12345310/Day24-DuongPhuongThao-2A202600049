# AI Prompts Used (Academic Integrity Log)

This file logs all AI assistant (Claude) prompts used to build this project,
per the Lab 24 academic integrity policy.

---

## Phase A — RAGAS Evaluation

**Prompt 1 — Requirements extraction:**
> Read `lab24-student-edition.pdf` and extract the required repository structure,
> acceptance criteria, and deliverables for all 4 phases.

**Prompt 2 — Synthetic test set generation:**
> Write `phase-a/generate_testset_openai.py` that:
> - Loads Markdown docs from `./docs`
> - Calls OpenAI gpt-4o-mini to generate 50 questions with 50/25/25 distribution
>   (simple/reasoning/multi_context)
> - Saves testset_v1.csv with columns: question_id, question, ground_truth,
>   contexts, evolution_type, review_status
> - Marks at least 1 question as "edited"

**Prompt 3 — RAG pipeline:**
> Write `phase-a/openai_rag.py`: a minimal OpenAI-backed RAG pipeline that
> loads Markdown docs, chunks them (~220 words), embeds with text-embedding-3-small,
> retrieves top-k chunks by cosine similarity, and answers with gpt-4o-mini
> using only retrieved context. Cache embeddings to avoid re-embedding.

**Prompt 4 — RAGAS evaluation runner:**
> Write `phase-a/run_ragas_openai.py` that runs the RAG pipeline on testset_v1.csv,
> evaluates with RAGAS (faithfulness, answer_relevancy, context_precision,
> context_recall) when ragas is installed, or falls back to an OpenAI rubric.
> Save ragas_results.csv and ragas_summary.json. Log token usage and estimated cost.

**Prompt 5 — Offline artifact generation:**
> Write `scripts/generate_artifacts.py` to create deterministic sample docs,
> evaluation outputs, judge outputs, calibration files, guardrail test results,
> latency benchmark, blueprint, and README without any API calls.

---

## Phase B — LLM-as-Judge

**Prompt 6 — Pairwise judge with swap-and-average:**
> Write `phase-b/openai_judge.py` that:
> - Implements `pairwise_judge_with_swap()`: runs 2 comparisons per question
>   (normal order + swapped) and aggregates with majority vote
> - Implements `absolute_score()`: scores on 4 dimensions (accuracy, relevance,
>   conciseness, helpfulness) with overall = average
> - Saves pairwise_results.csv and absolute_scores.csv

**Prompt 7 — Cohen's kappa analysis:**
> Write `phase-b/kappa_analysis.py` that reads pairwise_results.csv and
> human_labels.csv using absolute paths (Path(__file__).resolve()), computes
> cohen_kappa_score, interprets it per the standard scale, and writes
> kappa_result.json. Print root cause analysis if kappa < 0.6.

---

## Phase C — Guardrails Stack

**Prompt 8 — Input guardrail (PII + topic + injection):**
> Write `phase-c/input_guard.py` with:
> - `InputGuard`: VN-specific regex (CCCD 12 digits, phone, tax code, email) +
>   Presidio NER fallback; `sanitize()` returns (scrubbed_text, latency_ms, labels)
> - `TopicGuard`: OpenAI embedding similarity against allowed topic list; graceful refusal on off-topic
> - `InjectionGuard`: regex patterns for DAN, jailbreak, roleplay, encoding attacks

**Prompt 9 — Output guardrail (Llama Guard 3):**
> Write `phase-c/output_guard.py` with `LlamaGuard3OutputGuard`:
> - Uses Groq API (llama-guard-3-8b) when GROQ_API_KEY is set
> - Falls back to deterministic regex patterns for unsafe content categories
> - Async `check_async()` for parallel execution

**Prompt 10 — Full stack integration + latency benchmark:**
> Write `phase-c/full_pipeline.py` with async `guarded_pipeline()`:
> - L1: run PII, topic, injection guards in parallel (asyncio.create_task)
> - L2: RAG pipeline call
> - L3: output guard check
> - L4: fire-and-forget audit log
> - Benchmark 100 requests and report P50/P95/P99 for L1, L2, L3, total

---

## Phase D — Blueprint

**Prompt 11 — Production blueprint:**
> Write `phase-d/blueprint.md` with:
> - Section 1: ≥5 SLOs with alert thresholds and severity levels
> - Section 2: Mermaid architecture diagram showing 4 defense-in-depth layers
> - Section 3: ≥3 incident playbooks (faithfulness drop, guardrail regression, P95 spike)
> - Section 4: monthly cost breakdown for 100k queries/month

---

## Bonus

**Prompt 12 — Bonus analyses:**
> Write `bonus/bonus_analysis.py` implementing:
> - Cross-judge protocol: 3 simulated judge profiles with majority vote
> - SelfCheckGPT-style consistency: 5 answer samples per question, Jaccard similarity
> - Semantic entropy: cluster samples and compute normalized entropy
> - Prompt Guard: pattern-based injection classifier with 5 attack categories

**Prompt 13 — Streamlit dashboard:**
> Write `bonus/dashboard.py`: a Streamlit app showing RAGAS scores, judge
> outcomes, guardrail test rates, latency P50/P95/P99, and bonus uncertainty
> metrics across all phases.

---

## Review Process

All AI-generated code was reviewed manually:
- Checked logic against PDF acceptance criteria for each task
- Verified CSV column names match required format
- Ran deterministic scripts locally to confirm outputs
- Fixed path resolution in kappa_analysis.py (absolute paths via Path(__file__))
- Confirmed latency targets: L1 P95 < 50ms, L3 P95 < 100ms
