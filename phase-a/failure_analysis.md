# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | How should answer relevancy and context precision work tog | multi_context | 0.565 | 0.571 | 0.451 | 0.532 | 0.53 | C1 |
| 2 | How should audit log and faithfulness work together in Lab | multi_context | 0.596 | 0.596 | 0.473 | 0.489 | 0.538 | C1 |
| 3 | How should pairwise judge and cohen kappa work together in | multi_context | 0.565 | 0.593 | 0.474 | 0.556 | 0.547 | C1 |
| 4 | How should prompt injection and llama guard work together  | multi_context | 0.546 | 0.571 | 0.535 | 0.535 | 0.547 | C1 |
| 5 | Why is pairwise judge important for a production RAG syste | reasoning | 0.642 | 0.619 | 0.535 | 0.548 | 0.586 | C2 |
| 6 | Why is prompt injection important for a production RAG sys | reasoning | 0.61 | 0.607 | 0.57 | 0.58 | 0.592 | C2 |
| 7 | Why is faithfulness important for a production RAG system? | reasoning | 0.581 | 0.687 | 0.518 | 0.617 | 0.601 | C2 |
| 8 | How should cohen kappa and pii redaction work together in  | multi_context | 0.735 | 0.724 | 0.605 | 0.657 | 0.68 | C1 |
| 9 | How should context recall and pairwise judge work together | multi_context | 0.702 | 0.716 | 0.621 | 0.738 | 0.694 | C1 |
| 10 | What does Lab 24 say about prompt injection? | simple | 0.717 | 0.763 | 0.634 | 0.678 | 0.698 | C2 |

## Clusters Identified

### Cluster C1: Multi-context retrieval gaps

**Pattern:** Cross-document questions have lower context precision and recall because the retriever often returns only one useful chunk.

**Examples:**
- How should prompt injection and llama guard work together in Lab 24?
- How should latency benchmark and audit log work together in Lab 24?

**Root cause:** Dense top-k retrieval is too narrow for questions that require two components.

**Proposed fix:** Increase `top_k` from 3 to 6, add Maximal Marginal Relevance, and add a reranker before generation.

### Cluster C2: Reasoning and abstraction failures

**Pattern:** Reasoning questions score lower on faithfulness when the generated answer adds generic production advice not present in retrieved context.

**Examples:**
- Why is Cohen kappa important for a production RAG system?
- Why is topic guard important for a production RAG system?

**Root cause:** The answer template over-generalizes from the question instead of quoting retrieved evidence.

**Proposed fix:** Add citation-required answer formatting, reject answers with no supporting span, and run a second retrieval pass for low recall questions.
