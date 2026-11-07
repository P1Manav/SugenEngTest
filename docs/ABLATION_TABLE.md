# Empirical Ablation Study — ICD-11 Pipeline Architecture

Evaluated on 60 outpatient notes (`data/dev_notes.jsonl`) using the official grader (`score.py`) rules against WHO ICD-11 MMS 2026-01 (`data/mms-2026-01.json`).

All decision rules use strictly data-driven clinical concepts (`rules/clinical_concepts.yaml`) with qualifier-differential matching. Zero hardcoded ICD code heuristics.

## Ablation Matrix (95% Empirical Bootstrap CIs, 1,000 Resamples)

| Ablation Variant | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Score / 75.0 (95% CI) | Recall@30 | Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Full Pipeline** | **83.3%**<br>[73.3%, 91.7%] | **88.3%**<br>[80.0%, 95.0%] | **63.75**<br>[56.67, 69.58] | 96.7% | 26.0s |
| Dense-only (no BM25) | **73.3%**<br>[61.7%, 85.0%] | **83.3%**<br>[73.3%, 91.7%] | **57.50**<br>[49.58, 65.00] | 85.0% | 7.7s |
| BM25-only (no Dense) | **80.0%**<br>[70.0%, 90.0%] | **88.3%**<br>[80.0%, 95.0%] | **62.08**<br>[55.00, 68.33] | 96.7% | 22.6s |
| No Abbreviations | **80.0%**<br>[70.0%, 90.0%] | **90.0%**<br>[81.7%, 96.7%] | **62.50**<br>[55.83, 68.75] | 96.7% | 27.4s |
| No Assertion Filter | **76.7%**<br>[65.0%, 86.7%] | **88.3%**<br>[80.0%, 95.0%] | **60.42**<br>[52.92, 67.08] | 100.0% | 29.0s |
| No Specificity Descent | **43.3%**<br>[31.7%, 56.7%] | **48.3%**<br>[35.0%, 61.7%] | **33.75**<br>[24.58, 42.94] | 83.3% | 22.2s |
| No Reranker | **46.7%**<br>[35.0%, 60.0%] | **70.0%**<br>[58.3%, 81.7%] | **40.83**<br>[32.92, 49.58] | 96.7% | 24.2s |

## Key Insights
1. **Specificity Descent is Vital (+30.00 pts):** Without qualifier-differential descent and .Z resolution, non-leaf categories score 0 on the exact match grader.
2. **Rule-Based Reranker (+22.92 pts):** Candidate prioritization based on section presence, biological consistency, and distinguishing qualifiers is crucial to promote exact diagnoses into Top-1.
3. **Hybrid Sparse + Dense Retrieval (+1.67 pts over BM25, +6.25 pts over Dense):** Combining BM25 lexical precision with MiniLM semantic generalization yields 100% Candidate Recall@50 and 96.7% Recall@30.
4. **Assertion & Negation Filter (+3.33 pts):** Prevents negated conditions, unconfirmed differentials, and family history from being erroneously coded as active primary diagnoses.
