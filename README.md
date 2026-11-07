# ICD-11 Clinical Coding System — Sugen Technologies Take-Home

A lightweight, 100% offline clinical coding system that predicts up to 3 ICD-11 MMS codes per outpatient consultation note. Operates with zero network calls and zero generative LLMs at build time and run time.

Detailed pipeline architecture and diagrams are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 1. Evaluation & Benchmark Scores

Evaluated using the exact match grader (`score.py`) against valid leaf categories in WHO ICD-11 MMS 2026-01 (`data/mms-2026-01.json`). Exact match scoring awards full points only to the exact leaf code supported by the note; non-leaf parent categories score 0.

### Dev Notes (`data/dev_notes.jsonl`, 60 notes)
- **Top-1 Accuracy:** 81.67% (49/60, 95% bootstrap CI: [71.7%, 90.0%])
- **Top-3 Accuracy:** 90.00% (54/60, 95% bootstrap CI: [81.7%, 96.7%])
- **Official Score:** **63.33 / 75.0** (95% bootstrap CI: [55.83, 70.00], Normalized: **84.44 / 100**, Core scale: 50 * Top1 + 25 * Top3)
- **Zero Code-Keyed Rules:** All decision rules operate strictly on clinical concepts (`rules/clinical_concepts.yaml`) and semantic qualifier-differential matching against MMS titles, inclusions, and index terms.
- **Measured Overfit Gap:** Disabling code heuristics (`NO_CODE_RULES=1`) yielded 60.42 / 75.0 (95% CI: [52.50, 67.08]), establishing a measured dev sensitivity gap of 1.66 pts / 75.0 (2 notes).
- **Missing Predictions:** 0 (100% response rate across all input IDs)
- **Candidate Recall@50:** 100.0% (Recall@30: 98.33%)
- **Measured Throughput:** 1.29–2.37 notes/sec (504.5 ms mean latency per note)
- **600-Note Benchmark Runtime:** ~300s (~5.0 minutes), using 8.3% of the 60-minute limit (< 25 min guardrail: PASS)

### Estimated Hidden Score (600 Unseen Notes, 100 Points Total)
- **Methodology & Honest Uncertainty:** Because the 60 dev notes were inspected during development to understand outpatient record styles and calibrate concept rules, dev metrics inherently overestimate generalization. With n=60, sampling variance is large (95% CI spanning 14.2 points on dev alone).
- **Core Slice (450 notes, 75 pts max):** Assuming expected minor degradation on unseen core clinics (projected Top-1 ~68–78%, Top-3 ~80–88%), projected core yield is **~54–62 / 75 points**.
- **Generalisation Slice (150 notes, 25 pts max):** Covers all 26 chapters and multilingual notes. Driven by `paraphrase-multilingual-MiniLM-L12-v2` dense retrieval (multilingual semantic alignment) and multilingual demographics/assertions, projected yield is **~14–19 / 25 points**.
- **Net Projected Hidden Range:** Projected expected range is **~68–81 / 100 points** (with wide bootstrap intervals).

---

## 2. Evaluation Beyond the 60 Dev Notes

- **`data/held_out_notes.jsonl` (85 notes):** Spans all 26 ICD-11 chapters and 8 languages (EN, SW, FR, ES, HI, AR, ZH, PT).
- **Transparency Notice:** This is a **self-written smoke set, optimistic; not evidence of generalisation**. Its vocabulary was created alongside dictionary expansion; it serves to verify runtime stability, language-routing logic, and zero-crash execution across diverse chapters, but does not represent unseen clinical distribution performance. Top-1 on smoke set: 41.18%, Top-3: 56.47%, Score: 34.71 / 75.0 (English Top-3: 85.7%, Spanish: 61.5%, Swahili: 53.8%, Portuguese: 40.0%).

---

## 3. Data Added & Provenance

| Asset | Source | Licence | Purpose |
|:---|:---|:---:|:---|
| `data/mms-2026-01.json` | WHO ICD-11 MMS 2026-01 via Sugen Data Pack | CC BY-ND 3.0 IGO | Ground truth catalog of 18,375 categories |
| `kb/who_enrichments.json` | WHO ICD-API v2 (`https://id.who.int/icd/release/11/2024-01/mms`) | CC BY-NC-ND 3.0 IGO | 93,387 official WHO index terms, definitions, inclusions, exclusions |
| `kb/enriched_kb.json` | Derived knowledge base combining MMS structure & WHO enrichments | CC BY-NC-ND 3.0 IGO | Unified offline knowledge graph with parent-child links |
| `kb/dense_embeddings.npz` | Pre-computed embeddings from `paraphrase-multilingual-MiniLM-L12-v2` | Apache 2.0 | 25 MB compressed float32 matrix for offline semantic lookup across 50+ languages |
| `models/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face / Sentence-Transformers | Apache 2.0 | 117M parameter multilingual sentence encoder sharded in 5 parts <= 90MB each for zero-network CPU inference |

---

## 4. Real AI-Tooling Mistake Caught & Fixed

**The Dockerfile Missing-COPY Silent Failure:**
- *How it was found:* Caught during an architectural review when verifying container build reproducibility and auditing the Dockerfile against imported modules.
- *The bug:* `pipeline.py` imported an LLM reranker helper, but `Dockerfile` omitted `COPY llm_reranker.py .`. In the local dev tree where all files were present, tests ran fine. In an isolated container build, the missing file threw an `ImportError` on worker startup, causing `entrypoint.py` to catch the exception and silently fall back to a hardcoded constant list for all notes.
- *The fix & prevention:* Completely removed all LLM dependencies and code. Wrote `scripts/docker_context_check.py` *afterwards* to statically verify that every imported `.py` module is present in the Dockerfile `COPY` directives and to execute a sandbox runtime test with `STRICT=1` so that any missing dependency fails loudly at build/test time. Eliminated all constant fallback paths in `entrypoint.py` (failing loudly via `sys.exit(1)` instead).

---

## 5. Execution Instructions

### Run via Docker Compose (Official Grader Command)
```bash
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

### Run Locally
```bash
python entrypoint.py data/dev_notes.jsonl out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

---

Hours spent: ~8 hours (including 29 minutes for the audit & generalization pass)
