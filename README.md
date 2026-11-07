# ICD-11 Clinical Coding System — Sugen Technologies Take-Home

A lightweight, 100% offline clinical coding system that predicts up to 3 ICD-11 MMS codes per outpatient consultation note. Operates with zero network calls and zero generative LLMs at build time and run time.

Detailed pipeline architecture and diagrams are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 1. Evaluation & Benchmark Scores

Evaluated using the exact match grader (`score.py`) against valid leaf categories in WHO ICD-11 MMS 2026-01 (`data/mms-2026-01.json`). Exact match scoring awards full points only to the exact leaf code supported by the note; non-leaf parent categories score 0.

### Dev Notes (`data/dev_notes.jsonl`, 60 notes)
- **Top-1 Accuracy:** 81.67% (95% CI: [70.0%, 90.0%])
- **Top-3 Accuracy:** 85.00% (95% CI: [75.0%, 93.3%])
- **Official Score:** **62.08 / 75.0** (Normalized: **82.78 / 100**, Core scale: 50 * Top1 + 25 * Top3)
- **Missing Predictions:** 0 (100% response rate across all input IDs)
- **Candidate Recall@50:** 100.0% (Recall@30: 96.67%, Recall@10: 88.33%)
- **Measured Throughput:** 2.37 notes/sec (380.8 ms mean latency per note)
- **600-Note Benchmark Runtime:** 253.40s (4.22 minutes), using 7.0% of the 60-minute limit (< 25 min guardrail: PASS)

### Estimated Hidden Score (600 Unseen Notes, 100 Points Total)
- **Core Slice (450 notes, 75 pts):** Similar clinic distribution and styles as dev. Projected: **~55–58 points**.
- **Generalisation Slice (150 notes, 25 pts):** All 26 chapters, multilingual consults. Projected: **~12–15 points**.
- **Total Estimated Score:** **~67–73 / 100**

---

## 2. Evaluation Beyond the 60 Dev Notes

- **`data/held_out_notes.jsonl` (85 notes):** Spans all 26 ICD-11 chapters and 8 languages (EN, SW, FR, ES, HI, AR, ZH, PT).
- **Transparency Notice:** This is a **self-written smoke set, optimistic; not evidence of generalisation**. Its vocabulary was created alongside dictionary expansion; it serves to verify runtime stability, language-routing logic, and zero-crash execution across diverse chapters, but does not represent unseen clinical distribution performance. Top-1 on smoke set: 32.94%, Top-3: 56.47%, Score: 30.59 / 75.0.

---

## 3. Data Added & Provenance

| Asset | Source | Licence | Purpose |
|:---|:---|:---:|:---|
| `data/mms-2026-01.json` | WHO ICD-11 MMS 2026-01 via Sugen Data Pack | CC BY-ND 3.0 IGO | Ground truth catalog of 18,375 categories |
| `kb/who_enrichments.json` | WHO ICD-API v2 (`https://id.who.int/icd/release/11/2024-01/mms`) | CC BY-NC-ND 3.0 IGO | 93,387 official WHO index terms, definitions, inclusions, exclusions |
| `kb/enriched_kb.json` | Derived knowledge base combining MMS structure & WHO enrichments | CC BY-NC-ND 3.0 IGO | Unified offline knowledge graph with parent-child links |
| `kb/dense_embeddings.npz` | Pre-computed embeddings from `all-MiniLM-L6-v2` | Apache 2.0 | 26 MB compressed float32 matrix for offline semantic lookup |
| `models/all-MiniLM-L6-v2` | Hugging Face / Sentence-Transformers | Apache 2.0 | 22M parameter sentence encoder vendored for zero-network CPU inference |

---

## 4. Real AI-Tooling Mistake Caught & Fixed

**The Dockerfile Missing-COPY Silent Failure:**
- *How it was found:* Running `scripts/docker_context_check.py --run`, which copies strictly the files declared in the Dockerfile `COPY` instructions into an isolated temporary directory and executes `entrypoint.py` on `data/dev_notes.jsonl`.
- *The bug:* `pipeline.py` imported `llm_reranker`, but `Dockerfile` omitted `COPY llm_reranker.py .`. In the local workspace where both files existed, tests passed. In the isolated container build, the missing file threw an `ImportError` on worker startup, causing `entrypoint.py` to silently catch the exception and fall back to the constant default `["1F40.Z", "CA40.Z", "1A07.Z"]` for all 60 notes.
- *The fix:* Completely removed all LLM dependencies and code. Added `STRICT=1` environment mode to fail loudly on unhandled errors, and added dynamic per-note BM25 fallback in `entrypoint.py` so constant fallback predictions are never emitted.

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

Hours spent: <ask me>
