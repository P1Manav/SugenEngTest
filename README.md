# ICD-11 Clinical Coding System — Sugen Technologies Take-Home

A production-grade, offline-first clinical coding system that ingests outpatient consultation notes and predicts up to 3 ICD-11 MMS codes per note, best first.

The system is designed with a **strictly offline baseline** requiring **zero network access and zero generative LLMs** at build time and run time, accompanied by an **optional LLM reranking stage** that activates only when external keys are supplied and operates within strict safety, budget, and time boundaries.

---

## 1. System Performance & Evaluation

All metrics are computed using the exact match evaluation script (`score.py`) against valid leaf categories in WHO ICD-11 MMS 2026-01 (`data/mms-2026-01.json`). Exact match scoring awards full points only to the exact leaf code supported by the note; non-leaf parent categories and invalid codes score 0.

### 1.1 Performance Summary

| Evaluation Suite | Size & Scope | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Official Grader Score | Missing Preds | Invalid Codes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Dev Notes (`data/dev_notes.jsonl`)** | 60 notes (Kenyan outpatient clinic flavour, SOAP, short consults, ambient transcripts) | **53.33%**<br>`[40.0%, 66.7%]` | **71.67%**<br>`[60.0%, 83.3%]` | **44.58 / 75.0**<br>(Normalized: **59.44 / 100**) | **0** (0.0%) | **0** (0.0%) |
| **Held-Out Benchmark (`data/held_out_notes.jsonl`)** | 85 notes spanning **all 26 chapters** (01–26) in **8 languages** (EN, SW, FR, ES, HI, AR, ZH, PT) | **32.94%**<br>`[23.5%, 43.5%]` | **56.47%**<br>`[45.9%, 67.1%]` | **30.59 / 75.0**<br>(Normalized: **40.78 / 100**) | **0** (0.0%) | **0** (0.0%) |

- **Candidate Recall on Dev:**
  - Recall@30: **95.0%**
  - Recall@50: **98.3%**
- **Bootstrap Confidence Intervals:** Calculated over 1,000 resamples at the note level via `evaluate.py`.
- **Zero Hallucination / Zero Missing:** Every single input note produces a valid output record with 1–3 strictly validated ICD-11 leaf codes.

### 1.2 Estimated Score on Hidden Grading Set (600 Notes)

The evaluation harness grades 600 unseen notes:
1. **Core Slice (450 notes, 75 pts):** Similar clinic distribution and styles as dev. Based on our dev score of 44.58 / 75, we project **~44–48 points**.
2. **Generalisation Slice (150 notes, 25 pts):** All 26 chapters, multi-language consults. Based on our held-out 26-chapter 8-language score of 40.78 / 100 (which scales to 10.2 / 25), we project **~10–12 points**.
3. **Total Estimated Score:** **~55–60 / 100** for the baseline offline system alone.

---

## 2. Architecture Overview

```mermaid
flowchart TD
    subgraph INGESTION["1. Input & Multilingual Processing"]
        IN[Raw Clinical Note] --> LID{Language ID<br/>Unicode Script Gate + Lingua}
        LID -->|Non-Latin / Multilingual| TRANS[Clinical Normalizer<br/>Swahili, Hindi, Arabic, Chinese, Spanish, French]
        LID -->|English| EXP[Abbreviation & Section Expansion]
        TRANS --> EXP
    end

    subgraph CLINICAL["2. Clinical Extraction & Assertion"]
        EXP --> EXT[Diagnosis Extractor<br/>Typed SOAP + Ambient Transcripts]
        EXP --> FILT[Assertion Filter<br/>Negation, Differentials, Misdiagnoses, Immunization Guard]
        EXT --> QUERIES[Primary Query + Filtered Text]
        FILT --> QUERIES
    end

    subgraph RETRIEVAL["3. Hybrid Retrieval & Candidate Expansion"]
        QUERIES --> BM25[BM25 Sparse Retrieval<br/>18,375 Categories + 93k Synonyms]
        QUERIES --> DENSE[Dense Semantic Retrieval<br/>all-MiniLM-L6-v2 + NPZ Index]
        BM25 --> RRF[Reciprocal Rank Fusion<br/>k=60]
        DENSE --> RRF
        RRF --> EXPAND[Hierarchy Candidate Expansion<br/>Parent, Children, .Y, .Z]
    end

    subgraph RESOLUTION["4. Hierarchy Descent & Reranking"]
        EXPAND --> RESOLVE[Specificity Descent Engine<br/>Distinguishing Qualifiers & Child .Z Precedence]
        RESOLVE --> RERANK[Specificity-Aware Rule Reranker<br/>Alias Matching, Clinical Alignment, Symptom Demotion]
        RERANK --> DIV[Diversity Selector<br/>Top 1: Primary Leaf | Top 2: Specificity Alt | Top 3: Secondary]
    end

    subgraph OPTIONAL["5. Optional LLM Reranking (Gated)"]
        DIV --> GATE{USE_LLM_RERANK=1 &<br/>API Key Present?}
        GATE -->|No / Fallback| OUT[Output JSONL]
        GATE -->|Yes| LLM[LLMReranker<br/>Top-20 Candidate Restricted<br/>Hard $4.00 Budget Guard<br/>15s Timeout]
        LLM --> OUT
    end
```

### Key Architectural Stages

1. **Deterministic Multilingual & Script Normalization (`multilingual.py`):**
   - Instant Unicode block gating routes non-Latin scripts (Devanagari, Arabic, CJK, Cyrillic) immediately away from English fall-through.
   - Combined with Lingua n-gram detection for Latin-script European and African languages (Swahili, Spanish, French, Portuguese).
   - Domain-specific clinical dictionaries translate diagnostic terms into standardized ICD-11 concepts before retrieval.
2. **Clinical Assertion Filtering (`assertion_filter.py`):**
   - Identifies and excludes negated statements (`"mRDT NEG"`, `"ruled out"`, `"no signs of"`), previous misdiagnoses (`"initially thought to be malaria"`), and screening questions.
   - Shields confirmed diagnoses (`"Dx:"`, `"Assessment:"`) and immunization history (`"not vaccinated vs mumps"`) from spurious negation.
3. **Hybrid Sparse + Dense Retrieval (`retrieval.py`):**
   - **BM25Okapi:** Indexed over enriched search documents (official titles, parent ancestor chains, WHO index terms, inclusions, definitions).
   - **Dense MiniLM-L6-v2:** 384-dimensional cosine similarity embeddings pre-computed for all 18,375 MMS categories and stored as a portable 26 MB compressed NumPy array (`dense_embeddings.npz`).
   - Combined using Reciprocal Rank Fusion ($k=60$).
4. **ICD-11 Hierarchy & Specificity Descent (`icd_hierarchy.py`):**
   - Navigates the MMS category tree. When a retrieved candidate is a non-leaf category, the descent engine searches child nodes for distinguishing clinical qualifiers and descendants' synonyms matching the note.
   - Strictly enforces child `.Z` priority (inspects the category's own children for an unspecified leaf before falling back to sibling metadata).
5. **Specificity-Aware Reranker (`reranker.py`):**
   - Parses bracketed WHO index terms (e.g. `[urinary schistosomiasis]`, `[acute exacerbation of chronic obstructive pulmonary disease]`) into discrete clinical aliases.
   - Awards specificity bonuses (+0.04) when a specific leaf's distinguishing qualifiers match confirmed clinical findings.
   - Demotes generic symptom codes (Chapter 21 / `M` codes) when definitive disease categories are identified.
6. **Optional LLM Reranker (`llm_reranker.py`):**
   - Strict candidate gating: prompts OpenAI/Anthropic models only with the top-20 baseline candidates.
   - Validates all output codes against `valid_codes.json`; drops any hallucinated or non-leaf codes.
   - Hard budget cap: tracks cumulative spend with a hard $4.00 shutoff (well below the $6.00 brief budget).
   - Enforces a 15-second per-note timeout with automatic graceful fallback to the baseline candidates.

---

## 3. Data Sources & Provenance

All knowledge bases, model weights, and indexes are vendored directly in the repository for 100% offline self-containment.

| Asset | Source / Origin | Licence | Purpose |
|:---|:---|:---:|:---|
| `data/mms-2026-01.json` | WHO ICD-11 MMS (2026-01 release) via Sugen Data Pack | CC BY-ND 3.0 IGO | Ground truth catalog of 18,375 diagnostic categories |
| `kb/who_enrichments.json` | WHO ICD-API v2 (`https://id.who.int/icd/release/11/2024-01/mms`) | CC BY-NC-ND 3.0 IGO | 93,387 official WHO index terms, definitions, inclusions, exclusions |
| `kb/enriched_kb.json` | Derived knowledge base combining MMS structure & WHO enrichments | CC BY-NC-ND 3.0 IGO | Unified offline knowledge graph with parent-child links and search text |
| `kb/dense_embeddings.npz` | Pre-computed embeddings from `all-MiniLM-L6-v2` | Apache 2.0 | 26 MB compressed float32 matrix for instant offline semantic lookup |
| `models/all-MiniLM-L6-v2` | Hugging Face / Sentence-Transformers | Apache 2.0 | 22M parameter sentence encoder cached locally for zero-network CPU inference |
| `models/fasttext/lid.176.ftz` | Facebook Research fastText | CC BY-SA 3.0 | 917 KB compressed language identification model |
| `rank_bm25` | pypi / rank-bm25 | Apache 2.0 | BM25Okapi sparse lexical scoring |
| `lingua-language-detector` | pypi / lingua | Apache 2.0 | Offline n-gram language detection for African and European languages |

---

## 4. Rejection Risk Defenses & Operational Guarantees

The pipeline is hardened against all automated grader rejection conditions:

1. **Clean Clone Docker Build Guarantee:**
   - All model weights, language detectors, and dense embeddings are committed in the repository.
   - Pinned CPU-only dependencies in `requirements.txt` (`torch==2.5.1+cpu`, `sentence-transformers==6.2.0`, `onnxruntime==1.31.1`, `numpy==2.4.6`).
   - `docker compose build` executes 100% offline with zero network calls to HuggingFace or external repositories.
2. **Prediction Completeness Guarantee (>90% threshold):**
   - Every input ID in the input JSONL is guaranteed to appear in the output JSONL.
   - Handled corrupted input lines using raw string regex extraction to salvage original note IDs rather than inventing synthetic IDs.
   - Every emitted code is strictly validated against `valid_codes.json` and verified to be a leaf category.
3. **No Silent Failures or Constant Fallbacks:**
   - Eliminated all silent `except Exception: return RA01` traps.
   - Pipeline runs a startup self-test note on initialization (`{"id": "__self_test__"}`).
   - If an unexpected error occurs during note processing, a structured error trace is logged to `sys.stderr` and dynamic BM25 fallback candidates for that specific note text are emitted.
4. **Time Budget & CPU Resource Management:**
   - Multi-process worker pool using `multiprocessing.Pool` with `imap_unordered`.
   - Single-thread CPU pinning per worker (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`) to eliminate thread contention across the 8-CPU grading box.
   - Incremental line-by-line output flushing and resume support from partial outputs.
   - Throughput: ~0.4s per note on a single CPU core; completes 600 notes in < 3 minutes across 8 cores (well within the 60-minute limit).

---

## 5. Concrete AI Tooling Errors Caught & Fixed

1. **Immunization History Negation Trap (Caught in Dev Evaluation):**
   - *Symptom:* In note `dev-019`, a pediatric patient presented with `"Not vaccinated vs mumps. Dx: Mumps."` The regex negation filter saw `"not ... mumps"`, negated mumps, dropped `1D80.0` from candidate consideration, and predicted symptom code `MD93 Dysphagia`.
   - *Fix:* Added an explicit exception regex shielding vaccination and immunization history (`"vaccinated"`, `"immuniz"`, `"vaccine"`) from clinical assertion negation, and enforced that confirmed diagnostic statements (`Dx:`, `Imp:`, `Assessment:`) cannot be negated by upstream clinical history.
2. **Forearm Fracture Subtree Sibling Collision (Caught in Specificity Descent Testing):**
   - *Symptom:* In distal radius fractures (`NC32.5`), the hierarchy engine resolved `NC32` to `NC32.Z` (unspecified forearm fracture) because `info.get('z_code')` stored the parent's sibling `.Z` code instead of inspecting `NC32`'s own children for `NC32.5Z`.
   - *Fix:* Refactored `get_z_code()` to inspect direct children first before checking sibling metadata, ensuring that fracture subtrees correctly resolve to child leaves.
3. **Multi-Word WHO Index Term Substring Omission (Caught in Reranker Analysis):**
   - *Symptom:* In `dev-012` (`Urinary schistosomiasis`) and `dev-013` (`AECOPD`), the official WHO index terms contained bracketed forms (`[urinary schistosomiasis]` and `AECOPD – [acute exacerbation of chronic obstructive pulmonary disease]`). Contiguous substring matching failed because the raw text was 70+ characters long.
   - *Fix:* Added bracket/delimiter alias extraction in `reranker.py`, splitting entries into canonical aliases and enabling bidirectional token overlap.

---

## 6. How to Run

### Run Inference via Docker Compose (Official Grader Command)

```bash
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

### Run Inference Locally (Python CLI)

```bash
python entrypoint.py /in/notes.jsonl /out/predictions.jsonl
```

### Run Evaluation Suite & Official Grader

```bash
# Evaluate on dev notes with score.py
python evaluate.py --gold data/dev_notes.jsonl

# Evaluate on 85-note 26-chapter held-out suite
python evaluate.py --gold data/held_out_notes.jsonl --out out/held_out_predictions.jsonl

# Run official scoring harness
python score.py --gold data/dev_notes.jsonl --pred out/dev_predictions.jsonl --mms data/mms-2026-01.json
```

### Run Unit Tests

```bash
python -m unittest discover tests
```

---

## 7. Engineering Decisions Log

Every architectural decision, rationale, rejected alternative, and verification result is tracked with immutable timestamps and commit hashes in [`DECISIONS.md`](file:///d:/Projects/SugenEngTest/DECISIONS.md).
