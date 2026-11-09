# System Architecture — ICD-11 Clinical Coding Engine

This document details the architectural design, pipeline stages, and component interactions of the 100% offline ICD-11 MMS Clinical Coding System at current HEAD.

---

## 1. High-Level Architecture

The system executes as a deterministic, multi-stage offline pipeline operating without generative LLMs or external network access at runtime. It ingests outpatient consultation notes across multiple languages and consultation styles and predicts up to three verified ICD-11 MMS leaf categories per note.

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion & Multilingual Normalization"]
        IN[Raw Clinical Note: id, text, style] --> LID{Language Identification<br/>Unicode Script Gating + Lexicon Markers}
        LID -->|9 Languages: EN, ES, FR, DE, SW, ZH, AR, HI, PT| NORM[Clinical Term Normalizer<br/>Deterministic Translation & Aliases]
        NORM --> EXP[Clinical Abbreviation Expansion<br/>e.g. GDM -> Gestational Diabetes, CCF, thrush]
    end

    subgraph CLINICAL["2. Clinical Extraction & Assertion Scoping"]
        EXP --> EXT[Diagnosis Extractor<br/>Typed SOAP, Short Consults, Ambient Dialogues]
        EXP --> FILT[Clause-Level Assertion Filter<br/>Punctuation Split: Commas & Semicolons]
        FILT --> SCOPE[Assertion Scoping<br/>• Drops: Negations, Rule-Outs, History, Family<br/>• Triggers: non-tender, wnl, unremarkable<br/>• Protects: Current Diagnoses & Vaccination State]
        EXT --> QUERIES[Dual Retrieval Queries<br/>• Primary Affirmed Query<br/>• Full Filtered Note Text<br/>• Native Non-English Query]
        SCOPE --> QUERIES
    end

    subgraph RETRIEVAL["3. Hybrid Dual-Query Retrieval & RRF"]
        QUERIES --> BM25[BM25 Lexical Retrieval<br/>18,375 MMS Categories + 93,387 WHO Terms]
        QUERIES --> DENSE[Dense Semantic Retrieval<br/>paraphrase-multilingual-MiniLM-L12-v2<br/>384-dim Cosine Similarity via dense_embeddings.npz]
        BM25 --> RRF[Reciprocal Rank Fusion<br/>k = 60 | Top 40–50 Candidate Pool]
        DENSE --> RRF
    end

    subgraph RERANK["4. Knowledge-Guided Reranking"]
        RRF --> EXCL[WHO Exclusion Conflict Graph<br/>3,042 Conflict Pairs from kb/exclusion_graph.json<br/>Mutual Exclusivity Demotion: -0.25 to -0.35]
        RRF --> CONCEPTS[Clinical Concepts Engine<br/>Declarative Rules from rules/clinical_concepts.yaml<br/>BP Staging, Gestational Diabetes, Acquired vs Congenital]
        RRF --> DEMOG[Demographic & Clinical Guards<br/>• Age Guards: Pediatric vs Adult vs Geriatric<br/>• Obstetric Guards: Non-Pregnant / Post-Menopausal<br/>• Complication Grading: .0/.Z vs Complicated Leaves]
        EXCL --> SCORED[Reranked & Filtered Candidates]
        CONCEPTS --> SCORED
        DEMOG --> SCORED
    end

    subgraph RESOLUTION["5. Specificity Descent & Top-3 Assembly"]
        SCORED --> DESCEND[Specificity Descent Engine<br/>BFS Traversal of MMS Hierarchy DAG<br/>Full Note Qualifier Differential: Tokens(Child) minus Tokens(Parent)]
        DESCEND --> LEAF_CHECK{Valid Terminal Leaf?<br/>15,912 Valid Leaves in MMS}
        LEAF_CHECK -->|Qualifiers Present| CHILD[Resolved Specific Child Leaf]
        LEAF_CHECK -->|Qualifiers Absent| RESIDUAL[Resolved Unspecified .Z Leaf]
        CHILD --> DIVERSE[Top-3 Hedging & Diversity Assembly<br/>Rank 1: Primary Leaf | Rank 2: Specificity Alt | Rank 3: Stem-Sharing]
        RESIDUAL --> DIVERSE
    end

    DIVERSE --> OUT[Output JSONL: id, codes[3]]
```

---

## 2. Pipeline Stages in Detail

### Stage 1: Ingestion & Multilingual Normalization (`multilingual.py`, `text_utils.py`)
- **Unicode Script Gating:** Immediately classifies non-Latin scripts (Devanagari, Arabic, CJK, Cyrillic) via character code points before vocabulary analysis.
- **Language Identification:** Combines character n-gram profiling with high-precision clinical stop-word dictionaries across **9 target languages**: English (`en`), Spanish (`es`), French (`fr`), German (`de`), Swahili (`sw`), Chinese (`zh`), Arabic (`ar`), Hindi (`hi`), and Portuguese (`pt`).
- **Deterministic Translation & Aliasing:** Maps regional outpatient clinical terms (e.g. *homa ya vipindi* $\rightarrow$ typhoid fever, *tension arterial* $\rightarrow$ hypertension) into canonical English medical terms without generative translation hallucination risks.
- **Clinical Abbreviation Expansion:** Expands 40+ common outpatient shorthand terms (`GDM`, `CCF`, `AECOPD`, `thrush`, `dysentery`, `frozen shoulder`).

### Stage 2: Clinical Extraction & Assertion Scoping (`diagnosis_extractor.py`, `assertion_filter.py`)
- **Diagnostic Section Extraction:** Recognizes diagnostic section headers across typed clinical notes (`Assessment:`, `Impression:`, `Diagnose:`, `Beurteilung:`, `Diagnostic:`, `Plan:`) and doctor concluding turns in ambient dialogue transcripts.
- **Clause-Level Segmentation:** Splits sentences at commas, semicolons, and coordinating conjunctions (`split_sentences_and_clauses`). This ensures trailing negative clauses (e.g. *"Assessment: Viral URTI, malaria excluded"*) do not taint affirmed primary conditions.
- **Assertion Filtering:**
  - Identifies and discards ruled-out findings, differential suspects, past conditions, and family medical history.
  - Recognizes clinical post-negation triggers (`non-tender`, `wnl`, `within normal limits`, `unremarkable`, `ni ya kawaida`).
  - **Immunization Protection:** Shields vaccination history (`"not vaccinated against mumps"`) from erroneously negating active infection codes.

### Stage 3: Hybrid Dual-Query Retrieval & Fusion (`retrieval.py`)
- **Dual-Query Generation:**
  - Passes native non-English text to the multilingual dense retriever to exploit cross-lingual semantic alignment.
  - Passes translated, normalized English queries to the BM25 lexical retriever.
- **Dense Vector Search:** Employs `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (117M parameters, 384 dimensions). Cosine similarity is computed against precomputed, L2-normalized float32 vectors for all 18,375 MMS categories stored in `kb/dense_embeddings.npz` (25 MB).
- **Sparse BM25 Index:** BM25Okapi inverted index built over search documents combining category titles, 93,387 official WHO index terms, inclusions, definitions, and ancestor titles.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse rankings with smoothing parameter $k=60$, creating a high-recall candidate pool of 40–50 stems (Overall Recall@50: **93.3%** on dev, **89.2%** on blind).

### Stage 4: Knowledge-Guided Reranking (`reranker.py`)
- **WHO Official Exclusion Conflict Graph:** Evaluates candidate pairs against `kb/exclusion_graph.json` (3,042 explicit exclusion conflict pairs across 1,836 categories derived from WHO ICD-11 foundation metadata). Candidates refuted by the primary diagnosis receive a calibrated penalty (-0.25 to -0.35).
- **Clinical Concepts Engine:** Applies declarative clinical knowledge from `rules/clinical_concepts.yaml`:
  - Blood pressure staging (Stage 1 vs Stage 2 vs Crisis).
  - Gestational diabetes vs pre-existing diabetes in pregnancy.
  - Acquired vs congenital distinctions.
- **Demographic & Obstetric Guards:**
  - Demotes maternal Chapter 18 codes for non-pregnant or post-menopausal presentations.
  - Prioritizes neonatal conditions (Chapter 19) for infant encounters.
- **Complication Alignment:** Awards bonuses to `.0` or `.Z` for uncomplicated presentations and rewards specific complication leaves only when clinical evidence is affirmed in the note.

### Stage 5: Specificity Descent & Leaf Resolution (`icd_hierarchy.py`, `pipeline.py`)
- **Mandatory Leaf Resolution:** Every candidate must resolve to one of the 15,912 valid terminal leaves in MMS 2026-01 (`is_leaf = true`). Non-leaf parent categories score 0 on the exact match grader.
- **Qualifier Differential Matching:** When descending from parent to children, the engine computes:
  $$\Delta Q_{\text{child}} = \text{Tokens}(\text{Child Title}) \setminus \text{Tokens}(\text{Parent Title})$$
  The full filtered note text is evaluated against $\Delta Q_{\text{child}}$. Child codes are chosen only if their unique qualifying tokens are substantiated.
- **Unspecified (`.Z`) Fallback:** If clinical documentation lacks child qualifiers, the engine automatically resolves to the terminal unspecified category (`.Z`).

### Stage 6: Top-3 Assembly & Hedging (`pipeline.py`)
- **Rank 1:** The highest-scoring primary diagnostic leaf.
- **Rank 2:** A specificity alternative (e.g. specific child when Rank 1 is `.Z`) or the top secondary confirmed comorbidity.
- **Rank 3:** A stem-sharing category or high-probability differential leaf.
- **Output Validation:** Formats output into JSONL (`{"id": ..., "codes": [...]}`) guaranteeing exactly matching IDs, valid leaf codes, and zero missing records.

---

## 3. Runtime Characteristics & Resource Profile

| Metric | Measured Specification | Guardrail / Limit | Status |
|:---|:---:|:---:|:---:|
| **Runtime Network Calls** | **0 (Completely Offline)** | Model APIs allowed; zero used | PASS |
| **Throughput (1 CPU Core)** | 1.44–2.20 notes/sec | > 0.4 notes/sec | PASS |
| **Throughput (8 CPU Cores)** | ~10–12 notes/sec | > 1.0 notes/sec | PASS |
| **600-Note Projected Runtime** | **~1.0 minute (8 CPUs)** | < 60 minutes | PASS |
| **Resident Memory (RSS)** | **~1.2 GB RAM** | < 16 GB RAM | PASS |
| **Leaf Code Validity** | **100% (15,912/15,912 MMS)** | 100% valid leaves | PASS |
| **Missing Prediction Rate** | **0.0% (0 / 600)** | < 10% auto-reject threshold | PASS |
