# System Architecture

This document details the architectural design and pipeline stages of the offline ICD-11 Clinical Coding System.

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

    DIV --> OUT[Output JSONL]
```

## Pipeline Stages

### 1. Ingestion & Multilingual Normalization (`multilingual.py`)
- **Unicode Script Gating:** Instantly detects non-Latin scripts (Devanagari, Arabic, CJK, Cyrillic) via character ranges before n-gram analysis, preventing non-Latin text from silently falling through to English.
- **Language Identification:** Uses Lingua n-gram detection for Latin-script languages (Swahili, Spanish, French, Portuguese).
- **Deterministic Clinical Normalization:** High-frequency clinical term mapping into standard ICD-11 English concepts without generative translation hallucination risks.

### 2. Clinical Extraction & Assertion Filtering (`diagnosis_extractor.py`, `assertion_filter.py`)
- **Diagnosis Extractor:** Extracts diagnostic sections across SOAP notes (`Assessment`, `Dx`, `Impression`) and dialogue transcripts.
- **Assertion Filter:** Detects and removes negated findings (`mRDT NEG`, `ruled out`), past misdiagnoses (`initially thought to be malaria`), and family history, while strictly protecting confirmed diagnoses and vaccination history (`not vaccinated vs mumps`).

### 3. Hybrid Sparse + Dense Retrieval (`retrieval.py`)
- **Sparse BM25:** BM25Okapi over search documents combining category titles, WHO index terms (synonyms), definitions, and ancestor titles.
- **Dense MiniLM-L6-v2:** 384-dimensional cosine similarity embeddings pre-computed for all 18,375 MMS categories and stored as a compressed 26 MB NumPy array (`dense_embeddings.npz`).
- **Reciprocal Rank Fusion (RRF):** Fuses sparse and dense rankings ($k=60$) to build the candidate pool.

### 4. Hierarchy Specificity Descent & Leaf Resolution (`icd_hierarchy.py`)
- Navigates the MMS category tree from retrieved parent categories down to valid leaf codes.
- Distinguishes leaf categories using qualifier token overlap between the clinical note and child titles.
- Enforces child `.Z` priority (checks the category's own children for an unspecified category before falling back to sibling metadata).

### 5. Rule-Based Clinical Reranking & Diversity Selection (`reranker.py`)
- Evaluates canonical aliases extracted from bracketed WHO index terms.
- Awards specificity bonuses for explicit clinical qualifiers.
- Demotes generic symptom/sign codes (Chapter 21) when definitive disease categories are present.
- Emits up to 3 leaf codes: Rank 1 is the primary leaf, Rank 2 is an alternate specificity variant or top secondary diagnosis, Rank 3 is an additional supported leaf.

