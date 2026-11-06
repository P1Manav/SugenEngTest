# ICD-11 Clinical Coding System — Sugen Technologies Take-Home

An offline ICD-11 coding system that reads outpatient consult notes and proposes up to 3 ICD-11 MMS codes. **No LLM is used at build time or run time.** All intelligence comes from retrieval (BM25 + dense), small encoder models, rule-based diagnosis extraction, and classical ML.

## Dev Score

| Metric | Value |
|--------|-------|
| Top-1 Accuracy (Slice A) | 45.00% (95% CI: [31.7%, 75.0%]) |
| Top-3 Accuracy (Slice A) | 63.33% (95% CI: [51.7%, 75.0%]) |
| Score (dev, /75) | 38.33 / 75.0 (Normalized: 51.11 / 100) |
| Candidate Recall @ 30 | 85.0% |
| Candidate Recall @ 50 | 90.0% |

**Estimated hidden score:** **~52 / 100** — We estimate ~42 points on the Core slice (450 notes) and ~10 points on the Generalisation slice (150 notes). The generalisation slice covers unseen ICD-11 chapters and non-English consultations where exact lexical match degrades, but semantic dense embeddings and hierarchy fallback maintain robust candidate generation.

## No LLM

This system does not call any LLM API (OpenAI, Anthropic, Gemini, or any other) at build time or run time. No API keys are needed, and runtime works 100% offline. The pipeline uses:
- **BM25** sparse lexical retrieval (`rank-bm25`) over an enriched ICD-11 category database with ancestor contexts.
- **Sentence-transformers** (`sentence-transformers/all-MiniLM-L6-v2`) with pre-computed unit-normalized embeddings for 18,375 categories.
- **Reciprocal Rank Fusion (RRF)** (`k=60`) combining sparse and dense candidate lists across primary assessment and full-note queries.
- **ICD-11 Hierarchy Engine**: derives parent-child links, sibling sets, and `.Z` (unspecified) / `.Y` (other specified) fallbacks; enforces strict leaf-only output.
- **Rule-based clinical extraction**: extracts diagnoses from typed notes (`A:`, `Imp:`, `Dx:` sections) and ambient doctor transcripts (`"this is"`, `"what you have is"`).
- **Assertion filter**: handles negation, ruled-out conditions, and previous misdiagnoses without corrupting vaccination history.

## Data Sources

| Source | Licence | What |
|--------|---------|------|
| `mms-2026-01.json` (provided) | WHO / Sugen Data Pack | 18,375 ICD-11 MMS diagnostic categories with code, title, chapter, leaf/residual flags, depth, and URIs |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | 80MB multilingual sentence embedding model pre-cached locally for offline CPU inference |
| `rank_bm25` | Apache 2.0 | Pure Python BM25Okapi implementation for lexical matching |

## Evaluation Beyond Dev

- **Bootstrap Confidence Intervals:** 1,000 resamples with replacement to establish 95% confidence intervals on score, Top-1, and Top-3 accuracy.
- **Candidate Recall Tracking:** Recall@30 (85.0%) and Recall@50 (90.0%) measured to ensure candidate generation is not the bottleneck.
- **Failure Taxonomy:** Every miss is automatically tagged (`wrong_specificity`: 57.6%, `wrong_diagnosis`: 36.4%, `language_or_slang_issue`: 6.1%).
- **Diversity Selection:** Ranks primary candidate first, reserves slot 2 for `.Z`/sibling specificity alternative, and slot 3 for distinct secondary comorbidities.

## AI Tooling Error Caught

In early iterations, the AI assertion filter flagged confirmed diagnoses as negated when notes contained negative vaccination history. In note `dev-019`, the text stated `"Not vaccinated vs mumps. Dx mumps."` The tooling regex `\bnot\b` matched `mumps` across the clause, negating the true diagnosis, completely dropping `1D80.0 Mumps without complication` from the candidate pool, and erroneously predicting `MD93 Dysphagia` from background chewing symptoms. We caught this during failure analysis, added immunization exceptions to negation checking, and enforced that confirmed diagnostic statements (`Dx:`, `Imp:`) are shielded from upstream history negation.

## Hours Spent

3.5 hours
