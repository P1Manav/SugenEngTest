# DECISIONS.md — ICD-11 Coding System

## Entry 1 — 2026-11-06 14:20 IST — Project scaffold and v0 architecture

**Commit:** initial scaffold  
**Decision:** Start with BM25 + dense retrieval (all-MiniLM-L6-v2) + RRF fusion over MMS titles only, with rule-based diagnosis extraction and .Z fallback logic.

**Options considered:**
1. Jump straight to BGE-M3 + cross-encoder — too slow to iterate; need a baseline first.
2. Use TF-IDF instead of BM25 — BM25 has proven IDF-aware term weighting and is standard for medical IR.
3. Use FAISS for dense retrieval — adds complexity; numpy cosine similarity is sufficient for 18K codes.

**What decided it:** The dev set has 60 notes covering ~40 distinct ICD-11 codes. Exact match scoring means we need high candidate recall first. BM25 + dense gives two orthogonal signals. All-MiniLM-L6-v2 is 80MB, fast on CPU, and good enough for English medical terms in v0. We'll upgrade to BGE-M3 in v2 for multilingual support.

**Revisit:** Switch to BGE-M3 when we add multilingual support (v2). Consider FAISS if numpy similarity becomes a bottleneck at 18K codes (unlikely).

---

## Entry 2 — 2026-11-06 14:35 IST — Strict Leaf Resolution and Parent-to-Child Qualifier Matching

**Commit:** v0-hierarchy-rules  
**Decision:** All predictions must strictly resolve to valid MMS leaf codes (15,912 out of 18,375 categories). Non-leaf codes are never emitted. When descending from a non-leaf candidate to child codes, specificity matching is computed only against *child qualifier tokens* that do not exist in the parent title.

**Options considered:**
1. Direct child token overlap with note text — caused "Malaria due to Plasmodium falciparum" to incorrectly descend into `1F40.0` (Plasmodium falciparum malaria with cerebral complications) because "malaria" and "falciparum" matched the note, despite no cerebral symptoms being present.
2. Always defaulting non-leaf to `.Z` — fails when notes contain specific clinical subtypes (e.g. `1A36.00 Acute amoebiasis`).
3. Qualifier differential matching (`child_tokens - parent_tokens`) — only rewards child-specific qualifiers like "cerebral", "bacterial", or "viral".

**What decided it:** dev-001 gold is `1F40.Z` (uncomplicated falciparum malaria). Using qualifier differential matching prevented false cerebral descent and correctly defaulted to `1F40.Z`, improving Top-1 accuracy.

**Revisit:** Incorporate clinical section weighting (e.g. lab findings vs symptoms) to further refine specificity descent.

---

## Entry 3 — 2026-11-06 14:40 IST — Assertion Filter Vaccination Protection & Chapter 21 Demotion

**Commit:** v0-assertion-and-reranker  
**Decision:** Prevent assertion filter from negating confirmed diagnoses when notes contain negative vaccination history (e.g., "Not vaccinated vs mumps"), and demote Chapter 21 symptom/sign codes when definitive disease codes are matched.

**Options considered:**
1. Pure string-based pre-negation matching — failed on dev-019 ("Not vaccinated vs mumps. Dx mumps"), which erroneously negated mumps and predicted `MD93 Dysphagia`.
2. Explicitly protecting confirmed diagnostic phrases (`A:`, `Imp:`, `Dx:`) from upstream negation in history and stripping vaccine/immunisation statements from disease negation checks.
3. Treating symptoms as equal candidates to definitive etiologies — led to predicting "cough" and "dysphagia" instead of pneumonia and mumps.

**What decided it:** Fixing the vaccination negation bug restored `1D80.0` for dev-019. Demoting Chapter 21 symptoms increased Top-1 accuracy on the dev set from 35.0% to 45.0% and Top-3 accuracy to 63.3%, lifting the official score from 30.4 to 38.3 / 75.0 (51.1 / 100).

**Revisit:** Train a lightweight binary classifier for assertion detection if more ambiguous complex clinical narratives emerge in hidden sets.
