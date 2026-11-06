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


---

## Entry 4 — 2026-11-06 14:50 IST — Intra-Sentence Assertion Splitting & Confirmed Clause Isolation

**Commit:** v1-assertion-clause-splitting  
**Decision:** Split sentences on commas and semicolons when evaluating diagnostic assertions (`split_sentences_and_clauses`), ensuring that trailing negative clauses (e.g., `malaria excluded` or `mRDT NEG`) do not falsely negate the primary diagnosis in compound diagnostic sentences like `Dx viral URTI, malaria excluded`.

**Options considered:**
1. Sentence-level assertion windowing — negations within 6 tokens negated the entire sentence, dropping `viral URTI` when `malaria excluded` appeared on the same line.
2. Comma and punctuation-aware clause segmentation — isolating `Dx viral URTI` (affirmed) from `malaria excluded` (negated).

**What decided it:** In `dev-002`, `Dx viral URTI, malaria excluded` was incorrectly flagged as negated, which suppressed `CA07.0` and `CA00`. With clause-level assertion isolation, `viral URTI` was affirmed as the primary retrieval query, while `malaria` was safely dropped.

**Revisit:** Maintain clause isolation rules if multi-clause complex compound diagnoses appear in generalisation sets.

---

## Entry 5 — 2026-11-06 15:00 IST — Clinical Abbreviation Expansion, .Y Demotion, and Complication Alignment

**Commit:** v1-reranker-and-abbreviations  
**Decision:** Expand high-frequency East African / outpatient clinical abbreviations (e.g. `thrush`, `vaginal thrush`, `dysentery`, `frozen shoulder`, `plantar fasciitis`, `GDM`, `CCF`) in `text_utils.py`, apply a penalty to `.Y` ("Other specified") codes relative to `.Z` ("Unspecified"), and add uncomplicated/complication clinical alignment bonuses in `RuleReranker`.

**Options considered:**
1. Leave abbreviations unexpanded — dense embeddings and BM25 missed specific matches like `JA63.2` for `GDM` or `1A36.00` for `amoebic dysentery`.
2. Treat `.Y` and `.Z` equally — in real outpatient practice, `.Y` codes are almost never used when `.Z` or specific children apply; dev set has 0 `.Y` gold codes versus 14 `.Z` gold codes.
3. Align uncomplicated presentations with `.0` / `.Z` and penalize complicated/cerebral codes unless complications are explicitly evidenced.

**What decided it:** This combination increased Top-1 from 45.0% to 55.0% and Top-3 from 63.3% to 73.3%, raising the official score to 45.83 / 75.0 (Normalized: **61.11 / 100**), with candidate Recall@30 at 88.3% and Recall@50 at 93.3%.

**Revisit:** Monitor candidate recall on non-English / generalisation inputs.

---

## Entry 6 — 2026-11-06 15:10 IST — Specific Child Descent vs Blind .Z Promotion

**Commit:** v1-specificity-descent-analysis  
**Decision:** Specificity descent from a family to a specific child must be strictly evidence-based (requiring differential qualifier matches in the note text) rather than blind ranking promotion of specific siblings over `.Z`.

**Options considered:**
1. Unconditionally swap .Z with highest-ranked specific child — tested via ablation, which dropped dev score from 45.83 to 44.17 because true unspecified cases (e.g. `1F40.Z` Falciparum malaria, unspecified) were incorrectly promoted to false specific subtypes.
2. Keep normalized token overlap in `RuleReranker` to prevent length bias — unnormalized token overlap was tested and dropped score to 32.08 because long category titles accumulated spurious word matches.
3. Retain principled balance between retrieval relevance and targeted clinical alignment.

**What decided it:** Empirical ablation confirmed that raw length-unnormalized matching degrades general retrieval stability. Controlled clinical alignment preserves the 61.11 / 100 benchmark score.


---

## Entry 7 — 2026-11-06 15:20 IST — Deterministic Multilingual Clinical Normalization (Swahili, Spanish, French)

**Commit:** v2-multilingual-clinical-normalization  
**Decision:** Implement a 100% offline, zero-network multilingual normalization module (`multilingual.py`) using marker-word language identification and high-frequency clinical term translation for Swahili, Spanish, and French consultations.

**Options considered:**
1. Heavy translation models (e.g. NLLB-200 or Opus-MT) via CTranslate2 — introduces substantial model weight bloat (~600MB+ per language), slow CPU inference, and potential translation hallucinations on clinical acronyms.
2. Rely solely on multilingual embeddings — dense embeddings without term alignment struggle on specific medical terminology and abbreviations across disparate languages.
3. Lightweight deterministic clinical translation mapping — maps key diagnostic, anatomical, and symptomatologic phrases into standard English ICD-11 terminology before BM25 and dense retrieval, keeping the pipeline ultra-fast and completely free of generative hallucinations.

**What decided it:** Validated against a hand-crafted multilingual test suite (`tests/test_multilingual.py`) covering Swahili, Spanish, and French notes across infectious, respiratory, musculoskeletal, and gynecological chapters. Achieved **83.3% Top-1** (5/6) and **100.0% Top-3** (6/6) accuracy with zero degradation to dev scores.

**Revisit:** Extend clinical mapping dictionaries if additional regional dialect markers emerge.


---

## Entry 8 — 2026-11-06 17:05 IST — Localized Evaluation Data, Official Grader, and Evaluation Runner

**Task:** 1.1  
**Decision:** Migrate `dev_notes.jsonl`, `predictions.example.jsonl`, and `score.py` directly into the repository root and `data/` directory, update all path references across `evaluate.py`, `build_kb.py`, and analysis scripts, and add `scripts/run_eval.sh`.

**Options considered:**
1. Keep evaluation assets in `public/` — violates self-contained repo requirement when `public/` is excluded from git.
2. Store `dev_notes.jsonl` in `data/` and `score.py` at repository root — makes the repo completely autonomous and runnable from a fresh clone.

**What decided it:** Satisfies P0 requirement 1.1. Verified that `python score.py --gold data/dev_notes.jsonl --pred out/dev_predictions.jsonl --mms data/mms-2026-01.json` reproduces the exact 45.83 / 75.0 (61.11 / 100) dev benchmark score.


---

## Entry 9 — 2026-11-06 17:10 IST — Corrected Bootstrap Confidence Interval Bug & Synthetic Validation

**Task:** 1.5  
**Decision:** Correct the bootstrap confidence interval calculation in `evaluate.py`. The upper bound for `top1_ci` was inadvertently referencing `b_top3s` instead of `b_top1s`, artificially inflating the 55% Top-1 upper bound to 85.0%. Added unit test `tests/test_bootstrap_ci.py` validating percentile estimation on synthetic binomial cases.

**Options considered:**
1. Leave reporting uncorrected — misleadingly claims Top-1 CI reaches 85.0% on 60 notes.
2. Fix index array reference and validate against exact normal approximation ($0.55 \pm 1.96 \sqrt{0.55 \times 0.45 / 60}$) — confirms symmetrical distribution `[43.3%, 68.3%]`.

**What decided it:** Fixed bug and verified via unit test. Re-evaluated dev notes: Top-1 95% CI is now `[43.3%, 68.3%]`, Top-3 95% CI is `[61.7%, 85.0%]`, and Score 95% CI is `[37.92, 54.18]`.
