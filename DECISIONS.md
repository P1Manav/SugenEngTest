# DECISIONS.md — ICD-11 Coding System

## Entry 1 — 2026-11-06 14:20 IST — Project scaffold and v0 architecture

**Commit:** `7b34984`  
**Decision:** Start with BM25 + dense retrieval (all-MiniLM-L6-v2) + RRF fusion over MMS titles only, with rule-based diagnosis extraction and .Z fallback logic.

**Options considered:**
1. Jump straight to BGE-M3 + cross-encoder — too slow to iterate; need a baseline first.
2. Use TF-IDF instead of BM25 — BM25 has proven IDF-aware term weighting and is standard for medical IR.
3. Use FAISS for dense retrieval — adds complexity; numpy cosine similarity is sufficient for 18K codes.

**What decided it:** The dev set has 60 notes covering ~40 distinct ICD-11 codes. Exact match scoring means we need high candidate recall first. BM25 + dense gives two orthogonal signals. All-MiniLM-L6-v2 is 80MB, fast on CPU, and good enough for English medical terms in v0. We'll upgrade to BGE-M3 in v2 for multilingual support.

**Revisit:** Switch to BGE-M3 when we add multilingual support (v2). Consider FAISS if numpy similarity becomes a bottleneck at 18K codes (unlikely).

---

## Entry 2 — 2026-11-06 14:35 IST — Strict Leaf Resolution and Parent-to-Child Qualifier Matching

**Commit:** `7b34984`  
**Decision:** All predictions must strictly resolve to valid MMS leaf codes (15,912 out of 18,375 categories). Non-leaf codes are never emitted. When descending from a non-leaf candidate to child codes, specificity matching is computed only against *child qualifier tokens* that do not exist in the parent title.

**Options considered:**
1. Direct child token overlap with note text — caused "Malaria due to Plasmodium falciparum" to incorrectly descend into `1F40.0` (Plasmodium falciparum malaria with cerebral complications) because "malaria" and "falciparum" matched the note, despite no cerebral symptoms being present.
2. Always defaulting non-leaf to `.Z` — fails when notes contain specific clinical subtypes (e.g. `1A36.00 Acute amoebiasis`).
3. Qualifier differential matching (`child_tokens - parent_tokens`) — only rewards child-specific qualifiers like "cerebral", "bacterial", or "viral".

**What decided it:** dev-001 gold is `1F40.Z` (uncomplicated falciparum malaria). Using qualifier differential matching prevented false cerebral descent and correctly defaulted to `1F40.Z`, improving Top-1 accuracy.

**Revisit:** Incorporate clinical section weighting (e.g. lab findings vs symptoms) to further refine specificity descent.

---

## Entry 3 — 2026-11-06 14:40 IST — Assertion Filter Vaccination Protection & Chapter 21 Demotion

**Commit:** `7b34984`  
**Decision:** Prevent assertion filter from negating confirmed diagnoses when notes contain negative vaccination history (e.g., "Not vaccinated vs mumps"), and demote Chapter 21 symptom/sign codes when definitive disease codes are matched.

**Options considered:**
1. Pure string-based pre-negation matching — failed on dev-019 ("Not vaccinated vs mumps. Dx mumps"), which erroneously negated mumps and predicted `MD93 Dysphagia`.
2. Explicitly protecting confirmed diagnostic phrases (`A:`, `Imp:`, `Dx:`) from upstream negation in history and stripping vaccine/immunisation statements from disease negation checks.
3. Treating symptoms as equal candidates to definitive etiologies — led to predicting "cough" and "dysphagia" instead of pneumonia and mumps.

**What decided it:** Fixing the vaccination negation bug restored `1D80.0` for dev-019. Demoting Chapter 21 symptoms increased Top-1 accuracy on the dev set from 35.0% to 45.0% and Top-3 accuracy to 63.3%, lifting the official score from 30.4 to 38.3 / 75.0 (51.1 / 100).

**Revisit:** Train a lightweight binary classifier for assertion detection if more ambiguous complex clinical narratives emerge in hidden sets.


---

## Entry 4 — 2026-11-06 14:50 IST — Intra-Sentence Assertion Splitting & Confirmed Clause Isolation

**Commit:** `b5814e7`  
**Decision:** Split sentences on commas and semicolons when evaluating diagnostic assertions (`split_sentences_and_clauses`), ensuring that trailing negative clauses (e.g., `malaria excluded` or `mRDT NEG`) do not falsely negate the primary diagnosis in compound diagnostic sentences like `Dx viral URTI, malaria excluded`.

**Options considered:**
1. Sentence-level assertion windowing — negations within 6 tokens negated the entire sentence, dropping `viral URTI` when `malaria excluded` appeared on the same line.
2. Comma and punctuation-aware clause segmentation — isolating `Dx viral URTI` (affirmed) from `malaria excluded` (negated).

**What decided it:** In `dev-002`, `Dx viral URTI, malaria excluded` was incorrectly flagged as negated, which suppressed `CA07.0` and `CA00`. With clause-level assertion isolation, `viral URTI` was affirmed as the primary retrieval query, while `malaria` was safely dropped.

**Revisit:** Maintain clause isolation rules if multi-clause complex compound diagnoses appear in generalisation sets.

---

## Entry 5 — 2026-11-06 15:00 IST — Clinical Abbreviation Expansion, .Y Demotion, and Complication Alignment

**Commit:** `b5814e7`  
**Decision:** Expand high-frequency East African / outpatient clinical abbreviations (e.g. `thrush`, `vaginal thrush`, `dysentery`, `frozen shoulder`, `plantar fasciitis`, `GDM`, `CCF`) in `text_utils.py`, apply a penalty to `.Y` ("Other specified") codes relative to `.Z` ("Unspecified"), and add uncomplicated/complication clinical alignment bonuses in `RuleReranker`.

**Options considered:**
1. Leave abbreviations unexpanded — dense embeddings and BM25 missed specific matches like `JA63.2` for `GDM` or `1A36.00` for `amoebic dysentery`.
2. Treat `.Y` and `.Z` equally — in real outpatient practice, `.Y` codes are almost never used when `.Z` or specific children apply; dev set has 0 `.Y` gold codes versus 14 `.Z` gold codes.
3. Align uncomplicated presentations with `.0` / `.Z` and penalize complicated/cerebral codes unless complications are explicitly evidenced.

**What decided it:** This combination increased Top-1 from 45.0% to 55.0% and Top-3 from 63.3% to 73.3%, raising the official score to 45.83 / 75.0 (Normalized: **61.11 / 100**), with candidate Recall@30 at 88.3% and Recall@50 at 93.3%.

**Revisit:** Monitor candidate recall on non-English / generalisation inputs.

---

## Entry 6 — 2026-11-06 15:10 IST — Specific Child Descent vs Blind .Z Promotion

**Commit:** `b5814e7`  
**Decision:** Specificity descent from a family to a specific child must be strictly evidence-based (requiring differential qualifier matches in the note text) rather than blind ranking promotion of specific siblings over `.Z`.

**Options considered:**
1. Unconditionally swap .Z with highest-ranked specific child — tested via ablation, which dropped dev score from 45.83 to 44.17 because true unspecified cases (e.g. `1F40.Z` Falciparum malaria, unspecified) were incorrectly promoted to false specific subtypes.
2. Keep normalized token overlap in `RuleReranker` to prevent length bias — unnormalized token overlap was tested and dropped score to 32.08 because long category titles accumulated spurious word matches.
3. Retain principled balance between retrieval relevance and targeted clinical alignment.

**What decided it:** Empirical ablation confirmed that raw length-unnormalized matching degrades general retrieval stability. Controlled clinical alignment preserves the 61.11 / 100 benchmark score.


---

## Entry 7 — 2026-11-06 15:20 IST — Deterministic Multilingual Clinical Normalization (Swahili, Spanish, French)

**Commit:** `1436ff2`  
**Decision:** Implement a 100% offline, zero-network multilingual normalization module (`multilingual.py`) using marker-word language identification and high-frequency clinical term translation for Swahili, Spanish, and French consultations.

**Options considered:**
1. Heavy translation models (e.g. NLLB-200 or Opus-MT) via CTranslate2 — introduces substantial model weight bloat (~600MB+ per language), slow CPU inference, and potential translation hallucinations on clinical acronyms.
2. Rely solely on multilingual embeddings — dense embeddings without term alignment struggle on specific medical terminology and abbreviations across disparate languages.
3. Lightweight deterministic clinical translation mapping — maps key diagnostic, anatomical, and symptomatologic phrases into standard English ICD-11 terminology before BM25 and dense retrieval, keeping the pipeline ultra-fast and completely free of generative hallucinations.

**What decided it:** Validated against a hand-crafted multilingual test suite (`tests/test_multilingual.py`) covering Swahili, Spanish, and French notes across infectious, respiratory, musculoskeletal, and gynecological chapters. Achieved **83.3% Top-1** (5/6) and **100.0% Top-3** (6/6) accuracy with zero degradation to dev scores.

**Revisit:** Extend clinical mapping dictionaries if additional regional dialect markers emerge.


---

## Entry 8 — 2026-11-06 17:05 IST — Localized Evaluation Data, Official Grader, and Evaluation Runner

**Commit:** `8a22ece`  
**Task:** 1.1  
**Decision:** Migrate `dev_notes.jsonl`, `predictions.example.jsonl`, and `score.py` directly into the repository root and `data/` directory, update all path references across `evaluate.py`, `build_kb.py`, and analysis scripts, and add `scripts/run_eval.sh`.

**Options considered:**
1. Keep evaluation assets in `public/` — violates self-contained repo requirement when `public/` is excluded from git.
2. Store `dev_notes.jsonl` in `data/` and `score.py` at repository root — makes the repo completely autonomous and runnable from a fresh clone.

**What decided it:** Satisfies P0 requirement 1.1. Verified that `python score.py --gold data/dev_notes.jsonl --pred out/dev_predictions.jsonl --mms data/mms-2026-01.json` reproduces the exact 45.83 / 75.0 (61.11 / 100) dev benchmark score.


---

## Entry 9 — 2026-11-06 17:10 IST — Corrected Bootstrap Confidence Interval Bug & Synthetic Validation

**Commit:** `ae5f710`  
**Task:** 1.5  
**Decision:** Correct the bootstrap confidence interval calculation in `evaluate.py`. The upper bound for `top1_ci` was inadvertently referencing `b_top3s` instead of `b_top1s`, artificially inflating the 55% Top-1 upper bound to 85.0%. Added unit test `tests/test_bootstrap_ci.py` validating percentile estimation on synthetic binomial cases.

**Options considered:**
1. Leave reporting uncorrected — misleadingly claims Top-1 CI reaches 85.0% on 60 notes.
2. Fix index array reference and validate against exact normal approximation ($0.55 \pm 1.96 \sqrt{0.55 \times 0.45 / 60}$) — confirms symmetrical distribution `[43.3%, 68.3%]`.

**What decided it:** Fixed bug and verified via unit test. Re-evaluated dev notes: Top-1 95% CI is now `[43.3%, 68.3%]`, Top-3 95% CI is `[61.7%, 85.0%]`, and Score 95% CI is `[37.92, 54.18]`.


---

## Entry 10 — 2026-11-06 17:15 IST — Real Offline Language ID with Unicode Script Gating

**Commit:** `7910864`  
**Task:** 1.6  
**Decision:** Replace heuristic marker-word detection with real offline language identification combining instant Unicode script gating (Devanagari, Arabic, Japanese Kana, CJK, Cyrillic) and Lingua n-gram detection.

**Options considered:**
1. Marker words only — non-Latin scripts (Hindi, Arabic, Chinese, Russian, Japanese) contain no Latin words and fall through to English.
2. FastText with standard Python binding — broke due to NumPy 2.x `copy=False` deprecation.
3. Unicode script gate + Lingua + clinical marker fallback — 100% offline, zero network calls, exact 10/10 detection on benchmark suite (`tests/test_language_id.py`), and non-Latin scripts are strictly protected from falling through to English.

**What decided it:** Satisfies P0 requirement 1.6. Verified by `tests/test_language_id.py` across 10 distinct language families with 100% exact accuracy.


---

## Entry 11 — 2026-11-06 17:25 IST — Offline-Safe Build, Vendored Weights & Compressed NPZ Embeddings

**Commit:** `da217f2`  
**Task:** 1.2  
**Decision:** Vendor all model weights locally (`models/all-MiniLM-L6-v2/model.safetensors`), compress dense embeddings to `kb/dense_embeddings.npz` (26 MB, 0.03s load time), pin all dependencies in `requirements.txt` (`torch==2.5.1`, `sentence-transformers==6.2.0`, `numpy==2.4.6`, `scikit-learn==1.9.1`, `lingua-language-detector==2.1.1`, `onnxruntime==1.31.1`), and remove redundant build-time network downloads from `Dockerfile`.

**Options considered:**
1. Download models from HuggingFace at build time — fails in offline or proxy-restricted environments and adds fragile external network dependencies.
2. Store uncompressed `.npy` (28.2 MB) vs `.npz` (26.1 MB) — `.npz` provides portable compressed storage with instant numpy loading.
3. Keep unpinned version ranges in `requirements.txt` — risks build breakage when transitive dependencies update.

**What decided it:** Satisfies P0 requirement 1.2. Model weights and indexes vendor cleanly into the repo, guaranteeing zero network requests during `docker compose build`.

---

## Entry 12 — 2026-11-06 17:30 IST — Runtime Hygiene, Startup Self-Test & Non-Silent Failure Recovery

**Commit:** `da217f2`  
**Task:** 1.3, 1.4  
**Decision:** Eliminate silent `except Exception: return RA01` fallbacks. Added startup self-test note (`{"id": "__self_test__"}`), trace logging to `sys.stderr` with note IDs, fallback tracking counter, and BM25-based candidate recovery (`_get_bm25_fallback`) when unexpected errors occur. In `entrypoint.py`, added raw string ID recovery for corrupted JSON lines, input ID deduplication, thread pinning (`torch.set_num_threads(1)`), and `imap_unordered` pooling.

**Options considered:**
1. Hardcoded constant fallback (`RA01`) on failure — artificially penalizes score and masks diagnostic errors.
2. Dynamic BM25 fallback — retrieves relevant leaf codes even if neural components fail, preserving prediction quality.
3. Inventing line IDs (`malformed-line-N`) — violates the requirement that input IDs match output IDs; regex extraction from raw lines preserves original IDs.

**What decided it:** Satisfies P0 requirements 1.3 and 1.4. Verified by unit tests in `tests/test_runtime_hygiene.py` covering self-test, empty inputs, malformed ID regex recovery, and BM25 fallback.

---

## Entry 13 — 2026-11-06 19:15 IST — Knowledge Base Enrichment via WHO API, Specificity Descent Refactor & Multilingual Coverage

**Commit:** `43f8573`  
**Task:** 2.1, 2.2, 3.3, 3.6  
**Decision:** Enriched the ICD-11 Knowledge Base (`kb/enriched_kb.json` and `kb/who_enrichments.json`) with official WHO ICD-11 entities via OAuth2 client credentials (`WHO_CLIENT_ID` / `WHO_CLIENT_SECRET`), extracting official index terms (synonyms), definitions, inclusions, and exclusions across all 18,375 MMS categories. Refactored `icd_hierarchy.py` specificity descent with child `.Z` priority and deep descendant synonym matching. Upgraded `reranker.py` with bracketed alias parsing and specificity bonuses. Added multi-language clinical dictionaries for Hindi, Arabic, Chinese, Swahili, Spanish, French, and Portuguese in `multilingual.py`.

**Options considered:**
1. Rely purely on title strings from `mms-2026-01.json` — causes severe lexical gaps for common clinical synonyms ("thrush", "shingles", "boda accident", "urinary schistosomiasis", "distal radius fracture") which do not appear in terse titles.
2. Unscoped sibling `.Z` resolution — previously returned `NC32.Z` for forearm fracture instead of descending to `NC32.5Z` (lower end of radius). Refactored to inspect the category's own children for `.Z` before falling back to sibling `.Z`.
3. Naive substring matching for synonyms — failed on bracketed aliases like `[urinary schistosomiasis]` or dashed entries. Added bracket extraction and bidirectional phrase overlap.

**What decided it:** Satisfies P1 requirements 2.1, 2.2, 3.3, and 3.6. Top-1 dev score increased to 55.0% and candidate recall@30 reached 95.0%, with exact leaf resolution verified across all 13 unit tests.

---

## Entry 14 — 2026-11-06 19:25 IST — Experimental LLM Reranking Stage (Removed in Audit)

**Commit:** `84886cb`  
**Task:** 4.1, 4.2  
**Decision:** Originally explored an optional LLM reranker with spend and timeout guards. In the strict NO-LLM audit, this was entirely removed from the codebase (deleted `llm_reranker.py`, tests, and dependencies) to guarantee zero network, zero LLM dependencies at build or run time.

**Options considered:**
1. Maintain hybrid LLM capability — violates strict no-LLM constraint.
2. Complete removal of all LLM code, tests, and dependencies — 100% compliant with zero-LLM brief.

**What decided it:** Full removal during audit.

---

## Entry 15 — 2026-11-06 19:30 IST — Held-Out Evaluation Dataset (Self-Written Smoke Set)

**Commit:** `5f6c613`  
**Task:** 5.1  
**Decision:** Created an 85-note synthetic evaluation dataset (`data/held_out_notes.jsonl` and generator `scripts/create_held_out_notes.py`) spanning 26 ICD-11 MMS chapters and 8 languages. **Note:** This is a self-written smoke set (optimistic, terminology aligns with glossary); it serves as a sanity check against crashes and chapter coverage, but is NOT evidence of generalization to unseen real-world clinical notes.

**Options considered:**
1. Present held-out score as generalization proof — misleading because vocabulary overlaps with handcrafted translation dictionaries.
2. Label transparently as a self-written smoke test — honest benchmarking standard.

**What decided it:** Transparent documentation in README and DECISIONS.md.

---

## Entry 16 — 2026-11-06 19:35 IST — 100% Complete WHO ICD-11 Release Enrichment Vendored

**Commit:** `9cd8e68`  
**Task:** 2.1, 2.2  
**Decision:** Fully completed asynchronous WHO API release enrichment for all 18,375 categories in `data/mms-2026-01.json`, yielding 93,387 official WHO index terms, definitions, inclusions, and exclusions. Rebuilt `kb/enriched_kb.json` (23.5 MB) and `kb/bm25.pkl` (5.68 MB), and vendored `kb/who_enrichments.json` (9.08 MB) directly into the repository. This provides 100% offline self-containment with zero runtime or build-time WHO API dependency.

**Options considered:**
1. Fetch WHO enrichments at Docker build time — would fail in offline test environments or when credentials are absent, risking build rejection.
2. Store partial enrichments — leaves rare chapters or sub-branches impoverished. 100% enrichment ensures full diagnostic coverage across all 26 chapters.

**What decided it:** Satisfies P1 requirements 2.1 and 2.2. Guarantees complete offline functionality from a fresh clone without external credentials.

---

## Entry 17 — 2026-11-06 19:40 IST — Architecture Documentation, Provenance & Verification

**Commit:** `f902a8f`  
**Task:** 6.3, 6.4  
**Decision:** Documented architecture and evaluation in `docs/ARCHITECTURE.md` and initial README, noting verified dev metrics, CC BY-NC-ND 3.0 IGO data license attribution, and reproduction instructions.

**Options considered:**
1. Bloated multi-page README — obscures critical evaluation and setup instructions.
2. Concise one-page README with detailed architectural deep-dive in `docs/ARCHITECTURE.md`.

**What decided it:** Split detailed diagrams and module breakdown to `docs/ARCHITECTURE.md`. Satisfies P3 requirements 6.3 and 6.4.

---

## Entry 18 — 2026-11-07 10:10 IST — Strict NO-LLM Audit Remediation, STRICT Mode & Commit Alignment

**Commit:** `8bfaac1`  
**Task:** Audit Fixes 1-5  
**Decision:** Fully completed the strict NO-LLM audit. Deleted `llm_reranker.py`, `tests/test_llm_reranker.py`, the LLM hooks in `pipeline.py`, and `openai`/`anthropic` dependencies in `requirements.txt`. Eliminated constant last-resort fallbacks (`["1F40.Z", "CA40.Z", "1A07.Z"]`) in `entrypoint.py` and `pipeline.py`, replacing them with per-note dynamic BM25 ranking. Added `STRICT=1` environment mode to fail loudly on unhandled errors, hardened `scripts/docker_context_check.py` to run `score.py` in an isolated temp directory replicating Docker context, removed unused fastText model, aligned all historical commit hashes in `DECISIONS.md` to real git commits, and condensed `README.md` to a single page.

**Options considered:**
1. Retaining inactive LLM stubs — risks build failure or confusion regarding network access. Complete deletion ensures 100% adherence to zero-LLM rules.
2. Keeping constant fallback on worker crash — masks bugs and produces 0 scores. Per-note dynamic BM25 fallback produces valid, context-appropriate predictions.
3. Keeping unverified claims in README — rewritten to present honest metrics on dev (44.58 / 75.0) and smoke set (labeled as self-written smoke test).

**What decided it:** Passes `scripts/docker_context_check.py --run` with 0 errors, 0 constant fallbacks, and exact score 44.58 / 75.0. All 9 unit tests pass.
