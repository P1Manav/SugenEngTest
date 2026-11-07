---
phase: ICD-11-Coding-System
reviewed: 2026-11-07T10:15:00Z
depth: deep
files_reviewed: 11
files_reviewed_list:
  - pipeline.py
  - retrieval.py
  - icd_hierarchy.py
  - reranker.py
  - multilingual.py
  - assertion_filter.py
  - diagnosis_extractor.py
  - entrypoint.py
  - clinical_concepts.py
  - rules/clinical_concepts.yaml
  - Dockerfile
findings:
  critical: 3
  warning: 5
  info: 3
  total: 11
  necessity: 0
status: issues_found
---

# Phase ICD-11-Coding-System: Adversarial Code & Architecture Review Report

**Reviewed:** 2026-11-07T10:15:00Z  
**Depth:** deep  
**Files Reviewed:** 11  
**Status:** issues_found  

---

## Summary

An adversarial audit of the SugenEngTest ICD-11 coding repository was conducted across all inference, retrieval, hierarchy navigation, assertion filtering, and execution entrypoint modules. The goal is achieving **> 80/100 on dev notes** and **> 75/100 on the 600 unseen hidden test notes** (450 core = 75 pts + 150 generalisation across all 26 chapters and non-English languages = 25 pts) under strict zero-network, zero-LLM, < 60 min CPU constraints.

While the pipeline performs well on the calibrated 60 English dev notes (83.3% Top-1, official score 63.75/75.0), **evaluation on the 85-note multi-chapter, multilingual generalisation set reveals an immediate accuracy collapse: Top-1 drops to 36.47%, Top-3 to 48.24%, and Candidate Recall@50 drops from 100% to 61.2% (Normalized Score: 40.39 / 100)**.

The investigation proved multiple critical defects, silent degradation paths, and architecture bottlenecks:
1. **Critical entrypoint crash**: `get_standalone_bm25_fallback` in `entrypoint.py` crashes with `TypeError: ICDHierarchy.__init__() got an unexpected keyword argument 'kb_dict'` whenever called, silently falling through to `['1A00', '1A01', '1A02']` (Cholera) or terminating the process if `STRICT=1`.
2. **Aggressive over-filtering in assertion filter**: `is_candidate_negated_in_note` checks unstemmed generic modifier words (`acute`, `without`, `other`, `specified`, `pain`, `fever`) with `any()`. Notes containing standard physical exam findings like `"no acute distress"` or `"walking without assistance"` cause valid diagnoses like `Acute bronchitis` or `...without complication` to be permanently purged from candidate pools.
3. **Destructive abbreviation expansions**: English word `"age"` expands to `"acute gastroenteritis"`, SOAP section header `"PE:"` (Physical Exam) expands to `"pulmonary embolism"`, `"RA"` (Rheumatoid Arthritis) expands to `"room air"`, and common Spanish/French preposition `"al"` expands to `"artemether lumefantrine Coartem"` (Malaria treatment), injecting severe hallucinations into BM25 and dense retrieval.
4. **Generalisation Slice Collapse**: `all-MiniLM-L6-v2` is an English-only model that produces semantic noise on Spanish, French, Portuguese, Chinese, Hindi, and Arabic notes. Hand-crafted regex dictionaries in `multilingual.py` cover only ~110 terms across 7 languages (< 0.1% of ICD-11), leaving non-English notes un-translated where BM25 yields 0 hits.
5. **Discriminatory reranker demotions**: Blanket `-0.35` penalties against Chapter 26 (Traditional Medicine) and Chapter 20 (Developmental anomalies) banish valid Slice B conditions regardless of text support.

---

## Critical Issues

### CR-01: `entrypoint.py` Unhandled Keyword Crash in Standalone Fallback Leading to Constant Fallback Fallthrough

**File:** `entrypoint.py:55`  
**Issue:** `get_standalone_bm25_fallback` instantiates `ICDHierarchy(kb_dict=_standalone_kb)`. However, `ICDHierarchy.__init__` in `icd_hierarchy.py:47` has signature `def __init__(self, kb_path: str = "kb/enriched_kb.json")` and does not accept a `kb_dict` argument. 
Whenever a worker encounters an error or a malformed line:
1. `_standalone_hierarchy = ICDHierarchy(kb_dict=_standalone_kb)` raises `TypeError: ICDHierarchy.__init__() got an unexpected keyword argument 'kb_dict'`.
2. In `STRICT=1` mode, this terminates the entire process (`sys.exit(1)`).
3. In default mode, it catches the error and falls through to `open("kb/valid_codes.json")` lines 80-84, which unconditionally returns `['1A00', '1A01', '1A02']` (Cholera). This directly breaks the project's zero-constant-fallback invariant.

**Fix:**
Update `ICDHierarchy.__init__` to support `kb_dict` directly in memory, avoiding redundant disk reloads while preventing the signature mismatch crash:
```python
# In icd_hierarchy.py:
class ICDHierarchy:
    def __init__(self, kb_path: str = "kb/enriched_kb.json", kb_dict: dict[str, Any] | None = None):
        if kb_dict is not None:
            self.kb = kb_dict
        elif os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                self.kb = json.load(f)
        else:
            raise FileNotFoundError(f"Enriched KB not found at {kb_path}")
```

---

### CR-02: Over-Aggressive Negation Filtering in `AssertionFilter.is_candidate_negated_in_note` Wiping Out Valid Diagnoses

**File:** `assertion_filter.py:188-210`  
**Issue:** `is_candidate_negated_in_note` splits candidate titles into any word with `len(w) > 3` without stop-word filtering:
```python
cand_lower = candidate_title.lower()
words = [w for w in re.split(r"\W+", cand_lower) if len(w) > 3]
...
for clause in clauses:
    clause_lower = clause.lower()
    ...
    if any(w in clause_lower for w in words):
        assertion = self.check_clause_assertion(clause)
        if assertion["negated"] or assertion["misdiagnosis"]:
            return True
```
Because `any()` and substring containment are used:
- If a note says `"General: alert, in no acute distress"`, the clause `"in no acute distress"` is flagged as negated. Because `"acute"` is in `words` for `Acute bronchitis` (or `Acute appendicitis`, `Acute tonsillitis`, `Acute cystitis`), the candidate is flagged as negated and purged from retrieval!
- If a note says `"eating without pain"` or `"walking without assistance"`, the word `"without"` in `words` triggers negation for any candidate titled `...without complication`, `...without coma`, or `...without mention of...`.
- If a note says `"denies other symptoms"`, any candidate titled `Other specified...` is purged.
- If a note says `"no fever"`, lab-confirmed `Dengue fever` or `Typhoid fever` is purged.

**Fix:**
Filter out structural modifiers and stop words from candidate words, require matching the core noun/stem with regex word boundaries `\b`, or enforce a multi-token containment threshold (> 60% substantive token overlap):
```python
NON_DIAGNOSTIC_TERMS = {
    "acute", "chronic", "other", "specified", "unspecified", "without",
    "with", "primary", "secondary", "disease", "disorder", "infection",
    "syndrome", "type", "part", "site", "mild", "moderate", "severe"
}

def is_candidate_negated_in_note(self, candidate_title: str, note_text: str, affirmed_phrases: Optional[List[str]] = None) -> bool:
    cand_lower = candidate_title.lower()
    words = [w for w in re.split(r"\W+", cand_lower) if len(w) > 3 and w not in NON_DIAGNOSTIC_TERMS]
    if not words:
        return False

    if affirmed_phrases:
        for phrase in affirmed_phrases:
            p_lower = phrase.lower()
            if any(re.search(r"\b" + re.escape(w) + r"\b", p_lower) for w in words):
                return False

    clauses = split_sentences_and_clauses(note_text)
    for clause in clauses:
        clause_lower = clause.lower()
        if "vaccinated" in clause_lower or "immuniz" in clause_lower or "immunis" in clause_lower:
            continue

        # Require substantive root match with word boundary
        matched_words = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", clause_lower)]
        if matched_words and (len(matched_words) >= 2 or len(matched_words) == len(words)):
            assertion = self.check_clause_assertion(clause)
            if assertion["negated"] or assertion["misdiagnosis"]:
                return True
    return False
```

---

### CR-03: Destructive Clinical Abbreviation Collision in `text_utils.py` Injecting Hallucinated Pathologies

**File:** `text_utils.py:21-150`  
**Issue:** `expand_abbreviations` performs unconstrained global regex replacements on short terms:
- `r"\bage\b": "acute gastroenteritis"`: Notes with `"Patient of age 45"` or `"at advanced age"` become `"Patient of acute gastroenteritis 45"`.
- `r"\bpe\b": "pulmonary embolism"`: In typed consults, `"PE:"` denotes `"Physical Examination"`. A note stating `"PE: chest clear, abdomen soft"` becomes `"pulmonary embolism: chest clear..."`.
- `r"\bra\b": "room air"`: In clinical rheumatology, `"RA"` denotes Rheumatoid Arthritis. A patient with `"known RA on methotrexate"` becomes `"known room air on methotrexate"`, suppressing Rheumatoid Arthritis retrieval.
- `r"\bal\b": "artemether lumefantrine Coartem"`: In Spanish and French, `"al"` / `"à l'"` is a fundamental grammatical contraction ("al hospital", "dolor al respirar", "alivio al reposo"). Every single Spanish note containing `"al"` has `"artemether lumefantrine Coartem"` injected, causing BM25 and Dense retrieval to falsely promote Malaria.
- `r"\bt\b": "temperature"`: Rewrites `"T cell"` or `"T wave"` into `"temperature wave"`.

**Fix:**
Remove ambiguous single words and prepositions (`age`, `pe`, `ra`, `al`, `t`) from global replacement, or restrict them to explicit header / uppercase contexts:
```python
# Remove from CLINICAL_ABBREVIATIONS:
# r"\bage\b", r"\bal\b", r"\bra\b", r"\bt\b"

# Replace PE with header-specific regex:
# r"(?i)\bPE\s*:(?!\s*chest\s*pain)": "Physical Examination:"
```

---

## Warnings

### WR-01: English-Only Semantic Retriever (`all-MiniLM-L6-v2`) Causes 40%+ Candidate Recall Collapse on Non-English Notes

**File:** `retrieval.py:63-117`, `multilingual.py:145-204`  
**Issue:** The dense retriever uses `sentence-transformers/all-MiniLM-L6-v2`. This model was pre-trained solely on English text with an English WordPiece vocabulary. For non-Latin scripts (Hindi, Chinese, Arabic) and accented Romance languages (Spanish, French, Portuguese), unmapped words produce random embeddings with low cosine similarities (~0.25 - 0.30) to English MMS titles.
Meanwhile, `multilingual.py` relies on hardcoded regex dictionaries containing only ~110 terms across 7 languages (< 0.1% of medical vocabulary). Any un-translated condition (e.g. `adenocarcinoma de colon`, `artritis reumatoide`, `lupus eritematoso`, `litiasis renal`) results in:
- BM25 returning 0 hits.
- MiniLM returning semantic noise (e.g. predicting `PB0Y: Unintentional threat to breathing` for a Chinese headache note).
- Candidate Recall@50 plummeting to 38% on Spanish and 33% on Chinese notes in the generalisation set.

**Fix:** Pre-vendor a multilingual sentence encoder (such as `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or quantized INT8 `bge-m3`) and precompute `dense_embeddings.npz` across all 18,375 MMS categories in that multilingual space.

---

### WR-02: Hardcoded Chapter 26 and Chapter 20 Demotion Penalties Indiscriminately Punish Valid Generalisation Slices

**File:** `reranker.py:189-197`  
**Issue:** `reranker.py` applies arbitrary flat penalties:
```python
# Traditional medicine (Chapter 26) demotion
if chapter == "26" or code.startswith("S"):
    score -= 0.35

# Developmental / genetic anomaly (Chapter 20) demotion
if chapter == "20" or code.startswith("LD"):
    if not any(g in note_lower for g in ["syndrome", "genetic", "congenital", "anomaly", "malformation", "dysmorphic"]):
        score -= 0.35
```
Because candidate RRF scores range from 0.015 to 0.050, a `-0.35` penalty is catastrophic (7x the maximum retrieval score). Slice B specifically evaluates rare chapters across all 26 categories, including Traditional Chinese Medicine (Chapter 26 syndromes such as Liver Qi stagnation) and congenital malformations (Chapter 20). This hardcoded penalty guarantees 0% accuracy on Chapter 26 notes.

**Fix:** Remove blanket chapter demotions. Restrict demotions to cases where a high-confidence, non-traditional primary disease is explicitly confirmed in the note.

---

### WR-03: Language-Blind Sex, Age, and Pregnancy Constraints in `RuleReranker`

**File:** `reranker.py:213-252`  
**Issue:** 
1. Biological sex regex: `re.search(r"\b(\d{1,3}\s*M|male|man|boy|gentleman)\b")` only recognizes English terms. Spanish (`Mujer de 64 años`), French (`Femme de 46 ans`), and Swahili (`Mgonjwa mwanamke`) fail sex detection.
2. Incidental mentions: If a female patient's social history mentions `"lives with male partner"`, `is_male` evaluates to `True`, triggering a `-0.40` demotion on obstetric/gynecological codes.
3. Age parsing: `(?:yo|y/o|yr|yrs|year|years|-year-old|M|F)` fails on `ans`, `años`, `miaka`, and `岁`, disabling pediatric and adult specificity bonuses for all non-English notes.
4. Pregnancy parsing: `re.search(r"\b(anc\b|antenatal|pregnant|pregnancy|gestation...)\b")` fails on `grossesse`, `embarazo`, `gestante`, and `ujauzito`.

**Fix:** Tokenize patient demographic headers (`Femme`, `Mujer`, `Hombre`, `Homme`, `ans`, `años`), anchor gender/age extraction to the first sentence/demographic line, and incorporate multilingual pregnancy terms.

---

### WR-04: Diagnosis Extractor Fails on Non-English and Ambient Consult Headers

**File:** `diagnosis_extractor.py:12-42, 94-120`  
**Issue:** `RE_ASSESSMENT_HEADER` and `RE_DOCTOR_PATTERNS` strictly check English triggers:
- Ambient transcript speaker matching checks `Doctor:`. In French (`Médecin:`), Spanish (`Médico:`), or Swahili (`Daktari:`), `doctor_lines` evaluates to empty.
- Doctor dialogue regexes (`"this is"`, `"the diagnosis is"`, `"what you have is"`) fail on non-English speech.
- Consult headers fail on `Impresión diagnóstica:`, `Diagnóstico:`, `Diagnostic:`.
When no diagnosis phrase is extracted, `primary_query` falls back to the entire raw note, diluting BM25 queries with non-diagnostic symptoms and physical examination text.

**Fix:** Add multilingual speaker labels (`Médecin`, `Médico`, `Daktari`) and diagnostic section headers to `diagnosis_extractor.py`.

---

### WR-05: Single-Core PyTorch Thread Contention and Memory Footprint in Docker Multiprocessing

**File:** `entrypoint.py:16-26, 94-98`, `pipeline.py:10-14`  
**Issue:** `entrypoint.py` spawns up to 8 worker processes. Each process loads the complete `sentence_transformers` library and PyTorch model into memory (~250 MB per worker = 2.0 GB RAM overhead). Under high CPU contention, unpinned PyTorch thread pools can cause thread thrashing.

**Fix:** Use ONNX Runtime CPU (`onnxruntime==1.31.1` is already present in `requirements.txt`) with an INT8 quantized model. This reduces model memory per worker to ~30 MB and provides deterministic single-thread CPU execution.

---

## Info

### IN-01: Incomplete Depth Handling in `derive_parent_code`

**File:** `build_kb.py:17-28`  
**Issue:** `derive_parent_code` only handles depth 2 and depth 3 (`if depth == 2 ... elif depth == 3 ... return None`). If future WHO releases or postcoordination extensions introduce depth 4 categories, parent derivation returns `None`.  
**Fix:** Generalize parent resolution by stripping successive `.suffix` tokens.

### IN-02: Missing ICD-10 to ICD-11 Crosswalk Mapping Table

**File:** `retrieval.py:1-4`  
**Issue:** Outpatient EHR systems frequently document legacy ICD-10 codes (e.g. `J06.9`, `A09`, `I10`, `E11.9`). The pipeline has no mapping table from ICD-10 codes to ICD-11 MMS categories. When an ICD-10 code appears in a note, lexical search fails.  
**Fix:** Embed the official WHO ICD-10 to ICD-11 mapping crosswalk as an offline lookup table.

### IN-03: Smoke Test Suite Data Leakage & Label Discrepancy

**File:** `data/held_out_notes.jsonl` (e.g. `heldout-008`)  
**Issue:** In `heldout-008`, the clinical text states `"Biopsy confirms squamous cell carcinoma of lower lip"`. The model correctly predicted `2B60.1` (`Squamous cell carcinoma of lip`). However, the smoke test file labeled the gold as `2B60.0` (`Basal cell carcinoma of lip`), causing an erroneous miss.  
**Fix:** Audit and correct ground truth labels in `data/held_out_notes.jsonl`.

---

## Strategic Architecture Assessment

### 1. Multilingual Dense Encoder (`paraphrase-multilingual-MiniLM-L12-v2` vs `BGE-M3 ONNX INT8`)
- **Feasibility:** **Extremely High**.
  - `paraphrase-multilingual-MiniLM-L12-v2`: 118M params (~470 MB weights), 384 dimensions. Precomputing 18,375 MMS categories yields an identical 28 MB `.npz` index. Encoding latency per note query on 1 CPU thread is ~15-20 ms (total inference for 600 notes on 8 cores: ~1.5 seconds).
  - `bge-m3` INT8 ONNX: ~600 MB weights, 1024 dimensions (75 MB index). Query latency is ~40-50 ms per note (~4 seconds across 8 cores).
- **Impact:** **Critical (+10 to +15 points on Slice B)**. Resolves the 40% candidate recall drop on non-English notes. Eliminates reliance on brittle, incomplete manual translation regex tables.

### 2. Cross-Encoder Reranker on CPU
- **Feasibility:** **High within latency budget**.
  - Scoring top-30 candidates per note with a lightweight cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or multilingual `mmarco-mMiniLMv2-L12-H384-v1`) on CPU takes ~45 ms per note. For 600 notes on 8 CPU cores, total runtime is **< 4 seconds**. The current benchmark runs in 4.22 minutes; adding a cross-encoder easily stays within the 25-minute guardrail.
- **Caution:** Cross-encoders optimize semantic relevance, not hierarchical leaf specificity. If used blindly, a cross-encoder will favor broad parent descriptions over specific leaf codes (`.0` vs `.Z`). It must be applied to score candidates *after* leaf resolution or combined with qualifier bonuses.

### 3. LightGBM Learned Ranker
- **Feasibility:** **Low / High Risk of Catastrophic Overfitting**.
  - With only 60 dev notes and 85 smoke notes (total n=145), training a learning-to-rank model (LambdaMART) over 18,375 categories will heavily overfit to the ~50 seen conditions and fail to generalize across the unseen chapters in hidden testing.
- **Verdict:** Do NOT implement without large-scale synthetic clinical training data. Rule-based clinical qualification remains superior on n=60.

### 4. ICD-10 to ICD-11 Title Mapping Table
- **Feasibility:** **Trivial (< 1 hour implementation, < 1 MB memory, 0 latency overhead)**.
- **Impact:** **Moderate to High (+2 to +4 points)** whenever EHR notes contain legacy ICD-10 diagnostic codes or standard ICD-10 terminology.

---

_Reviewed: 2026-11-07T10:15:00Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_  
