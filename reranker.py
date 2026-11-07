"""Candidate re-ranking and scoring.
Combines retrieval scores with lexical overlap, section presence, and specificity cues.
"""

import re
from typing import Any

from clinical_concepts import extract_clinical_qualifiers
from text_utils import tokenize


class RuleReranker:
    """Rerank candidates using exact phrase matches, assessment section alignment, and cue bonuses."""

    def __init__(self, kb: dict[str, dict[str, Any]]):
        self.kb = kb

    def score_candidate(
        self,
        code: str,
        base_score: float,
        extracted_phrases: list[str],
        note_text: str,
    ) -> float:
        """Calculate adjusted score for a candidate."""
        info = self.kb.get(code)
        if not info:
            return base_score

        title = info.get("title", "").lower()
        title_tokens = set(tokenize(title)) - {"other", "specified", "unspecified", "due", "with", "and", "the", "for", "to", "from", "in", "of", "by"}
        if not title_tokens:
            return base_score

        score = base_score

        # 1. Exact phrase match or substring match bonus (title and WHO synonyms)
        note_lower = note_text.lower()
        extracted_text = " ".join(extracted_phrases).lower()
        extracted_tokens = set(tokenize(extracted_text)) - {"other", "specified", "unspecified", "due", "with", "and", "the", "for", "mild", "patient", "alert"}

        is_unspecified = (code.endswith(".Z") or code.endswith("Z") or "unspecified" in title)
        is_other_specified = (code.endswith(".Y") or code.endswith("Y") or "other specified" in title)
        is_residual = info.get("residual", False) or is_unspecified or is_other_specified

        matched_synonym = False
        matched_specific = False

        for phrase in extracted_phrases:
            phrase_lower = phrase.lower().strip()
            if not phrase_lower:
                continue
            if title in phrase_lower or phrase_lower in title:
                score += 0.06
                matched_synonym = True
                if not is_residual:
                    matched_specific = True
                break

        # Check official WHO synonyms and inclusions if title didn't match directly
        GENERIC_STOP = {"due", "with", "and", "the", "for", "other", "specified", "unspecified", "from", "to", "in", "of", "by"}
        GENERIC_MODIFIERS = {"stage", "grade", "class", "type", "level", "degree", "group", "phase", "mild", "moderate", "severe"}

        if not matched_synonym:
            for syn in info.get("synonyms", []) + info.get("inclusions", []):
                syn_lower = syn.lower().strip()
                if len(syn_lower) < 4:
                    continue

                # Extract aliases: stripped brackets [xyz] or dash separations
                bracketed = re.findall(r"\[(.*?)\]|\((.*?)\)", syn_lower)
                aliases = [syn_lower]
                clean_syn = re.sub(r"\[.*?\]|\(.*?\)", "", syn_lower).strip()
                if clean_syn and clean_syn != syn_lower and len(clean_syn) >= 4:
                    aliases.append(clean_syn)
                for b in bracketed:
                    for item in b:
                        if item and len(item.strip()) >= 3:
                            aliases.append(item.strip())
                if " – " in syn_lower:
                    aliases.extend([p.strip() for p in syn_lower.split(" – ") if len(p.strip()) >= 3])
                elif " - " in syn_lower:
                    aliases.extend([p.strip() for p in syn_lower.split(" - ") if len(p.strip()) >= 3])

                syn_hit = False
                for term in aliases:
                    if term in extracted_text or (len(term) >= 5 and term in note_lower):
                        syn_hit = True
                        break
                    for phrase in extracted_phrases:
                        p_clean = phrase.lower().strip()
                        if len(p_clean) >= 4 and (p_clean in term or term in p_clean):
                            syn_hit = True
                            break
                        # Bag-of-words concept matching for terms with connectives
                        term_words = set(re.findall(r"\b[a-z]{3,}\b", term)) - GENERIC_STOP
                        p_words = set(re.findall(r"\b[a-z]{3,}\b", p_clean))
                        # Prevent matching solely on generic severity words like 'stage' or 'grade'
                        substantive_term_words = term_words - GENERIC_MODIFIERS
                        if substantive_term_words:
                            term_stems = {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in term_words}
                            p_stems = {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in p_words}
                            if term_stems.issubset(p_stems):
                                syn_hit = True
                                break
                    if syn_hit:
                        break

                if syn_hit:
                    score += 0.065
                    matched_synonym = True
                    if not is_residual:
                        matched_specific = True
                    break

        # 2. Token overlap with stemming (singular / plural)
        def _token_matches(tok: str, tok_set: set[str]) -> bool:
            if tok in tok_set:
                return True
            if tok.endswith("s") and tok[:-1] in tok_set:
                return True
            if (tok + "s") in tok_set:
                return True
            return False

        overlap = sum(1 for t in title_tokens if _token_matches(t, extracted_tokens))
        if overlap > 0:
            recall = overlap / len(title_tokens)
            precision = overlap / len(extracted_tokens) if extracted_tokens else 0
            score += 0.03 * recall + 0.02 * precision

        # Specificity verification: if candidate is specific, check that distinguishing qualifiers exist in note
        parent_code = info.get("parent")
        parent_info = self.kb.get(parent_code) if parent_code else None
        if parent_info and not is_residual:
            parent_title = parent_info.get("title", "").lower()
            parent_toks = set(tokenize(parent_title))
            GENERIC_BODY_SYSTEMS = {"respiratory", "rate", "system", "tract", "pressure", "pulse", "organ", "tissue", "disease", "disorder", "infection"}
            distinguishing = title_tokens - parent_toks - GENERIC_STOP - GENERIC_MODIFIERS - GENERIC_BODY_SYSTEMS
            if distinguishing:
                has_dist_support = False
                for dt in distinguishing:
                    if dt in note_lower or (dt.endswith("s") and dt[:-1] in note_lower) or ((dt + "s") in note_lower):
                        has_dist_support = True
                        break
                    # Clinical equivalents
                    if dt == "food" and any(fd in note_lower for fd in ["peanut", "groundnut", "nut", "biscuit", "egg", "milk", "fish", "wheat", "food"]):
                        has_dist_support = True
                        break
                    if dt.startswith("allerg") and "allerg" in note_lower:
                        has_dist_support = True
                        break
                    if dt == "bladder" and "cystitis" in note_lower:
                        has_dist_support = True
                        break
                    if dt == "arising" and any(ar in note_lower for ar in ["pregnancy", "gestational", "come with", "developing"]):
                        has_dist_support = True
                        break
                    if dt in ("systolic", "diastolic", "combined") and re.search(r"\b(1[4-9]\d|[2-9]\d\d)\s*/\s*(9\d|1\d\d)\b", note_lower):
                        has_dist_support = True
                        break
                    if dt in ("adult", "adults") and not any(ped in note_lower for ped in ["child", "infant", "paed", "pediatric"]):
                        has_dist_support = True
                        break
                    if dt == "acquired" and (any(acq in note_lower for acq in ["primary", "adult", "hashimoto"]) or not any(cong in note_lower for cong in ["congenital", "infant", "child"])):
                        has_dist_support = True
                        break
                # Negative qualifiers: "without coma", "without complication", "without mention of"
                if not has_dist_support and any(w_neg in title for w_neg in ["without coma", "without complication", "without mention"]):
                    has_dist_support = True

                if not has_dist_support:
                    # Candidate has distinguishing qualifiers that are absent from clinical note
                    matched_specific = False
                    score -= 0.06

        # 3. Specificity bonus: specific leaf whose qualifiers / synonyms matched the clinical note
        if not is_residual and matched_specific:
            score += 0.05
        elif is_unspecified and not matched_synonym:
            # Demote unspecified if note had specific diagnostic phrases
            score -= 0.02

        # 4. Chapter 21 (Symptoms and signs) demotion when definitive disease is present
        chapter = info.get("chapter", "")
        if chapter == "21" or code.startswith("M"):
            score -= 0.05

        # Traditional medicine (Chapter 26) demotion
        if chapter == "26" or code.startswith("S"):
            score -= 0.35

        # Developmental / genetic anomaly (Chapter 20) demotion for standard outpatient notes
        if chapter == "20" or code.startswith("LD"):
            if not any(g in note_lower for g in ["syndrome", "genetic", "congenital", "anomaly", "malformation", "dysmorphic"]):
                score -= 0.35

        # 5. Demote .Y (Other specified) codes relative to .Z (Unspecified)
        if is_other_specified:
            score -= 0.06

        # 6. Demote critical clinical mismatch (complications / conditions not in note)
        critical_mismatch = [
            "abscess", "fracture", "neoplasm", "carcinoma", "malignant", "congenital",
            "orchitis", "meningitis", "encephalitis", "pancreatitis", "osteomyelitis",
            "peritonitis", "mastoiditis", "dementia"
        ]
        for term in critical_mismatch:
            if term in title and term not in note_lower:
                score -= 0.06

        # 7. Biological sex consistency checks
        is_male = bool(re.search(r"\b(\d{1,3}\s*M|male|man|boy|gentleman)\b", note_text, re.IGNORECASE))
        is_female = bool(re.search(r"\b(\d{1,3}\s*F|female|woman|girl|lady|primigravida|gravida|para|lmp)\b", note_text, re.IGNORECASE))
        if is_male:
            # Male patients cannot have obstetric/puerperal codes (Ch 18) or female genital (GA)
            if chapter == "18" or code.startswith(("JA", "JB", "GA")):
                score -= 0.40
        if is_female:
            # Female patients cannot have male genital diseases (GB)
            if code.startswith("GB"):
                score -= 0.40

        # 8. Age & life-stage consistency
        age_match = re.search(r"\b(\d{1,3})\s*(?:yo|y/o|yr|yrs|year|years|-year-old|M|F)\b", note_text, re.IGNORECASE)
        age = int(age_match.group(1)) if age_match else None
        if age is not None:
            if age >= 18:
                # Adults cannot have neonatal diseases or conditions of prematurity
                if chapter == "19" or code.startswith("KA") or "prematurity" in title or "neonatal" in title:
                    score -= 0.40
                elif "in adults" in title or "adult" in title:
                    score += 0.06
            elif age < 12:
                if "in adults" in title:
                    score -= 0.40
                elif "in children" in title or "child" in title:
                    score += 0.06

        # 9. Pregnancy & obstetric consistency
        is_pregnant = bool(re.search(r"\b(anc\b|antenatal|pregnant|pregnancy|gestation|trimester|fundal|fhr|edd|g\d+p\d+|\d+/\d+\s*weeks?)\b", note_text, re.IGNORECASE))
        is_not_pregnant = bool(re.search(r"\b(upt\s*neg|pregnancy\s*test\s*neg|not\s*pregnant)\b", note_text, re.IGNORECASE))
        if is_not_pregnant:
            if chapter == "18" or code.startswith(("JA", "JB")):
                score -= 0.40
        elif is_pregnant:
            # In pregnancy, obstetric chapter conditions take precedence
            if chapter == "18" or code.startswith(("JA", "JB")) or "pregnancy" in title or "maternal" in title:
                score += 0.08
            elif "cystitis" in title and ("pregnancy" not in title and "maternal" not in title and "obstetric" not in title):
                score -= 0.08

        # 10. Chapter 24 administrative encounter vs definitive disease
        if (chapter == "24" or code.startswith("Q")) and not is_pregnant:
            if any(dx in note_lower for dx in ["hiv", "hypertension", "diabetes", "malaria", "pneumonia", "asthma", "ulcer", "mumps", "measles", "varicella", "infection", "tonsillitis", "otitis", "bronchitis"]):
                score -= 0.25

        # 11. Species & Clinical rules
        if "falciparum" in note_lower:
            if "falciparum" in title:
                score += 0.08
            elif "malaria" in title and ("unspecified" in title or "other" in title):
                score -= 0.08
            elif any(other_sp in title for other_sp in ["malariae", "vivax", "ovale"]):
                score -= 0.12
        elif "malariae" in note_lower:
            if any(other_sp in title for other_sp in ["falciparum", "vivax", "ovale"]):
                score -= 0.12

        # 12. Uncomplicated / complicated clinical alignment
        if any(uncomp in note_lower for uncomp in ["uncomplicated", "no complication", "no complications", "without complication"]):
            if "without complication" in title or "uncomplicated" in title:
                score += 0.06
            elif code.endswith(".Z") or code.endswith("Z"):
                score += 0.02
            if "with complication" in title or "cerebral" in title:
                score -= 0.07

        # 13. Acute encounter in Impression vs background history
        if "hypoglycaemia" in note_lower and any(h in extracted_text for h in ["hypoglycaemia", "low sugar"]):
            if "hypoglycaemia" in title:
                score += 0.08

        # 14. Peptic ulcer unconfirmed site
        if "peptic ulcer" in note_lower and any(unsp in note_lower for unsp in ["not confirmed", "haven't confirmed", "site unspecified", "stomach or duodenum"]):
            if "peptic ulcer" in title and ("site unspecified" in title or "unspecified" in title):
                score += 0.08

        # 15. Anatomical site specificity (osteoarthritis, fractures, etc.)
        if "osteoarthritis" in note_lower:
            for site in ["knee", "hip", "hand", "wrist", "shoulder", "ankle", "spine"]:
                if site in note_lower:
                    if site in title:
                        score += 0.08
                    elif "unspecified" in title or "other" in title:
                        score -= 0.06

        # 16. Adult obesity
        if any(ob in note_lower for ob in ["obesity", "bmi"]) and not any(ped in note_lower for ped in ["child", "infant", "pediatric"]):
            if "in adults" in title or "adult" in title:
                score += 0.08

        # 17. Active clinical concept qualifiers
        active_qualifiers = extract_clinical_qualifiers(note_text)
        for q in active_qualifiers:
            if q in title:
                score += 0.06
                break

        # 18. Bacteriologically confirmed alignment
        if "confirmed" in note_lower and "confirmed" in title:
            score += 0.04

        # 18. Prefer leaf codes over non-leaf
        if not info.get("leaf", False):
            score -= 0.05

        return score

    def rerank(
        self,
        candidates: list[tuple[str, float]],
        extracted_phrases: list[str],
        note_text: str,
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Rerank candidate list."""
        scored = []
        for code, base_score in candidates:
            final_score = self.score_candidate(code, base_score, extracted_phrases, note_text)
            scored.append((code, final_score))

        sorted_candidates = sorted(scored, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]
