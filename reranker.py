"""Candidate re-ranking and scoring.
Combines retrieval scores with lexical overlap, section presence, and specificity cues.
"""

import re
from typing import Any

from clinical_concepts import extract_clinical_qualifiers
from icd_hierarchy import CLINICAL_QUALIFIER_EQUIVALENTS
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
            phrase_clean = " ".join(dict.fromkeys(phrase_lower.split()))
            if title in phrase_lower or phrase_lower in title or phrase_clean in title or title in phrase_clean:
                num_substantive = len([w for w in title.split() if w not in {"due", "with", "and", "the", "for", "other", "specified", "unspecified", "from", "to", "in", "of", "by"} and len(w) > 2])
                score += 0.06 + min(0.04, 0.015 * max(0, num_substantive - 1))
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

                # Extract aliases: stripped brackets [xyz], stripped NOS, or dash separations
                bracketed = re.findall(r"\[(.*?)\]|\((.*?)\)", syn_lower)
                aliases = [syn_lower]
                clean_syn = re.sub(r"\[.*?\]|\(.*?\)", "", syn_lower).strip()
                if clean_syn and clean_syn != syn_lower and len(clean_syn) >= 4:
                    aliases.append(clean_syn)
                clean_nos = re.sub(r"\bnos\b", "", syn_lower).strip()
                if clean_nos and clean_nos != syn_lower and len(clean_nos) >= 4:
                    aliases.append(clean_nos)
                clean_both = re.sub(r"\bnos\b", "", clean_syn).strip()
                if clean_both and clean_both not in aliases and len(clean_both) >= 4:
                    aliases.append(clean_both)
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
                    num_substantive = len([w for w in term.split() if w not in GENERIC_STOP and len(w) > 2])
                    syn_bonus = 0.065 + min(0.04, 0.015 * max(0, num_substantive - 1))
                    score += syn_bonus
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
                # 1. Check concept equivalents across compound title that distinguish child from parent
                for concept, eqs in CLINICAL_QUALIFIER_EQUIVALENTS.items():
                    if parent_title and (concept in parent_title or concept.replace("-", " ") in parent_title):
                        continue
                    if concept in title or concept.replace("-", " ") in title:
                        if any(re.search(r"\b" + re.escape(eq) + r"\b", note_lower) for eq in eqs + [concept]):
                            has_dist_support = True
                            break

                if not has_dist_support:
                    for dt in distinguishing:
                        if dt in note_lower or (dt.endswith("s") and dt[:-1] in note_lower) or ((dt + "s") in note_lower):
                            has_dist_support = True
                            break
                        # Check dictionary equivalents
                        eqs = CLINICAL_QUALIFIER_EQUIVALENTS.get(dt, [])
                        if any(eq in note_lower for eq in eqs):
                            has_dist_support = True
                            break
                        # Clinical equivalents
                        if dt in ("infectious", "infective") and any(inf in note_lower for inf in ["infection", "infective", "infectious", "infecciosa", "infeccioso", "infectieuse", "infectieux", "maambukizi", "uti", "bacterial", "viral"]):
                            has_dist_support = True
                            break
                        if dt == "food" and any(fd in note_lower for fd in ["peanut", "groundnut", "nut", "biscuit", "egg", "milk", "fish", "wheat", "food"]):
                            has_dist_support = True
                            break
                        if dt.startswith("allerg") and any(al in note_lower for al in ["allerg", "atopic", "atopique", "alérgic"]):
                            has_dist_support = True
                            break
                        if dt == "bladder" and any(bl in note_lower for bl in ["cystitis", "bladder", "mkojo", "vejiga", "vessie"]):
                            has_dist_support = True
                            break
                        if dt == "arising" and any(ar in note_lower for ar in ["pregnancy", "gestational", "come with", "developing", "gestante", "grossesse"]):
                            has_dist_support = True
                            break
                        if dt in ("systolic", "diastolic", "combined") and re.search(r"\b(1[4-9]\d|[2-9]\d\d)\s*/\s*(9\d|1\d\d)\b", note_lower):
                            has_dist_support = True
                            break
                        if dt in ("adult", "adults"):
                            # Only support adult if note does not indicate pediatric patient
                            is_ped = any(ped in note_lower for ped in ["child", "infant", "paed", "pediatric", "girl", "boy", "son", "daughter"])
                            ped_age = re.search(r"\b([1-9]|1[0-7])\s*(?:yo|y/o|yr|yrs|year|ans|años|anos|miaka|岁)\b", note_lower)
                            if not is_ped and not ped_age:
                                has_dist_support = True
                                break
                        if dt == "acquired" and any(acq in note_lower for acq in ["acquired", "adult-onset", "post-surgical", "secundaria", "secondaire"]):
                            has_dist_support = True
                            break
                        if dt in ("closed", "simple") and any(cl in note_lower for cl in ["closed", "simple", "fermee", "fermée", "cerrada"]):
                            has_dist_support = True
                            break
                # Negative qualifiers: "without [complication]", "uncomplicated", etc.
                if not has_dist_support:
                    if any(w_neg in title for w_neg in ["without", "uncomplicated", "no mention"]):
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
            score -= 0.20

        # Traditional medicine (Chapter 26) demotion only if no traditional medicine terms in note
        if chapter == "26" or code.startswith("S"):
            if not any(g in note_lower for g in ["traditional", "tcm", "qi", "yin", "yang", "herbal", "acupuncture", "kampo", "ayurveda"]):
                score -= 0.08

        # Developmental / genetic anomaly (Chapter 20) demotion for standard outpatient notes
        if chapter == "20" or code.startswith("LD"):
            if not any(g in note_lower for g in ["syndrome", "genetic", "congenital", "anomaly", "malformation", "dysmorphic", "trisomy", "chromosom"]):
                score -= 0.08

        # Manifestation vs etiology check:
        # For codes with 'X due to Y' (e.g. 'Chorea due to SLE', 'Syncope due to autonomic failure'),
        # BOTH manifestation X and cause Y must be documented in the note.
        if " due to " in title:
            parts = title.split(" due to ")
            x_part, y_part = parts[0].strip(), parts[1].strip()
            x_words = [w for w in re.findall(r"\b[a-z]{4,}\b", x_part) if w not in GENERIC_STOP and w not in GENERIC_MODIFIERS]
            y_words = [w for w in re.findall(r"\b[a-z]{4,}\b", y_part) if w not in GENERIC_STOP and w not in GENERIC_MODIFIERS]
            x_miss = x_words and not any(mw in note_lower for mw in x_words)
            y_miss = y_words and not any(mw in note_lower for mw in y_words)
            if x_miss or y_miss:
                score -= 0.30
        elif " in " in title and any(term in title for term in ["dementia", "retinopathy", "nephropathy", "neuropathy"]):
            manifestation = title.split(" in ")[0].strip()
            m_words = [w for w in re.findall(r"\b[a-z]{4,}\b", manifestation) if w not in GENERIC_STOP and w not in GENERIC_MODIFIERS]
            if m_words and not any(mw in note_lower for mw in m_words):
                score -= 0.30

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
                # If term is preceded by "without", it defines the absence of the complication, not a mismatch
                if re.search(r"\bwithout\b.*?\b" + re.escape(term) + r"\b", title):
                    continue
                score -= 0.06

        # 7. Biological sex consistency checks (multilingual support)
        is_male = bool(re.search(r"\b(\d{1,3}\s*M|male|man|boy|gentleman|homme|garçon|garcon|masculino|hombre|niño|nino|varón|varon|kiume|mwanaume|男性|男)\b", note_text, re.IGNORECASE))
        is_female = bool(re.search(r"\b(\d{1,3}\s*F|female|woman|girl|lady|femme|fille|femenino|mujer|niña|nina|kike|mwanamke|primigravida|gravida|para|lmp|anc|antenatal|enceinte|embarazada|gravidez|gestante|ujauzito|mimba|女性|女)\b", note_text, re.IGNORECASE))
        if is_male:
            # Male patients cannot have obstetric/puerperal codes (Ch 18) or female genital (GA)
            if chapter == "18" or code.startswith(("JA", "JB", "GA")):
                score -= 0.40
        if is_female:
            # Female patients cannot have male genital diseases (GB)
            if code.startswith("GB"):
                score -= 0.40

        # 8. Age & life-stage consistency (multilingual support)
        age_match = re.search(r"\b(\d{1,3})\s*(?:yo|y/o|yr|yrs|year|years|-year-old|ans|années|annees|años|anos|miaka|岁|歲)\b|\b(?:age[d\s:]+|edad\s*[:\s]+|âge\s*[:\s]+|idade\s*[:\s]+|miaka\s+)(\d{1,3})\b|\b(\d{1,3})\s*[MF]\b", note_text, re.IGNORECASE)
        age = None
        if age_match:
            age_str = next((g for g in age_match.groups() if g is not None), None)
            if age_str:
                age = int(age_str)
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

        # 9. Pregnancy & obstetric consistency (multilingual support)
        is_neonate = bool(re.search(r"\b(newborn|neonate|infant|nouveau-n[eé]|mtoto\s+mchanga|reci[eé]n\s+nacido)\b", note_text, re.IGNORECASE))
        if is_neonate:
            if chapter == "18" or code.startswith(("JA", "JB")):
                score -= 0.40
            elif chapter == "19" or code.startswith("KA") or "fetus or newborn" in title:
                score += 0.15

        is_pregnant = bool(re.search(r"\b(anc\b|antenatal|pregnant|pregnancy|gestation|trimester|fundal|fhr|edd|g\d+p\d+|\d+/\d+\s*weeks?|enceinte|grossesse|embarazada|embarazo|gravidez|gestante|ujauzito|mimba)\b", note_text, re.IGNORECASE))
        is_not_pregnant = bool(re.search(r"\b(upt\s*neg|pregnancy\s*test\s*neg|not\s*pregnant)\b", note_text, re.IGNORECASE))
        if is_not_pregnant or (age is not None and age >= 50 and not is_pregnant):
            if chapter == "18" or code.startswith(("JA", "JB")):
                score -= 0.40
        elif not is_pregnant:
            if chapter == "18" or code.startswith(("JA", "JB")):
                score -= 0.35
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
        for concept, eqs in CLINICAL_QUALIFIER_EQUIVALENTS.items():
            if concept in title or concept.replace("-", " ") in title:
                if any(re.search(r"\b" + re.escape(eq) + r"\b", note_lower) for eq in eqs + [concept]):
                    score += 0.05
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
