"""Assertion filter adapted for clinical notes (including East African outpatient records).
Handles Negation, History, Comorbidity, Ruled Out, Family History, and Previous Misdiagnosis.
"""

import re
from typing import Dict, List, Optional, Tuple

# Pre-negation triggers (negate following terms within scope)
PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bneg\b",
    r"\bnegative\b",
    r"\bdenied\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\brule\s+out\b",
    r"\bruled\s+out\b",
    r"\br/o\b",
    r"\bexclude\b",
    r"\bexcluded\b",
    r"\bfree\s+of\b",
    r"\bunlikely\b",
    r"\bnever\b",
    r"\bno\s+evidence\s+of\b",
    r"\bno\s+features\s+of\b",
    r"\bno\s+signs?\s+of\b",
    r"\bno\s+history\s+of\b",
    r"\bno\s+h/o\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bzero\b",
]

# Post-negation triggers (negate preceding terms within scope)
POST_NEGATION_TRIGGERS = [
    r"\bneg\b",
    r"\bnegative\b",
    r"\bexcluded\b",
    r"\bruled\s+out\b",
    r"\bwas\s+ruled\s+out\b",
    r"\bnot\s+seen\b",
    r"\bunlikely\b",
    r"\babsent\b",
    r"\bnormal\b",
    r"\bwnl\b",
    r"\bwithin\s+normal\s+limits\b",
    r"\bunremarkable\b",
    r"\bnon-?tender\b",
    r"\bni\s+ya\s+kawaida\b",
    r"\bya\s+kawaida\b",
    r"\bnormale?s?\b",
]

# Ruled out / query triggers (differential / uncertainty)
RULED_OUT_TRIGGERS = [
    r"\?",
    r"\bquery\b",
    r"\bpossible\b",
    r"\brule\s+out\b",
    r"\br/o\b",
    r"\bexclude\b",
    r"\bto\s+exclude\b",
    r"\bdifferential\b",
    r"\bddx\b",
]

# Comorbidity triggers (background diseases, not acute reason for visit)
COMORBIDITY_TRIGGERS = [
    r"\bknown\b",
    r"\bknown\s+case\s+of\b",
    r"\bbackground\s+of\b",
    r"\bunderlying\b",
    r"\bco-morbidity\b",
    r"\bcomorbidity\b",
    r"\balso\s+known\b",
    r"\balso\s+has\b",
    r"\bon\s+(?:amlodipine|metformin|insulin|nifedipine|losartan|enalapril|atorvastatin)\b",
]

# History triggers (past events / distant history)
HISTORY_TRIGGERS = [
    r"\bhistory\s+of\b",
    r"\bh/o\b",
    r"\bprevious\b",
    r"\bpast\b",
    r"\bhx\b",
    r"\bpast\s+medical\s+history\b",
    r"\bpmhx\b",
    r"\bpast\s+hx\b",
    r"\blast\s+antimalarial\b",
    r"\bfinished\s+\w+\s+last\s+(?:wk|week|month)\b",
]

# Family history triggers
FAMILY_TRIGGERS = [
    r"\bfamily\s+history\b",
    r"\bfhx\b",
    r"\bmother\b",
    r"\bfather\b",
    r"\bbrother\b",
    r"\bsister\b",
    r"\bparent\b",
    r"\bsibling\b",
    r"\bcousin\b",
]

# Earlier misdiagnosis triggers
MISDIAGNOSIS_TRIGGERS = [
    r"\bwas\s+treated\s+for\b.*?\bwithout\s+improvement\b",
    r"\btreated\s+twice\s+for\b.*?\bwithout\s+relief\b",
    r"\btreated\s+for\b.*?\bwithout\s+(?:relief|improvement|resolution)\b",
    r"\bprevious\s+[\"']?\w+[\"']?\s+was\s+likely\s+misdiagnosis\b",
    r"\bmisdiagnosis\b",
    r"\bwrongly\s+treated\b",
    r"\binitial\s+diagnosis\s+of\b.*?\brevised\b",
]

RE_PRE_NEG = re.compile(r"|".join(PRE_NEGATION_TRIGGERS), re.IGNORECASE)
RE_POST_NEG = re.compile(r"|".join(POST_NEGATION_TRIGGERS), re.IGNORECASE)
RE_RULED_OUT = re.compile(r"|".join(RULED_OUT_TRIGGERS), re.IGNORECASE)
RE_COMORBIDITY = re.compile(r"|".join(COMORBIDITY_TRIGGERS), re.IGNORECASE)
RE_HISTORY = re.compile(r"|".join(HISTORY_TRIGGERS), re.IGNORECASE)
RE_FAMILY = re.compile(r"|".join(FAMILY_TRIGGERS), re.IGNORECASE)
RE_MISDIAG = re.compile(r"|".join(MISDIAGNOSIS_TRIGGERS), re.IGNORECASE)

# Termination cues for negation scope
CONJUNCTIONS = re.compile(r"\b(but|however|although|except|whereas|save\s+for|presented\s+with|imp:|impression:|a:|assessment:|dx:|diagnosis:)\b", re.IGNORECASE)


def split_sentences_and_clauses(text: str) -> list[str]:
    """Split text into sentences, short clause fragments, parentheticals, and question-mark queries."""
    # Split by periods, semicolons, colons, newlines, tabs, commas, parentheses, and question marks
    # Parentheticals like "(no alarm features)" and queries like "?Fe def" should be separate subclauses
    clauses = re.split(r"[\n\r;.]+|,\s*|\s*[\(\)]\s*|\s*\?\s*", text)
    cleaned = []
    for c in clauses:
        c = c.strip()
        if c:
            cleaned.append(c)
    return cleaned


NON_DIAGNOSTIC_TERMS = {
    "acute", "chronic", "other", "specified", "unspecified", "without",
    "with", "primary", "secondary", "disease", "disorder", "infection",
    "syndrome", "type", "part", "site", "mild", "moderate", "severe",
    "stage", "level", "state", "general", "generalised", "generalized",
    "mention", "due", "post", "status", "recurrent", "persistent",
    "pregnancy", "pregnant", "childbirth", "puerperium", "maternal",
    "infant", "child", "adult", "adults", "onset", "arising", "developing",
    "associated", "related",
}


SYMPTOM_OR_LAB_TERMS = {
    "blood", "pain", "fever", "cough", "headache", "vomiting", "diarrhea", "diarrhoea",
    "nausea", "swelling", "edema", "rash", "discharge", "sugar", "protein", "glucose",
    "cells", "fluid", "urine", "stool", "weight", "loss", "vision", "sound", "breath",
}


class AssertionFilter:
    """Classify clinical statements and candidate diagnoses by assertion status."""

    def __init__(self):
        pass

    def check_clause_assertion(self, clause: str) -> Dict[str, bool]:
        """Check status of a single clause."""
        # In clinical grammar, 'X without Y' (e.g. 'migraine without aura', 'asthma without complication')
        # affirms disease X while specifying the absence of complication Y.
        # Strip trailing 'without ...' to evaluate assertion of the primary condition X.
        clean_clause = re.sub(r"\bwithout\b.*$", "", clause, flags=re.IGNORECASE).strip()
        if not clean_clause:
            clean_clause = clause

        is_misdiag = bool(RE_MISDIAG.search(clean_clause))
        is_family = bool(RE_FAMILY.search(clean_clause))
        is_comorbidity = bool(RE_COMORBIDITY.search(clean_clause))
        is_history = bool(RE_HISTORY.search(clean_clause))
        is_ruled_out = bool(RE_RULED_OUT.search(clean_clause))
        is_negated = bool(RE_PRE_NEG.search(clean_clause) or RE_POST_NEG.search(clean_clause))

        return {
            "negated": is_negated,
            "ruled_out": is_ruled_out,
            "comorbidity": is_comorbidity,
            "history": is_history,
            "family": is_family,
            "misdiagnosis": is_misdiag,
            "affirmed": not (is_negated or is_ruled_out or is_misdiag or is_family),
        }

    def filter_text_for_retrieval(self, text: str) -> Tuple[str, List[str]]:
        """Filter out negated and misdiagnosed clauses from text before retrieval.
        Returns:
            clean_text: text with negated/misdiagnosed portions removed or suppressed.
            rejected_phrases: phrases that were negated or marked as misdiagnosis.
        """
        clauses = split_sentences_and_clauses(text)
        accepted_clauses = []
        rejected_phrases = []

        for clause in clauses:
            assertion = self.check_clause_assertion(clause)
            if assertion["negated"] or assertion["misdiagnosis"]:
                rejected_phrases.append(clause)
            elif assertion["family"]:
                # Exclude family history from primary diagnostic query
                rejected_phrases.append(clause)
            else:
                accepted_clauses.append(clause)

        filtered_text = ". ".join(accepted_clauses)
        return filtered_text, rejected_phrases

    def is_candidate_negated_in_note(self, candidate_title: str, note_text: str, affirmed_phrases: Optional[List[str]] = None) -> bool:
        """Check if a specific disease title is explicitly negated or marked as misdiagnosed in the note.
        Protects candidates if core terms appear in affirmed diagnosis phrases.
        Incidental lab/symptom negatives (e.g. 'stool blood negative') never negate systemic diseases."""
        cand_lower = candidate_title.lower()
        substantive_words = [w for w in re.split(r"\W+", cand_lower) if len(w) > 3 and w not in NON_DIAGNOSTIC_TERMS]
        words = substantive_words if substantive_words else [w for w in re.split(r"\W+", cand_lower) if len(w) > 3]
        if not words:
            return False

        # If any primary substantive disease term appears in affirmed diagnosis phrases, candidate is protected
        if affirmed_phrases:
            for phrase in affirmed_phrases:
                p_lower = phrase.lower()
                affirmed_matches = [w for w in words if len(w) >= 4 and re.search(r"\b" + re.escape(w) + r"\b", p_lower)]
                # If at least 2 words or 40% of substantive words are affirmed, protect candidate
                if len(affirmed_matches) >= 2 or (len(words) <= 2 and len(affirmed_matches) >= 1):
                    return False

        clauses = split_sentences_and_clauses(note_text)
        for clause in clauses:
            clause_lower = clause.lower()
            if "vaccinated" in clause_lower or "immuniz" in clause_lower or "immunis" in clause_lower:
                continue

            matched = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", clause_lower)]
            if not matched:
                continue

            # Core fix: Incidental symptom/finding negation (e.g. 'blood negative', 'no cough')
            # must not negate a disease title if the disease's primary pathology is not matched
            non_symptom_words = [w for w in words if w not in SYMPTOM_OR_LAB_TERMS]
            if non_symptom_words:
                matched_non_symptoms = [w for w in matched if w not in SYMPTOM_OR_LAB_TERMS]
                if not matched_non_symptoms:
                    # Only symptom words matched in negated clause (e.g. 'blood' in 'stool blood negative')
                    continue

            if "without" in clause_lower:
                without_pos = clause_lower.find("without")
                all_before = all(re.search(r"\b" + re.escape(w) + r"\b", clause_lower[:without_pos]) for w in matched)
                if all_before:
                    continue

            assertion = self.check_clause_assertion(clause)
            if assertion["negated"] or assertion["misdiagnosis"]:
                return True
        return False
