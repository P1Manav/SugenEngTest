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


def split_sentences_and_clauses(text: str) -> List[str]:
    """Split text into sentences and short clause fragments based on punctuation."""
    # Split by periods, commas, semicolons, colons, newlines, tabs
    clauses = re.split(r"[\n\r;.]+|,\s*", text)
    cleaned = []
    for c in clauses:
        c = c.strip()
        if c:
            cleaned.append(c)
    return cleaned


class AssertionFilter:
    """Classify clinical statements and candidate diagnoses by assertion status."""

    def __init__(self):
        pass

    def check_clause_assertion(self, clause: str) -> Dict[str, bool]:
        """Check status of a single clause."""
        is_misdiag = bool(RE_MISDIAG.search(clause))
        is_family = bool(RE_FAMILY.search(clause))
        is_comorbidity = bool(RE_COMORBIDITY.search(clause))
        is_history = bool(RE_HISTORY.search(clause))
        is_ruled_out = bool(RE_RULED_OUT.search(clause))
        is_negated = bool(RE_PRE_NEG.search(clause) or RE_POST_NEG.search(clause))

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
        Never negates if the candidate title appears in the affirmed diagnosis phrases."""
        cand_lower = candidate_title.lower()
        words = [w for w in re.split(r"\W+", cand_lower) if len(w) > 3]
        if not words:
            return False

        # If candidate appears in affirmed diagnosis phrases, it cannot be negated
        if affirmed_phrases:
            for phrase in affirmed_phrases:
                p_lower = phrase.lower()
                if any(w in p_lower for w in words):
                    return False

        clauses = split_sentences_and_clauses(note_text)
        for clause in clauses:
            clause_lower = clause.lower()
            # Ignore vaccination history (e.g. "not vaccinated vs mumps")
            if "vaccinated" in clause_lower or "immuniz" in clause_lower or "immunis" in clause_lower:
                continue

            # If at least one key word of disease appears in this clause
            if any(w in clause_lower for w in words):
                assertion = self.check_clause_assertion(clause)
                if assertion["negated"] or assertion["misdiagnosis"]:
                    return True
        return False
