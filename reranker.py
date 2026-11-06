"""Candidate re-ranking and scoring.
Combines retrieval scores with lexical overlap, section presence, and specificity cues.
"""

import re
from typing import Any

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
        title_tokens = set(tokenize(title)) - {"other", "specified", "unspecified", "due", "with", "and", "the", "for"}
        if not title_tokens:
            return base_score

        score = base_score

        # 1. Exact phrase match or substring match bonus (title and WHO synonyms)
        note_lower = note_text.lower()
        extracted_text = " ".join(extracted_phrases).lower()
        extracted_tokens = set(tokenize(extracted_text)) - {"other", "specified", "unspecified", "due", "with", "and", "the", "for", "mild", "patient", "alert"}

        matched_synonym = False
        matched_specific = False

        for phrase in extracted_phrases:
            phrase_lower = phrase.lower().strip()
            if not phrase_lower:
                continue
            if title in phrase_lower or phrase_lower in title:
                score += 0.05
                matched_synonym = True
                if not (code.endswith(".Z") or code.endswith("Z") or "unspecified" in title):
                    matched_specific = True
                break

        # Check official WHO synonyms and inclusions if title didn't match directly
        if not matched_synonym:
            for syn in info.get("synonyms", []) + info.get("inclusions", []):
                syn_lower = syn.lower().strip()
                if len(syn_lower) < 4:
                    continue

                # Extract aliases from brackets [xyz] or dash separations
                bracketed = re.findall(r"\[(.*?)\]|\((.*?)\)", syn_lower)
                aliases = [syn_lower]
                for b in bracketed:
                    for item in b:
                        if item and len(item.strip()) >= 4:
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
                    if syn_hit:
                        break

                if syn_hit:
                    score += 0.055
                    matched_synonym = True
                    if not (code.endswith(".Z") or code.endswith("Z") or "unspecified" in title):
                        matched_specific = True
                    break

        # 2. Token overlap with extracted diagnostic phrases
        overlap = len(title_tokens.intersection(extracted_tokens))
        if overlap > 0:
            recall = overlap / len(title_tokens)
            precision = overlap / len(extracted_tokens) if extracted_tokens else 0
            score += 0.03 * recall + 0.02 * precision

        # 3. Specificity bonus: specific leaf whose qualifiers / synonyms matched the clinical note
        is_unspecified = (code.endswith(".Z") or code.endswith("Z") or "unspecified" in title)
        if not is_unspecified and matched_specific:
            score += 0.04
        elif is_unspecified and not matched_synonym:
            # Demote unspecified if note had specific diagnostic phrases
            score -= 0.01

        # 4. Chapter 21 (Symptoms and signs) demotion when definitive disease is present
        chapter = info.get("chapter", "")
        if chapter == "21" or code.startswith("M"):
            score -= 0.025

        # 5. Demote .Y (Other specified) codes relative to .Z (Unspecified)
        if code.endswith(".Y") or code.endswith("Y"):
            score -= 0.015

        # 5. Demote critical clinical mismatch (e.g. abscess when not mentioned)
        critical_mismatch = ["abscess", "fracture", "neoplasm", "carcinoma", "malignant", "congenital"]
        for term in critical_mismatch:
            if term in title and term not in note_lower:
                score -= 0.035

        # 6. Uncomplicated / complicated clinical alignment
        if any(uncomp in note_lower for uncomp in ["uncomplicated", "no complication", "no complications", "without complication"]):
            if "without complication" in title:
                score += 0.055
            elif code.endswith(".Z") or code.endswith("Z"):
                score += 0.02
            if "with complication" in title or "cerebral" in title:
                score -= 0.06

        # 7. Bacteriologically confirmed alignment
        if "confirmed" in note_lower and "confirmed" in title:
            score += 0.04

        # 8. Prefer leaf codes over non-leaf
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
