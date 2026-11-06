"""Candidate re-ranking and scoring.
Combines retrieval scores with lexical overlap, section presence, and specificity cues.
"""

import re
from typing import List, Tuple, Dict, Any, Optional

from text_utils import tokenize


class RuleReranker:
    """Rerank candidates using exact phrase matches, assessment section alignment, and cue bonuses."""

    def __init__(self, kb: Dict[str, Dict[str, Any]]):
        self.kb = kb

    def score_candidate(
        self,
        code: str,
        base_score: float,
        extracted_phrases: List[str],
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

        # 1. Exact phrase match or substring match bonus
        for phrase in extracted_phrases:
            phrase_lower = phrase.lower().strip()
            if not phrase_lower:
                continue
            # Either title in phrase or phrase in title
            if title in phrase_lower or phrase_lower in title:
                score += 0.04
                break

        # 2. Token overlap with extracted diagnostic phrases
        extracted_text = " ".join(extracted_phrases).lower()
        extracted_tokens = set(tokenize(extracted_text)) - {"other", "specified", "unspecified", "due", "with", "and", "the", "for", "mild", "patient", "alert"}
        overlap = len(title_tokens.intersection(extracted_tokens))
        if overlap > 0:
            recall = overlap / len(title_tokens)
            precision = overlap / len(extracted_tokens) if extracted_tokens else 0
            score += 0.03 * recall + 0.02 * precision

        # 3. Chapter 21 (Symptoms and signs) demotion when definitive disease is present
        chapter = info.get("chapter", "")
        if chapter == "21" or code.startswith("M"):
            score -= 0.025

        # 4. Prefer leaf codes over non-leaf
        if not info.get("leaf", False):
            score -= 0.05

        return score

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        extracted_phrases: List[str],
        note_text: str,
        top_k: int = 30,
    ) -> List[Tuple[str, float]]:
        """Rerank candidate list."""
        scored = []
        for code, base_score in candidates:
            final_score = self.score_candidate(code, base_score, extracted_phrases, note_text)
            scored.append((code, final_score))

        sorted_candidates = sorted(scored, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]
