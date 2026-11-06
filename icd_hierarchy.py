"""ICD-11 hierarchy navigation, sibling expansion, and specificity descent.
Ensures predictions resolve to valid leaf codes and appropriately leverage .Z and specific children.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set


class ICDHierarchy:
    """Navigate the ICD-11 category hierarchy and expand candidate sets."""

    def __init__(self, kb_path: str = "kb/enriched_kb.json"):
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Enriched KB not found at {kb_path}. Run build_kb.py first.")

        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb: Dict[str, Dict[str, Any]] = json.load(f)

    def get_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Get enriched metadata for a code."""
        return self.kb.get(code)

    def is_leaf(self, code: str) -> bool:
        """Check if code is a leaf category."""
        info = self.kb.get(code)
        return bool(info and info.get("leaf", False))

    def get_parent(self, code: str) -> Optional[str]:
        """Get parent code."""
        info = self.kb.get(code)
        return info.get("parent") if info else None

    def get_children(self, code: str) -> List[str]:
        """Get immediate children codes."""
        info = self.kb.get(code)
        return info.get("children", []) if info else []

    def get_siblings(self, code: str) -> List[str]:
        """Get sibling codes under same parent."""
        info = self.kb.get(code)
        return info.get("siblings", []) if info else []

    def get_z_code(self, code: str) -> Optional[str]:
        """Get the unspecified (.Z) code for this category."""
        info = self.kb.get(code)
        if not info:
            return None
        # If code already ends in Z
        if code.endswith(".Z") or code.endswith("Z"):
            return code
        # Check explicit z_code in metadata
        if info.get("z_code"):
            return info["z_code"]
        # If non-leaf, inspect children for .Z
        for ch in info.get("children", []):
            if ch.endswith(".Z") or ch.endswith("Z"):
                return ch
        # Inspect siblings for .Z
        for s in info.get("siblings", []):
            if s.endswith(".Z") or s.endswith("Z"):
                return s
        return None

    def get_y_code(self, code: str) -> Optional[str]:
        """Get other specified (.Y) code for this category."""
        info = self.kb.get(code)
        if not info:
            return None
        if code.endswith(".Y") or code.endswith("Y"):
            return code
        if info.get("y_code"):
            return info["y_code"]
        for ch in info.get("children", []):
            if ch.endswith(".Y") or ch.endswith("Y"):
                return ch
        for s in info.get("siblings", []):
            if s.endswith(".Y") or s.endswith("Y"):
                return s
        return None

    def resolve_to_leaf(self, code: str, note_text: str = "") -> str:
        """If code is not a leaf, descend to the best specific child or fallback to .Z."""
        info = self.kb.get(code)
        if not info:
            return code
        if info.get("leaf", False):
            return code

        # If note has text, try matching specific child
        children = info.get("children", [])
        if children and note_text:
            best_child, match_count = self.match_specific_child(children, note_text)
            if best_child and match_count > 0:
                # Recursively ensure child is a leaf
                return self.resolve_to_leaf(best_child, note_text)

        # Fallback to .Z child
        z_code = self.get_z_code(code)
        if z_code and self.is_leaf(z_code):
            return z_code

        # If no .Z leaf found, pick the first leaf child
        for ch in children:
            if self.is_leaf(ch):
                return ch
            # Or recurse into child
            leaf_descendant = self.resolve_to_leaf(ch, note_text)
            if self.is_leaf(leaf_descendant):
                return leaf_descendant

        return code

    def match_specific_child(self, children: List[str], text: str) -> Tuple[Optional[str], int]:
        """Score candidate children against specificity cues in text."""
        text_lower = text.lower()
        best_child = None
        best_score = 0

        for ch in children:
            ch_info = self.kb.get(ch)
            if not ch_info:
                continue
            # Don't pick residual (.Z / .Y) in specific child matching
            if ch_info.get("residual", False):
                continue

            title = ch_info.get("title", "").lower()
            words = set(re.findall(r"\b[a-z]{3,}\b", title)) - {"and", "the", "for", "with", "due", "other", "specified"}
            if not words:
                continue

            # Identify specific qualifier words not in parent title
            parent_code = ch_info.get("parent")
            parent_words = set()
            if parent_code and parent_code in self.kb:
                p_title = self.kb[parent_code].get("title", "").lower()
                parent_words = set(re.findall(r"\b[a-z]{3,}\b", p_title))

            qualifiers = words - parent_words
            target_words = qualifiers if qualifiers else words

            matches = sum(1 for w in target_words if w in text_lower)
            if matches > best_score:
                best_score = matches
                best_child = ch

        return best_child, best_score

    def expand_candidates(self, ranked_candidates: List[Tuple[str, float]], max_expansions: int = 40) -> List[Tuple[str, float]]:
        """Expand a ranked list of (code, score) tuples with .Z siblings, children, and parent."""
        expanded: Dict[str, float] = {}
        
        for rank, (code, score) in enumerate(ranked_candidates):
            if code not in self.kb:
                continue
            
            # Keep original code with its base score
            if code not in expanded or score > expanded[code]:
                expanded[code] = score

            info = self.kb[code]

            # 1. Add .Z code (crucial for ICD-11 unspecified fallback)
            z_code = self.get_z_code(code)
            if z_code and z_code in self.kb and z_code != code:
                # Score .Z slightly below or equal to base code
                z_score = score * 0.95
                if z_code not in expanded or z_score > expanded[z_code]:
                    expanded[z_code] = z_score

            # 2. Add children (more specific)
            children = info.get("children", [])
            for ch in children:
                ch_score = score * 0.90
                if ch not in expanded or ch_score > expanded[ch]:
                    expanded[ch] = ch_score

            # 3. Add parent (less specific context)
            parent = info.get("parent")
            if parent and parent in self.kb:
                p_score = score * 0.85
                if parent not in expanded or p_score > expanded[parent]:
                    expanded[parent] = p_score

            # 4. Add .Y sibling (other specified)
            y_code = self.get_y_code(code)
            if y_code and y_code in self.kb and y_code != code:
                y_score = score * 0.80
                if y_code not in expanded or y_score > expanded[y_code]:
                    expanded[y_code] = y_score

        # Sort all by score descending
        sorted_candidates = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:max_expansions]
