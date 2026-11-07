"""ICD-11 hierarchy navigation, sibling expansion, and specificity descent.
Ensures predictions resolve to valid leaf codes and appropriately leverage .Z and specific children.
Includes strict distinguishing qualifier matching and clinical anatomical synonym normalization.
"""

import json
import os
import re
from typing import Any

# Clinical and anatomical qualifiers and their equivalent expressions
CLINICAL_QUALIFIER_EQUIVALENTS: dict[str, list[str]] = {
    "distal": ["distal", "lower end", "inferior", "caudal"],
    "lower": ["distal", "lower end", "inferior", "lower"],
    "proximal": ["proximal", "upper end", "superior", "cranial"],
    "upper": ["proximal", "upper end", "superior", "upper"],
    "shaft": ["shaft", "diaphysis", "midshaft"],
    "anterior": ["anterior", "front", "ventral"],
    "posterior": ["posterior", "back", "dorsal"],
    "bilateral": ["bilateral", "both", "b/l"],
    "unilateral": ["unilateral", "one side", "left", "right", "single"],
    "acute": ["acute", "sudden", "recent"],
    "chronic": ["chronic", "longstanding", "persistent", "recurrent"],
    "open": ["open", "compound"],
    "closed": ["closed", "simple"],
    "primary": ["primary", "essential", "idiopathic"],
    "secondary": ["secondary"],
    "type 1": ["type 1", "type i", "t1dm", "iddm", "juvenile"],
    "type 2": ["type 2", "type ii", "t2dm", "niddm", "adult-onset"],
    "right": ["right", "rt", "r side"],
    "left": ["left", "lt", "l side"],
    "complication": ["complication", "complications"],
    "facial": ["facial", "face", "mandible", "jaw"],
}

STOP_WORDS = {
    "and", "the", "for", "with", "due", "other", "specified", "unspecified",
    "not", "elsewhere", "classified", "without", "part", "site", "condition",
}


class ICDHierarchy:
    """Navigate the ICD-11 category hierarchy and expand candidate sets."""

    def __init__(self, kb_path: str = "kb/enriched_kb.json"):
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Enriched KB not found at {kb_path}. Run build_kb.py first.")

        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb: dict[str, dict[str, Any]] = json.load(f)

    def get_info(self, code: str) -> dict[str, Any] | None:
        """Get enriched metadata for a code."""
        return self.kb.get(code)

    def is_leaf(self, code: str) -> bool:
        """Check if code is a leaf category."""
        info = self.kb.get(code)
        return bool(info and info.get("leaf", False))

    def get_parent(self, code: str) -> str | None:
        """Get parent code."""
        info = self.kb.get(code)
        return info.get("parent") if info else None

    def get_children(self, code: str) -> list[str]:
        """Get immediate children codes."""
        info = self.kb.get(code)
        return info.get("children", []) if info else []

    def get_siblings(self, code: str) -> list[str]:
        """Get sibling codes under same parent."""
        info = self.kb.get(code)
        return info.get("siblings", []) if info else []

    def get_z_code(self, code: str) -> str | None:
        """Get the unspecified (.Z) code for this category."""
        info = self.kb.get(code)
        if not info:
            return None
        # If code already ends in Z
        if code.endswith(".Z") or code.endswith("Z"):
            return code
        # If non-leaf, inspect own children for .Z FIRST
        for ch in info.get("children", []):
            if ch.endswith(".Z") or ch.endswith("Z"):
                return ch
        # Check explicit z_code in metadata (sibling .Z)
        if info.get("z_code"):
            return info["z_code"]
        # Inspect siblings for .Z
        for s in info.get("siblings", []):
            if s.endswith(".Z") or s.endswith("Z"):
                return s
        return None

    def get_subtree_z_code(self, code: str, max_depth: int = 4) -> str | None:
        """Recursively search descendants for an unspecified (.Z) leaf code."""
        if max_depth <= 0 or code not in self.kb:
            return None
        info = self.kb[code]
        # Check direct children first
        for ch in info.get("children", []):
            if (ch.endswith(".Z") or ch.endswith("Z")) and self.is_leaf(ch):
                return ch
        # Recurse down children
        for ch in info.get("children", []):
            leaf_z = self.get_subtree_z_code(ch, max_depth - 1)
            if leaf_z and self.is_leaf(leaf_z):
                return leaf_z
        return None

    def get_y_code(self, code: str) -> str | None:
        """Get other specified (.Y) code for this category."""
        info = self.kb.get(code)
        if not info:
            return None
        if code.endswith(".Y") or code.endswith("Y"):
            return code
        # If non-leaf, inspect own children for .Y FIRST
        for ch in info.get("children", []):
            if ch.endswith(".Y") or ch.endswith("Y"):
                return ch
        if info.get("y_code"):
            return info["y_code"]
        for s in info.get("siblings", []):
            if s.endswith(".Y") or s.endswith("Y"):
                return s
        return None

    def resolve_to_leaf(self, code: str, note_text: str = "") -> str:
        """If code is not a leaf, descend to the best specific child or fallback to .Z.

        Strictly requires distinguishing child qualifiers or synonyms to match in the text
        before descending to a specific child; otherwise falls back to the .Z unspecified leaf.
        """
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
                resolved_child = self.resolve_to_leaf(best_child, note_text)
                if self.is_leaf(resolved_child):
                    return resolved_child

        # Fallback 1: Direct .Z child leaf
        z_code = self.get_z_code(code)
        if z_code and self.is_leaf(z_code):
            return z_code
        elif z_code:
            # If z_code itself has children, resolve it to leaf
            resolved_z = self.resolve_to_leaf(z_code, note_text)
            if self.is_leaf(resolved_z):
                return resolved_z

        # Fallback 2: Subtree .Z leaf code
        subtree_z = self.get_subtree_z_code(code)
        if subtree_z:
            return subtree_z

        # Fallback 3: First leaf child (only when no .Z exists anywhere in subtree)
        for ch in children:
            if self.is_leaf(ch):
                return ch
            leaf_descendant = self.resolve_to_leaf(ch, note_text)
            if self.is_leaf(leaf_descendant):
                return leaf_descendant

        return code

    def match_specific_child(self, children: list[str], text: str) -> tuple[str | None, int]:
        """Score candidate children against specificity cues in text with strict word boundaries
        and anatomical/clinical qualifier normalization.
        """
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

            score = 0
            # 1. Exact phrase matching on enriched official WHO synonyms and inclusions
            synonyms = list(ch_info.get("synonyms", [])) + list(ch_info.get("inclusions", []))
            # Also include direct child synonyms if ch is not a leaf
            for sub in ch_info.get("children", []):
                sub_info = self.kb.get(sub)
                if sub_info:
                    synonyms.extend(sub_info.get("synonyms", []))
                    synonyms.extend(sub_info.get("inclusions", []))

            for syn in synonyms:
                syn_clean = syn.lower().strip()
                if len(syn_clean) >= 4:
                    # Check exact phrase with word boundaries
                    pattern = r"\b" + re.escape(syn_clean) + r"\b"
                    if re.search(pattern, text_lower):
                        score += 10
                        break

            # 2. Check "without complication" special handling
            if "without complication" in ch_info.get("title", "").lower():
                if any(phrase in text_lower for phrase in ["no complication", "no complications", "uncomplicated", "without complication"]):
                    score += 15

            # 2. Distinguishing qualifier tokens between child and parent
            title = ch_info.get("title", "").lower()
            words = set(re.findall(r"\b[a-z0-9]{3,}\b", title)) - STOP_WORDS
            if not words and score == 0:
                continue

            parent_code = ch_info.get("parent")
            parent_words = set()
            if parent_code and parent_code in self.kb:
                p_title = self.kb[parent_code].get("title", "").lower()
                parent_words = set(re.findall(r"\b[a-z0-9]{3,}\b", p_title))

            qualifiers = words - parent_words
            target_words = qualifiers if qualifiers else words

            # Match qualifiers with word boundaries & clinical synonyms
            qualifier_matches = 0
            for w in target_words:
                matched_w = False
                # Direct word boundary match
                if re.search(r"\b" + re.escape(w) + r"\b", text_lower):
                    matched_w = True
                else:
                    # Check clinical equivalents (e.g., distal <-> lower end)
                    eq_list = CLINICAL_QUALIFIER_EQUIVALENTS.get(w, [])
                    for eq in eq_list:
                        if re.search(r"\b" + re.escape(eq) + r"\b", text_lower):
                            matched_w = True
                            break

                if matched_w:
                    qualifier_matches += 1

            # Only count qualifier score if at least one distinguishing qualifier matched
            if qualifiers and qualifier_matches > 0:
                score += qualifier_matches * 2
            elif not qualifiers and qualifier_matches > 0:
                score += qualifier_matches

            if score > best_score:
                best_score = score
                best_child = ch

        return best_child, best_score

    def expand_candidates(self, ranked_candidates: list[tuple[str, float]], max_expansions: int = 40) -> list[tuple[str, float]]:
        """Expand a ranked list of (code, score) tuples with .Z siblings, children, and parent."""
        expanded: dict[str, float] = {}

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
