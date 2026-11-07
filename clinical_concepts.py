"""Clinical concept extraction and qualifier normalization from data-driven rules.
Provides concept-based qualifier extraction with zero hardcoded ICD codes.
"""

import os
import re
from typing import Any

_CONCEPTS_CACHE: dict[str, Any] | None = None


def load_concept_rules(rules_path: str = "rules/clinical_concepts.yaml") -> dict[str, Any]:
    """Load concept rules from YAML data table."""
    global _CONCEPTS_CACHE
    if _CONCEPTS_CACHE is not None:
        return _CONCEPTS_CACHE

    if not os.path.exists(rules_path):
        _CONCEPTS_CACHE = {}
        return _CONCEPTS_CACHE

    try:
        import yaml
        with open(rules_path, "r", encoding="utf-8") as f:
            _CONCEPTS_CACHE = yaml.safe_load(f) or {}
    except Exception:
        _CONCEPTS_CACHE = {}

    return _CONCEPTS_CACHE


def extract_clinical_qualifiers(text: str, rules_path: str = "rules/clinical_concepts.yaml") -> set[str]:
    """Extract clinical qualifier phrases from text based on concept rules table.
    Returns a set of normalized qualifier phrases for matching against WHO MMS entities.
    """
    concepts_data = load_concept_rules(rules_path)
    if not concepts_data or not text:
        return set()

    text_lower = text.lower()
    active_qualifiers: set[str] = set()

    # 1. Blood pressure phenotypes
    bp_match = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text)
    if bp_match:
        try:
            sys_val = int(bp_match.group(1))
            dia_val = int(bp_match.group(2))
            for rule in concepts_data.get("blood_pressure_phenotypes", {}).get("rules", []):
                sys_min = rule.get("systolic_min", 0)
                sys_max = rule.get("systolic_max", 999)
                dia_min = rule.get("diastolic_min", 0)
                dia_max = rule.get("diastolic_max", 999)
                if sys_min <= sys_val <= sys_max and dia_min <= dia_val <= dia_max:
                    for qt in rule.get("qualifier_terms", []):
                        active_qualifiers.add(qt.lower().strip())
        except (ValueError, TypeError):
            pass

    # 2. General concept categories
    for cat_name, cat_data in concepts_data.items():
        if cat_name == "blood_pressure_phenotypes":
            continue
        for rule in cat_data.get("rules", []):
            neg_patterns = rule.get("negative_patterns", [])
            if any(re.search(np, text_lower) for np in neg_patterns):
                continue

            patterns = rule.get("patterns", [])
            for p in patterns:
                if re.search(p, text_lower):
                    for qt in rule.get("qualifier_terms", []):
                        active_qualifiers.add(qt.lower().strip())
                    break

    return active_qualifiers

