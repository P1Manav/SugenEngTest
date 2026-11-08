"""Build ICD-11 Exclusion Conflict Graph from WHO ICD-11 enrichments.
Parses all 2,181 exclusion sets in kb/who_enrichments.json and resolves them
to target ICD-11 codes and exclusion condition phrases.

At inference time, the exclusion graph operates as:
1. Candidate-to-Candidate exclusion: If Candidate A excludes Candidate B, and Candidate B
   is supported by the note, Candidate A is penalized/suppressed.
2. Note-to-Candidate exclusion: If specific exclusion phrases for Candidate A match
   extracted phrases in the clinical note, Candidate A is penalized.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any


def clean_text(s: str) -> str:
    """Normalize text for matching: lower, strip brackets and extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\[.*?\]|\(.*?\)", "", s)
    s = re.sub(r"\bnos\b", "", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    return " ".join(s.split())


def build_exclusion_graph(
    mms_path: str = "data/mms-2026-01.json",
    enrichments_path: str = "kb/who_enrichments.json",
    output_path: str = "kb/exclusion_graph.json",
):
    print(f"Loading enrichments from {enrichments_path}...")
    with open(enrichments_path, "r", encoding="utf-8") as f:
        enrichments = json.load(f)

    print(f"Loading MMS data from {mms_path}...")
    with open(mms_path, "r", encoding="utf-8") as f:
        mms = json.load(f)

    # 1. Build comprehensive lookup indices: title -> code, synonym -> list[code]
    title_to_code: dict[str, str] = {}
    synonym_to_codes: dict[str, set[str]] = defaultdict(set)
    valid_codes: set[str] = set()

    for code, item in enrichments.items():
        valid_codes.add(code)
        title = item.get("title", "")
        if title:
            ct = clean_text(title)
            if ct:
                title_to_code[ct] = code

        for syn in item.get("synonyms", []):
            cs = clean_text(syn)
            if cs and len(cs) >= 3:
                synonym_to_codes[cs].add(code)

    # Also incorporate entities from MMS
    for entity in mms:
        code = entity.get("code")
        if not code:
            continue
        valid_codes.add(code)
        title = entity.get("title", "")
        if title:
            ct = clean_text(title)
            if ct and ct not in title_to_code:
                title_to_code[ct] = code

    print(f"Indexed {len(valid_codes)} valid codes, {len(title_to_code)} unique titles, {len(synonym_to_codes)} synonyms.")

    # 2. Extract and resolve exclusions
    code_to_exclusions: dict[str, list[dict[str, Any]]] = {}
    reverse_exclusions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phrase_exclusions: dict[str, list[str]] = defaultdict(list)

    total_exclusion_entries = 0
    resolved_to_code = 0
    resolved_by_explicit_code = 0

    code_pattern = re.compile(r"\b([0-9][A-Z0-9]{2,}(?:\.[0-9A-Z]+)?)\b")

    for src_code, item in enrichments.items():
        raw_exclusions = item.get("exclusions", [])
        if not raw_exclusions:
            continue

        resolved_list = []
        clean_phrases = []

        for excl_raw in raw_exclusions:
            total_exclusion_entries += 1
            excl_str = excl_raw.strip()
            if not excl_str:
                continue

            target_code = None
            resolution_method = "none"

            # Method A: Check for explicit code in parentheses or text, e.g. "Asthma (CA23)"
            matches = code_pattern.findall(excl_str)
            for m in matches:
                if m in valid_codes and m != src_code:
                    target_code = m
                    resolution_method = "explicit_code"
                    resolved_by_explicit_code += 1
                    break

            # Method B: Exact title match
            ct = clean_text(excl_str)
            if ct and len(ct) >= 3:
                clean_phrases.append(ct)

            if not target_code and ct:
                if ct in title_to_code and title_to_code[ct] != src_code:
                    target_code = title_to_code[ct]
                    resolution_method = "exact_title"

            # Method C: Exact synonym match (if unambiguous)
            if not target_code and ct in synonym_to_codes:
                candidates = synonym_to_codes[ct] - {src_code}
                if len(candidates) == 1:
                    target_code = next(iter(candidates))
                    resolution_method = "unique_synonym"
                elif 1 < len(candidates) <= 3:
                    # Pick closest or primary
                    target_code = sorted(candidates, key=lambda c: len(c))[0]
                    resolution_method = "multi_synonym"

            if target_code:
                resolved_to_code += 1
                entry = {
                    "target_code": target_code,
                    "target_title": enrichments.get(target_code, {}).get("title", ""),
                    "raw_text": excl_str,
                    "method": resolution_method,
                }
                resolved_list.append(entry)
                reverse_exclusions[target_code].append({
                    "source_code": src_code,
                    "source_title": item.get("title", ""),
                    "raw_text": excl_str,
                })

        if resolved_list or clean_phrases:
            code_to_exclusions[src_code] = resolved_list
            phrase_exclusions[src_code] = list(dict.fromkeys(clean_phrases))

    stats = {
        "total_source_codes_with_exclusions": len(code_to_exclusions),
        "total_raw_exclusions": total_exclusion_entries,
        "total_resolved_to_target_code": resolved_to_code,
        "resolved_by_explicit_code": resolved_by_explicit_code,
        "total_unique_target_codes_excluded": len(reverse_exclusions),
    }

    print("\n--- Exclusion Graph Build Statistics ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    output_data = {
        "stats": stats,
        "code_to_exclusions": code_to_exclusions,
        "reverse_exclusions": dict(reverse_exclusions),
        "phrase_exclusions": dict(phrase_exclusions),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved exclusion graph to {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB).")


if __name__ == "__main__":
    build_exclusion_graph()

