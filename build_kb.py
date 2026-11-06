"""Build the enriched knowledge base from mms-2026-01.json and official WHO API enrichments.

Creates:
- kb/enriched_kb.json: Enriched dictionary of ICD-11 categories with hierarchy, WHO synonyms,
  inclusions, exclusions, definitions, search text, siblings, .Y/.Z codes
- kb/valid_codes.json: List of all valid normalized MMS codes
"""

import json
import os
import sys
from typing import Any

from text_utils import clean_text, normalize_code


def derive_parent_code(code: str, depth: int) -> str | None:
    """Derive parent code from code structure."""
    if depth == 1 or "." not in code:
        return None
    stem, suffix = code.split(".", 1)
    if depth == 2:
        return stem
    elif depth == 3:
        if len(suffix) >= 2:
            return f"{stem}.{suffix[0]}"
        return stem
    return None


def build_kb(mms_path: str, kb_dir: str):
    """Build enriched KB and save artifacts."""
    os.makedirs(kb_dir, exist_ok=True)

    print(f"Loading MMS data from {mms_path}...")
    with open(mms_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    print(f"Loaded {len(categories)} categories.")

    # Load WHO enrichments if available
    who_enrichments = {}
    who_path = os.path.join(kb_dir, "who_enrichments.json")
    if os.path.exists(who_path):
        try:
            with open(who_path, "r", encoding="utf-8") as f:
                who_enrichments = json.load(f)
            print(f"Loaded {len(who_enrichments)} WHO enrichments from {who_path}.")
        except Exception as e:
            print(f"Warning loading WHO enrichments: {e}")

    # 1. Map by normalized code
    code_map: dict[str, dict[str, Any]] = {}
    valid_codes = []

    for cat in categories:
        code = normalize_code(cat["code"])
        valid_codes.append(code)

        who_info = who_enrichments.get(code, {})

        code_map[code] = {
            "code": code,
            "title": cat.get("title", ""),
            "chapter": cat.get("chapter", ""),
            "leaf": bool(cat.get("leaf", False)),
            "residual": bool(cat.get("residual", False)),
            "depth": int(cat.get("depth", 1)),
            "uri": cat.get("uri", ""),
            "parent": None,
            "children": [],
            "siblings": [],
            "z_code": None,
            "y_code": None,
            "synonyms": who_info.get("synonyms", []),
            "inclusions": who_info.get("inclusions", []),
            "exclusions": who_info.get("exclusions", []),
            "definition": who_info.get("definition", ""),
            "search_text": "",
        }

    # 2. Derive parent-child relationships
    parent_to_children: dict[str, list] = {}
    for code, info in code_map.items():
        parent = derive_parent_code(code, info["depth"])
        if parent and parent in code_map:
            info["parent"] = parent
            parent_to_children.setdefault(parent, []).append(code)

    for parent, children in parent_to_children.items():
        code_map[parent]["children"] = sorted(children)

    # 3. Derive siblings and .Y / .Z links
    for code, info in code_map.items():
        parent = info["parent"]
        if parent and parent in code_map:
            siblings = [c for c in code_map[parent]["children"] if c != code]
            info["siblings"] = sorted(siblings)

            # Find .Z and .Y siblings
            for s in code_map[parent]["children"]:
                if s.endswith(".Z") or s.endswith("Z"):
                    info["z_code"] = s
                elif s.endswith(".Y") or s.endswith("Y"):
                    info["y_code"] = s
        else:
            # If depth 1, check children for .Z and .Y if non-leaf
            for ch in info["children"]:
                if ch.endswith(".Z") or ch.endswith("Z"):
                    info["z_code"] = ch
                elif ch.endswith(".Y") or ch.endswith("Y"):
                    info["y_code"] = ch

    # 4. Construct rich search text (ancestors + title + synonyms + inclusions + definition)
    for code, info in code_map.items():
        titles = [info["title"]]
        curr = info["parent"]
        while curr and curr in code_map:
            titles.append(code_map[curr]["title"])
            curr = code_map[curr]["parent"]
        titles.reverse()
        base_title = " - ".join(titles)

        search_parts = [base_title]
        syns = info.get("synonyms", [])
        if syns:
            search_parts.append(" ".join(syns[:12]))
        incs = info.get("inclusions", [])
        if incs:
            search_parts.append(" ".join(incs[:6]))
        definition = info.get("definition", "")
        if definition:
            search_parts.append(definition)

        full_search = " ".join(search_parts)
        info["search_text"] = clean_text(full_search)

    # 5. Save enriched_kb.json and valid_codes.json
    enriched_path = os.path.join(kb_dir, "enriched_kb.json")
    valid_path = os.path.join(kb_dir, "valid_codes.json")

    print(f"Saving enriched KB to {enriched_path}...")
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(code_map, f, ensure_ascii=False, indent=2)

    print(f"Saving valid codes to {valid_path}...")
    with open(valid_path, "w", encoding="utf-8") as f:
        json.dump(sorted(valid_codes), f, ensure_ascii=False, indent=2)

    print("KB build complete!")


if __name__ == "__main__":
    mms_candidates = [
        "data/mms-2026-01.json",
        "/data/mms-2026-01.json",
    ]
    mms_file = None
    for cand in mms_candidates:
        if os.path.exists(cand):
            mms_file = cand
            break

    if not mms_file and len(sys.argv) > 1:
        mms_file = sys.argv[1]

    if not mms_file:
        sys.exit("Error: Could not locate mms-2026-01.json")

    build_kb(mms_file, "kb")
