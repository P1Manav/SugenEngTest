#!/usr/bin/env python3
"""Detailed error diagnostics and failure triage report for ICD-11 coding pipeline.

Categorizes errors into 5 specific diagnostic buckets:
1. Retrieval Miss: Gold diagnosis missing from top-30 retrieval candidates.
2. Reranker Demotion: Gold in top-30 retrieval, but pushed outside top-3.
3. Specificity Error: Correct parent/family, but wrong child (.Z vs specific).
4. Assertion Failure: Coded a negated finding, differential, or historical diagnosis.
5. Language Issue: Non-English text or multilingual normalization failure.

Also measures:
- Candidate Recall@10, @30, @50
- Breakdown by note style (typed_consult, short_consult, ambient_transcript)
- Breakdown by ICD-11 chapter
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import ICD11Pipeline
from text_utils import normalize_code


def load_mms_info(mms_path: str) -> tuple[dict[str, dict], dict[str, str]]:
    with open(mms_path, "r", encoding="utf-8") as f:
        categories = json.load(f)
    mms = {}
    code_to_chapter = {}
    for cat in categories:
        code = normalize_code(cat.get("code", ""))
        mms[code] = cat
        code_to_chapter[code] = cat.get("chapter", "Unknown")
    return mms, code_to_chapter


def categorize_failure(
    note: dict[str, Any],
    pred_codes: list[str],
    gold_codes: list[str],
    debug_info: dict[str, Any],
    mms: dict[str, dict],
) -> tuple[str, str]:
    """Classify why prediction failed to match gold."""
    primary_gold = gold_codes[0] if gold_codes else ""
    raw_cands = [normalize_code(c[0]) for c in debug_info.get("raw_candidates", [])]
    resolved_cands = [normalize_code(c[0]) for c in debug_info.get("resolved_candidates", [])]
    lang = debug_info.get("lang", "en")
    rejected_clauses = debug_info.get("rejected_clauses", [])

    pred_top1 = pred_codes[0] if pred_codes else ""
    pred_info = mms.get(pred_top1, {})
    pred_title = pred_info.get("title", "").lower()
    gold_info = mms.get(primary_gold, {})
    gold_title = gold_info.get("title", "")

    # Check Category 5: Language issue
    if lang != "en" and any(ord(char) > 127 for char in note.get("text", "")):
        if primary_gold not in raw_cands[:30]:
            return "Cat 5 (Language Issue)", f"Detected language '{lang}'; gold {primary_gold} missed in retrieval"

    # Check Category 4: Failed assertion
    # Did we predict a diagnosis that appears in rejected_clauses or matches negated text?
    note_text_lower = note.get("text", "").lower()
    for cl in rejected_clauses:
        cl_clean = cl.lower()
        if pred_title and (pred_title in cl_clean or cl_clean in pred_title):
            return "Cat 4 (Assertion Failure)", f"Predicted '{pred_top1}' ({pred_title}) which was in negated/rejected clause '{cl}'"
    # Check common negation markers near predicted title
    neg_pattern = r"\b(?:no|denies|negative|ruled out|unlikely|without|free of|screened for)\s+([a-z\s]{3,30})"
    for m in re.finditer(neg_pattern, note_text_lower):
        neg_phrase = m.group(0)
        if any(w in neg_phrase for w in pred_title.split() if len(w) > 4):
            return "Cat 4 (Assertion Failure)", f"Predicted '{pred_top1}' overlaps negated context '{neg_phrase}'"

    # Check Category 3: Wrong specificity / family
    # Did we get the right 3-character stem or parent, but wrong child?
    gold_stem = primary_gold[:3] if len(primary_gold) >= 3 else primary_gold
    pred_stem = pred_top1[:3] if len(pred_top1) >= 3 else pred_top1
    if gold_stem == pred_stem:
        if pred_top1.endswith(".Z") and not primary_gold.endswith(".Z"):
            return "Cat 3 (Specificity Error)", f"Chose unspecified (.Z) {pred_top1} when specific {primary_gold} was gold"
        elif not pred_top1.endswith(".Z") and primary_gold.endswith(".Z"):
            return "Cat 3 (Specificity Error)", f"Chose specific {pred_top1} when unspecified (.Z) {primary_gold} was gold"
        else:
            return "Cat 3 (Specificity Error)", f"Chose wrong specific sibling {pred_top1} instead of {primary_gold}"

    # Check Category 1: Retrieval missed
    if primary_gold not in raw_cands[:30] and primary_gold not in resolved_cands[:30]:
        return "Cat 1 (Retrieval Miss)", f"Gold {primary_gold} ({gold_title}) not in top-30 retrieval candidates"

    # Check Category 2: Reranker demoted
    # Gold was in retrieval top 30, but not in top 3 predictions
    if primary_gold in raw_cands[:30] or primary_gold in resolved_cands[:30]:
        rank_in_retrieval = None
        for r_idx, c in enumerate(resolved_cands[:30], 1):
            if c == primary_gold:
                rank_in_retrieval = r_idx
                break
        return "Cat 2 (Reranker Demotion)", f"Gold {primary_gold} was rank {rank_in_retrieval} in retrieval, but reranker placed it outside top 3"

    return "Cat 1 (Retrieval Miss)", f"Gold {primary_gold} not identified by retrieval or resolution"


def run_failure_report(
    gold_path: str = "data/dev_notes.jsonl",
    mms_path: str = "data/mms-2026-01.json",
    out_report_path: str | None = None,
) -> str:
    mms, code_to_chapter = load_mms_info(mms_path)

    with open(gold_path, "r", encoding="utf-8") as f:
        notes = [json.loads(line) for line in f if line.strip()]

    pipeline = ICD11Pipeline(verbose=False)

    total_notes = len(notes)
    top1_correct = 0
    top3_correct = 0

    recall_at_10 = 0
    recall_at_30 = 0
    recall_at_50 = 0

    category_counts = defaultdict(int)
    style_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})
    chapter_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})

    triage_rows = []

    for note in notes:
        note_id = note["id"]
        gold_raw = note.get("gold", note.get("codes", []))
        gold_codes = [normalize_code(c) for c in gold_raw if c]
        style = note.get("style", "unknown")
        style_stats[style]["total"] += 1

        primary_gold = gold_codes[0] if gold_codes else ""
        chapter = code_to_chapter.get(primary_gold, "Unknown")
        chapter_stats[chapter]["total"] += 1

        pred_res = pipeline.predict_note(note, debug=True)
        pred_codes = [normalize_code(c) for c in pred_res.get("codes", [])]
        debug = pred_res.get("debug", {})

        is_top1 = bool(pred_codes and primary_gold and pred_codes[0] == primary_gold)
        is_top3 = any(c in gold_codes for c in pred_codes[:3])

        if is_top1:
            top1_correct += 1
            style_stats[style]["top1"] += 1
            chapter_stats[chapter]["top1"] += 1
        if is_top3:
            top3_correct += 1
            style_stats[style]["top3"] += 1
            chapter_stats[chapter]["top3"] += 1

        # Check candidate recall
        cands = [normalize_code(c[0]) for c in debug.get("raw_candidates", [])]
        resolved = [normalize_code(c[0]) for c in debug.get("resolved_candidates", [])]
        combined_cands = []
        for c in resolved + cands:
            if c not in combined_cands:
                combined_cands.append(c)

        if any(g in combined_cands[:10] for g in gold_codes):
            recall_at_10 += 1
        if any(g in combined_cands[:30] for g in gold_codes):
            recall_at_30 += 1
        if any(g in combined_cands[:50] for g in gold_codes):
            recall_at_50 += 1

        if not is_top1:
            cat, reason = categorize_failure(note, pred_codes, gold_codes, debug, mms)
            category_counts[cat] += 1
            text_snip = note.get("text", "").replace("\n", " ")[:65]
            triage_rows.append({
                "id": note_id,
                "style": style,
                "chapter": chapter,
                "gold": gold_codes,
                "pred": pred_codes,
                "cat": cat,
                "reason": reason,
                "snip": text_snip,
            })

    score_75 = (50.0 * top1_correct + 25.0 * top3_correct) / total_notes if total_notes else 0.0

    lines = []
    lines.append("=" * 80)
    lines.append("                ICD-11 PIPELINE FAILURE & DIAGNOSTICS REPORT")
    lines.append("=" * 80)
    lines.append(f"Notes Evaluated:    {total_notes}")
    lines.append(f"Top-1 Accuracy:     {top1_correct}/{total_notes} ({top1_correct / total_notes * 100:.2f}%)")
    lines.append(f"Top-3 Accuracy:     {top3_correct}/{total_notes} ({top3_correct / total_notes * 100:.2f}%)")
    lines.append(f"Official Grader:    {score_75:.2f} / 75.0  (Normalized: {score_75 * 100 / 75:.2f} / 100)")
    lines.append("-" * 80)
    lines.append("CANDIDATE RECALL ON GOLD TARGETS:")
    lines.append(f"  Recall@10:        {recall_at_10}/{total_notes} ({recall_at_10 / total_notes * 100:.2f}%)")
    lines.append(f"  Recall@30:        {recall_at_30}/{total_notes} ({recall_at_30 / total_notes * 100:.2f}%)")
    lines.append(f"  Recall@50:        {recall_at_50}/{total_notes} ({recall_at_50 / total_notes * 100:.2f}%)")
    lines.append("-" * 80)
    lines.append("FAILURE CATEGORIZATION TRIAGE (Non-Top-1 Notes):")
    for cat in sorted(category_counts.keys()):
        cnt = category_counts[cat]
        lines.append(f"  {cat:<30} : {cnt:>2} notes ({cnt / total_notes * 100:>5.1f}%)")
    lines.append("-" * 80)
    lines.append("PERFORMANCE BY CLINICAL STYLE:")
    for style, s_data in sorted(style_stats.items()):
        tot = s_data["total"]
        t1 = s_data["top1"]
        t3 = s_data["top3"]
        lines.append(f"  {style:<22} : n={tot:>2} | Top-1: {t1:>2}/{tot:>2} ({t1/tot*100:5.1f}%) | Top-3: {t3:>2}/{tot:>2} ({t3/tot*100:5.1f}%)")
    lines.append("-" * 80)
    lines.append("PERFORMANCE BY ICD-11 CHAPTER:")
    for ch, c_data in sorted(chapter_stats.items()):
        tot = c_data["total"]
        t1 = c_data["top1"]
        t3 = c_data["top3"]
        lines.append(f"  Chapter {ch:<14} : n={tot:>2} | Top-1: {t1:>2}/{tot:>2} ({t1/tot*100:5.1f}%) | Top-3: {t3:>2}/{tot:>2} ({t3/tot*100:5.1f}%)")
    lines.append("=" * 80)
    lines.append("DETAILED PER-NOTE FAILURE TRIAGE:")
    lines.append(f"{'Note ID':<10} | {'Style':<16} | {'Gold':<12} | {'Predicted':<12} | {'Category':<24} | {'Reason'}")
    lines.append("-" * 110)
    for r in triage_rows:
        gold_str = ",".join(r["gold"])
        pred_str = ",".join(r["pred"])
        lines.append(f"{r['id']:<10} | {r['style']:<16} | {gold_str:<12} | {pred_str:<12} | {r['cat']:<24} | {r['reason']}")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    if out_report_path:
        os.makedirs(os.path.dirname(out_report_path) or ".", exist_ok=True)
        with open(out_report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"Report written to {out_report_path}")

    return report_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run failure triage and diagnostics report.")
    parser.add_argument("--gold", default="data/dev_notes.jsonl", help="Path to gold notes JSONL")
    parser.add_argument("--mms", default="data/mms-2026-01.json", help="Path to MMS categories JSON")
    parser.add_argument("--out", default="docs/baseline_failure_report.txt", help="Path to write report output")
    args = parser.parse_args()

    report = run_failure_report(gold_path=args.gold, mms_path=args.mms, out_report_path=args.out)
    print(report)
