#!/usr/bin/env python3
"""Unified evaluation harness for ICD-11 coding engine.

Calculates:
1. Official score (score.py rules) + 1,000-sample bootstrap 95% CIs.
2. Candidate Recall@10, @30, @50 broken down:
   - Overall
   - Per language
   - Per ICD-11 chapter
3. Error triage categorization (Retrieval Miss, Reranker Demotion, Specificity, Assertion, Language).
4. Runtime metrics and 600-note extrapolation.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import ICD11Pipeline
from text_utils import normalize_code
from multilingual import detect_language


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_mms_info(mms_path: str) -> tuple[dict[str, dict], dict[str, str], set[str]]:
    with open(mms_path, "r", encoding="utf-8") as f:
        categories = json.load(f)
    mms = {}
    code_to_chapter = {}
    valid_codes = set()
    for cat in categories:
        code = normalize_code(cat.get("code", ""))
        mms[code] = cat
        code_to_chapter[code] = cat.get("chapter", "Unknown")
        valid_codes.add(code)
    return mms, code_to_chapter, valid_codes


def compute_official_score(results: list[dict[str, Any]]) -> dict[str, float]:
    stats = {}
    for r in results:
        s = r.get("slice", "A")
        st = stats.setdefault(s, {"n": 0, "top1": 0, "top3": 0})
        st["n"] += 1
        st["top1"] += int(r["top1"])
        st["top3"] += int(r["top3"])

    weights = {"A": 75.0, "B": 25.0}
    total_score = 0.0
    present_weight = 0.0

    slice_metrics = {}
    for s, st in stats.items():
        n = st["n"]
        t1 = st["top1"] / n if n > 0 else 0.0
        t3 = st["top3"] / n if n > 0 else 0.0
        w = weights.get(s, 0.0)
        pts = w * (2.0 / 3.0 * t1 + 1.0 / 3.0 * t3)
        total_score += pts
        present_weight += w
        slice_metrics[s] = {"top1": t1, "top3": t3, "points": pts, "max_points": w, "n": n}

    norm_score = total_score * (100.0 / present_weight) if present_weight > 0 else 0.0
    tot_notes = len(results)

    return {
        "score": total_score,
        "present_weight": present_weight,
        "normalized_score": norm_score,
        "slices": slice_metrics,
        "top1_acc": sum(st["top1"] for st in stats.values()) / tot_notes if tot_notes else 0.0,
        "top3_acc": sum(st["top3"] for st in stats.values()) / tot_notes if tot_notes else 0.0,
    }


def bootstrap_ci(results: list[dict[str, Any]], n_resamples: int = 1000, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    n = len(results)
    if n == 0:
        return {}

    b_scores = []
    b_top1s = []
    b_top3s = []

    rng = random.Random(42)
    for _ in range(n_resamples):
        sample = [results[rng.randint(0, n - 1)] for _ in range(n)]
        sc = compute_official_score(sample)
        b_scores.append(sc["score"])
        b_top1s.append(sc["top1_acc"])
        b_top3s.append(sc["top3_acc"])

    low_p = (alpha / 2.0) * 100
    high_p = (1.0 - alpha / 2.0) * 100

    return {
        "score_ci": (float(np.percentile(b_scores, low_p)), float(np.percentile(b_scores, high_p))),
        "top1_ci": (float(np.percentile(b_top1s, low_p)), float(np.percentile(b_top1s, high_p))),
        "top3_ci": (float(np.percentile(b_top3s, low_p)), float(np.percentile(b_top3s, high_p))),
    }


def categorize_failure(
    note: dict[str, Any],
    pred_codes: list[str],
    gold_codes: list[str],
    debug_info: dict[str, Any],
    mms: dict[str, dict],
) -> tuple[str, str]:
    primary_gold = gold_codes[0] if gold_codes else ""
    raw_cands = [normalize_code(c[0]) for c in debug_info.get("raw_candidates", [])]
    resolved_cands = [normalize_code(c[0]) for c in debug_info.get("resolved_candidates", [])]
    lang = debug_info.get("lang", note.get("language", "en"))
    rejected_clauses = debug_info.get("rejected_clauses", [])

    pred_top1 = pred_codes[0] if pred_codes else ""
    pred_info = mms.get(pred_top1, {})
    pred_title = pred_info.get("title", "").lower()
    gold_info = mms.get(primary_gold, {})
    gold_title = gold_info.get("title", "")

    # 1. Language issue
    if lang != "en" and any(ord(char) > 127 for char in note.get("text", "")):
        if primary_gold not in raw_cands[:30] and primary_gold not in resolved_cands[:30]:
            return "Cat 5 (Language Issue)", f"Detected language '{lang}'; gold {primary_gold} missed in retrieval"

    # 2. Assertion failure
    note_text_lower = note.get("text", "").lower()
    for cl in rejected_clauses:
        cl_clean = cl.lower()
        if pred_title and (pred_title in cl_clean or cl_clean in pred_title):
            return "Cat 4 (Assertion Failure)", f"Predicted '{pred_top1}' ({pred_title}) which was in negated clause '{cl}'"
    neg_pattern = r"\b(?:no|denies|negative|ruled out|unlikely|without|free of|screened for)\s+([a-z\s]{3,30})"
    for m in re.finditer(neg_pattern, note_text_lower):
        neg_phrase = m.group(0)
        if any(w in neg_phrase for w in pred_title.split() if len(w) > 4):
            return "Cat 4 (Assertion Failure)", f"Predicted '{pred_top1}' overlaps negated context '{neg_phrase}'"

    # 3. Wrong specificity
    gold_stem = primary_gold[:3] if len(primary_gold) >= 3 else primary_gold
    pred_stem = pred_top1[:3] if len(pred_top1) >= 3 else pred_top1
    if gold_stem == pred_stem:
        if pred_top1.endswith(".Z") and not primary_gold.endswith(".Z"):
            return "Cat 3 (Specificity Error)", f"Chose unspecified (.Z) {pred_top1} when specific {primary_gold} was gold"
        elif not pred_top1.endswith(".Z") and primary_gold.endswith(".Z"):
            return "Cat 3 (Specificity Error)", f"Chose specific {pred_top1} when unspecified (.Z) {primary_gold} was gold"
        else:
            return "Cat 3 (Specificity Error)", f"Chose wrong specific sibling {pred_top1} instead of {primary_gold}"

    # 4. Retrieval miss
    if primary_gold not in raw_cands[:50] and primary_gold not in resolved_cands[:50]:
        return "Cat 1 (Retrieval Miss)", f"Gold {primary_gold} ({gold_title}) not in top-50 retrieval pool"

    # 5. Reranker demotion
    if primary_gold in raw_cands[:50] or primary_gold in resolved_cands[:50]:
        rank_in_retrieval = None
        for r_idx, c in enumerate(resolved_cands[:50], 1):
            if c == primary_gold:
                rank_in_retrieval = r_idx
                break
        return "Cat 2 (Reranker Demotion)", f"Gold {primary_gold} was rank {rank_in_retrieval} in retrieval, but placed outside top 3"

    return "Cat 1 (Retrieval Miss)", f"Gold {primary_gold} missing"


def run_evaluation(
    gold_path: str,
    pred_path: str | None = None,
    out_pred_path: str = "out/eval_predictions.jsonl",
    mms_path: str = "data/mms-2026-01.json",
    n_resamples: int = 1000,
) -> dict[str, Any]:
    start_time = time.time()
    mms, code_to_chapter, valid_codes = load_mms_info(mms_path)
    notes = load_jsonl(gold_path)
    total_notes = len(notes)

    print(f"\nLoaded {total_notes} notes from {gold_path}.")

    pipeline = None
    preds_dict = {}

    if pred_path and os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}...")
        for row in load_jsonl(pred_path):
            preds_dict[row["id"]] = row
    else:
        print("Instantiating pipeline and running predictions...")
        pipeline = ICD11Pipeline(verbose=False)
        os.makedirs(os.path.dirname(out_pred_path) or ".", exist_ok=True)
        with open(out_pred_path, "w", encoding="utf-8") as out_f:
            for idx, note in enumerate(notes, 1):
                p = pipeline.predict_note(note, debug=True)
                preds_dict[p["id"]] = p
                out_f.write(json.dumps(p, ensure_ascii=False) + "\n")
                out_f.flush()
                if idx % 20 == 0 or idx == total_notes:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(f"  Processed {idx}/{total_notes} notes ({rate:.1f} notes/sec)")

    eval_duration = time.time() - start_time
    rate = total_notes / eval_duration if eval_duration > 0 else 0
    projected_600_sec = 600.0 / rate if rate > 0 else 0
    projected_600_8cpu = projected_600_sec / 7.0  # conservative 7x speedup on 8 CPUs

    results = []
    failures = []

    # Recall tracking
    recall_total = {"r10": 0, "r30": 0, "r50": 0, "count": 0}
    recall_by_lang = defaultdict(lambda: {"r10": 0, "r30": 0, "r50": 0, "count": 0})
    recall_by_chapter = defaultdict(lambda: {"r10": 0, "r30": 0, "r50": 0, "count": 0})
    acc_by_style = defaultdict(lambda: {"top1": 0, "top3": 0, "count": 0})
    acc_by_lang = defaultdict(lambda: {"top1": 0, "top3": 0, "count": 0})
    acc_by_chapter = defaultdict(lambda: {"top1": 0, "top3": 0, "count": 0})
    cat_counts = defaultdict(int)

    for note in notes:
        nid = note["id"]
        slice_name = note.get("slice", "A")
        raw_golds = note.get("gold", note.get("codes", []))
        gold_set = {normalize_code(c) for c in raw_golds if c}
        primary_gold = sorted(gold_set)[0] if gold_set else ""
        chapter = code_to_chapter.get(primary_gold, "Unknown")
        style = note.get("style", "unknown")

        pred_entry = preds_dict.get(nid, {})
        pred_codes = [normalize_code(c) for c in (pred_entry.get("codes") or [])[:3]]
        valid_pred_codes = [c for c in pred_codes if c in valid_codes]

        top1 = bool(valid_pred_codes and valid_pred_codes[0] in gold_set)
        top3 = any(c in gold_set for c in valid_pred_codes)

        debug = pred_entry.get("debug", {})
        lang = note.get("language") or debug.get("lang") or detect_language(note.get("text", ""))

        # Accuracies
        acc_by_style[style]["count"] += 1
        acc_by_lang[lang]["count"] += 1
        acc_by_chapter[chapter]["count"] += 1
        if top1:
            acc_by_style[style]["top1"] += 1
            acc_by_lang[lang]["top1"] += 1
            acc_by_chapter[chapter]["top1"] += 1
        if top3:
            acc_by_style[style]["top3"] += 1
            acc_by_lang[lang]["top3"] += 1
            acc_by_chapter[chapter]["top3"] += 1

        # Candidate recall
        cands = [normalize_code(c[0]) for c in debug.get("raw_candidates", [])]
        resolved = [normalize_code(c[0]) for c in debug.get("resolved_candidates", [])]
        combined_pool = []
        for c in resolved + cands:
            if c and c not in combined_pool:
                combined_pool.append(c)

        if combined_pool:
            hit10 = any(g in combined_pool[:10] for g in gold_set)
            hit30 = any(g in combined_pool[:30] for g in gold_set)
            hit50 = any(g in combined_pool[:50] for g in gold_set)

            recall_total["count"] += 1
            recall_by_lang[lang]["count"] += 1
            recall_by_chapter[chapter]["count"] += 1

            if hit10:
                recall_total["r10"] += 1
                recall_by_lang[lang]["r10"] += 1
                recall_by_chapter[chapter]["r10"] += 1
            if hit30:
                recall_total["r30"] += 1
                recall_by_lang[lang]["r30"] += 1
                recall_by_chapter[chapter]["r30"] += 1
            if hit50:
                recall_total["r50"] += 1
                recall_by_lang[lang]["r50"] += 1
                recall_by_chapter[chapter]["r50"] += 1

        res_row = {
            "id": nid,
            "slice": slice_name,
            "gold": sorted(gold_set),
            "pred": valid_pred_codes,
            "top1": top1,
            "top3": top3,
            "language": lang,
            "chapter": chapter,
            "style": style,
        }
        results.append(res_row)

        if not top1:
            cat, reason = categorize_failure(note, valid_pred_codes, sorted(gold_set), debug, mms)
            cat_counts[cat] += 1
            failures.append({
                "id": nid,
                "style": style,
                "lang": lang,
                "chapter": chapter,
                "gold": sorted(gold_set),
                "pred": valid_pred_codes,
                "cat": cat,
                "reason": reason,
                "snip": note.get("text", "").replace("\n", " ")[:60],
            })

    score_dict = compute_official_score(results)
    ci = bootstrap_ci(results, n_resamples=n_resamples)

    # Print summary
    print("\n" + "=" * 80)
    print("                     EVALUATION & HARNESS REPORT")
    print("=" * 80)
    print(f"Notes Evaluated:    {total_notes}")
    print(f"Official Score:     {score_dict['score']:.2f} / {score_dict['present_weight']:.1f}  (Normalized: {score_dict['normalized_score']:.2f} / 100)")
    print(f"Score 95% CI:       [{ci['score_ci'][0]:.2f}, {ci['score_ci'][1]:.2f}]")
    print(f"Top-1 Accuracy:     {score_dict['top1_acc']*100:.2f}%  (95% CI: [{ci['top1_ci'][0]*100:.1f}%, {ci['top1_ci'][1]*100:.1f}%])")
    print(f"Top-3 Accuracy:     {score_dict['top3_acc']*100:.2f}%  (95% CI: [{ci['top3_ci'][0]*100:.1f}%, {ci['top3_ci'][1]*100:.1f}%])")
    print(f"Elapsed Time:       {eval_duration:.2f}s ({rate:.1f} notes/sec)")
    print(f"Projected 600-Note: {projected_600_sec/60:.1f} min (1 CPU) | {projected_600_8cpu/60:.1f} min (8 CPUs)")

    print("\n" + "-" * 80)
    print("CANDIDATE RECALL ON GOLD TARGETS:")
    if recall_total["count"] > 0:
        c_tot = recall_total["count"]
        print(f"  Overall Recall@10: {recall_total['r10']/c_tot*100:.1f}% ({recall_total['r10']}/{c_tot})")
        print(f"  Overall Recall@30: {recall_total['r30']/c_tot*100:.1f}% ({recall_total['r30']}/{c_tot})")
        print(f"  Overall Recall@50: {recall_total['r50']/c_tot*100:.1f}% ({recall_total['r50']}/{c_tot})")

    print("\nRECALL & ACCURACY BY LANGUAGE:")
    print(f"{'Lang':<8} | {'Count':<6} | {'Top-1':<8} | {'Top-3':<8} | {'Recall@10':<10} | {'Recall@30':<10} | {'Recall@50':<10}")
    print("-" * 75)
    for lang_code in sorted(acc_by_lang.keys()):
        cnt = acc_by_lang[lang_code]["count"]
        t1 = acc_by_lang[lang_code]["top1"] / cnt * 100 if cnt else 0
        t3 = acc_by_lang[lang_code]["top3"] / cnt * 100 if cnt else 0
        rcnt = recall_by_lang[lang_code]["count"]
        r10 = recall_by_lang[lang_code]["r10"] / rcnt * 100 if rcnt else 0
        r30 = recall_by_lang[lang_code]["r30"] / rcnt * 100 if rcnt else 0
        r50 = recall_by_lang[lang_code]["r50"] / rcnt * 100 if rcnt else 0
        print(f"{lang_code:<8} | {cnt:<6} | {t1:>5.1f}%  | {t3:>5.1f}%  | {r10:>7.1f}%  | {r30:>7.1f}%  | {r50:>7.1f}%")

    print("\nRECALL & ACCURACY BY CHAPTER (Top 10):")
    print(f"{'Chapter':<10} | {'Count':<6} | {'Top-1':<8} | {'Top-3':<8} | {'Recall@10':<10} | {'Recall@30':<10} | {'Recall@50':<10}")
    print("-" * 77)
    for ch in sorted(acc_by_chapter.keys(), key=lambda x: str(x))[:12]:
        cnt = acc_by_chapter[ch]["count"]
        t1 = acc_by_chapter[ch]["top1"] / cnt * 100 if cnt else 0
        t3 = acc_by_chapter[ch]["top3"] / cnt * 100 if cnt else 0
        rcnt = recall_by_chapter[ch]["count"]
        r10 = recall_by_chapter[ch]["r10"] / rcnt * 100 if rcnt else 0
        r30 = recall_by_chapter[ch]["r30"] / rcnt * 100 if rcnt else 0
        r50 = recall_by_chapter[ch]["r50"] / rcnt * 100 if rcnt else 0
        print(f"{ch:<10} | {cnt:<6} | {t1:>5.1f}%  | {t3:>5.1f}%  | {r10:>7.1f}%  | {r30:>7.1f}%  | {r50:>7.1f}%")

    print("\nFAILURE TRIAGE (Top-1 Misses):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat:<26}: {cnt:>2} notes ({cnt/total_notes*100:>4.1f}%)")

    return {
        "score": score_dict,
        "ci": ci,
        "recall_total": recall_total,
        "recall_by_lang": dict(recall_by_lang),
        "recall_by_chapter": dict(recall_by_chapter),
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/dev_notes.jsonl")
    parser.add_argument("--pred", default=None)
    parser.add_argument("--out", default="out/eval_predictions.jsonl")
    parser.add_argument("--mms", default="data/mms-2026-01.json")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()

    run_evaluation(
        gold_path=args.gold,
        pred_path=args.pred,
        out_pred_path=args.out,
        mms_path=args.mms,
        n_resamples=args.resamples,
    )

