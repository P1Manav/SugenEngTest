"""Evaluation script for ICD-11 coding system with bootstrap confidence intervals,
candidate recall metrics, and per-note failure taxonomy.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np

from text_utils import normalize_code
from pipeline import ICD11Pipeline


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_score(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute score following score.py rules."""
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

    return {
        "score": total_score,
        "normalized_score": norm_score,
        "slices": slice_metrics,
        "total_top1": sum(st["top1"] for st in stats.values()) / len(results) if results else 0.0,
        "total_top3": sum(st["top3"] for st in stats.values()) / len(results) if results else 0.0,
    }


def bootstrap_ci(results: List[Dict[str, Any]], n_resamples: int = 1000, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap percentile confidence intervals for score, top1, and top3."""
    n = len(results)
    if n == 0:
        return {}

    b_scores = []
    b_top1s = []
    b_top3s = []

    rng = random.Random(42)
    for _ in range(n_resamples):
        sample = [results[rng.randint(0, n - 1)] for _ in range(n)]
        sc = compute_score(sample)
        b_scores.append(sc["score"])
        b_top1s.append(sc["total_top1"])
        b_top3s.append(sc["total_top3"])

    low_p = (alpha / 2.0) * 100
    high_p = (1.0 - alpha / 2.0) * 100

    return {
        "score_ci": (float(np.percentile(b_scores, low_p)), float(np.percentile(b_scores, high_p))),
        "top1_ci": (float(np.percentile(b_top1s, low_p)), float(np.percentile(b_top1s, high_p))),
        "top3_ci": (float(np.percentile(b_top3s, low_p)), float(np.percentile(b_top3s, high_p))),
    }


def evaluate(
    gold_path: str = "data/dev_notes.jsonl",
    pred_path: Optional[str] = None,
    output_pred_path: str = "out/dev_predictions.jsonl",
    mms_path: str = "kb/valid_codes.json",
):
    print("=" * 60)
    print("ICD-11 EVALUATION")
    print("=" * 60)

    with open(mms_path, "r", encoding="utf-8") as f:
        valid_codes = set(json.load(f))

    gold_notes = load_jsonl(gold_path)
    print(f"Loaded {len(gold_notes)} gold notes from {gold_path}.")

    pipeline = None
    if not pred_path:
        pipeline = ICD11Pipeline()
        os.makedirs(os.path.dirname(output_pred_path) or ".", exist_ok=True)
        print(f"Running pipeline and saving predictions to {output_pred_path}...")
        preds_dict = {}
        with open(output_pred_path, "w", encoding="utf-8") as out_f:
            for note in gold_notes:
                p = pipeline.predict_note(note)
                preds_dict[p["id"]] = p
                out_f.write(json.dumps(p, ensure_ascii=False) + "\n")
                out_f.flush()
    else:
        print(f"Loading pre-generated predictions from {pred_path}...")
        preds = load_jsonl(pred_path)
        preds_dict = {p["id"]: p for p in preds}

    # Evaluate notes
    results = []
    failures = []
    recall30_hits = 0
    recall50_hits = 0

    for note in gold_notes:
        nid = note["id"]
        gold_set = {normalize_code(g) for g in note.get("gold", [])}
        slice_name = note.get("slice", "A")

        p = preds_dict.get(nid, {})
        pred_codes = [normalize_code(c) for c in (p.get("codes") or [])[:3]]
        # Filter invalid codes
        valid_pred_codes = [c for c in pred_codes if c in valid_codes]

        top1 = bool(valid_pred_codes) and valid_pred_codes[0] in gold_set
        top3 = any(c in gold_set for c in valid_pred_codes)

        result_row = {
            "id": nid,
            "slice": slice_name,
            "gold": sorted(gold_set),
            "pred": valid_pred_codes,
            "top1": top1,
            "top3": top3,
            "text": note.get("text", ""),
        }
        results.append(result_row)

        # Check candidate recall if pipeline is available
        if pipeline:
            try:
                extracted = pipeline.extractor.extract_diagnoses(note.get("text", ""), note.get("style"))
                cands = pipeline.retriever.retrieve(
                    primary_query=extracted["primary_query"],
                    fallback_query=extracted["expanded_text"],
                    top_k=50,
                )
                cand_codes = [c[0] for c in cands]
                # Check with hierarchy expansion
                expanded_cands = [c[0] for c in pipeline.hierarchy.expand_candidates(cands, max_expansions=50)]
                resolved_50 = [pipeline.hierarchy.resolve_to_leaf(c) for c in expanded_cands[:50]]
                resolved_30 = [pipeline.hierarchy.resolve_to_leaf(c) for c in expanded_cands[:30]]

                if any(g in resolved_30 for g in gold_set):
                    recall30_hits += 1
                if any(g in resolved_50 for g in gold_set):
                    recall50_hits += 1
            except Exception:
                pass

        if not top1:
            # Failure categorization
            tag = "wrong_diagnosis"
            gold_stems = {g.split(".")[0] for g in gold_set}
            pred_stems = {c.split(".")[0] for c in valid_pred_codes if c}

            if gold_stems.intersection(pred_stems):
                tag = "wrong_specificity"
            elif any(word in note.get("text", "").lower() for word in ["mzee", "daktari", "karibu", "mama", "boda", "uji"]):
                tag = "language_or_slang_issue"

            failures.append({
                "id": nid,
                "gold": sorted(gold_set),
                "pred": valid_pred_codes,
                "tag": tag,
                "top1": top1,
                "top3": top3,
                "text_snippet": note.get("text", "")[:120].replace("\n", " "),
            })

    # Overall Metrics
    metrics = compute_score(results)
    ci = bootstrap_ci(results, n_resamples=1000)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total notes evaluated: {len(results)}")
    print(f"Top-1 Accuracy: {metrics['total_top1'] * 100:.2f}% (95% CI: [{ci['top1_ci'][0]*100:.1f}%, {ci['top1_ci'][1]*100:.1f}%])")
    print(f"Top-3 Accuracy: {metrics['total_top3'] * 100:.2f}% (95% CI: [{ci['top3_ci'][0]*100:.1f}%, {ci['top3_ci'][1]*100:.1f}%])")
    print(f"Official Score: {metrics['score']:.2f} / 75.0 (Normalized: {metrics['normalized_score']:.2f} / 100)")
    print(f"Score 95% CI:   [{ci['score_ci'][0]:.2f}, {ci['score_ci'][1]:.2f}]")

    if pipeline:
        r30 = recall30_hits / len(gold_notes) * 100
        r50 = recall50_hits / len(gold_notes) * 100
        print(f"Candidate Recall@30: {r30:.1f}%")
        print(f"Candidate Recall@50: {r50:.1f}%")

    print("\n" + "-" * 60)
    print(f"FAILURE ANALYSIS ({len(failures)} notes with Top-1 misses)")
    print("-" * 60)
    tag_counts = {}
    for f in failures:
        tag_counts[f["tag"]] = tag_counts.get(f["tag"], 0) + 1
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tag}: {cnt} notes ({cnt/len(failures)*100:.1f}%)")

    print("\nSample Failures:")
    for f in failures[:10]:
        print(f"  [{f['id']}] [{f['tag']}] Top3 Hit: {f['top3']}")
        print(f"      Gold: {f['gold']}")
        print(f"      Pred: {f['pred']}")
        print(f"      Note: {f['text_snippet']}...")

    return metrics, ci, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/dev_notes.jsonl")
    parser.add_argument("--pred", default=None)
    parser.add_argument("--out", default="out/dev_predictions.jsonl")
    parser.add_argument("--mms", default="kb/valid_codes.json")
    args = parser.parse_args()

    evaluate(gold_path=args.gold, pred_path=args.pred, output_pred_path=args.out, mms_path=args.mms)
