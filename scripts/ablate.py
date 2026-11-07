#!/usr/bin/env python3
"""Ablation experiments runner for the ICD-11 coding pipeline.

Runs pipeline on dev notes with one architectural switch flipped at a time:
- Full Baseline (all features enabled)
- Dense-only (no BM25)
- BM25-only (no Dense)
- No abbreviation expansion
- No assertion filter
- No specificity descent
- No reranker

Outputs a markdown table of Top-1, Top-3, Score / 75.0, Recall@30, and runtime with 95% bootstrap CIs.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import ICD11Pipeline
from text_utils import normalize_code


def bootstrap_ci(scores: np.ndarray, n_bootstraps: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and empirical bootstrap percentile confidence intervals."""
    rng = np.random.default_rng(42)
    n = len(scores)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    boot_means = np.array(boot_means)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    mean_val = float(np.mean(scores))
    return mean_val, low, high


def evaluate_variant(pipeline: ICD11Pipeline, notes: list[dict]) -> dict:
    start_time = time.time()
    top1_list = []
    top3_list = []
    score_list = []
    recall30_count = 0

    for note in notes:
        gold_raw = note.get("gold", note.get("codes", []))
        gold_codes = [normalize_code(c) for c in gold_raw if c]
        primary_gold = gold_codes[0] if gold_codes else ""

        pred_res = pipeline.predict_note(note, debug=True)
        pred_codes = [normalize_code(c) for c in pred_res.get("codes", [])]
        debug = pred_res.get("debug", {})

        is_top1 = 1.0 if (pred_codes and primary_gold and pred_codes[0] == primary_gold) else 0.0
        is_top3 = 1.0 if any(c in gold_codes for c in pred_codes[:3]) else 0.0
        note_score = 50.0 * is_top1 + 25.0 * is_top3

        top1_list.append(is_top1)
        top3_list.append(is_top3)
        score_list.append(note_score)

        cands = [normalize_code(c[0]) for c in debug.get("raw_candidates", [])]
        resolved = [normalize_code(c[0]) for c in debug.get("resolved_candidates", [])]
        combined = []
        for c in resolved + cands:
            if c not in combined:
                combined.append(c)
        if any(g in combined[:30] for g in gold_codes):
            recall30_count += 1

    elapsed = time.time() - start_time
    n = len(notes)

    top1_mean, top1_low, top1_high = bootstrap_ci(np.array(top1_list))
    top3_mean, top3_low, top3_high = bootstrap_ci(np.array(top3_list))
    score_mean, score_low, score_high = bootstrap_ci(np.array(score_list))

    return {
        "top1": top1_mean * 100,
        "top1_ci": (top1_low * 100, top1_high * 100),
        "top3": top3_mean * 100,
        "top3_ci": (top3_low * 100, top3_high * 100),
        "score": score_mean,
        "score_ci": (score_low, score_high),
        "recall30": (recall30_count / n) * 100,
        "runtime_s": elapsed,
    }


def run_ablations(gold_path: str = "data/dev_notes.jsonl"):
    with open(gold_path, "r", encoding="utf-8") as f:
        notes = [json.loads(line) for line in f if line.strip()]

    variants = [
        ("Full Baseline", {}),
        ("Dense-only (no BM25)", {"use_bm25": False, "use_dense": True}),
        ("BM25-only (no Dense)", {"use_bm25": True, "use_dense": False}),
        ("No Abbreviations", {"use_abbreviations": False}),
        ("No Assertion Filter", {"use_assertion_filter": False}),
        ("No Specificity Descent", {"use_specificity_descent": False}),
        ("No Reranker", {"use_reranker": False}),
    ]

    results = []
    print(f"Running ablations across {len(variants)} configurations on {len(notes)} notes...")

    for name, kwargs in variants:
        print(f"Evaluating: {name}...")
        p = ICD11Pipeline(verbose=False, **kwargs)
        res = evaluate_variant(p, notes)
        results.append((name, res))
        print(f"  -> Score: {res['score']:.2f}/75, Top-1: {res['top1']:.1f}%, Top-3: {res['top3']:.1f}%, Time: {res['runtime_s']:.1f}s")

    # Generate Markdown Table
    lines = [
        "| Ablation Variant | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Score / 75.0 (95% CI) | Recall@30 | Runtime |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for name, r in results:
        t1_str = f"**{r['top1']:.1f}%**<br>[{r['top1_ci'][0]:.1f}%, {r['top1_ci'][1]:.1f}%]"
        t3_str = f"**{r['top3']:.1f}%**<br>[{r['top3_ci'][0]:.1f}%, {r['top3_ci'][1]:.1f}%]"
        sc_str = f"**{r['score']:.2f}**<br>[{r['score_ci'][0]:.2f}, {r['score_ci'][1]:.2f}]"
        rec_str = f"{r['recall30']:.1f}%"
        rt_str = f"{r['runtime_s']:.1f}s"
        lines.append(f"| {name} | {t1_str} | {t3_str} | {sc_str} | {rec_str} | {rt_str} |")

    table_md = "\n".join(lines)
    print("\n" + table_md + "\n")
    return table_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/dev_notes.jsonl")
    args = parser.parse_args()
    run_ablations(args.gold)
