#!/usr/bin/env python3
"""Zero-shot NLI Go / No-Go Feasibility Experiment for SugenEngTest (Step 0).

Evaluates whether zero-shot multilingual NLI (mDeBERTa-v3-base-xnli) outperforms
or complements the current lexical/concept reranker on the 60 dev notes.

Measures:
1. Per-note rank of the gold code under:
   - Current Reranker Score
   - NLI Entailment Differential (Focus Span): P(entail) - P(contradict)
   - NLI Entailment Differential (Full Note): P(entail) - P(contradict)
   - Ensemble: (1 - alpha) * Reranker + alpha * NLI
2. Top-1, Top-3, and Mean Reciprocal Rank (MRR)
3. Pairwise AUC: Probability that gold outscores a non-gold candidate
4. CPU latency per candidate pair and projected runtime for 600 notes x 15 candidates
5. Clear Go / No-Go Decision Verdict
"""

import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import ICD11Pipeline
from text_utils import normalize_code


def load_mms_titles(mms_path: str) -> dict[str, str]:
    with open(mms_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {normalize_code(d["code"]): d.get("title", "") for d in data if "code" in d}


def main():
    print("=" * 80)
    print("   ZERO-SHOT NLI FEASIBILITY & GO / NO-GO EXPERIMENT (STEP 0)")
    print("=" * 80)

    mms_path = os.path.join(ROOT, "data", "mms-2026-01.json")
    dev_path = os.path.join(ROOT, "data", "dev_notes.jsonl")
    code_titles = load_mms_titles(mms_path)

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_notes = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(dev_notes)} notes from {dev_path}.")

    # 1. Load pipeline and retrieve top-20 candidates
    print("\n[1/4] Running retrieval and baseline candidate scoring on dev notes...")
    pipeline = ICD11Pipeline(verbose=False)

    note_data = []
    total_candidates_evaluated = 0

    for note in dev_notes:
        nid = note["id"]
        raw_text = note.get("text", "")
        style = note.get("style")
        gold_codes = [normalize_code(c) for c in note.get("gold", []) if c]
        primary_gold = gold_codes[0] if gold_codes else ""

        # Extract focus span & expanded text
        extracted = pipeline.extractor.extract_diagnoses(raw_text, style=style)
        focus_span = extracted.get("primary_query", "") or raw_text[:200]
        expanded_text = extracted.get("expanded_text", raw_text)

        # Retrieval
        raw_candidates = pipeline.retriever.retrieve(
            primary_query=focus_span,
            fallback_query=expanded_text,
            top_k=40,
        )
        expanded_candidates = pipeline.hierarchy.expand_candidates(raw_candidates, max_expansions=120)

        # Resolve to leaf
        resolved_candidates = []
        seen = set()
        for code, sc in expanded_candidates:
            leaf = pipeline.hierarchy.resolve_to_leaf(code, focus_span or expanded_text)
            norm_leaf = normalize_code(leaf)
            if norm_leaf and norm_leaf in pipeline.valid_codes and norm_leaf not in seen:
                seen.add(norm_leaf)
                resolved_candidates.append((norm_leaf, sc))

        top20_resolved = resolved_candidates[:20]

        # Get baseline reranker scores
        reranked_pool = pipeline.reranker.rerank(
            top20_resolved,
            extracted_phrases=extracted.get("phrases") or [focus_span],
            note_text=expanded_text,
            top_k=20,
        )
        reranker_score_map = {c: s for c, s in reranked_pool}

        note_data.append({
            "id": nid,
            "gold": primary_gold,
            "gold_set": set(gold_codes),
            "focus_span": focus_span,
            "full_text": raw_text,
            "candidates": [c for c, _ in top20_resolved],
            "reranker_scores": reranker_score_map,
        })
        total_candidates_evaluated += len(top20_resolved)

    print(f"  Extracted {total_candidates_evaluated} candidate pairs across {len(dev_notes)} notes.")

    # 2. Load mDeBERTa-v3 NLI model
    model_id = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    print(f"\n[2/4] Loading NLI model: {model_id} (CPU evaluation)...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    print(f"  Model loaded in {time.time() - t_load:.2f}s. Config: {model.config.id2label}")

    # Determine label mapping
    id2label = {idx: str(lbl).lower() for idx, lbl in model.config.id2label.items()}
    entail_idx = next(idx for idx, lbl in id2label.items() if "entail" in lbl)
    contra_idx = next(idx for idx, lbl in id2label.items() if "contra" in lbl)

    # 3. Score candidate pairs with NLI
    print("\n[3/4] Scoring candidates with Zero-Shot NLI (Focus Span & Full Note)...")
    time_records = []

    for idx, nd in enumerate(note_data, 1):
        focus_span = nd["focus_span"]
        full_text = nd["full_text"]
        cands = nd["candidates"]

        nli_focus_scores = {}
        nli_full_scores = {}

        for code in cands:
            title = code_titles.get(code, "Unknown Condition")
            hypothesis = f"The patient has {title}."

            # Measure CPU time on focus span
            t0 = time.time()
            inputs_focus = tokenizer(
                focus_span,
                hypothesis,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits_focus = model(**inputs_focus).logits
                probs_focus = torch.softmax(logits_focus, dim=-1)[0]
                p_e_focus = probs_focus[entail_idx].item()
                p_c_focus = probs_focus[contra_idx].item()
                s_nli_focus = p_e_focus - p_c_focus
            dt = time.time() - t0
            time_records.append(dt)

            # Full note inference
            inputs_full = tokenizer(
                full_text,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits_full = model(**inputs_full).logits
                probs_full = torch.softmax(logits_full, dim=-1)[0]
                p_e_full = probs_full[entail_idx].item()
                p_c_full = probs_full[contra_idx].item()
                s_nli_full = p_e_full - p_c_full

            nli_focus_scores[code] = s_nli_focus
            nli_full_scores[code] = s_nli_full

        nd["nli_focus_scores"] = nli_focus_scores
        nd["nli_full_scores"] = nli_full_scores

        if idx % 10 == 0 or idx == len(note_data):
            avg_ms = np.mean(time_records) * 1000
            print(f"  Processed {idx}/{len(note_data)} notes ({avg_ms:.1f} ms/pair)")

    # 4. Comparative Evaluation
    print("\n[4/4] Computing comparative ranking metrics...")

    def evaluate_ranking(score_fn, name: str):
        top1_hits = 0
        top3_hits = 0
        mrr_sum = 0.0
        auc_sum = 0.0
        eval_notes = 0

        for nd in note_data:
            gold_set = nd["gold_set"]
            cands = nd["candidates"]
            if not any(g in cands for g in gold_set):
                continue  # skip notes where gold wasn't in top-20 pool for fair ranking comparison

            eval_notes += 1
            scored_cands = [(c, score_fn(nd, c)) for c in cands]
            scored_cands.sort(key=lambda x: x[1], reverse=True)

            ranked_codes = [c for c, _ in scored_cands]
            top1_hits += int(ranked_codes[0] in gold_set)
            top3_hits += int(any(c in gold_set for c in ranked_codes[:3]))

            # Rank of first gold hit
            first_rank = next((r for r, c in enumerate(ranked_codes, 1) if c in gold_set), len(ranked_codes))
            mrr_sum += 1.0 / first_rank

            # Pairwise AUC
            gold_scores = [score_fn(nd, c) for c in cands if c in gold_set]
            neg_scores = [score_fn(nd, c) for c in cands if c not in gold_set]
            if gold_scores and neg_scores:
                pair_hits = sum(1.0 for gs in gold_scores for ns in neg_scores if gs > ns)
                pair_ties = sum(0.5 for gs in gold_scores for ns in neg_scores if gs == ns)
                auc = (pair_hits + pair_ties) / (len(gold_scores) * len(neg_scores))
                auc_sum += auc

        n = eval_notes if eval_notes > 0 else 1
        return {
            "name": name,
            "notes": eval_notes,
            "top1": top1_hits / n * 100,
            "top3": top3_hits / n * 100,
            "mrr": mrr_sum / n,
            "auc": auc_sum / n * 100,
        }

    # Evaluate individual signals
    res_rerank = evaluate_ranking(lambda nd, c: nd["reranker_scores"].get(c, -1.0), "Baseline Lexical Reranker")
    res_nli_focus = evaluate_ranking(lambda nd, c: nd["nli_focus_scores"].get(c, -1.0), "Zero-Shot NLI (Focus Span)")
    res_nli_full = evaluate_ranking(lambda nd, c: nd["nli_full_scores"].get(c, -1.0), "Zero-Shot NLI (Full Note)")

    # Normalize scores for ensembling
    # Min-max scale per note
    def get_norm_scores(score_dict):
        vals = list(score_dict.values())
        if not vals:
            return {}
        vmin, vmax = min(vals), max(vals)
        denom = vmax - vmin if vmax > vmin else 1.0
        return {k: (v - vmin) / denom for k, v in score_dict.items()}

    for nd in note_data:
        nd["norm_rerank"] = get_norm_scores(nd["reranker_scores"])
        nd["norm_nli_focus"] = get_norm_scores(nd["nli_focus_scores"])

    res_ens_20 = evaluate_ranking(lambda nd, c: 0.8 * nd["norm_rerank"].get(c, 0) + 0.2 * nd["norm_nli_focus"].get(c, 0), "Ensemble (80% Rerank + 20% NLI)")
    res_ens_35 = evaluate_ranking(lambda nd, c: 0.65 * nd["norm_rerank"].get(c, 0) + 0.35 * nd["norm_nli_focus"].get(c, 0), "Ensemble (65% Rerank + 35% NLI)")
    res_ens_50 = evaluate_ranking(lambda nd, c: 0.5 * nd["norm_rerank"].get(c, 0) + 0.5 * nd["norm_nli_focus"].get(c, 0), "Ensemble (50% Rerank + 50% NLI)")

    # Latency projections
    avg_latency_ms = np.mean(time_records) * 1000
    med_latency_ms = np.median(time_records) * 1000
    p95_latency_ms = np.percentile(time_records, 95) * 1000

    # 600 notes x 15 candidates = 9000 pairs
    total_pairs = 600 * 15
    proj_sec_1cpu = (avg_latency_ms / 1000.0) * total_pairs
    proj_min_8cpu = (proj_sec_1cpu / 7.0) / 60.0  # conservative 7x scaling on 8 CPUs

    # Print Report Table
    print("\n" + "=" * 80)
    print("                     ZERO-SHOT NLI EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"{'Method / Signal':<35} | {'Top-1':<8} | {'Top-3':<8} | {'MRR':<8} | {'Pairwise AUC':<12}")
    print("-" * 78)
    for r in [res_rerank, res_nli_focus, res_nli_full, res_ens_20, res_ens_35, res_ens_50]:
        print(f"{r['name']:<35} | {r['top1']:>5.1f}%  | {r['top3']:>5.1f}%  | {r['mrr']:>6.3f} | {r['auc']:>9.1f}%")

    print("\n" + "-" * 80)
    print("CPU LATENCY & RUNTIME PROFILE:")
    print(f"  Mean latency per candidate pair : {avg_latency_ms:.2f} ms")
    print(f"  Median latency                  : {med_latency_ms:.2f} ms")
    print(f"  P95 latency                     : {p95_latency_ms:.2f} ms")
    print(f"  Projected 600-note runtime: {proj_sec_1cpu/60.0:.2f} min (1 CPU) | {proj_min_8cpu:.2f} min (8 CPUs)")
    print(f"  Run box limit: 60 min (target < 25 min) -> Status: {'PASS (<25m)' if proj_min_8cpu < 25.0 else 'FAIL (>25m)'}")

    # Decision Rule Logic
    print("\n" + "=" * 80)
    print("                     GO / NO-GO DECISION VERDICT")
    print("=" * 80)
    mrr_gain = res_ens_20["mrr"] - res_rerank["mrr"]
    top1_gain = res_ens_20["top1"] - res_rerank["top1"]
    fits_budget = proj_min_8cpu < 25.0

    print(f"Top-1 Delta with NLI Ensemble: {top1_gain:+.2f}% ({res_rerank['top1']:.1f}% -> {res_ens_20['top1']:.1f}%)")
    print(f"MRR Delta with NLI Ensemble  : {mrr_gain:+.4f} ({res_rerank['mrr']:.3f} -> {res_ens_20['mrr']:.3f})")
    print(f"Pairwise AUC of NLI alone    : {res_nli_focus['auc']:.1f}% (vs Reranker: {res_rerank['auc']:.1f}%)")
    print(f"8-CPU Latency Budget Check   : {proj_min_8cpu:.2f} min vs 25 min target ({'OK' if fits_budget else 'EXCEEDED'})")

    if (mrr_gain > 0 or top1_gain > 0 or res_nli_focus['auc'] > 75.0) and fits_budget:
        verdict = "GO"
        decision_notes = (
            "NLI provides strong, complementary discriminative signal to lexical reranking "
            "and comfortably satisfies the 25-minute runtime ceiling on 8 CPUs."
        )
    else:
        verdict = "NO-GO"
        decision_notes = (
            "NLI does not convincingly beat the tuned lexical/concept reranker or exceeds latency bounds; "
            "focus should remain on recall expansion, exclusion conflict graphs, and decision-theoretic calibration."
        )

    print(f"\nVERDICT: >>> {verdict} <<<")
    print(f"Rationale: {decision_notes}")
    print("=" * 80 + "\n")

    return {
        "verdict": verdict,
        "results": {
            "rerank": res_rerank,
            "nli_focus": res_nli_focus,
            "nli_full": res_nli_full,
            "ens_20": res_ens_20,
            "ens_35": res_ens_35,
        },
        "latency": {
            "avg_ms": avg_latency_ms,
            "proj_8cpu_min": proj_min_8cpu,
        },
    }


if __name__ == "__main__":
    main()

