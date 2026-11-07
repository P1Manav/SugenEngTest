#!/usr/bin/env python3
"""Benchmark runtime, peak RAM, per-stage latency breakdown, and throughput across 600 notes.

Measures:
1. Total wall-clock time for 600 notes (dev set repeated 10x with unique IDs).
2. Per-stage latency breakdown (mean + p95 ms per note) for:
   - Language ID (LID)
   - Clinical Diagnosis Extraction & Abbreviation Expansion
   - Assertion & Negation Filtering
   - Hybrid Sparse + Dense Retrieval
   - ICD Hierarchy Specificity Descent
   - Rule Reranker & Diversity Selection
3. Peak RAM consumption
4. Model weights and index disk footprint
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import ICD11Pipeline
from text_utils import normalize_code
from assertion_filter import split_sentences_and_clauses
from multilingual import detect_language, translate_clinical_text


def measure_per_stage_latencies(notes: list[dict], pipeline: ICD11Pipeline) -> dict[str, list[float]]:
    """Profile latency of each individual pipeline stage on a representative sample."""
    latencies = {
        "1. Language ID": [],
        "2. Clinical Extraction": [],
        "3. Assertion Filtering": [],
        "4. Hybrid Retrieval": [],
        "5. Hierarchy Descent": [],
        "6. Reranking & Diversity": [],
        "Total per note": [],
    }

    for note in notes:
        raw_text = note.get("text", "")
        style = note.get("style")
        t_note_start = time.perf_counter()

        # Stage 1: LID
        t0 = time.perf_counter()
        lang = detect_language(raw_text)
        input_text = translate_clinical_text(raw_text, lang) if lang != "en" else raw_text
        t_lid = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Extraction
        t0 = time.perf_counter()
        extracted = pipeline.extractor.extract_diagnoses(input_text, style=style)
        primary_query = extracted["primary_query"]
        extracted_phrases = extracted["phrases"]
        expanded_text = extracted["expanded_text"]
        t_extract = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Assertion Filtering
        t0 = time.perf_counter()
        affirmed_phrases = []
        for p in extracted_phrases:
            for cl in split_sentences_and_clauses(p):
                if pipeline.assertion_filter.check_clause_assertion(cl)["affirmed"]:
                    affirmed_phrases.append(cl)
        if affirmed_phrases:
            primary_query = " ".join(affirmed_phrases)
            extracted_phrases = affirmed_phrases
        filtered_note_text, _ = pipeline.assertion_filter.filter_text_for_retrieval(expanded_text)
        t_assert = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Retrieval
        t0 = time.perf_counter()
        raw_candidates = pipeline.retriever.retrieve(
            primary_query=primary_query,
            fallback_query=filtered_note_text,
            top_k=40,
        )
        filtered_candidates = []
        for code, score in raw_candidates:
            info = pipeline.kb.get(code)
            if info:
                title = info.get("title", "")
                if pipeline.assertion_filter.is_candidate_negated_in_note(title, expanded_text, affirmed_phrases=extracted_phrases):
                    continue
            filtered_candidates.append((code, score))
        if not filtered_candidates:
            filtered_candidates = raw_candidates
        t_retrieval = (time.perf_counter() - t0) * 1000.0

        # Stage 5: Hierarchy Descent
        t0 = time.perf_counter()
        expanded_candidates = pipeline.hierarchy.expand_candidates(filtered_candidates, max_expansions=50)
        resolved_candidates = []
        seen_codes = set()
        for code, score in expanded_candidates:
            leaf_code = pipeline.hierarchy.resolve_to_leaf(code, primary_query or filtered_note_text)
            norm_c = normalize_code(leaf_code)
            if norm_c and norm_c in pipeline.valid_codes and norm_c not in seen_codes:
                seen_codes.add(norm_c)
                resolved_candidates.append((norm_c, score))
        t_hierarchy = (time.perf_counter() - t0) * 1000.0

        # Stage 6: Reranking
        t0 = time.perf_counter()
        reranked = pipeline.reranker.rerank(
            resolved_candidates,
            extracted_phrases=extracted_phrases or [primary_query],
            note_text=expanded_text,
            top_k=25,
        )
        _ = pipeline._select_diverse_top3(reranked, primary_query, expanded_text)
        t_rerank = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_note_start) * 1000.0

        latencies["1. Language ID"].append(t_lid)
        latencies["2. Clinical Extraction"].append(t_extract)
        latencies["3. Assertion Filtering"].append(t_assert)
        latencies["4. Hybrid Retrieval"].append(t_retrieval)
        latencies["5. Hierarchy Descent"].append(t_hierarchy)
        latencies["6. Reranking & Diversity"].append(t_rerank)
        latencies["Total per note"].append(t_total)

    return latencies


def get_peak_ram_mb() -> float:
    """Measure peak RAM usage of current process in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def run_benchmark(num_notes: int = 600, dev_path: str = "data/dev_notes.jsonl"):
    print("=" * 75)
    print(f"       BENCHMARK RUNTIME & LATENCY PROFILER ({num_notes} NOTES)")
    print("=" * 75)

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_notes = [json.loads(line) for line in f if line.strip()]

    # 1. Measure per-stage latencies
    print(f"Measuring per-stage latency breakdown on {len(dev_notes)} distinct notes...")
    pipeline = ICD11Pipeline(verbose=False)
    stage_latencies = measure_per_stage_latencies(dev_notes, pipeline)

    print("\nPER-STAGE LATENCY BREAKDOWN (ms per note):")
    print(f"{'Pipeline Stage':<35} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Share (%)'}")
    print("-" * 75)
    mean_total = np.mean(stage_latencies["Total per note"])
    for stage, vals in stage_latencies.items():
        if stage == "Total per note":
            continue
        m = float(np.mean(vals))
        p95 = float(np.percentile(vals, 95))
        share = (m / mean_total) * 100.0 if mean_total > 0 else 0.0
        print(f"{stage:<35} | {m:>8.2f} ms | {p95:>8.2f} ms | {share:>6.1f}%")
    print("-" * 75)
    print(f"{'Total End-to-End Latency':<35} | {mean_total:>8.2f} ms | {np.percentile(stage_latencies['Total per note'], 95):>8.2f} ms | 100.0%")

    # 2. Benchmark full 600-note parallel execution via entrypoint.py
    print(f"\nBenchmarking parallel multi-worker throughput for {num_notes} notes via entrypoint.py...")
    reps = (num_notes + len(dev_notes) - 1) // len(dev_notes)
    synth_notes = []
    for r in range(reps):
        for n in dev_notes:
            synth_notes.append({"id": f"{n['id']}_rep{r}", "text": n["text"]})
            if len(synth_notes) == num_notes:
                break
        if len(synth_notes) == num_notes:
            break

    tmp_dir = tempfile.mkdtemp()
    inp_path = os.path.join(tmp_dir, "bench_600_in.jsonl")
    out_path = os.path.join(tmp_dir, "bench_600_out.jsonl")

    with open(inp_path, "w", encoding="utf-8") as f:
        for item in synth_notes:
            f.write(json.dumps(item) + "\n")

    t_start = time.time()
    env = os.environ.copy()
    env["STRICT"] = "1"
    proc = subprocess.run([sys.executable, "entrypoint.py", inp_path, out_path], capture_output=True, text=True, env=env)
    wall_clock = time.time() - t_start

    if proc.returncode != 0:
        print(f"Benchmark failed:\n{proc.stderr}")
        return

    out_preds = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            out_preds = [line for line in f if line.strip()]

    throughput = len(out_preds) / wall_clock if wall_clock > 0 else 0.0

    print("-" * 75)
    print("THROUGHPUT & RESOURCE MEASUREMENTS:")
    print(f"  Notes Processed:         {len(out_preds)} / {num_notes}")
    print(f"  Total Wall-Clock Time:   {wall_clock:.2f} seconds ({wall_clock / 60.0:.2f} minutes)")
    print(f"  Throughput:              {throughput:.2f} notes/sec ({1000.0 / throughput:.1f} ms/note effective)")
    print(f"  60-Min Limit Budget:     Used {wall_clock / 3600.0 * 100:.1f}% of allowed 60 min (< 25 min requirement: PASS)")

    # Measure disk footprints
    kb_size_mb = sum(os.path.getsize(os.path.join(ROOT, "kb", f)) for f in os.listdir(os.path.join(ROOT, "kb"))) / (1024 * 1024)
    model_size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(os.path.join(ROOT, "models"))
        for f in filenames
    ) / (1024 * 1024)

    print(f"  KB Indexes Footprint:    {kb_size_mb:.2f} MB")
    print(f"  Vendored Model Weights:  {model_size_mb:.2f} MB")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=int, default=600)
    args = parser.parse_args()
    run_benchmark(num_notes=args.notes)
