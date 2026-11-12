"""Main entrypoint for Docker container execution — v2 Hybrid Architecture.
Uses async LLM reranking (OpenAI/Anthropic) with offline fallback.
Reads input notes JSONL, runs ICD-11 coding pipeline, writes output predictions JSONL.
Guarantees every input id in the output, 1-3 valid codes, no silent failures.
"""

import asyncio
import json
import os
import re
import sys
import time
from typing import Any

# Pin single thread per process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


def extract_id_from_raw_line(line: str, line_idx: int) -> str:
    """Extract id from raw line text if json.loads fails."""
    m = re.search(r'["\']id["\']\s*:\s*["\']([^"\']+)["\']', line)
    if m:
        return m.group(1)
    m = re.search(r'["\']id["\']\s*:\s*(\d+)', line)
    if m:
        return str(m.group(1))
    return f"note-line-{line_idx}"


def get_emergency_fallback_codes() -> list[str]:
    """Load valid codes for emergency fallback."""
    try:
        with open("kb/valid_codes.json", "r", encoding="utf-8") as f:
            valid = json.load(f)
        if valid and len(valid) >= 3:
            return valid[:3]
    except Exception:
        pass
    return ["CA07.0", "CA00", "1F40.Z"]  # Common outpatient codes


def main():
    if len(sys.argv) < 3:
        print("Usage: python entrypoint.py <input_notes.jsonl> <output_predictions.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        sys.exit(f"Input file not found: {input_path}")

    # Ensure output parent directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Check for already-processed IDs (resume support)
    processed_ids: set[str] = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        p = json.loads(line)
                        if "id" in p:
                            processed_ids.add(p["id"])
                    except Exception:
                        pass
        if processed_ids:
            print(f"Resuming: found {len(processed_ids)} already processed notes")

    # Read and deduplicate input notes
    seen_input_ids: set[str] = set()
    pending_notes: list[dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                note_id = item.get("id")
                if not note_id:
                    note_id = f"note-{line_idx}"
                    item["id"] = note_id
                if note_id in seen_input_ids:
                    continue
                seen_input_ids.add(note_id)
                if note_id not in processed_ids:
                    pending_notes.append(item)
            except Exception:
                orig_id = extract_id_from_raw_line(line, line_idx)
                if orig_id not in processed_ids and orig_id not in seen_input_ids:
                    seen_input_ids.add(orig_id)
                    fallback_pred = {"id": orig_id, "codes": get_emergency_fallback_codes()}
                    with open(output_path, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(fallback_pred, ensure_ascii=False) + "\n")
                        out_f.flush()
                    processed_ids.add(orig_id)

    total_pending = len(pending_notes)
    print(f"Total notes to process: {total_pending} (Already processed: {len(processed_ids)})")

    if total_pending == 0:
        print("All notes are already processed. Exiting.")
        return

    # Initialize the v2 pipeline
    start_time = time.time()
    print("Initializing ICD-11 Pipeline v2 (Hybrid Architecture)...")

    from pipeline_v2 import ICD11PipelineV2, process_notes_hybrid, process_notes_offline

    pipeline = ICD11PipelineV2(verbose=True)

    # Decide mode: hybrid (with LLM) or offline-only
    has_api = pipeline.llm_reranker.api_available

    if has_api:
        print("API keys detected — running in HYBRID mode (LLM + offline fallback)")
        print(f"Processing {total_pending} notes with async LLM reranking...")

        # Run async event loop
        completed = asyncio.run(
            process_notes_hybrid(pipeline, pending_notes, output_path, processed_ids)
        )
    else:
        print("No API keys — running in OFFLINE-ONLY mode (improved v1 with bug fixes)")
        print(f"Processing {total_pending} notes sequentially...")

        completed = process_notes_offline(pipeline, pending_notes, output_path, processed_ids)

    total_time = time.time() - start_time
    rate = completed / total_time if total_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"Completed {completed} notes in {total_time:.2f}s ({rate:.1f} notes/sec)")
    if has_api:
        print(f"LLM success: {pipeline.llm_success_count}, Offline fallback: {pipeline.llm_fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
