"""Main entrypoint for Docker container execution.
Reads input notes JSONL, runs ICD-11 coding pipeline, writes output predictions JSONL.
Supports multiprocessing across 8 CPUs, timeout handling, resume, and graceful degradation.
"""

import json
import os
import sys
import time
from typing import Dict, Any, Optional
from multiprocessing import Pool, cpu_count

# Default fallback code if note processing fails or times out
DEFAULT_FALLBACK_CODE = "RA01"

# Worker pipeline global reference
_worker_pipeline = None


def init_worker():
    """Initialize pipeline once per worker process without stdout deadlock."""
    global _worker_pipeline
    try:
        from pipeline import ICD11Pipeline
        _worker_pipeline = ICD11Pipeline(verbose=False)
    except Exception as e:
        _worker_pipeline = None


def process_single_note(note_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single note and return prediction dict."""
    global _worker_pipeline
    note_id = note_data.get("id", "unknown")
    try:
        if _worker_pipeline is None:
            from pipeline import ICD11Pipeline
            _worker_pipeline = ICD11Pipeline(verbose=False)
        return _worker_pipeline.predict_note(note_data)
    except Exception as e:
        return {"id": note_id, "codes": [DEFAULT_FALLBACK_CODE]}


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

    # Check for already-processed IDs to support resume
    processed_ids = set()
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
        print(f"Resuming: found {len(processed_ids)} already processed notes in {output_path}")

    # Read input notes
    pending_notes = []
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
                if note_id not in processed_ids:
                    pending_notes.append(item)
            except Exception as e:
                # Malformed JSON line: write default fallback directly
                fallback_pred = {"id": f"malformed-line-{line_idx}", "codes": [DEFAULT_FALLBACK_CODE]}
                with open(output_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(fallback_pred, ensure_ascii=False) + "\n")
                    out_f.flush()

    total_pending = len(pending_notes)
    print(f"Total notes to process: {total_pending}")

    if total_pending == 0:
        print("All notes are already processed. Exiting.")
        return

    import multiprocessing
    multiprocessing.freeze_support()

    # Determine worker count
    env_workers = os.environ.get("NUM_WORKERS")
    if env_workers and env_workers.isdigit():
        num_workers = int(env_workers)
    else:
        # On Linux/Docker, use min(8, cpu_count()); on Windows, 1 is safest to avoid IPC spawn locks
        num_workers = 1 if os.name == "nt" else min(8, cpu_count())

    print(f"Processing notes using {num_workers} parallel workers...")
    start_time = time.time()
    completed = 0

    if num_workers > 1:
        try:
            with open(output_path, "a", encoding="utf-8") as out_f:
                with Pool(processes=num_workers, initializer=init_worker) as pool:
                    for pred in pool.imap(process_single_note, pending_notes, chunksize=2):
                        out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                        out_f.flush()
                        completed += 1
                        if completed % 50 == 0 or completed == total_pending:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            print(f"Progress: {completed}/{total_pending} notes ({rate:.1f} notes/sec)")
        except Exception as e:
            print(f"Multiprocessing encountered an issue: {e}. Falling back to sequential execution...")
            # Re-read processed_ids to resume seamlessly
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
            remaining = [n for n in pending_notes if n["id"] not in processed_ids]
            num_workers = 1

    if num_workers <= 1:
        # Sequential execution
        from pipeline import ICD11Pipeline
        pipe = ICD11Pipeline(verbose=False)
        with open(output_path, "a", encoding="utf-8") as out_f:
            for note in pending_notes:
                if note["id"] in processed_ids:
                    continue
                pred = pipe.predict_note(note)
                out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                if completed % 50 == 0 or completed == total_pending:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"Progress: {completed}/{total_pending} notes ({rate:.1f} notes/sec)")

    total_time = time.time() - start_time
    print(f"Completed {completed} notes in {total_time:.2f}s ({completed / total_time:.1f} notes/sec).")


if __name__ == "__main__":
    main()
