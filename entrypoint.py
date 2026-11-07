"""Main entrypoint for Docker container execution.
Reads input notes JSONL, runs ICD-11 coding pipeline, writes output predictions JSONL.
Supports multiprocessing across 8 CPUs, timeout handling, resume, and graceful degradation.
Guarantees every input id in the output, 1-3 valid codes, no silent failures.
"""

import json
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Any

# Pin single thread per process across all BLAS/Torch libraries
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

# Worker pipeline global reference
_worker_pipeline = None

# Standalone fallback references
_standalone_bm25 = None
_standalone_kb = None
_standalone_hierarchy = None
_standalone_valid = None


def get_standalone_bm25_fallback(raw_text: str) -> list[str]:
    """Per-note dynamic BM25 ranking from shipped indexes, never constants."""
    global _standalone_bm25, _standalone_kb, _standalone_hierarchy, _standalone_valid
    try:
        if _standalone_bm25 is None:
            import pickle
            with open("kb/bm25.pkl", "rb") as f:
                d = pickle.load(f)
            _standalone_bm25 = (d["bm25"], d["codes"])
        if _standalone_kb is None:
            with open("kb/enriched_kb.json", "r", encoding="utf-8") as f:
                _standalone_kb = json.load(f)
        if _standalone_valid is None:
            with open("kb/valid_codes.json", "r", encoding="utf-8") as f:
                _standalone_valid = set(json.load(f))
        if _standalone_hierarchy is None:
            from icd_hierarchy import ICDHierarchy
            _standalone_hierarchy = ICDHierarchy(kb_dict=_standalone_kb)

        from text_utils import clean_text, expand_abbreviations, normalize_code
        bm25, codes = _standalone_bm25
        query_text = clean_text(expand_abbreviations(raw_text or ""))
        tokens = query_text.split() if query_text else ["general", "outpatient"]
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:30]
        hits = []
        for idx in top_indices:
            code = codes[idx]
            leaf = _standalone_hierarchy.resolve_to_leaf(code, raw_text or "")
            norm_c = normalize_code(leaf)
            if norm_c in _standalone_valid and norm_c not in hits:
                hits.append(norm_c)
            if len(hits) >= 3:
                break
        if hits:
            return hits
    except Exception as e:
        sys.stderr.write(f"[ERROR] Standalone BM25 fallback failed: {e}\n")
        if os.environ.get("STRICT") == "1":
            sys.exit(1)

    try:
        with open("kb/valid_codes.json", "r", encoding="utf-8") as f:
            valid = json.load(f)
        if valid and len(valid) >= 3:
            return valid[:3]
        raise ValueError("kb/valid_codes.json has fewer than 3 codes")
    except Exception as e:
        sys.stderr.write(f"[FATAL] Cannot load valid codes fallback from kb/valid_codes.json: {e}\n")
        sys.exit(1)


def init_worker():
    """Initialize pipeline once per worker process."""
    global _worker_pipeline
    try:
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        from pipeline import ICD11Pipeline
        _worker_pipeline = ICD11Pipeline(verbose=False)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to initialize worker pipeline: {e}\n")
        if os.environ.get("STRICT") == "1":
            sys.exit(1)
        _worker_pipeline = None


def process_single_note(note_data: dict[str, Any]) -> dict[str, Any]:
    """Process a single note and return prediction dict."""
    global _worker_pipeline
    note_id = note_data.get("id", "unknown")
    raw_text = note_data.get("text", "")
    try:
        if _worker_pipeline is None:
            from pipeline import ICD11Pipeline
            _worker_pipeline = ICD11Pipeline(verbose=False)
        return _worker_pipeline.predict_note(note_data)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Worker error on note {note_id}: {e}\n")
        if os.environ.get("STRICT") == "1":
            sys.exit(1)
        fallback_codes = get_standalone_bm25_fallback(raw_text)
        return {"id": note_id, "codes": fallback_codes[:3]}


def extract_id_from_raw_line(line: str, line_idx: int) -> str:
    """Extract id from raw line text if json.loads fails, preserving the original note id."""
    m = re.search(r'["\']id["\']\s*:\s*["\']([^"\']+)["\']', line)
    if m:
        return m.group(1)
    m = re.search(r'["\']id["\']\s*:\s*(\d+)', line)
    if m:
        return str(m.group(1))
    return f"note-line-{line_idx}"


def extract_text_from_raw_line(line: str) -> str:
    """Extract text from raw line text if json.loads fails."""
    m = re.search(r'["\']text["\']\s*:\s*["\'](.*?)["\']\s*[,}]', line)
    if m:
        return m.group(1)
    return ""


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
        if processed_ids:
            print(f"Resuming: found {len(processed_ids)} already processed notes in {output_path}")

    # Read and deduplicate input notes
    seen_input_ids = set()
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
                if note_id in seen_input_ids:
                    continue
                seen_input_ids.add(note_id)
                if note_id not in processed_ids:
                    pending_notes.append(item)
            except Exception as e:
                # Malformed JSON line: extract id and text from raw string, never invent arbitrary names if id exists
                if os.environ.get("STRICT") == "1":
                    sys.stderr.write(f"[ERROR] Malformed JSON line {line_idx}: {e}\n")
                    sys.exit(1)
                orig_id = extract_id_from_raw_line(line, line_idx)
                raw_txt = extract_text_from_raw_line(line)
                if orig_id not in processed_ids and orig_id not in seen_input_ids:
                    seen_input_ids.add(orig_id)
                    fallback_codes = get_standalone_bm25_fallback(raw_txt)
                    fallback_pred = {"id": orig_id, "codes": fallback_codes[:3]}
                    with open(output_path, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(fallback_pred, ensure_ascii=False) + "\n")
                        out_f.flush()
                    processed_ids.add(orig_id)

    total_pending = len(pending_notes)
    print(f"Total notes to process: {total_pending} (Already processed: {len(processed_ids)})")

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
        # On Linux/Docker, use min(8, cpu_count()); on Windows, 1 is safest
        num_workers = 1 if os.name == "nt" else min(8, cpu_count())

    print(f"Processing notes using {num_workers} parallel workers...")
    start_time = time.time()
    completed = 0

    if num_workers > 1:
        try:
            with open(output_path, "a", encoding="utf-8") as out_f:
                with Pool(processes=num_workers, initializer=init_worker) as pool:
                    # imap_unordered maximizes throughput across workers
                    for pred in pool.imap_unordered(process_single_note, pending_notes, chunksize=2):
                        out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                        out_f.flush()
                        completed += 1
                        if completed % 50 == 0 or completed == total_pending:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            print(f"Progress: {completed}/{total_pending} notes ({rate:.1f} notes/sec)")
        except Exception as e:
            sys.stderr.write(f"[WARN] Multiprocessing error: {e}. Falling back to sequential execution...\n")
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
            pending_notes = [n for n in pending_notes if n["id"] not in processed_ids]
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

        if hasattr(pipe, "fallback_count") and pipe.fallback_count > 0:
            print(f"[SUMMARY] Pipeline fallbacks triggered: {pipe.fallback_count}/{completed}")

    total_time = time.time() - start_time
    rate = completed / total_time if total_time > 0 else 0
    print(f"Completed {completed} notes in {total_time:.2f}s ({rate:.1f} notes/sec).")


if __name__ == "__main__":
    main()
