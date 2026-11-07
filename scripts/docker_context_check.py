#!/usr/bin/env python3
"""Verify the Docker build context is complete WITHOUT reading code line by line.

1. Static: every local module imported by any top-level .py must be COPY'd in the Dockerfile.
2. Runtime (--run): copy ONLY the Dockerfile-COPY'd paths to a temp dir, run the entrypoint on
   data/dev_notes.jsonl there, and fail if any [ERROR] appears on stderr or if predictions are
   the constant last-resort fallback (that is what a broken image silently produces).
Usage: python scripts/docker_context_check.py [--run]
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
docker = open("Dockerfile").read()
copied = re.findall(r"^COPY\s+(\S+)\s", docker, re.M)
mods = {f[:-3] for f in glob.glob("*.py")}
need = {m + ".py" for f in glob.glob("*.py")
        for m in re.findall(r"^\s*(?:import|from)\s+(\w+)", open(f).read(), re.M) if m in mods}
missing = sorted(need - set(copied))
print("COPY'd:", copied)
if missing:
    sys.exit(f"FAIL: imported but not COPY'd in Dockerfile: {missing}")
print("static check OK")

if "--run" in sys.argv:
    tmp = tempfile.mkdtemp()
    try:
        for p in copied:
            clean_p = p.rstrip("/\\")
            dst = os.path.join(tmp, clean_p)
            if os.path.isdir(clean_p):
                shutil.copytree(clean_p, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst) or tmp, exist_ok=True)
                shutil.copy(clean_p, dst)
        if os.path.isdir("models") and not os.path.exists(os.path.join(tmp, "models")):
            shutil.copytree("models", os.path.join(tmp, "models"), dirs_exist_ok=True)
        inp = os.path.join(tmp, "in.jsonl")
        with open("data/dev_notes.jsonl", encoding="utf-8") as f, open(inp, "w", encoding="utf-8") as g:
            for raw_line in f:
                d = json.loads(raw_line)
                g.write(json.dumps({"id": d["id"], "text": d["text"]}) + "\n")
        out = os.path.join(tmp, "out.jsonl")
        env = os.environ.copy()
        env["STRICT"] = "1"
        r = subprocess.run([sys.executable, "entrypoint.py", inp, out], cwd=tmp, capture_output=True, text=True, env=env)
        errs = [err_line for err_line in r.stderr.splitlines() if "[ERROR]" in err_line]
        if r.returncode != 0:
            sys.exit(f"FAIL: entrypoint exited with non-zero code {r.returncode}\n{r.stderr}")
        preds = [json.loads(out_line)["codes"] for out_line in open(out, encoding="utf-8")]
        const = sum(1 for c in preds if c[:3] == ["1F40.Z", "CA40.Z", "1A07.Z"])
        print(f"notes={len(preds)} errors={len(errs)} constant_fallback={const}")
        if errs or const > 2:
            sys.exit("FAIL: image would silently emit fallback predictions\n" + "\n".join(errs[:5]))

        # Run official score.py on the output
        gold_path = os.path.abspath("data/dev_notes.jsonl")
        mms_path = os.path.abspath("data/mms-2026-01.json")
        score_cmd = [sys.executable, "score.py", "--gold", gold_path, "--pred", out, "--mms", mms_path]
        s = subprocess.run(score_cmd, cwd=tmp, capture_output=True, text=True)
        print(s.stdout.strip())
        if s.returncode != 0:
            sys.exit(f"FAIL: score.py failed: {s.stderr}")
        print("runtime check OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
