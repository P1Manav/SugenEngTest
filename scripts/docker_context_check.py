#!/usr/bin/env python3
"""Verify the Docker build context is complete WITHOUT reading code line by line.

1. Static: every local module imported by any top-level .py must be COPY'd in the Dockerfile.
2. Runtime (--run): copy ONLY the Dockerfile-COPY'd paths to a temp dir, run the entrypoint on
   data/dev_notes.jsonl there, and fail if any [ERROR] appears on stderr or if predictions are
   the constant last-resort fallback (that is what a broken image silently produces).
Usage: python scripts/docker_context_check.py [--run]
"""
import glob, json, os, re, shutil, subprocess, sys, tempfile

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
    for p in copied:
        dst = os.path.join(tmp, p)
        os.makedirs(os.path.dirname(dst) or tmp, exist_ok=True)
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, dst)
    inp = os.path.join(tmp, "in.jsonl")
    with open("data/dev_notes.jsonl") as f, open(inp, "w") as g:
        for l in f:
            d = json.loads(l); g.write(json.dumps({"id": d["id"], "text": d["text"]}) + "\n")
    out = os.path.join(tmp, "out.jsonl")
    r = subprocess.run([sys.executable, "entrypoint.py", inp, out], cwd=tmp, capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if "[ERROR]" in l]
    preds = [json.loads(l)["codes"] for l in open(out)]
    const = sum(1 for c in preds if c[:3] == ["1F40.Z", "CA40.Z", "1A07.Z"])
    print(f"notes={len(preds)} errors={len(errs)} constant_fallback={const}")
    if errs or const > 2:
        sys.exit("FAIL: image would silently emit fallback predictions\n" + "\n".join(errs[:5]))
    print("runtime check OK")
