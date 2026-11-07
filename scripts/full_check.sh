#!/usr/bin/env bash
# Merge gate verification script
# Runs:
# 1. ruff linter
# 2. unit tests
# 3. docker_context_check.py --run (verifies strict build context and lack of constant fallbacks)
# 4. docker compose run coder (if docker daemon running) on a clean tree with score.py grading
# Exits non-zero if ANY step fails.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "                   STAGE 1: RUFF LINTER"
echo "============================================================"
ruff check .
echo "[PASS] Ruff check passed cleanly."

echo "============================================================"
echo "                   STAGE 2: UNIT TESTS"
echo "============================================================"
python -m unittest discover tests
echo "[PASS] Unit tests passed."

echo "============================================================"
echo "          STAGE 3: ISOLATED DOCKER CONTEXT CHECK"
echo "============================================================"
python scripts/docker_context_check.py --run
echo "[PASS] Docker context and runtime execution passed."

echo "============================================================"
echo "          STAGE 4: DOCKER COMPOSE CLEAN RUN (IF DAEMON UP)"
echo "============================================================"
if docker info >/dev/null 2>&1; then
    echo "Docker daemon is active. Running clean build and execution..."
    CLEAN_TMP=$(mktemp -d)
    trap 'rm -rf "$CLEAN_TMP"' EXIT
    
    echo "Cloning clean worktree into $CLEAN_TMP..."
    git archive HEAD | tar -x -C "$CLEAN_TMP"
    
    mkdir -p "$CLEAN_TMP/in" "$CLEAN_TMP/out"
    python -c '
import json
with open("data/dev_notes.jsonl") as f, open("'"$CLEAN_TMP"'/in/notes.jsonl", "w") as g:
    for line in f:
        d = json.loads(line)
        g.write(json.dumps({"id": d["id"], "text": d["text"]}) + "\n")
'
    (
        cd "$CLEAN_TMP"
        docker compose build coder
        docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl
        python score.py --gold data/dev_notes.jsonl --pred "$CLEAN_TMP/out/predictions.jsonl" --mms data/mms-2026-01.json
    )
    echo "[PASS] Docker compose test on clean tree passed!"
else
    echo "[INFO] Docker daemon is not active on host. Stage 3 (docker_context_check.py --run) verified container isolation."
fi

echo "============================================================"
echo "       ALL VERIFICATION CHECKS PASSED (MERGE GATE OK)"
echo "============================================================"
