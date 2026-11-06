#!/usr/bin/env bash
set -euo pipefail

DEV_NOTES="${1:-data/dev_notes.jsonl}"
PRED_OUT="${2:-out/dev_predictions.jsonl}"
MMS_PATH="${3:-data/mms-2026-01.json}"

mkdir -p out

echo "============================================================"
echo "RUNNING ICD-11 EVALUATION PIPELINE"
echo "Input notes:  ${DEV_NOTES}"
echo "Predictions:  ${PRED_OUT}"
echo "============================================================"

python entrypoint.py "${DEV_NOTES}" "${PRED_OUT}"

echo "============================================================"
echo "SCORING WITH OFFICIAL GRADER"
echo "============================================================"

python score.py --gold "${DEV_NOTES}" --pred "${PRED_OUT}" --mms "${MMS_PATH}"
