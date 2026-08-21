#!/bin/bash
# =============================================================================
# run_local_eval.sh — Score a run's preds.json with the OFFICIAL local
# SWE-bench harness (Docker on this Mac, via Colima), instead of sb-cli's
# hosted evaluator. sb-cli's cloud service has been returning
# completed_instances: 0 / failed_instances: 100% on every submission
# (confirmed via swe-bench/sb-cli#27, #28, #31 — a known, unresolved outage;
# see report.md), so its 0% Pass@1 numbers are not trustworthy. This script
# runs the same predictions through swebench's own harness locally instead.
#
# Usage:
#   scripts/run_local_eval.sh runs/run_qwen_100_merged.json run_qwen_100 [workers]
#
# Requires: scripts/install_eval_venv.sh has been run once (Colima + Docker +
# eval-venv). Generation itself still happens on Tillicum GPUs — this script
# only does CPU-side Docker evaluation, on your Mac.
# =============================================================================

set -euo pipefail

PREDS_JSON="${1:?Usage: run_local_eval.sh <preds.json> <run_id> [workers]}"
RUN_ID="${2:?Usage: run_local_eval.sh <preds.json> <run_id> [workers]}"
WORKERS="${3:-2}"
DATASET="${DATASET:-verified}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source eval-venv/bin/activate

mkdir -p local_eval/preds local-eval-reports
HARNESS_PREDS="local_eval/preds/${RUN_ID}.harness.json"

echo ">>> Converting predictions to harness format..."
python scripts/prepare_harness_preds.py "$PREDS_JSON" "$HARNESS_PREDS"

echo ">>> Pre-pulling amd64 eval images (Apple Silicon has no arm64 manifests on Docker Hub)..."
python scripts/prefetch_harness_images.py "$HARNESS_PREDS" --dataset "$DATASET"

echo ">>> Running official SWE-bench harness (run_id=$RUN_ID, workers=$WORKERS)..."
cd local_eval
swebench eval "$DATASET" -p "preds/${RUN_ID}.harness.json" --run-id "$RUN_ID" -j "$WORKERS"
cd "$PROJECT_DIR"

REPORT_GLOB=(local_eval/*."${RUN_ID}".json)
if [ -e "${REPORT_GLOB[0]}" ]; then
  cp "${REPORT_GLOB[0]}" "local-eval-reports/${RUN_ID}.json"
  echo ">>> Report copied to local-eval-reports/${RUN_ID}.json"
else
  echo ">>> WARNING: could not find a report file matching local_eval/*.${RUN_ID}.json"
fi
