#!/bin/bash
# =============================================================================
# submit_all_models.sh — Submit one eval_lcb_v6.slurm job per model.
#
# Run from the project root on the Hyak LOGIN NODE:
#   cd /gscratch/scrubbed/$USER/benchmarking-code-llms
#   bash /path/to/repo/scripts/submit_all_models.sh
#
# Each model gets its own Slurm job with a descriptive --job-name.
# Monitor with: squeue -u $USER
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_SCRIPT="$REPO_ROOT/scripts/eval_lcb_v6.slurm"
LOG_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms/logs"
mkdir -p "$LOG_DIR"

# ── Model definitions: NAME | HF_ID | JOB_LABEL ──────────────────────────────
declare -A MODELS
MODELS["qwen36_fp8"]="Qwen/Qwen3.6-35B-A3B-FP8"
MODELS["north_mini_w4a16"]="CohereLabs/North-Mini-Code-1.0-w4a16"
MODELS["devstral_small2"]="mistralai/Devstral-Small-2-24B-Instruct-2512"
MODELS["laguna_xs2_nvfp4"]="poolside/Laguna-XS.2-NVFP4"
MODELS["gemma4_27b"]="google/gemma-4-27b-it"   # TODO: verify HF ID

echo "Submitting LCB v6 eval jobs..."
echo ""

for JOB_LABEL in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$JOB_LABEL]}"
  JOB_ID=$(sbatch \
    --job-name="lcb_${JOB_LABEL}" \
    --output="$LOG_DIR/lcb_${JOB_LABEL}_%j.out" \
    --error="$LOG_DIR/lcb_${JOB_LABEL}_%j.err" \
    --export="MODEL_NAME=${MODEL_NAME}" \
    "$SLURM_SCRIPT" | awk '{print $NF}')
  echo "  Submitted $JOB_LABEL (${MODEL_NAME}) → job $JOB_ID"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel -u \$USER"
