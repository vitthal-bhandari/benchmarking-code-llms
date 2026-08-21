#!/bin/bash
# =============================================================================
# run_apptainer_eval.slurm — Score a SWE-bench run on UW Klone via Apptainer.
#
# CPU-ONLY job (no GPU): patch-apply + test execution inside a per-instance
# Apptainer container. Klone allows CPU-only jobs (unlike Tillicum), and its
# compute nodes are x86_64, so the swebench x86_64 images run natively — no
# QEMU emulation and no --platform workaround (both needed on the Mac's M3).
#
# Defaults to the checkpoint partition (ckpt-all): ~1024 idle CPUs there, so it
# schedules immediately, and the runner is resumable (per-instance report cache
# under $WORKDIR/reports), so a preemption just means resubmitting — it picks up
# where it left off. Override PARTITION=compute for a non-preemptible run.
#
# Submit (after scripts/install_klone_eval_venv.sh has been run once):
#   sbatch --export=PREDS=runs/run_qwen_20_216375/preds.json,RUN_ID=run_qwen_20_216375 \
#     scripts/run_apptainer_eval.slurm
#
# Optional --export knobs: WORKERS (default 4), INSTANCE_IDS (comma list subset),
# OVERLAY_MODE (overlay|tmpfs, default overlay), TIMEOUT (per-instance s),
# DATASET (default SWE-bench/SWE-bench_Verified).
# =============================================================================
#SBATCH --job-name=sweb-eval
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/apptainer_eval_%j.out

set -euo pipefail

# Submit from the repo root (sbatch scripts/run_apptainer_eval.slurm), so
# SLURM_SUBMIT_DIR is the project dir.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

: "${PREDS:?set PREDS=path/to/preds.json via --export}"
: "${RUN_ID:?set RUN_ID=<run id> via --export}"
WORKERS="${WORKERS:-4}"
OVERLAY_MODE="${OVERLAY_MODE:-overlay}"
TIMEOUT="${TIMEOUT:-1800}"
DATASET="${DATASET:-SWE-bench/SWE-bench_Verified}"

SCRATCH="/gscratch/scrubbed/$USER"
# All heavy/scratch state off of the 10GB home quota.
export APPTAINER_CACHEDIR="$SCRATCH/.cache/apptainer"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TMPDIR="$SCRATCH/tmp"
export APPTAINER_TMPDIR="$SCRATCH/tmp"
export APPTAINER_EVAL_WORKDIR="$SCRATCH/apptainer_eval/$RUN_ID"
export APPTAINER_EVAL_SIFDIR="$SCRATCH/apptainer_eval/$RUN_ID/sif"
mkdir -p "$APPTAINER_CACHEDIR" "$HF_HOME" "$TMPDIR" "$APPTAINER_EVAL_SIFDIR" logs

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || \
  echo ">>> WARNING: could not module-load apptainer; assuming it's on PATH"
command -v apptainer >/dev/null || { echo "ERROR: apptainer not found"; exit 1; }

source klone-eval-venv/bin/activate

echo ">>> Scoring $RUN_ID from $PREDS (workers=$WORKERS, overlay=$OVERLAY_MODE)"
EXTRA_ARGS=()
[ -n "${INSTANCE_IDS:-}" ] && EXTRA_ARGS+=(--instance-ids "$INSTANCE_IDS")

python scripts/run_apptainer_eval.py \
  --preds "$PREDS" \
  --run-id "$RUN_ID" \
  --dataset "$DATASET" \
  --workers "$WORKERS" \
  --overlay-mode "$OVERLAY_MODE" \
  --timeout "$TIMEOUT" \
  --report-dir local-eval-reports \
  "${EXTRA_ARGS[@]}"

echo ">>> Report written to local-eval-reports/$RUN_ID.json"
