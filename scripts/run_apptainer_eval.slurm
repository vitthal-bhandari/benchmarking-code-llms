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
# schedules immediately, and the runner is resumable (per-instance reports cached
# under logs/apptainer_eval/<run_id>/, tracked in git), so a preemption just means
# resubmitting — it picks up where it left off. Edit the #SBATCH --partition line
# below to compute/cpu-g2 for a non-preemptible run.
#
# Submit (after scripts/install_klone_eval_venv.sh has been run once):
#   sbatch --export=PREDS=runs/run_qwen_20_216375/preds.json,RUN_ID=run_qwen_20_216375 \
#     scripts/run_apptainer_eval.slurm
#
# Optional --export knobs: WORKERS (default 4), INSTANCE_IDS (comma list subset),
# OVERLAY_MODE (overlay|tmpfs, default overlay), TIMEOUT (per-instance s),
# DATASET (default SWE-bench/SWE-bench_Verified), OVERWRITE=1 (re-grade every
# instance from scratch, ignoring cached per-instance reports — use it to redo a
# run whose earlier attempt errored; omit it to resume/skip already-graded ones).
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

# Docker Hub auth: raises the pull limit 100->200/6h. Combined with the shared,
# persistent APPTAINER_CACHEDIR above, the 125 UNIQUE images download once and
# every model + every parallel slice reuses them — so scoring 3 models doesn't
# multiply into 375 pulls and hit the rate limit. Put DOCKER_USERNAME +
# DOCKER_TOKEN (a Docker Hub access token) in .env. Harmless if absent.
[ -f .env ] && source <(grep -v '^#' .env | sed 's/ *#.*//')
if [ -n "${DOCKER_USERNAME:-}" ] && [ -n "${DOCKER_TOKEN:-}" ]; then
  export APPTAINER_DOCKER_USERNAME="$DOCKER_USERNAME"; export APPTAINER_DOCKER_PASSWORD="$DOCKER_TOKEN"
  export SINGULARITY_DOCKER_USERNAME="$DOCKER_USERNAME"; export SINGULARITY_DOCKER_PASSWORD="$DOCKER_TOKEN"
  echo ">>> Docker Hub auth enabled for $DOCKER_USERNAME"
else
  echo ">>> WARNING: no DOCKER_USERNAME/DOCKER_TOKEN in .env — anonymous Docker Hub pulls may rate-limit"
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || \
  echo ">>> WARNING: could not module-load apptainer; assuming it's on PATH"
command -v apptainer >/dev/null || { echo "ERROR: apptainer not found"; exit 1; }

source klone-eval-venv/bin/activate

echo ">>> Scoring $RUN_ID from $PREDS (workers=$WORKERS, overlay=$OVERLAY_MODE, slice=${EVAL_SLICE:-all}, prefetch=${PREFETCH:-0})"
EXTRA_ARGS=()
[ -n "${INSTANCE_IDS:-}" ] && EXTRA_ARGS+=(--instance-ids "$INSTANCE_IDS")
[ "${OVERWRITE:-0}" = "1" ] && EXTRA_ARGS+=(--overwrite)
# PREFETCH=1: pull the images into the shared cache and exit (no scoring). Run
# once before the parallel scoring jobs. EVAL_SLICE (start:end) is read directly
# from the env by the python (splits one run across parallel jobs).
[ "${PREFETCH:-0}" = "1" ] && EXTRA_ARGS+=(--prefetch-only)

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
