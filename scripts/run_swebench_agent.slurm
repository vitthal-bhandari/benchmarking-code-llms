#!/bin/bash
#SBATCH --job-name=swebench_agent
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# run_swebench_agent.slurm — drives mini-swe-agent against an already-running
# vllm_serve job. Makes HTTP calls to the vLLM server and builds/runs Apptainer
# sandboxes; it does not use a GPU itself.
#
# NOT SUBMITTABLE AS-IS ON TILLICUM: "CPU-only jobs are not allowed" — every
# job needs --gres=gpu:h200:1+. For full scored runs, use
# scripts/serve_and_run_swebench.slurm instead, which starts the server and
# drives it in one GPU job (so the driver doesn't silently bill a second GPU
# it never uses). This script is kept for clusters that do allow CPU-only jobs
# (it was the original two-job Klone pattern), or as a reference for manually
# driving an already-running interactive serve_vllm.slurm session on Tillicum
# by adding --gres=gpu:h200:1 yourself (accept the extra GPU-hour cost).
#
# Prerequisite: a serve_vllm.slurm job is RUNNING and configs/api_override.yaml
# points at its node.
#
# Usage:
#   sbatch --export=SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_qwen_20 \
#          scripts/run_swebench_agent.slurm
#   # a second model against its own server:
#   sbatch --export=MODEL_NAME="hosted_vllm/google/gemma-4-26B-A4B-it",\
#          API_OVERRIDE=configs/api_override_gemma4.yaml,\
#          SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_gemma4_20 \
#          scripts/run_swebench_agent.slurm
#
# Put new runs under runs/ (gitignored).
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gpfs/projects/stf/$USER/benchmarking-code-llms"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

MODEL_NAME="${MODEL_NAME:-hosted_vllm/Qwen/Qwen3.6-35B-A3B}"
SLICE="${SLICE:-0:20}"
WORKERS="${WORKERS:-4}"
# Job id appended so repeated runs get distinct dirs (mini-swe-agent resumes an
# existing OUTPUT_DIR rather than starting fresh).
OUTPUT_DIR="${OUTPUT_DIR:-runs/run}"
OUTPUT_DIR="${OUTPUT_DIR}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
API_OVERRIDE="${API_OVERRIDE:-configs/api_override.yaml}"

# Keep the SWE-Bench dataset cache off the 10GB home quota.
export HF_HOME="/gpfs/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

source agent-venv/bin/activate

export MSWEA_COST_TRACKING='ignore_errors'

# If the cluster routes outbound HTTP through a proxy, bypass it for the vLLM
# server. httpx matches NO_PROXY by STRING and won't resolve a hostname against
# a CIDR, so we add the api_base host VERBATIM (a bare CIDR silently failed to
# bypass a hostname api_base on Klone, sending requests through a flaky Squid
# proxy). Harmless if Tillicum has no proxy.
API_HOST="$(grep -oE 'https?://[^:/]+' "$API_OVERRIDE" | head -1 | sed -E 's#https?://##')"
export NO_PROXY="${API_HOST},localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
echo ">>> no_proxy set to: $NO_PROXY  (api host parsed: ${API_HOST:-NONE})"

echo ">>> Model: $MODEL_NAME | Slice: $SLICE | Workers: $WORKERS | Output: $OUTPUT_DIR"
echo ">>> $API_OVERRIDE points at:"; grep api_base "$API_OVERRIDE"

LITELLM_MODEL_REGISTRY_PATH=configs/registry.json mini-extra swebench \
  -m "$MODEL_NAME" \
  -c swebench.yaml -c "$API_OVERRIDE" \
  --subset verified --split test --slice "$SLICE" \
  --environment-class singularity --workers "$WORKERS" \
  --output "$OUTPUT_DIR/"

echo ""
echo ">>> Done. Predictions at $OUTPUT_DIR/preds.json"
