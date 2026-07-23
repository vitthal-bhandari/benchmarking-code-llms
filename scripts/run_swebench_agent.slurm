#!/bin/bash
#SBATCH --job-name=swebench_agent
#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# run_swebench_agent.slurm — drives mini-swe-agent against an already-running
# vllm_serve job. CPU-only: this job just makes HTTP calls to the vLLM server
# and builds/runs Singularity sandboxes; it does not need a GPU itself. 
#
# Prerequisite: a serve_vllm.slurm job must already be RUNNING (check with
# `squeue --me`) and api_override.yaml must point at its node.
#
# Usage:
#   sbatch --export=SLICE="0:20",WORKERS=4,OUTPUT_DIR=validate_run1 \
#          scripts/run_swebench_agent.slurm
#   sbatch --export=SLICE="0:500",WORKERS=4,OUTPUT_DIR=full_qwen36 \
#          scripts/run_swebench_agent.slurm
#   # running a second model against its own server, in parallel:
#   sbatch --export=MODEL_NAME="hosted_vllm/poolside/Laguna-XS.2-NVFP4",\
#          TOOL_CALL_PARSER=poolside_v1,API_OVERRIDE=api_override_poolside.yaml,\
#          SLICE="0:20",WORKERS=4,OUTPUT_DIR=run_poolside_20 \
#          scripts/run_swebench_agent.slurm
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

MODEL_NAME="${MODEL_NAME:-hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8}"
SLICE="${SLICE:-0:20}"
WORKERS="${WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-run_$(date +%Y%m%d_%H%M%S)}"
API_OVERRIDE="${API_OVERRIDE:-api_override.yaml}"

source agent-venv/bin/activate

export MSWEA_COST_TRACKING='ignore_errors'

# Hyak routes outbound HTTP through a Squid proxy (klone-dip1) by default,
# including requests between two compute nodes on the cluster's own internal
# network. Under concurrent workers this intermittently fails with
# "Connection refused" (litellm.ServiceUnavailableError wrapping a Squid
# ERR_CONNECT_FAIL page) — pure network flakiness, not a model/agent bug.
# Bypass the proxy only for Hyak's internal range so requests to the vLLM
# node go direct; leave the proxy in place for genuine external traffic
# (Singularity still needs it to pull SWE-Bench images from docker.io).
export NO_PROXY="10.64.0.0/16,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

echo ">>> Model: $MODEL_NAME"
echo ">>> Slice: $SLICE"
echo ">>> Workers: $WORKERS"
echo ">>> Output: $OUTPUT_DIR"
echo ">>> $API_OVERRIDE points at:"
grep api_base "$API_OVERRIDE"

LITELLM_MODEL_REGISTRY_PATH=registry.json mini-extra swebench \
  -m "$MODEL_NAME" \
  -c swebench.yaml -c "$API_OVERRIDE" \
  --subset verified --split test --slice "$SLICE" \
  --environment-class singularity --workers "$WORKERS" \
  --output "$OUTPUT_DIR/"

echo ""
echo ">>> Done. Predictions at $OUTPUT_DIR/preds.json"
