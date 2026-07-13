#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gpus=l40s:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# serve_vllm.slurm — OpenAI-compatible vLLM server for agent baselines
# (mini-swe-agent on SWE-Bench Verified talks to this endpoint).
#
# Usage:
#   sbatch scripts/serve_vllm.slurm
#   sbatch --export=MODEL_NAME="CohereLabs/North-Mini-Code-1.0-w4a16" scripts/serve_vllm.slurm
#
# Find the endpoint after it starts:
#   squeue --me                       # get the node, e.g. g3113
#   grep "Uvicorn running" logs/vllm_serve_<jobid>.out
#   → base URL is http://<node>:8000/v1  (reachable from login/other nodes)
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8000}"
# Model defaults to 262144 max context, but after loading 35B FP8 weights on a
# single L40S (48GB) there's only ~4.6 GiB left for KV cache — not enough for
# that full window. 64K is generous for mini-swe-agent's tool-call trajectories
# and leaves plenty of KV cache headroom. Override via --export if needed.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"
mkdir -p "$PROJECT_DIR/logs"

export HF_HOME="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export CUDA_HOME="$LCB_DIR/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# Native sampler — avoids FlashInfer nvcc JIT in the spawned EngineCore worker.
export VLLM_USE_FLASHINFER_SAMPLER=0
# flashinfer-python and flashinfer-cubin got installed at mismatched versions
# (0.6.14 / 0.6.13) during earlier debugging; this bypasses their hard version
# check so `import flashinfer` doesn't crash. TODO: pin both to matching
# versions properly once this pipeline is proven end-to-end.
export FLASHINFER_DISABLE_VERSION_CHECK=1

[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# Reuse the LCB venv — it already has the working vllm/transformers combo.
cd "$LCB_DIR"
source .venv/bin/activate

echo ">>> Serving $MODEL_NAME on $(hostname):$PORT (job ${SLURM_JOB_ID:-local})"
vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype auto \
  --trust-remote-code \
  --enforce-eager
