#!/bin/bash
#SBATCH --job-name=lcb_v6_eval
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# eval_lcb_v6.slurm — Run LiveCodeBench v6 code generation eval for one model.
#
# Required env vars (pass via --export):
#   MODEL_NAME   : the model_name key in lm_styles.py
#                  e.g. "Qwen/Qwen3.6-35B-A3B-FP8"
#
# Optional env vars:
#   TENSOR_PARALLEL_SIZE : number of GPUs for tensor parallelism (default: 1)
#   QUANTIZATION         : vLLM quantization flag, e.g. "fp8", "gptq", "awq"
#                          (default: auto-detected from model name)
#
# Example:
#   sbatch --export=MODEL_NAME="Qwen/Qwen3.6-35B-A3B-FP8" \
#          --job-name=lcb_qwen36 \
#          scripts/eval_lcb_v6.slurm
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ── Validate required input ───────────────────────────────────────────────────
if [ -z "${MODEL_NAME:-}" ]; then
  echo "ERROR: MODEL_NAME env var not set. Use --export=MODEL_NAME=<name>" >&2
  exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"
RESULTS_DIR="$PROJECT_DIR/results/lcb_v6"
mkdir -p "$PROJECT_DIR/logs" "$RESULTS_DIR"

# ── Cache dirs ────────────────────────────────────────────────────────────────
export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
export HF_HOME="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# FlashInfer JIT-compiles some kernels at runtime and needs nvcc. Hyak's system
# CUDA modules only go up to 12.9.1, but our torch build is cu130 — using them
# would be a toolkit/ABI mismatch. Point at the CUDA 13 nvcc pip already
# installed inside the venv (nvidia-cuda-nvcc) instead of a system module.
export CUDA_HOME="$LCB_DIR/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"

# The FlashInfer sampling kernel JIT-compiles via nvcc inside vLLM's spawned
# EngineCore worker, which doesn't resolve our CUDA_HOME. Use vLLM's native
# sampler instead — slightly slower sampling, no compilation at all.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Load HF token from .env
[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# ── Tensor parallel size ──────────────────────────────────────────────────────
TP="${TENSOR_PARALLEL_SIZE:-1}"

# Quantization is auto-detected by vLLM from the model's config.json.
# LCB's main.py has no --quantization flag; --dtype auto handles all formats.
QUANTIZATION=""

cd "$LCB_DIR"

# Activate the venv directly. Do NOT use `uv run` — it re-syncs the env to the
# project's pinned deps and reverts our git-source transformers (needed for
# qwen3_5_moe) back to the release that doesn't recognize the arch.
source .venv/bin/activate

echo "================================================================="
echo " LiveCodeBench v6 — Code Generation Eval"
echo " Model      : $MODEL_NAME"
echo " Quantization: auto-detected by vLLM from model config"
echo " TP size     : $TP"
echo " Job ID      : ${SLURM_JOB_ID:-local}"
echo " Start       : $(date)"
echo "================================================================="

# ── Sanity check ──────────────────────────────────────────────────────────────
python -c "
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
from lcb_runner.lm_styles import LanguageModelStore
assert '$MODEL_NAME' in LanguageModelStore, \
    f'Model $MODEL_NAME not found in LanguageModelStore. Check lcb_patch/lm_styles_additions.py'
print('Model registered: OK')
"

# ── Build eval command ────────────────────────────────────────────────────────
EVAL_CMD=(
  python -m lcb_runner.runner.main
  --model "$MODEL_NAME"
  --scenario codegeneration
  --evaluate
  --release_version release_v6
  --n 1
  --temperature 0
  --tensor_parallel_size "$TP"
  --use_cache
  --dtype auto
  --trust_remote_code
)

echo ""
echo ">>> Running: ${EVAL_CMD[*]}"
echo ""

"${EVAL_CMD[@]}"

# ── Copy results to our results dir ──────────────────────────────────────────
MODEL_SLUG=$(echo "$MODEL_NAME" | tr '/' '_')
DEST="$RESULTS_DIR/$MODEL_SLUG"
mkdir -p "$DEST"

# LCB writes outputs to lcb_dir/output/ by default
if [ -d "$LCB_DIR/output" ]; then
  cp -r "$LCB_DIR/output/"*"${MODEL_SLUG}"* "$DEST/" 2>/dev/null || \
  cp -r "$LCB_DIR/output/" "$DEST/raw_output/" 2>/dev/null || true
fi

echo ""
echo ">>> Eval complete for $MODEL_NAME at $(date)"
echo ">>> Results saved to $DEST"
