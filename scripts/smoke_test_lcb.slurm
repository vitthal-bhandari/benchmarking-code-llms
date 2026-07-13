#!/bin/bash
#SBATCH --job-name=lcb_smoke
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gpus=l40s:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# smoke_test_lcb.slurm — Quick sanity check before full v6 eval runs.
#
# What this tests:
#   - vLLM loads the model without OOM
#   - lm_styles patch is applied (model is recognized)
#   - Prompt formatting is correct for the model's LMStyle
#   - LCB generates code and evaluates it end to end
#   - Results land in the expected output directory
#
# Uses release_v1 (400 problems, lite mode by default) — much faster than v6.
# Expect ~45-90 min depending on GPU queue and model size.
#
# Default model: Qwen3.6-35B-A3B-FP8 (most likely to work cleanly at FP8)
# Override via --export:
#   sbatch --export=MODEL_NAME="CohereLabs/North-Mini-Code-1.0-w4a16" \
#          scripts/smoke_test_lcb.slurm
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"
mkdir -p "$PROJECT_DIR/logs"

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

[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# Quantization is auto-detected by vLLM from the model's config.json.
# LCB's main.py has no --quantization flag; use --dtype auto (default).
QUANTIZATION=""

cd "$LCB_DIR"

# Activate the venv directly. Do NOT use `uv run` — it re-syncs the env to the
# project's pinned deps and reverts our git-source transformers (needed for
# qwen3_5_moe) back to the release that doesn't recognize the arch.
source .venv/bin/activate

echo ">>> CUDA_HOME=$CUDA_HOME"
echo ">>> which nvcc: $(which nvcc 2>&1)"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print('>>> torch cpp_extension CUDA_HOME:', CUDA_HOME)"

echo "================================================================="
echo " LCB SMOKE TEST"
echo " Model       : $MODEL_NAME"
echo " Quant       : ${QUANTIZATION:-none}"
echo " Release     : release_v1 (400 problems, lite mode)"
echo " Job ID      : ${SLURM_JOB_ID:-local}"
echo " Node        : $(hostname)"
echo " Start       : $(date)"
echo "================================================================="

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Sanity checks..."
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA — wrong partition?'
print('  CUDA OK:', torch.cuda.get_device_name(0))
print('  VRAM   :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

python -c "
from lcb_runner.lm_styles import LanguageModelStore
model = '$MODEL_NAME'
assert model in LanguageModelStore, f'Model not registered: {model}. Check lcb_patch.'
print('  Model registered OK:', LanguageModelStore[model].model_repr)
"

# ── Run eval ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> Starting generation + evaluation..."

EVAL_CMD=(
  python -m lcb_runner.runner.main
  --model "$MODEL_NAME"
  --scenario codegeneration
  --evaluate
  --release_version release_v1
  --n 1
  --temperature 0
  --tensor_parallel_size 1
  --use_cache
  --dtype auto
  --trust_remote_code
)

echo ">>> CMD: ${EVAL_CMD[*]}"
echo ""
"${EVAL_CMD[@]}"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo " SMOKE TEST COMPLETE for $MODEL_NAME"
echo " End: $(date)"
echo " Output dir: $LCB_DIR/output/"
echo ""
echo " Results files:"
ls -lh "$LCB_DIR/output/" 2>/dev/null || echo "  (no output dir found)"
echo ""
echo " Next step: if pass@1 looks reasonable (>0), run full v6 eval."
echo "   bash scripts/submit_all_models.sh"
echo "================================================================="
