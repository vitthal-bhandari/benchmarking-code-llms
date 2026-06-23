#!/bin/bash
#SBATCH --job-name=lcb_smoke
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
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

[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# ── Auto-detect quantization ──────────────────────────────────────────────────
case "$MODEL_NAME" in
  *FP8*|*fp8*)     QUANTIZATION="fp8" ;;
  *w4a16*|*W4A16*) QUANTIZATION="gptq" ;;
  *NVFP4*|*nvfp4*) QUANTIZATION="nvfp4" ;;
  *AWQ*|*awq*)     QUANTIZATION="awq" ;;
  *GPTQ*|*gptq*)   QUANTIZATION="gptq" ;;
  *)               QUANTIZATION="" ;;
esac

cd "$LCB_DIR"

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
uv run python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA — wrong partition?'
print('  CUDA OK:', torch.cuda.get_device_name(0))
print('  VRAM   :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

uv run python -c "
from lcb_runner.lm_styles import LanguageModelStore
model = '$MODEL_NAME'
assert model in LanguageModelStore, f'Model not registered: {model}. Check lcb_patch.'
print('  Model registered OK:', LanguageModelStore[model].model_repr)
"

# ── Run eval ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> Starting generation + evaluation..."

EVAL_CMD=(
  uv run python -m lcb_runner.runner.main
  --model "$MODEL_NAME"
  --scenario codegeneration
  --evaluate
  --release_version release_v1
  --n 1
  --temperature 0
  --tensor_parallel_size 1
  --use_cache
)
[ -n "${QUANTIZATION:-}" ] && EVAL_CMD+=(--quantization "$QUANTIZATION")

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
