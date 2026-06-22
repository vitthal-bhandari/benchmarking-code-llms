#!/bin/bash
#SBATCH --job-name=prewarm_models
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# prewarm_models.slurm — Pre-download all 5 model checkpoints to HF cache.
#
# Run this ONCE before submitting eval jobs so inference jobs don't burn
# wall-time on multi-GB downloads.
#
# Submit from project root on Hyak:
#   cd /gscratch/scrubbed/$USER/benchmarking-code-llms
#   sbatch scripts/prewarm_models.slurm
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"

# ── Cache dirs ────────────────────────────────────────────────────────────────
export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"
export HF_HOME="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"

# Load HF token
[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

cd "$LCB_DIR"

# ── Model checkpoints to pre-warm ────────────────────────────────────────────
MODELS=(
  "Qwen/Qwen3.6-35B-A3B-FP8"
  "CohereLabs/North-Mini-Code-1.0-w4a16"
  "mistralai/Devstral-Small-2-24B-Instruct-2512"
  "poolside/Laguna-XS.2-NVFP4"
  "google/gemma-4-27b-it"        # TODO: verify HF ID
)

echo ">>> Starting model pre-warm at $(date)"

uv run python - << 'EOF'
import os
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "CohereLabs/North-Mini-Code-1.0-w4a16",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "poolside/Laguna-XS.2-NVFP4",
    "google/gemma-4-27b-it",    # TODO: verify HF ID
]

token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

for model_id in models:
    print(f"\n>>> Downloading {model_id} ...")
    try:
        path = snapshot_download(
            repo_id=model_id,
            token=token,
            ignore_patterns=["*.pt", "original/*"],   # skip redundant formats
        )
        print(f"    Saved to: {path}")
    except Exception as e:
        print(f"    ERROR downloading {model_id}: {e}")

print("\n>>> Pre-warm complete.")
EOF

echo ">>> Finished at $(date)"
