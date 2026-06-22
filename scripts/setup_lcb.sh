#!/bin/bash
# =============================================================================
# setup_lcb.sh — One-time environment setup for LiveCodeBench on Hyak
#
# Run this from the LOGIN NODE (light commands only).
# The venv install step requires a GPU allocation — the script will prompt
# you to run that part inside an salloc session.
#
# Usage:
#   bash scripts/setup_lcb.sh
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"
CACHE_BASE="/gscratch/scrubbed/$USER/.cache"

# ── 1. Create directory layout ────────────────────────────────────────────────
echo ">>> Creating project directories..."
mkdir -p "$PROJECT_DIR"/{logs,results}
mkdir -p "$CACHE_BASE"/{uv,huggingface/hub,huggingface/datasets,torch}

# ── 2. Export cache env vars (also add to ~/.bashrc for persistence) ──────────
export UV_CACHE_DIR="$CACHE_BASE/uv"
export HF_HOME="$CACHE_BASE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$CACHE_BASE/torch"

for line in \
  "export UV_CACHE_DIR=\"$CACHE_BASE/uv\"" \
  "export HF_HOME=\"$CACHE_BASE/huggingface\"" \
  "export HF_HUB_CACHE=\"\$HF_HOME/hub\"" \
  "export HF_DATASETS_CACHE=\"\$HF_HOME/datasets\"" \
  "export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"" \
  "export HUGGINGFACE_HUB_CACHE=\"\$HF_HUB_CACHE\"" \
  "export TORCH_HOME=\"$CACHE_BASE/torch\""; do
  grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done
echo ">>> Cache env vars written to ~/.bashrc"

# ── 3. Clone LiveCodeBench ────────────────────────────────────────────────────
if [ ! -d "$LCB_DIR" ]; then
  echo ">>> Cloning LiveCodeBench..."
  git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
else
  echo ">>> LiveCodeBench already cloned at $LCB_DIR — skipping."
fi

# ── 4. Apply lm_styles patch ──────────────────────────────────────────────────
# Resolve repo root relative to this script (works from any cwd)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo ">>> Applying lm_styles patch..."
python "$REPO_ROOT/lcb_patch/apply_patch.py" --lcb-dir "$LCB_DIR"

# ── 5. Create .env for secrets ───────────────────────────────────────────────
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" << 'EOF'
HF_TOKEN=hf_xxx          # replace with your HuggingFace token
# WANDB_API_KEY=xxx       # uncomment if using W&B
EOF
  echo ">>> Created $ENV_FILE — fill in your HF_TOKEN before running evals."
else
  echo ">>> $ENV_FILE already exists — skipping."
fi

# ── 6. venv + vLLM install (must run inside salloc on gpu-l40s) ───────────────
echo ""
echo "================================================================="
echo "  NEXT STEP: Install the Python environment inside a GPU session."
echo "  Run the following:"
echo ""
echo "  salloc -A stf -p gpu-l40s -N 1 -c 8 --mem=32G --gpus=1 -t 01:00:00"
echo "  # then inside the allocation:"
echo "  bash $REPO_ROOT/scripts/install_venv.sh"
echo "================================================================="
