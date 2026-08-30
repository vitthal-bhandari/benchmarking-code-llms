#!/bin/bash
# =============================================================================
# install_venv_new.sh — Build a SECOND, newer-vLLM serving venv on Tillicum.
#
#   .venv-new   newer vLLM (unpinned by default) — for 2026 models that the
#               pinned .venv (vLLM 0.21.0) can't load: Gemma4 (per-layer
#               head_dim) and North-Mini-Code (hybrid dense/MoE). Both failed
#               .venv at WEIGHT LOADING, not config — a version wall, so the
#               only fix is model code newer than 0.21 ships.
#
# .venv stays exactly as-is (Qwen3.6 keeps serving on the stable stack). Select
# this venv per-job with SERVE_VENV=.venv-new on serve_and_run_swebench.slurm.
# The agent/driver venv (agent-venv) is unchanged and shared across all models.
#
# Run INSIDE a Tillicum GPU allocation (vLLM wants CUDA present at install):
#   salloc -A stf --qos=normal --gres=gpu:h200:1 -c 8 --mem=64G -t 02:00:00
#   bash scripts/install_venv_new.sh
#
# Python stays 3.11 so the serve script's CUDA_HOME path
# ($SERVE_VENV/lib/python3.11/...) resolves for either venv. Override the vLLM
# version if "latest" regresses:  VLLM_VERSION=0.23.0 bash scripts/install_venv_new.sh
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gpfs/projects/stf/$USER/benchmarking-code-llms"
export UV_CACHE_DIR="/gpfs/scrubbed/$USER/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
cd "$PROJECT_DIR"

VENV="${VENV:-.venv-new}"
VLLM_VERSION="${VLLM_VERSION:-}"   # empty = latest; else an exact pin like 0.23.0

module load gcc/11.5.0 2>/dev/null || echo ">>> WARNING: could not module-load gcc/11.5.0"

echo ">>> [$VENV] Creating serving venv with Python 3.11..."
uv venv --python 3.11 --seed "$VENV"
source "$VENV/bin/activate"

if [ -n "$VLLM_VERSION" ]; then
  echo ">>> [$VENV] Installing vLLM==$VLLM_VERSION (wheels only)..."
  pip install "vllm==$VLLM_VERSION" --only-binary=:all: --no-cache-dir
else
  echo ">>> [$VENV] Installing latest vLLM (wheels only)..."
  pip install -U vllm --only-binary=:all: --no-cache-dir
fi
echo ">>> [$VENV] vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"

echo ">>> [$VENV] Installing CUDA compiler pieces (nvcc + cccl headers)..."
pip install nvidia-cuda-nvcc nvidia-cuda-cccl --no-cache-dir

echo ">>> [$VENV] Installing transformers from git (newest model archs)..."
# After vLLM so vLLM doesn't pin it back. This is what carries brand-new model
# support (Gemma4/Cohere2) that a released transformers may lag.
pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"

NVIDIA_DIR="$PROJECT_DIR/$VENV/lib/python3.11/site-packages/nvidia"
echo ">>> [$VENV] Adding unversioned .so symlinks + lib64 for CUDA libs..."
if [ -d "$NVIDIA_DIR/cu13/lib" ]; then
  ( cd "$NVIDIA_DIR/cu13/lib" \
    && for lib in libcudart libnvrtc; do
         target=$(ls -1 "$lib".so.[0-9]* 2>/dev/null | head -1)
         [ -n "$target" ] && ln -sf "$target" "$lib.so"
       done )
  [ -e "$NVIDIA_DIR/cu13/lib64" ] || ln -sf lib "$NVIDIA_DIR/cu13/lib64"
fi

# Snapshot for reproducibility (parallel to requirements-working.txt for .venv).
pip freeze > "requirements-${VENV#.}.txt"

echo ">>> [$VENV] Done. vLLM $(python -c 'import vllm; print(vllm.__version__)') ready."
echo ">>> Serve a 2026 model from it with, e.g.:"
echo ">>>   SERVE_VENV=$VENV MODEL_NAME=google/gemma-4-26B-A4B-it TOOL_CALL_PARSER=gemma4 \\"
echo ">>>     AGENT_MODEL_NAME=hosted_vllm/google/gemma-4-26B-A4B-it SLICE=0:1 WORKERS=1 \\"
echo ">>>     OUTPUT_DIR=runs/smoke_gemma4 bash scripts/serve_and_run_swebench.slurm"
