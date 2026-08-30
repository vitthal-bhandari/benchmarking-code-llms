#!/bin/bash
# =============================================================================
# install_venv.sh — Build the two project venvs on Tillicum (UW RCC H200).
#
#   .venv        serving env (vLLM 0.21.0) — used by scripts/serve_vllm.slurm
#   agent-venv   mini-swe-agent + sb-cli    — used by run_swebench_agent.slurm
#
# Run INSIDE a Tillicum GPU allocation (vLLM needs CUDA present at install; the
# DeepGEMM build needs nvcc + a modern gcc from the module system):
#   salloc -A stf --qos=normal --gres=gpu:h200:1 -c 16 --mem=64G -t 02:00:00
#   bash scripts/install_venv.sh
#
# This script IS the reproducible recipe — keep it in sync with reality.
# requirements-working.txt (a pip freeze of a known-good serving env) is the
# authoritative snapshot; re-run `pip freeze > requirements-working.txt` after
# any interactive fix. Do NOT `uv sync` the serving env: it reverts vLLM to a 
# pinned-but-broken version that doesn't recognize Qwen3.6 (qwen3_5_moe). 
#
# Default is full-weights (BF16) models, which don't need DeepGEMM at all — on
# H200's 141GB there's no need to fight FP8 for memory the way L40S required.
# Set BUILD_DEEPGEMM=1 only if you specifically need to serve a pre-quantized
# FP8 checkpoint (e.g. a *-FP8 repo) on Hopper — it JIT-compiles a CUDA kernel
# from source and was the single biggest source of setup pain on Klone.
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gpfs/projects/stf/$USER/benchmarking-code-llms"
export UV_CACHE_DIR="/gpfs/scrubbed/$USER/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
cd "$PROJECT_DIR"

# Tillicum ships a CUDA-13 toolkit + modern gcc as modules. We still install a
# self-contained pip CUDA toolkit into the venv (proven, cluster-agnostic), and
# use the module gcc only as the host compiler for JIT/DeepGEMM builds.
module load gcc/11.5.0 2>/dev/null || echo ">>> WARNING: could not module-load gcc/11.5.0 (check 'module avail gcc')"

# ── Serving venv ─────────────────────────────────────────────────────────────
echo ">>> [.venv] Creating serving venv with Python 3.11..."
# --seed installs pip so the pip-based CUDA/vLLM installs below work.
uv venv --python 3.11 --seed .venv
source .venv/bin/activate

echo ">>> [.venv] Installing vLLM 0.21.0 (wheels only)..."
# Pin 0.21.0 exactly: ">=0.8" resolves to 0.8.4, which doesn't recognize
# qwen3_5_moe and ships an incompatible fastapi/starlette. --only-binary avoids
# a source build that fails on the llguidance/alloca Rust crate. 0.21.0 pulls a
# compatible torch (2.11.0) itself — no separate torch reinstall needed.
pip install "vllm==0.21.0" --only-binary=:all: --no-cache-dir

echo ">>> [.venv] Installing CUDA toolkit pieces pip splits out..."
# vLLM brings CUDA *runtime* libs but not the compiler/headers, which FlashInfer
# and DeepGEMM need to JIT-compile kernels at runtime.
#   nvidia-cuda-nvcc : nvcc at .venv/.../nvidia/cu13/bin (CUDA_HOME in serve script).
#                      NB: the "-cu13"-suffixed package is deprecated and fails.
#   nvidia-cuda-cccl : <nv/target> etc. under cu13/include/cccl.
pip install nvidia-cuda-nvcc nvidia-cuda-cccl --no-cache-dir

echo ">>> [.venv] Installing bitsandbytes (INT8 fallback for some models)..."
pip install bitsandbytes --no-cache-dir

echo ">>> [.venv] Installing transformers from git (PyPI lacks qwen3_5_moe)..."
# Must run AFTER vLLM so vLLM doesn't downgrade it.
pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"

NVIDIA_DIR="$PROJECT_DIR/.venv/lib/python3.11/site-packages/nvidia"

echo ">>> [.venv] Adding unversioned .so symlinks + lib64 for CUDA libs..."
# pip CUDA packages ship versioned-only libs (libcudart.so.13); the linker wants
# the unversioned name, and some JIT builds hardcode -L.../lib64 (absent in pip
# layout). Both are needed for from-source CUDA builds (DeepGEMM, GDN kernel).
if [ -d "$NVIDIA_DIR/cu13/lib" ]; then
  ( cd "$NVIDIA_DIR/cu13/lib" \
    && for lib in libcudart libnvrtc; do
         target=$(ls -1 "$lib".so.[0-9]* 2>/dev/null | head -1)
         [ -n "$target" ] && ln -sf "$target" "$lib.so"
       done )
  [ -e "$NVIDIA_DIR/cu13/lib64" ] || ln -sf lib "$NVIDIA_DIR/cu13/lib64"
fi

if [ "${BUILD_DEEPGEMM:-0}" = "1" ]; then
  echo ">>> [.venv] Building DeepGEMM (Hopper/H200 FP8 support)..."
  # Needed to serve pre-quantized FP8 checkpoints on Hopper. Don't
  # `pip install deep_gemm` — the PyPI sdist omits its CUTLASS submodule.
  # vLLM's installer clones --recursive at a pinned commit.
  export CUDA_HOME="$NVIDIA_DIR/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"   # Hopper sm_90a
  # nvcc's version check vs the pip CCCL headers can mismatch; the header
  # documents this opt-out. gcc from the module is the host compiler.
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
  curl -fsSL -o /tmp/install_deepgemm.sh \
    https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh
  bash /tmp/install_deepgemm.sh
  python -c "import deep_gemm; print('deep_gemm: OK')"
fi

echo ">>> [.venv] Sanity check..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
ls -l "$NVIDIA_DIR/cu13/bin/nvcc" >/dev/null 2>&1 || echo "  WARNING: nvcc missing — FlashInfer JIT kernels will fail at runtime"
deactivate

# ── Agent venv ───────────────────────────────────────────────────────────────
echo ">>> [agent-venv] Creating driver venv (mini-swe-agent + sb-cli)..."
uv venv --python 3.11 --seed agent-venv
source agent-venv/bin/activate
# litellm's completion() eagerly imports MCP/proxy code needing fastapi+orjson,
# so pull the full proxy extra even though we only use the plain client.
pip install --no-cache-dir mini-swe-agent "litellm[proxy]" fastapi sb-cli
python -c "import minisweagent; print('mini-swe-agent OK')"
deactivate

echo ""
echo ">>> Setup complete."
echo ">>>   serving env : $PROJECT_DIR/.venv"
echo ">>>   agent env   : $PROJECT_DIR/agent-venv"
echo ">>> Snapshot serving env:  source .venv/bin/activate && pip freeze > requirements-working.txt"
