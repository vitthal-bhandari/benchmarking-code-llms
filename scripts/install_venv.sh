#!/bin/bash
# =============================================================================
# install_venv.sh — Install uv venv + LCB + vLLM
#
# Run INSIDE a gpu-l40s salloc session (vLLM needs CUDA at install time):
#   salloc -A stf -p gpu-l40s -N 1 -c 8 --mem=32G --gpus=1 -t 01:00:00
#   bash scripts/install_venv.sh
#
# IMPORTANT — this script is the reproducible recipe; keep it in sync with
# reality. `requirements-working.txt` (pip freeze of a known-good env) is the
# authoritative snapshot; this script is the narrative version of how to get
# there. On Jul 2026 a `rm -rf .venv && uv sync` wiped a hand-assembled env
# because several manual pip steps had never been recorded here — the rebuild
# silently reverted vLLM 0.21.0 -> 0.8.4 (the version pinned in uv.lock) and
# broke Qwen3.6 support entirely. If you fix an env problem interactively,
# add it here and re-run `pip freeze > requirements-working.txt`. 
#
# Optional: set BUILD_DEEPGEMM=1 to also build DeepGEMM (needed only to serve
# pre-quantized FP8 checkpoints on Hopper/H200; not needed on L40S/Ada).
# That step needs a modern GCC via `module load gcc`, which is unavailable on
# login nodes — run inside an allocation (a cheap cpu-g2 one is fine; the
# build itself does not need a GPU).
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"

export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"

# LCB recommends Python 3.11 
cd "$LCB_DIR"
echo ">>> Creating venv with Python 3.11..."
# --seed installs pip into the venv so the pip-based vLLM/bitsandbytes installs
# below work (uv venv omits pip by default). The pip route for vLLM is
# deliberate — it avoids a uv/GPFS flashinfer-cubin copy failure on scrubbed.
uv venv --python 3.11 --seed
source .venv/bin/activate

echo ">>> Pinning transformers to git source in LCB pyproject.toml..."
# Rewrites LCB's transformers requirement to the git URL so uv never reverts our
# source build (needed for qwen3_5_moe) on a later `uv run`/`uv sync`. Idempotent.
python "$PROJECT_DIR/lcb_patch/pin_transformers.py" --lcb-dir "$LCB_DIR"

echo ">>> Installing LCB dependencies..."
uv pip install -e .

echo ">>> Installing vLLM 0.21.0 (CUDA build) via pip --no-cache-dir..."
# Use pip (not uv pip) to avoid a GPFS cross-path copy failure that hits
# flashinfer-cubin's extremely long filenames on Hyak's scrubbed filesystem.
# --no-cache-dir forces pip to extract to a tmp dir and move atomically.
#
# PIN 0.21.0 EXACTLY. The old ">=0.8.0" resolved to 0.8.4, which does not
# recognize Qwen3.6's qwen3_5_moe architecture and ships an incompatible
# fastapi/starlette pair (TypeError: Router.__init__() got an unexpected
# keyword argument 'on_startup' at `vllm serve` startup).
#
# --only-binary=:all: forces wheels: building from source fails on the
# llguidance dependency (broken `alloca v0.4.0` Rust crate).
#
# 0.21.0 pulls a compatible torch (2.11.0) as its own dependency — no separate
# torch force-reinstall step is needed (an earlier "NCCL undefined symbol"
# workaround was really just vLLM/torch version skew).
pip install "vllm==0.21.0" --only-binary=:all: --no-cache-dir

echo ">>> Installing CUDA toolkit pieces pip splits into separate packages..."
# vLLM's dependency tree brings CUDA *runtime* libs but NOT the compiler or the
# C++ core headers. Both are needed because FlashInfer JIT-compiles some kernels
# on first use (notably the bf16-query/fp8-KV-cache prefill kernel when serving
# with --kv-cache-dtype fp8) and dies at runtime without nvcc.
#   - nvidia-cuda-nvcc  : provides nvcc at nvidia/cu13/bin/nvcc (the path
#                         serve_vllm.slurm sets CUDA_HOME to). NOTE the package
#                         name: `nvidia-cuda-nvcc-cu13` is deprecated and its
#                         install intentionally fails.
#   - nvidia-cuda-cccl  : provides <nv/target> etc. under cu13/include/cccl;
#                         without it any CUDA C++ compile fails on cuda_fp16.h.
pip install nvidia-cuda-nvcc nvidia-cuda-cccl --no-cache-dir

echo ">>> Installing bitsandbytes (for Devstral INT8 fallback)..."
pip install bitsandbytes --no-cache-dir

echo ">>> Installing transformers from source (PyPI release lacks qwen3_5_moe and other new archs)..."
# The latest PyPI transformers does NOT yet recognize qwen3_5_moe (Qwen3.6 FP8).
# vLLM loads model configs via AutoConfig.from_pretrained, so transformers must
# know the arch. Install from git; run this AFTER vLLM so it isn't downgraded.
uv pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"

NVIDIA_DIR="$LCB_DIR/.venv/lib/python3.11/site-packages/nvidia"

echo ">>> Adding unversioned .so symlinks for CUDA libs..."
# pip's CUDA packages ship only versioned libraries (libcudart.so.13), but the
# linker resolves -lcudart / -lnvrtc via the unversioned name. Needed for any
# from-source CUDA build against this venv (e.g. DeepGEMM below). Harmless
# otherwise. Runtime loading additionally needs cu13/lib on LD_LIBRARY_PATH —
# scripts/serve_vllm.slurm exports that.
if [ -d "$NVIDIA_DIR/cu13/lib" ]; then
  ( cd "$NVIDIA_DIR/cu13/lib" \
    && for lib in libcudart libnvrtc; do
         target=$(ls -1 "$lib".so.[0-9]* 2>/dev/null | head -1)
         [ -n "$target" ] && ln -sf "$target" "$lib.so"
       done )
fi

if [ "${BUILD_DEEPGEMM:-0}" = "1" ]; then
  echo ">>> Building DeepGEMM (Hopper/H200 FP8 support)..."
  # Only needed to serve pre-quantized FP8 checkpoints on Hopper: vLLM selects
  # FlashInferFp8DeepGEMMDynamicBlockScaledKernel there and hard-fails weight
  # loading with "DeepGEMM backend is not available or outdated" without it.
  # Ada/L40S uses a different kernel path and does not need this at all.
  #
  # Don't `pip install deep_gemm` — the PyPI sdist omits its vendored CUTLASS
  # submodule and cannot build. vLLM's installer clones --recursive at a pinned
  # commit, which is the only reliable route.
  #
  # Requires GCC 9+ (`module load gcc`; unavailable on login nodes) and nvcc on
  # PATH. TORCH_CUDA_ARCH_LIST targets Hopper without needing a GPU present.
  export CUDA_HOME="$NVIDIA_DIR/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
  curl -fsSL -o /tmp/install_deepgemm.sh \
    https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh
  bash /tmp/install_deepgemm.sh
  python -c "import deep_gemm; print('deep_gemm: OK')"
fi

echo ">>> Sanity check..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "from lcb_runner.lm_styles import LanguageModelStore; print('LCB models registered:', len(LanguageModelStore))"
ls -l "$NVIDIA_DIR/cu13/bin/nvcc" 2>/dev/null || echo "  WARNING: nvcc missing — FlashInfer JIT kernels will fail at runtime"

echo ""
echo ">>> Setup complete. venv at $LCB_DIR/.venv"
echo ">>> Snapshot this environment:  pip freeze > $PROJECT_DIR/requirements-working.txt"
