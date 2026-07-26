#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gpus=l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# serve_vllm.slurm —  OpenAI-compatible vLLM server for agent baselines
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
# mini-swe-agent (via litellm) sends tool_choice="auto" with tool defs, which
# vLLM refuses unless explicitly enabled with a parser matching how the model
# formats tool calls in its raw text output. Qwen3.6 emits
# <tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>
# — confirmed by diffing a real trajectory against vllm's tool_parsers source;
# qwen3coder_tool_parser.py hardcodes these exact sentinel tokens
# (tool_call_start_token="<tool_call>", tool_call_prefix="<function=").
# "hermes" (JSON-based <tool_call>{"name":...}</tool_call>) doesn't match this
# shape and left tool_calls null, causing RepeatedFormatError.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
# Empty by default (Qwen/Poolside checkpoints are already pre-quantized and
# vLLM auto-detects that from config.json). Set to "fp8" for a native/BF16
# checkpoint that needs on-the-fly quantization to fit on a 48GB L40S — e.g.
# gemma-4-26B-A4B-it is ~49GB unquantized, doesn't fit without this.
QUANTIZATION="${QUANTIZATION:-}"
# Empty by default (vLLM's own default of 2048 is fine for text-only models).
# Multimodal models (e.g. Gemma4) force --disable-chunked-mm-input, which
# requires max_num_batched_tokens >= max_tokens_per_mm_item (one multimodal
# item must fit in a single scheduling step) — Gemma4 needs at least 2496;
# set a comfortable margin above that when serving it.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
# Empty by default (vLLM's native KV cache dtype matches model dtype). Set to
# "fp8" to store the KV cache itself in FP8 instead of the model's native
# precision — roughly halves KV-cache memory per token, independent of
# --quantization (which only affects the weights). This is what actually
# raises the safe --max-model-len ceiling on a memory-constrained GPU like a
# 48GB L40S; untested combination as of Jul 15 2026, watch server startup for
# OOM and back off MAX_MODEL_LEN if needed.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"
mkdir -p "$PROJECT_DIR/logs"

export HF_HOME="/gscratch/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gscratch/scrubbed/$USER/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# vLLM defaults its cache (incl. DeepGEMM's JIT kernel cache) to ~/.cache/vllm.
# $HOME is capped at 10GB on Hyak and filling it corrupts things badly — keep
# it on scratch like every other cache.
export VLLM_CACHE_ROOT="/gscratch/scrubbed/$USER/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"

# A modern host compiler must be on PATH before nvcc runs. DeepGEMM (Hopper FP8)
# JIT-compiles kernels at *runtime*, during KV-cache profiling — not just at
# build time — and nvcc shells out to whatever g++ it finds. The node default
# (/usr/bin/g++, GCC 8) predates C++20, so nvcc silently drops its -std=c++20
# flag and CCCL then fails with "CUDA compiler and CUDA toolkit headers are
# incompatible" / "libcu++ requires at least C++ 17".
#
# `module load gcc` is NOT reliable here: it works in an interactive salloc but
# exits 0 without changing PATH in a non-interactive batch shell. Point at the
# module tree's compiler directly instead; override with GCC_BIN if the path
# changes. Verify with: ls -d /sw/gcc/*/bin
GCC_BIN="${GCC_BIN:-/sw/gcc/11.2.0/bin}"
if [ -x "$GCC_BIN/g++" ]; then
  export PATH="$GCC_BIN:$PATH"
  # Putting the compiler on PATH is only the compile-time half. Kernels built
  # with it require symbol versions (e.g. GLIBCXX_3.4.26) that only exist in
  # THIS gcc's own libstdc++.so.6 — the runtime loader otherwise falls back to
  # the node's older system libstdc++ and fails with "version `GLIBCXX_3.4.26'
  # not found", even though the .so itself compiled fine. Put the matching
  # lib64 first on the search path.
  GCC_LIB64="${GCC_BIN%/bin}/lib64"
  [ -d "$GCC_LIB64" ] && export LD_LIBRARY_PATH="$GCC_LIB64:${LD_LIBRARY_PATH:-}"
else
  echo ">>> WARNING: no g++ at $GCC_BIN — DeepGEMM JIT will fail on Hopper (H200)"
fi

export CUDA_HOME="$LCB_DIR/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# Pin nvcc's host compiler explicitly rather than trusting PATH order — vLLM
# spawns EngineCore as a subprocess and this survives into it.
if command -v g++ >/dev/null 2>&1; then
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin $(command -v g++)"
  echo ">>> nvcc host compiler: $(command -v g++) ($(g++ -dumpversion 2>/dev/null))"
fi
# CCCL hard-errors unless nvcc's version exactly equals the CUDA runtime headers'
# CUDART_VERSION ("CUDA compiler and CUDA toolkit headers are incompatible"). Our
# venv has mixed CUDA 12 and 13 pip packages, so that check fails even though the
# combination works. The header documents this define as the intended opt-out for
# "a newer CTK than the compiler ships".
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# deep_gemm's compiled _C extension links against libnvrtc.so.13/libcudart.so.13
# in cu13/lib, which aren't on the default runtime search path (pip's CUDA
# packages ship versioned .so files only; we also added unversioned symlinks
# there for the link step of the DeepGEMM source build). Needed for FP8 weight
# loading on Hopper (H200) — harmless elsewhere.
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
# pip's nvidia-cu13 package only ships cu13/lib, not the traditional cu13/lib64
# — but some JIT-compiled kernels (e.g. Qwen3.6's GDN/linear-attention prefill
# kernel, compiled by FlashInfer at first real use) hardcode -L.../lib64 and
# silently fail to link (ld: cannot find -lcudart), non-fatal but forces a
# slower fallback path with an explicit "may OOM due to autotuner" risk.
# Symlinking the whole directory covers this and any similarly-affected future
# kernel.
[ -e "$CUDA_HOME/lib64" ] || ln -sf lib "$CUDA_HOME/lib64"
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

# A newer fastapi/starlette landed transitively during earlier torch/vllm
# reinstalls; prometheus-fastapi-instrumentator's routing.py wasn't compatible
# with it (AttributeError: '_IncludedRouter' object has no attribute 'path'
# on every request). Upgrading it was the narrow fix at the time — but on a
# freshly `uv sync`'d venv this now backfires: pulling in the latest
# prometheus-fastapi-instrumentator (8.0.2) drags in starlette>=1.0, which
# breaks fastapi's own pinned starlette<0.47 requirement
# (TypeError: Router.__init__() got an unexpected keyword argument
# 'on_startup'). A fresh `uv sync` installs a mutually-consistent, locked
# version set already — only run this upgrade if that old _IncludedRouter
# bug actually resurfaces (e.g. after further ad-hoc pip installs drift the
# venv again), not unconditionally on every launch.
SKIP_PROM_UPGRADE="${SKIP_PROM_UPGRADE:-1}"
if [ "$SKIP_PROM_UPGRADE" != "1" ]; then
  pip install --upgrade --no-cache-dir prometheus-fastapi-instrumentator
fi

QUANT_ARGS=()
if [ -n "$QUANTIZATION" ]; then
  QUANT_ARGS=(--quantization "$QUANTIZATION")
fi

BATCH_ARGS=()
if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
  BATCH_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

KV_CACHE_ARGS=()
if [ -n "$KV_CACHE_DTYPE" ]; then
  KV_CACHE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

echo ">>> Serving $MODEL_NAME on $(hostname):$PORT (job ${SLURM_JOB_ID:-local})"
echo ">>> Quantization: ${QUANTIZATION:-auto-detected from checkpoint}"
echo ">>> Max num batched tokens: ${MAX_NUM_BATCHED_TOKENS:-vLLM default}"
echo ">>> KV cache dtype: ${KV_CACHE_DTYPE:-vLLM default (matches model dtype)}"
echo ">>> Max model len: $MAX_MODEL_LEN"
vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype auto \
  --trust-remote-code \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  "${QUANT_ARGS[@]}" \
  "${BATCH_ARGS[@]}" \
  "${KV_CACHE_ARGS[@]}"
