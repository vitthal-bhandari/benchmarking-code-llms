#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# serve_vllm.slurm — OpenAI-compatible vLLM server on Tillicum (UW RCC H200).
# run_swebench_agent.slurm talks to this endpoint.
#
# Tillicum uses QoS (not partitions) and has NO checkpoint/preemptible queue,
# so this server won't be killed mid-run. GPUs are requested with
# --gres=gpu:h200:1.
#
# Tillicum nodes are provisioned at a fixed 8 CPUs / 200GB RAM PER GPU
# requested — sbatch rejects any job exceeding that ratio (not a soft limit).
# For 1 GPU: max 8 CPUs, max 200GB RAM. Request more GPUs to get more of either.
#
# Usage:
#   sbatch scripts/serve_vllm.slurm                                  # native context (default)
#   sbatch --export=MAX_MODEL_LEN=65536 scripts/serve_vllm.slurm     # cap it explicitly if desired
#
# Find the endpoint after it starts:
#   squeue --me                                   # get the node
#   grep "Application startup complete" logs/vllm_serve_<jobid>.out
#   -> base URL http://<node>:8000/v1  (put it in configs/api_override.yaml)
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"
# Empty = omit --max-model-len entirely, so vLLM reads the model's own native
# max from its config (generalizes across models with different native limits).
# Safe as the default on Tillicum/H200 (141GB) — the old 65536 default existed
# only to fit Klone's memory-constrained 48GB L40S. Set explicitly to cap it.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
# Tool-call parser must match how the model emits calls in raw text. Qwen3.6
# uses <tool_call><function=NAME><parameter=KEY>...  -> qwen3_coder. Gemma4 ->
# gemma4. Verify choices with: vllm serve --help=tool-call-parser
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
# Empty = auto-detect from checkpoint. Set "fp8" to on-the-fly-quantize a native
# BF16 checkpoint (not needed on H200 for models that fit natively).
QUANTIZATION="${QUANTIZATION:-}"
# Multimodal models (Gemma4) force --disable-chunked-mm-input, requiring this to
# be >= their max tokens-per-mm-item (Gemma4 needs >= 2496).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
# "fp8" stores the KV cache in FP8 (~half the bytes/token) — a memory trick for
# small GPUs. Not needed on H200; leave empty for full-precision KV cache.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
# vLLM's auto MoE backend (FlashInfer CUTLASS, for unquantized/BF16 MoE on
# Hopper) segfaults at the native level on Tillicum during startup profiling.
# Force Triton — confirmed clean startup. Verify choices: vllm serve --help=moe-backend
MOE_BACKEND="${MOE_BACKEND:-triton}"

PROJECT_DIR="/gpfs/projects/stf/$USER/benchmarking-code-llms"
mkdir -p "$PROJECT_DIR/logs"

export HF_HOME="/gpfs/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gpfs/scrubbed/$USER/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Keep vLLM's cache (incl. DeepGEMM/FlashInfer JIT kernels) off the 10GB home
# quota — filling home corrupts things.
export VLLM_CACHE_ROOT="/gpfs/scrubbed/$USER/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT" "$HF_HUB_CACHE" "$TORCH_HOME"

# On Tillicum this is a no-op in practice (system /usr/bin/g++ IS already
# 11.5.0 / Red Hat build) — kept as a harmless safety net in case that ever
# changes. It was NOT the fix for the startup crashes hit during initial setup;
# see MOE_BACKEND and LIMIT_MM below for the actual root causes found.
module load gcc/11.5.0 2>/dev/null || true

# Self-contained pip CUDA toolkit lives in the venv (nvcc, headers, libs).
export CUDA_HOME="$PROJECT_DIR/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
[ -e "$CUDA_HOME/lib64" ] || ln -sf lib "$CUDA_HOME/lib64" 2>/dev/null || true
# Pin nvcc's host compiler + skip the CCCL toolkit/compiler version check (our
# pip CUDA mixes minor versions but works).
if command -v g++ >/dev/null 2>&1; then
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin $(command -v g++)"
fi
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# Native sampler (avoids a FlashInfer nvcc JIT in the spawned worker); bypass a
# flashinfer version-check that trips on mixed pip versions.
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

cd "$PROJECT_DIR"
source .venv/bin/activate

QUANT_ARGS=(); [ -n "$QUANTIZATION" ] && QUANT_ARGS=(--quantization "$QUANTIZATION")
BATCH_ARGS=(); [ -n "$MAX_NUM_BATCHED_TOKENS" ] && BATCH_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
KV_CACHE_ARGS=(); [ -n "$KV_CACHE_DTYPE" ] && KV_CACHE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
# Text-only workload on a vision-language model: cap every modality at 0 so vLLM
# skips the multimodal encoder warmup (image profiling hung 28 min on Tillicum;
# capping image alone just moved it to video, which crashed). Override LIMIT_MM="".
if [ -z "${LIMIT_MM+x}" ]; then LIMIT_MM='{"image":0,"video":0}'; fi
MM_ARGS=(); [ -n "$LIMIT_MM" ] && MM_ARGS=(--limit-mm-per-prompt "$LIMIT_MM")
MOE_ARGS=(); [ -n "$MOE_BACKEND" ] && MOE_ARGS=(--kernel-config.moe_backend "$MOE_BACKEND")
MAX_LEN_ARGS=(); [ -n "$MAX_MODEL_LEN" ] && MAX_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")

echo ">>> Serving $MODEL_NAME on $(hostname):$PORT (job ${SLURM_JOB_ID:-local})"
echo ">>> gcc: $(command -v g++ 2>/dev/null || echo none)  |  max_model_len: ${MAX_MODEL_LEN:-native/model default}  |  kv_cache: ${KV_CACHE_DTYPE:-default}  |  moe_backend: ${MOE_BACKEND:-auto}"
vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --trust-remote-code \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  "${MAX_LEN_ARGS[@]}" \
  "${QUANT_ARGS[@]}" \
  "${BATCH_ARGS[@]}" \
  "${KV_CACHE_ARGS[@]}" \
  "${MM_ARGS[@]}" \
  "${MOE_ARGS[@]}"
