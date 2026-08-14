#!/bin/bash
#SBATCH --job-name=swebench_run
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# serve_and_run_swebench.slurm — Tillicum single-job pipeline: starts vLLM in
# the background, waits for it to be ready, then drives mini-swe-agent against
# it on localhost, in one GPU allocation.
#
# WHY combined, not two separate jobs (serve_vllm.slurm + run_swebench_agent.slurm):
# Tillicum requires --gres=gpu on every job — "CPU-only jobs are not allowed."
# A separate driver job would need its own GPU it never uses, silently doubling
# GPU-hour cost for the whole multi-hour driving run. One job = one GPU billed.
# Bonus: same-node localhost traffic can't hit a flaky inter-node proxy, which
# is the class of bug (Squid ERR_CONNECT_FAIL) that cost days on Klone.
#
# serve_vllm.slurm still exists standalone for interactive debugging (a
# long-lived server you curl/iterate against) — accept its extra GPU cost only
# for that short-lived use case, not for full scored runs.
#
# Usage:
#   sbatch --export=SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_qwen_20 \
#          scripts/serve_and_run_swebench.slurm
#   # second model, run in parallel as its own job (gets its own GPU):
#   sbatch --export=MODEL_NAME="google/gemma-4-26B-A4B-it",\
#          TOOL_CALL_PARSER=gemma4,MAX_NUM_BATCHED_TOKENS=4096,\
#          AGENT_MODEL_NAME="hosted_vllm/google/gemma-4-26B-A4B-it",\
#          SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_gemma4_20 \
#          scripts/serve_and_run_swebench.slurm
#
# 8 CPUs / 200GB RAM is the max Tillicum allows for 1 GPU (fixed ratio) — ample
# for a single-GPU server plus a handful of agent workers.
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# --- server config -----------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"
# Empty = omit --max-model-len entirely, so vLLM reads the model's own native
# max from its config (generalizes across models with different native limits,
# unlike hardcoding one number). Safe as the default specifically because we're
# on Tillicum/H200 (141GB) — the old 65536 default existed only to fit Klone's
# memory-constrained 48GB L40S and is no longer a relevant constraint here. Set
# explicitly (e.g. MAX_MODEL_LEN=65536) only if you deliberately want to cap it.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
QUANTIZATION="${QUANTIZATION:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
# vLLM's default MoE backend selection (FlashInfer CUTLASS, for unquantized/BF16
# MoE on Hopper) segfaults at the native level on Tillicum during the profiling
# forward pass ("Segfault encountered", tensorrt_llm cutlass_kernels frames).
# Force Triton instead — confirmed clean startup (weights load -> KV cache sizing
# -> Application startup complete, no crash). Verified via `vllm serve
# --help=moe-backend`; valid values: auto,triton,cutlass,deep_gemm,flashinfer_*,
# marlin,humming,aiter,emulation,triton_unfused. Only relevant for unquantized
# (BF16) MoE checkpoints — harmless to leave set for a quantized one too.
MOE_BACKEND="${MOE_BACKEND:-triton}"

# --- agent/driver config ------------------------------------------------------
# litellm model id used by mini-swe-agent; must match how MODEL_NAME above is
# registered in configs/registry.json (hosted_vllm/<repo id>).
AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-hosted_vllm/$MODEL_NAME}"
SLICE="${SLICE:-0:20}"
WORKERS="${WORKERS:-4}"
# Auto-append the job id so repeated runs of the same config get distinct dirs
# (mini-swe-agent RESUMES an existing OUTPUT_DIR — skipping instances already
# present — so without a unique suffix two runs would silently merge). The
# suffix also ties the run dir directly to its logs/swebench_run_<jobid>.* .
OUTPUT_DIR="${OUTPUT_DIR:-runs/run}"
OUTPUT_DIR="${OUTPUT_DIR}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"   # 30 min — full-weights loads varied 100-1100s on Klone under FS contention

PROJECT_DIR="/gpfs/projects/stf/$USER/benchmarking-code-llms"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

export HF_HOME="/gpfs/scrubbed/$USER/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/gpfs/scrubbed/$USER/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_CACHE_ROOT="/gpfs/scrubbed/$USER/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT" "$HF_HUB_CACHE" "$TORCH_HOME"

# No-op in practice on Tillicum (system g++ is already 11.5.0); kept as a
# harmless safety net. See MOE_BACKEND / LIMIT_MM below for the actual fixes
# to the startup crashes hit during initial setup.
module load gcc/11.5.0 2>/dev/null || true

export CUDA_HOME="$PROJECT_DIR/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
[ -e "$CUDA_HOME/lib64" ] || ln -sf lib "$CUDA_HOME/lib64" 2>/dev/null || true
if command -v g++ >/dev/null 2>&1; then
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin $(command -v g++)"
fi
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

[ -f "$PROJECT_DIR/.env" ] && source <(grep -v '^#' "$PROJECT_DIR/.env" | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

QUANT_ARGS=(); [ -n "$QUANTIZATION" ] && QUANT_ARGS=(--quantization "$QUANTIZATION")
BATCH_ARGS=(); [ -n "$MAX_NUM_BATCHED_TOKENS" ] && BATCH_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
KV_CACHE_ARGS=(); [ -n "$KV_CACHE_DTYPE" ] && KV_CACHE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
# Qwen3.6/Gemma4 are vision-language models, but SWE-Bench is text-only. The
# multimodal encoder warmup broke startup twice on Tillicum: with no limit it
# HUNG 28 min on "profiled with 1 image items"; with only image capped it moved
# on to "profiled with 1 video items" and the EngineCore worker crashed at the
# native level. Cap EVERY modality the model supports so the warmup is skipped
# entirely (also frees the encoder-cache memory). Override LIMIT_MM="" to
# re-enable, or edit the JSON if a model exposes other modalities.
if [ -z "${LIMIT_MM+x}" ]; then LIMIT_MM='{"image":0,"video":0}'; fi
MM_ARGS=(); [ -n "$LIMIT_MM" ] && MM_ARGS=(--limit-mm-per-prompt "$LIMIT_MM")
MOE_ARGS=(); [ -n "$MOE_BACKEND" ] && MOE_ARGS=(--kernel-config.moe_backend "$MOE_BACKEND")
MAX_LEN_ARGS=(); [ -n "$MAX_MODEL_LEN" ] && MAX_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")

SERVER_LOG="logs/vllm_serve_inline_${SLURM_JOB_ID:-local}.log"
echo ">>> [server] Starting $MODEL_NAME on $(hostname):$PORT (job ${SLURM_JOB_ID:-local})"
echo ">>> [server] gcc: $(command -v g++ 2>/dev/null || echo none)  |  max_model_len: ${MAX_MODEL_LEN:-native/model default}  |  kv_cache: ${KV_CACHE_DTYPE:-default}  |  moe_backend: ${MOE_BACKEND:-auto}"

(
  source .venv/bin/activate
  exec vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --dtype auto --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" \
    "${MAX_LEN_ARGS[@]}" "${QUANT_ARGS[@]}" "${BATCH_ARGS[@]}" "${KV_CACHE_ARGS[@]}" "${MM_ARGS[@]}" "${MOE_ARGS[@]}"
) > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo ">>> [cleanup] Stopping vLLM server (pid $VLLM_PID)..."
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo ">>> [server] Waiting for readiness (pid $VLLM_PID, timeout ${STARTUP_TIMEOUT_S}s)..."
waited=0
while ! grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ">>> [server] ERROR: process died during startup. Tail of $SERVER_LOG:"
    tail -80 "$SERVER_LOG"
    exit 1
  fi
  if [ "$waited" -ge "$STARTUP_TIMEOUT_S" ]; then
    echo ">>> [server] ERROR: not ready after ${STARTUP_TIMEOUT_S}s. Tail of $SERVER_LOG:"
    tail -80 "$SERVER_LOG"
    exit 1
  fi
  sleep 10; waited=$((waited + 10))
done
echo ">>> [server] Ready after ${waited}s."

# Point the agent at this same node — no manual api_override editing needed.
API_OVERRIDE="configs/api_override_inline_${SLURM_JOB_ID:-local}.yaml"
cat > "$API_OVERRIDE" << EOF
model:
  model_kwargs:
    api_base: http://localhost:${PORT}/v1
    temperature: 0
EOF

echo ">>> [driver] Model: $AGENT_MODEL_NAME | Slice: $SLICE | Workers: $WORKERS | Output: $OUTPUT_DIR"
(
  source agent-venv/bin/activate
  export MSWEA_COST_TRACKING='ignore_errors'
  # Same-node localhost traffic — no proxy bypass needed, but harmless if set.
  export NO_PROXY="localhost,127.0.0.1"; export no_proxy="$NO_PROXY"
  LITELLM_MODEL_REGISTRY_PATH=configs/registry.json mini-extra swebench \
    -m "$AGENT_MODEL_NAME" \
    -c swebench.yaml -c "$API_OVERRIDE" \
    --subset verified --split test --slice "$SLICE" \
    --environment-class singularity --workers "$WORKERS" \
    --output "$OUTPUT_DIR/"
)

echo ""
echo ">>> Done. Predictions at $OUTPUT_DIR/preds.json"
