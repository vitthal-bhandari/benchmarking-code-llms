#!/bin/bash
# =============================================================================
# dumpling_run.sh — one memory arm on one GPU of a scheduler-less box (dumpling).
# Serves vLLM on $GPU, drives scripts/run_memory_swebench.py, stops the server.
# Same vLLM flags, sampling override and agent CLI as serve_and_run_swebench.slurm,
# minus the Tillicum paths and modules.
#
#   GPU=0 SEED=1 bash scripts/dumpling_run.sh configs/models/qwen35-9b.env acm
#
# Knobs (env): GPU (0), PORT (8100+GPU), SEED (from cfg), HARNESS (v2), STEP_LIMIT (1000),
#   WORKERS (8), INSTANCE_FILE (configs/mem_subset_99.txt), ENV_CLASS (singularity|docker),
#   DATA_ROOT (/mnt/disk2/$USER), MARKER_ALL_ARMS (0), OUTPUT_DIR (resume an existing dir),
#   SMOKE_IDS (comma list: smoke on a few instances into runs/smoke_*).
# Run it under tmux (see scripts/dumpling_queue.sh); there is no scheduler to restart it.
# =============================================================================
set -euo pipefail
CFG="${1:?usage: dumpling_run.sh configs/models/<model>.env <none|summarize|acm|acm_nudge>}"
ARM="${2:?arm}"
cd "$(dirname "$0")/.."
PROJECT_DIR="$PWD"
set -a; source "$CFG"; set +a

GPU="${GPU:-0}"
PORT="${PORT:-$((8100 + GPU))}"
HARNESS="${HARNESS:-v2}"
STEP_LIMIT="${STEP_LIMIT:-1000}"
WORKERS="${WORKERS:-8}"
INSTANCE_FILE="${INSTANCE_FILE:-configs/mem_subset_99.txt}"
ENV_CLASS="${ENV_CLASS:-singularity}"
DATA_ROOT="${DATA_ROOT:-/mnt/disk2/$USER}"
TAG="$(basename "$CFG" .env)"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Everything heavy on the data disk: $HOME and /tmp are nearly full on dumpling.
export HF_HOME="$DATA_ROOT/.cache/huggingface" HF_HUB_CACHE="$DATA_ROOT/.cache/huggingface/hub"
export VLLM_CACHE_ROOT="$DATA_ROOT/.cache/vllm" TORCH_HOME="$DATA_ROOT/.cache/torch"
export APPTAINER_CACHEDIR="$DATA_ROOT/.cache/apptainer" SINGULARITY_CACHEDIR="$DATA_ROOT/.cache/apptainer"
export TMPDIR="$DATA_ROOT/tmp" APPTAINER_TMPDIR="$DATA_ROOT/tmp"
mkdir -p "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$APPTAINER_CACHEDIR" "$TMPDIR" logs runs
export CUDA_VISIBLE_DEVICES="$GPU"
# pip's CUDA toolkit inside the serving venv, for FlashInfer's JIT kernels (no nvcc on the host)
CU="$(ls -d "$PROJECT_DIR/$SERVE_VENV"/lib/python3.*/site-packages/nvidia/cu13 2>/dev/null | head -1)"
if [ -n "$CU" ]; then
  export CUDA_HOME="$CU" PATH="$CU/bin:$PATH" LD_LIBRARY_PATH="$CU/lib:${LD_LIBRARY_PATH:-}"
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_USE_FLASHINFER_SAMPLER=0
# mini-swe-agent's singularity env runs this binary (an unprivileged apptainer works)
command -v apptainer >/dev/null && export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-$(command -v apptainer)}"
[ -f .env ] && source <(grep -v '^#' .env | sed 's/ *#.*//')
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
if [ -n "${DOCKER_USERNAME:-}" ] && [ -n "${DOCKER_TOKEN:-}" ]; then
  export APPTAINER_DOCKER_USERNAME="$DOCKER_USERNAME" APPTAINER_DOCKER_PASSWORD="$DOCKER_TOKEN"
  export SINGULARITY_DOCKER_USERNAME="$DOCKER_USERNAME" SINGULARITY_DOCKER_PASSWORD="$DOCKER_TOKEN"
fi

if [ -n "${SMOKE_IDS:-}" ]; then
  OUT="${OUTPUT_DIR:-runs/smoke_${TAG}-${HARNESS}_${ARM}_s${SEED}_${STAMP}}"
  ID_ARGS=(--instance-ids "$SMOKE_IDS")
else
  OUT="${OUTPUT_DIR:-runs/${TAG}-${HARNESS}_${ARM}_s${SEED}_dumpling_${STAMP}}"
  ID_ARGS=(--instance-ids-file "$INSTANCE_FILE")
fi
RUN_ID="$(basename "$OUT")"
SERVER_LOG="logs/vllm_${RUN_ID}.log"

REASONING_ARGS=(); [ -n "${REASONING_PARSER:-}" ] && REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")
MAX_LEN_ARGS=(); [ -n "${MAX_MODEL_LEN:-}" ] && MAX_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
EAGER_ARGS=(); [ -n "${ENFORCE_EAGER:-}" ] && EAGER_ARGS=(--enforce-eager)
KV_ARGS=(); [ -n "${KV_CACHE_DTYPE:-}" ] && KV_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
BATCH_ARGS=(); [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && BATCH_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
LIMIT_MM='{"image":0,"video":0,"audio":0}'

echo ">>> [$RUN_ID] GPU $GPU port $PORT | $MODEL_NAME | arm=$ARM harness=$HARNESS seed=$SEED step_limit=$STEP_LIMIT env=$ENV_CLASS"
(
  source "$SERVE_VENV/bin/activate"
  exec vllm serve "$MODEL_NAME" --host 127.0.0.1 --port "$PORT" --dtype auto --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" "${REASONING_ARGS[@]}" \
    "${EAGER_ARGS[@]}" "${MAX_LEN_ARGS[@]}" "${KV_ARGS[@]}" "${BATCH_ARGS[@]}" \
    --limit-mm-per-prompt "$LIMIT_MM"
) > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo ">>> stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

waited=0
until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo ">>> vLLM died:"; tail -60 "$SERVER_LOG"; exit 1; }
  [ "$waited" -ge "${STARTUP_TIMEOUT_S:-1800}" ] && { echo ">>> vLLM not ready"; tail -60 "$SERVER_LOG"; exit 1; }
  sleep 10; waited=$((waited + 10))
done
echo ">>> vLLM ready after ${waited}s"

API_OVERRIDE="configs/api_override_${RUN_ID}.yaml"
{
  echo "model:"; echo "  model_kwargs:"; echo "    api_base: http://127.0.0.1:${PORT}/v1"
  [ -n "${TEMPERATURE:-}" ] && echo "    temperature: ${TEMPERATURE}"
  [ -n "${TOP_P:-}" ] && echo "    top_p: ${TOP_P}"
  [ -n "${SEED:-}" ] && echo "    seed: ${SEED}"
  [ -n "${PRESENCE_PENALTY:-}" ] && echo "    presence_penalty: ${PRESENCE_PENALTY}"
  [ -n "${MAX_TOKENS:-}" ] && echo "    max_tokens: ${MAX_TOKENS}"
  echo "    extra_body:"
  [ -n "${TOP_K:-}" ] && echo "      top_k: ${TOP_K}"
  [ -n "${MIN_P:-}" ] && echo "      min_p: ${MIN_P}"
  [ -n "${REPETITION_PENALTY:-}" ] && echo "      repetition_penalty: ${REPETITION_PENALTY}"
  if [ -n "${ENABLE_THINKING:-}" ]; then
    echo "      chat_template_kwargs:"; echo "        enable_thinking: ${ENABLE_THINKING}"
  fi
  printf 'agent:\n  step_limit: %s\n' "$STEP_LIMIT"
} > "$API_OVERRIDE"

(
  source agent-venv/bin/activate
  export MSWEA_COST_TRACKING='ignore_errors' NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
  MARKER_ARGS=(); [ "${MARKER_ALL_ARMS:-0}" = "1" ] && MARKER_ARGS=(--marker-all-arms)
  LITELLM_MODEL_REGISTRY_PATH=configs/registry.json \
  python scripts/run_memory_swebench.py \
    -m "$AGENT_MODEL_NAME" -c swebench.yaml -c "$API_OVERRIDE" \
    --subset verified --split test "${ID_ARGS[@]}" \
    --environment-class "$ENV_CLASS" --workers "$WORKERS" \
    --memory-policy "$ARM" --harness "$HARNESS" "${MARKER_ARGS[@]}" \
    --context-cap "${CONTEXT_CAP:?}" --summarize-at "${SUMMARIZE_AT:?}" --keep-last-k "${KEEP_LAST_K:-3}" \
    --output "$OUT"
) 2>&1 | tee "logs/agent_${RUN_ID}.log"
echo ">>> done: $OUT"
