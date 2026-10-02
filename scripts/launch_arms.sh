#!/usr/bin/env bash
# Submit the three memory arms (A1 none / A2 summarize / A3 acm) for one model.
#
#   bash scripts/launch_arms.sh configs/models/qwen35-9b.env
#   bash scripts/launch_arms.sh configs/models/qwen35-9b.env acm      # one arm only
#   DRY=1 bash scripts/launch_arms.sh configs/models/gemma4-12b-it.env
#
# Everything except the policy comes from the model's .env, so the three arms are
# identical by construction — the single swapped variable is MEMORY_POLICY.
# Instance ids go through INSTANCE_FILE (a path): sbatch --export is itself
# comma-separated, so a comma list passed through it is truncated to the first id.
set -euo pipefail

CFG="${1:?usage: launch_arms.sh configs/models/<model>.env [arm ...]}"
shift || true
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(none summarize acm)

[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }
# shellcheck disable=SC1090
set -a; source "$CFG"; set +a

TAG="$(basename "$CFG" .env)"
INSTANCE_FILE="${INSTANCE_FILE:-configs/mem_subset.txt}"
WORKERS="${WORKERS:-8}"
[ -f "$INSTANCE_FILE" ] || { echo "no instance file: $INSTANCE_FILE" >&2; exit 1; }

# Only forward variables that are actually set, so unset knobs fall through to
# the slurm script's own defaults instead of being pinned to empty strings.
pairs() {
  local out=""
  for v in MODEL_NAME AGENT_MODEL_NAME SERVE_VENV TOOL_CALL_PARSER REASONING_PARSER \
           MAX_NUM_BATCHED_TOKENS MOE_BACKEND ENFORCE_EAGER KV_CACHE_DTYPE \
           TEMPERATURE TOP_P TOP_K MIN_P PRESENCE_PENALTY REPETITION_PENALTY SEED \
           ENABLE_THINKING MAX_TOKENS MAX_MODEL_LEN CONTEXT_CAP KEEP_LAST_K \
           INSTANCE_FILE WORKERS; do
    [ -n "${!v:-}" ] && out="${out},${v}=${!v}"
  done
  printf '%s' "$out"
}

for ARM in "${ARMS[@]}"; do
  EXTRA=""
  # SUMMARIZE_AT only means anything for A2; keep it out of the other arms so
  # their logged config isn't misleading.
  [ "$ARM" = "summarize" ] && [ -n "${SUMMARIZE_AT:-}" ] && EXTRA=",SUMMARIZE_AT=${SUMMARIZE_AT}"
  EXPORTS="ALL$(pairs),MEMORY_POLICY=${ARM},OUTPUT_DIR=runs/${TAG}_${ARM}${EXTRA}"
  if [ -n "${DRY:-}" ]; then
    echo "sbatch --account=stf --export=${EXPORTS} scripts/serve_and_run_swebench.slurm"
  else
    echo ">>> submitting ${TAG} / ${ARM}"
    sbatch --account=stf --export="${EXPORTS}" scripts/serve_and_run_swebench.slurm
  fi
done
