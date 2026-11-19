#!/bin/bash
# dumpling_queue.sh — run several arms back to back on ONE GPU. Start one per GPU
# inside tmux so the queue survives logout:
#
#   tmux new -d -s gpu0 'GPU=0 bash scripts/dumpling_queue.sh qwen35-9b:acm:1 qwen35-9b:none:1'
#   tmux new -d -s gpu1 'GPU=1 bash scripts/dumpling_queue.sh qwen35-9b:summarize:1 qwen35-9b:acm_nudge:1'
#
# Each item is <config name>:<arm>:<seed>. A failed item is logged and the queue moves on.
cd "$(dirname "$0")/.."
for item in "$@"; do
  IFS=: read -r cfg arm seed <<< "$item"
  echo ">>> [$(date '+%F %T')] GPU ${GPU:-0}: $cfg / $arm / seed $seed"
  SEED="$seed" bash scripts/dumpling_run.sh "configs/models/${cfg}.env" "$arm" \
    || echo ">>> [$(date '+%F %T')] FAILED: $item"
done
echo ">>> [$(date '+%F %T')] queue on GPU ${GPU:-0} finished"
