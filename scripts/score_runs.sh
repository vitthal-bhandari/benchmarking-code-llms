#!/bin/bash
# =============================================================================
# score_runs.sh — score many runs on Klone in parallel, images fetched once.
#
#   bash scripts/score_runs.sh runs/*_3531*            # every run matching the glob
#   SLICES=4 WORKERS=4 bash scripts/score_runs.sh runs/qwen35-9b_acm_353104
#
# 1. One PREFETCH job pulls every image the runs need into a SHARED sif dir and
#    keeps it (Docker Hub counts each manifest fetch, so pulling per job would
#    hit the rate limit). Skipped when SKIP_PREFETCH=1.
# 2. Each run is split into SLICES jobs over its sorted instance ids
#    (EVAL_SLICE=start:end); all slices share one RUN_ID, which is safe because
#    per-instance reports and overlays are keyed by instance under that run.
#    Every scoring job waits for the prefetch to finish (afterany).
# Resumable: resubmitting skips instances that already have a report.json.
# =============================================================================
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash scripts/score_runs.sh runs/<run> [runs/<run> ...]"; exit 1; }

SLICES="${SLICES:-3}"
WORKERS="${WORKERS:-4}"
SCRATCH="${EVAL_SCRATCH:-/gscratch/scrubbed/$USER}"
export APPTAINER_EVAL_SIFDIR="${APPTAINER_EVAL_SIFDIR:-$SCRATCH/apptainer_eval/shared_sif}"
mkdir -p "$APPTAINER_EVAL_SIFDIR" logs

RUNS=()
for d in "$@"; do
  d="${d%/}"
  if [ -f "$d/preds.json" ]; then RUNS+=("$d"); else echo ">>> skip $d (no preds.json)"; fi
done
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs with preds.json"; exit 1; }

# Prefetch from the union of all instance ids, so one job covers every run.
UNION="$SCRATCH/apptainer_eval/union_preds_$$.json"
python3 - "$UNION" "${RUNS[@]}" <<'EOF'
import json, sys
out, runs = sys.argv[1], sys.argv[2:]
u = {}
for r in runs:
    p = json.load(open(f"{r}/preds.json"))
    items = p.items() if isinstance(p, dict) else ((x["instance_id"], x) for x in p)
    for iid, v in items:
        u.setdefault(iid, v)
json.dump(u, open(out, "w"))
print(f">>> {len(runs)} runs, {len(u)} unique instances")
EOF

DEP=""
if [ "${SKIP_PREFETCH:-0}" != "1" ]; then
  PJ=$(sbatch --parsable --export=ALL,PREDS="$UNION",RUN_ID=prefetch_shared,PREFETCH=1,KEEP_SIF=1,WORKERS=8 \
       scripts/run_apptainer_eval.slurm)
  echo ">>> prefetch job $PJ"
  # afterany: a few failed pulls must not cancel every scoring job; a scorer
  # pulls any image that is still missing.
  DEP="--dependency=afterany:$PJ"
fi

for d in "${RUNS[@]}"; do
  rid="$(basename "$d")"
  n=$(python3 -c "import json;p=json.load(open('$d/preds.json'));print(len(p))")
  per=$(( (n + SLICES - 1) / SLICES ))
  for ((s=0; s<n; s+=per)); do
    e=$(( s + per )); [ $e -gt $n ] && e=$n
    J=$(sbatch --parsable $DEP --job-name="sc-${rid: -6}" \
        --export=ALL,PREDS="$d/preds.json",RUN_ID="$rid",EVAL_SLICE="$s:$e",KEEP_SIF=1,WORKERS="$WORKERS" \
        scripts/run_apptainer_eval.slurm)
    echo ">>> $rid [$s:$e] -> job $J"
  done
done
echo ">>> watch: squeue -u \$USER ;  results land in logs/apptainer_eval/<run>/"
