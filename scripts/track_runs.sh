#!/usr/bin/env bash
# Progress table for in-flight / finished memory-arm jobs.
#   bash scripts/track_runs.sh              # auto: your queued jobs + recent logs
#   bash scripts/track_runs.sh 341301 341302
#   watch -n 60 'bash scripts/track_runs.sh'
set -uo pipefail
cd "$(dirname "$0")/.."

JOBS=("$@")
if [ ${#JOBS[@]} -eq 0 ]; then
  mapfile -t JOBS < <(squeue --me -h -o %i 2>/dev/null | sort -u)
  # plus anything that finished recently, so completed arms don't vanish
  mapfile -t DONE < <(ls -t logs/swebench_run_*.err 2>/dev/null | head -12 \
                      | sed -E 's/.*swebench_run_([0-9]+)\.err/\1/')
  JOBS=($(printf '%s\n' "${JOBS[@]}" "${DONE[@]:-}" | grep -E '^[0-9]+$' | sort -u))
fi
[ ${#JOBS[@]} -eq 0 ] && { echo "no jobs found"; exit 0; }

printf "%-8s %-26s %-10s %-6s %-7s %-6s %s\n" JOB MODEL POLICY READY CALIB DONE "EXIT MIX (running→)"
printf '%.0s─' {1..118}; echo
for J in "${JOBS[@]}"; do
  O=logs/swebench_run_$J.out; E=logs/swebench_run_$J.err
  [ -f "$O" ] || [ -f "$E" ] || continue
  MODEL=$(grep -ohE "Starting [^ ]+ on" "$O" 2>/dev/null | head -1 | sed 's/Starting //;s/ on//' | sed 's#^XiaomiMiMo/##;s#^Qwen/##;s#^google/##')
  POLICY=$(grep -ohE "policy=[a-z]+" "$E" 2>/dev/null | head -1 | cut -d= -f2)
  READY=$(grep -qs "Ready after" "$O" && echo yes || echo NO)
  CALIB=$(grep -ohE "ratio=[0-9.]+" "$E" 2>/dev/null | head -1 | cut -d= -f2)
  TOT=$(grep -ohE "Running [0-9]+ instances" "$E" 2>/dev/null | head -1 | grep -oE '[0-9]+')
  N=$(grep -c "done:" "$E" 2>/dev/null); N=${N:-0}
  MIX=$(grep -ohE "done: [A-Za-z]+" "$E" 2>/dev/null | sed 's/done: //' \
        | sort | uniq -c | sort -rn | awk '{printf "%s:%s ", $2, $1}')
  ST=$(squeue -h -j "$J" -o %T 2>/dev/null); [ -n "$ST" ] && MIX="[$ST] $MIX"
  printf "%-8s %-26s %-10s %-6s %-7s %-6s %s\n" \
    "$J" "${MODEL:-?}" "${POLICY:-?}" "$READY" "${CALIB:-—}" "$N/${TOT:-?}" "${MIX:-—}"
done

echo
echo "── A3 escalations (STOP_PROMPT firing before death) ──"
for J in "${JOBS[@]}"; do
  E=logs/swebench_run_$J.err; [ -f "$E" ] || continue
  C=$(grep -c "escalating to STOP_PROMPT" "$E" 2>/dev/null); C=${C:-0}
  [ "$C" -gt 0 ] 2>/dev/null && echo "  $J: $C escalation(s)"
done
echo "── server failures (never became ready) ──"
for J in "${JOBS[@]}"; do
  O=logs/swebench_run_$J.out; [ -f "$O" ] || continue
  if ! grep -qs "Ready after" "$O"; then
    R=$(grep -hoE "AttributeError:.*|ValueError:.*|RuntimeError:.*|Error:.*" "$O" 2>/dev/null | head -1)
    echo "  $J: ${R:-still loading or no error yet}"
  fi
done
