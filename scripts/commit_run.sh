#!/usr/bin/env bash
# Commit a finished run, including the trajectory files.
#
#   bash scripts/commit_run.sh runs/qwen35-9b_acm_350424 [more dirs...]
#   bash scripts/commit_run.sh runs/qwen35-9b*_3504*
#
# .gitignore excludes runs/**/*.traj.json and raw_messages.json so superseded
# runs do not bloat the repo (they were 2.8GB and broke GitHub's merge API).
# That means a NEW run's trajectories need an explicit -f, which is easy to
# forget: it has already cost two round trips of "the exit statuses are blank".
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: commit_run.sh <run_dir> [...]" >&2; exit 1; }

n=0
for d in "$@"; do
  [ -d "$d" ] || { echo "skip (not a dir): $d" >&2; continue; }
  git add "$d" 2>/dev/null || true
  git add -f "$d"/*/*.traj.json "$d"/*/raw_messages.json 2>/dev/null || true
  t=$(ls "$d"/*/*.traj.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  $(basename "$d"): $t traj files staged"
  n=$((n+1))
done
git add logs/ 2>/dev/null || true
git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
git commit -q -m "Add $n run(s): $(for d in "$@"; do basename "$d"; done | tr '\n' ' ')"
echo ">>> committed. push with: git push origin \$(git rev-parse --abbrev-ref HEAD)"
