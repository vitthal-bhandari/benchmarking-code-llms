#!/bin/bash
# =============================================================================
# install_klone_eval_venv.sh — One-time setup on UW Klone for scoring
# SWE-bench predictions with Apptainer (no Docker, no GPU).
#
# Run on a KLONE LOGIN NODE (needs internet for pip + the one-time HF dataset
# download; Apptainer image pulls happen later, inside the compute job).
#
#   klone-eval-venv/   lightweight venv (just `swebench` — pure CPU dataset +
#                      grading code, NO torch/vLLM). Separate from Tillicum's
#                      .venv / agent-venv.
#
# All caches go to /gscratch/scrubbed (home is 10GB and will choke). Apptainer
# itself is provided as a module on the compute node, not installed here.
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SCRATCH="/gscratch/scrubbed/$USER"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Python: prefer a module if the system python is old. Klone has python via
# module system; adjust the version if `module load` fails.
if ! command -v python3.11 >/dev/null 2>&1 && ! command -v python3.12 >/dev/null 2>&1; then
  module load python/3.11 2>/dev/null || module load Python 2>/dev/null || true
fi
PYBIN="$(command -v python3.12 || command -v python3.11 || command -v python3)"
echo ">>> Using python: $PYBIN ($($PYBIN --version))"

if [ ! -d klone-eval-venv ]; then
  echo ">>> Creating klone-eval-venv..."
  "$PYBIN" -m venv klone-eval-venv
fi
source klone-eval-venv/bin/activate
python -m pip install --upgrade pip -q
pip install swebench -q
pip freeze > requirements-klone-eval.txt

# Pre-download the Verified dataset into the scratch HF cache now (on the login
# node, which has reliable internet), so the compute job doesn't depend on HF
# connectivity — it only needs DockerHub for `apptainer pull`.
echo ">>> Pre-caching SWE-bench_Verified (test) into $HF_HOME ..."
python - <<'PY'
from swebench.harness.utils import load_swebench_dataset
ds = load_swebench_dataset("SWE-bench/SWE-bench_Verified", "test")
print(f"    cached {len(ds)} instances")
PY

echo ">>> Done. venv + dataset cache ready."
echo ">>> Next: sbatch --export=PREDS=runs/<run>/preds.json,RUN_ID=<id> scripts/run_apptainer_eval.slurm"
