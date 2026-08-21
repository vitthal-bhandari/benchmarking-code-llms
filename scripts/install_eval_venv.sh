#!/bin/bash
# =============================================================================
# install_eval_venv.sh — One-time local setup for scoring runs with the
# OFFICIAL SWE-bench Docker harness on this Mac, instead of sb-cli's hosted
# evaluator (which has been silently returning completed_instances: 0 on
# every submission — see report.md / swe-bench/sb-cli#27,#28,#31).
#
# This is CPU-side scoring only. Generation still happens on Tillicum GPUs;
# this machine just runs `docker` containers that apply a patch and run the
# repo's test suite.
#
#   eval-venv/   local venv (Python 3.12, NOT the repo's global Python) with
#                the `swebench` package — separate from Tillicum's .venv /
#                agent-venv, which only exist on the cluster.
#   Colima       lightweight Docker runtime (a Lima VM + docker daemon),
#                chosen over Docker Desktop to keep the footprint small and
#                the VM disk explicitly capped (see --disk below) so this
#                doesn't quietly eat your Mac's free space.
#
# Apple Silicon note: SWE-bench's prebuilt eval images on Docker Hub are
# x86_64-only (no arm64 manifest). Colima runs them via QEMU emulation
# (registered automatically at `colima start`); scripts/run_local_eval.sh
# pre-pulls each image with `docker pull --platform linux/amd64` because
# swebench's own docker-py pull call doesn't set a platform and fails against
# the arm64 host otherwise. This makes it slower than native x86_64 hardware,
# but correct.
#
# Disk: only run this if you have >~15GB genuinely free — check with `df -h /`
# first. Each repo (astropy, django, ...) pulls one large (~4GB) shared base
# image; additional instances of the *same* repo are cheap (~100MB, mostly
# shared layers). The --disk cap below is a hard ceiling on the Colima VM, not
# a request — Docker will fail to pull once it's full rather than spilling
# onto the rest of your disk, so raise it (`colima stop && colima start
# --disk N`) if you hit that.
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

FREE_GB=$(df -g / | awk 'NR==2 {print $4}')
echo ">>> Free disk on / : ${FREE_GB}GB"
if [ "$FREE_GB" -lt 15 ]; then
  echo ">>> WARNING: less than 15GB free. SWE-bench eval images are large;"
  echo ">>>          free up space before continuing or you may fill the disk."
fi

# ── Docker runtime (Colima) ─────────────────────────────────────────────────
if ! command -v colima &>/dev/null; then
  echo ">>> Installing colima + docker CLI + python@3.12 via Homebrew..."
  brew install colima docker python@3.12
fi

if ! colima status &>/dev/null; then
  # --disk is a hard cap on the VM's virtual disk, sized conservatively
  # against whatever's actually free on the host (see check above). Bump with
  # `colima stop && colima start --disk N` if a run needs more repos' worth
  # of base images than this leaves room for.
  echo ">>> Starting Colima (4 CPU, 4GB RAM, 28GB disk cap)..."
  colima start --cpu 4 --memory 4 --disk 28 --arch aarch64
else
  echo ">>> Colima already running."
fi

docker version >/dev/null || { echo "ERROR: docker CLI cannot reach Colima's daemon"; exit 1; }

# ── Python venv for the swebench harness ────────────────────────────────────
if [ ! -d eval-venv ]; then
  echo ">>> Creating eval-venv (Python 3.12)..."
  /opt/homebrew/bin/python3.12 -m venv eval-venv
fi

source eval-venv/bin/activate
python -m pip install --upgrade pip -q
pip install swebench -q

# requirements-eval.txt is the reproducible snapshot for this venv, parallel
# to requirements-working.txt for the Tillicum serving venv. Regenerate after
# any interactive change: `source eval-venv/bin/activate && pip freeze > requirements-eval.txt`
pip freeze > requirements-eval.txt

echo ">>> Done. eval-venv ready; requirements-eval.txt updated."
echo ">>> Next: scripts/run_local_eval.sh <preds.json> <run_id> [workers]"
