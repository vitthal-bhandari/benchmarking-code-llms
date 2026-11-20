#!/bin/bash
# =============================================================================
# dumpling_setup.sh — one-time setup on dumpling (2x A40, Ubuntu 22.04, no root,
# no scheduler). Run from the repo root, which should live on the data disk:
#
#   cd /mnt/disk2/$USER/benchmarking-code-llms && bash scripts/dumpling_setup.sh
#
# Builds, all under /mnt/disk2/$USER (home and /tmp are nearly full):
#   uv                  user-space install
#   .venv               vLLM 0.21.0 + pip CUDA toolkit, the version Tillicum served Qwen3.5 with
#   agent-venv          mini-swe-agent 2.4.6 (the version every scored run used) + litellm
#   apptainer           unprivileged install, IF user namespaces allow it (Docker is not
#                       available to our account; ask for the docker group in parallel)
# Re-runnable: each step skips what already exists. SKIP_APPTAINER=1 to skip that step.
# =============================================================================
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/mnt/disk2/$USER}"
cd "$(dirname "$0")/.."
export UV_INSTALL_DIR="$DATA_ROOT/.local/bin" UV_CACHE_DIR="$DATA_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$DATA_ROOT/.local/python" PIP_CACHE_DIR="$DATA_ROOT/.cache/pip"
export TMPDIR="$DATA_ROOT/tmp" PATH="$DATA_ROOT/.local/bin:$PATH"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$TMPDIR"

echo "=== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
uv --version

echo "=== .venv (vLLM 0.21.0)"
if [ ! -x .venv/bin/vllm ]; then
  uv venv --python 3.11 --seed .venv
  .venv/bin/pip install "vllm==0.21.0" --only-binary=:all:
  .venv/bin/pip install nvidia-cuda-nvcc nvidia-cuda-cccl
  # the transformers commit pinned in requirements-working.txt (Tillicum's serving env)
  .venv/bin/pip install "transformers @ git+https://github.com/huggingface/transformers.git@e8ea728a3eeeb903e77c7d1bd29267c80a1be71f"
fi
.venv/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')"

echo "=== agent-venv (mini-swe-agent 2.4.6)"
if [ ! -x agent-venv/bin/python ]; then
  uv venv --python 3.11 --seed agent-venv
  agent-venv/bin/pip install "mini-swe-agent==2.4.6" "litellm[proxy]" fastapi datasets transformers
fi
agent-venv/bin/python -c "import minisweagent, transformers; print('mini-swe-agent', minisweagent.__version__, '| transformers', transformers.__version__)"

echo "=== containers"
echo "user namespaces: max_user_namespaces=$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo ?)" \
     "unprivileged_userns_clone=$(cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || echo n/a)" \
     "apparmor_restrict=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo n/a)"
if docker info >/dev/null 2>&1; then
  echo "docker works for $USER -> use ENV_CLASS=docker"
elif [ "${SKIP_APPTAINER:-0}" != "1" ]; then
  if [ ! -x "$DATA_ROOT/apptainer/bin/apptainer" ]; then
    if command -v rpm2cpio >/dev/null && command -v cpio >/dev/null; then
      curl -fsSL https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh \
        | bash -s - "$DATA_ROOT/apptainer"
    else
      # No rpm2cpio on dumpling (and no root to install it): take the non-setuid
      # apptainer from conda-forge instead, via a user-space micromamba.
      MM="$DATA_ROOT/.local/bin/micromamba"
      if [ ! -x "$MM" ]; then
        curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$DATA_ROOT/.local" bin/micromamba
      fi
      MAMBA_ROOT_PREFIX="$DATA_ROOT/.mamba" "$MM" create -y -p "$DATA_ROOT/apptainer" -c conda-forge apptainer
    fi
  fi
  export PATH="$DATA_ROOT/apptainer/bin:$PATH" APPTAINER_CACHEDIR="$DATA_ROOT/.cache/apptainer"
  apptainer --version
  SB="$TMPDIR/apptainer_selftest"; rm -rf "$SB"
  apptainer build --sandbox "$SB" docker://alpine:3.19 >/dev/null
  # exactly how mini-swe-agent's singularity env runs every command
  apptainer exec --contain --cleanenv --fakeroot --writable "$SB" sh -c 'touch /probe && echo APPTAINER_OK'
  rm -rf "$SB"
  echo "add to ~/.bashrc:  export PATH=$DATA_ROOT/apptainer/bin:\$PATH"
fi

echo "=== GPUs"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "=== done. Put HF_TOKEN, DOCKER_USERNAME, DOCKER_TOKEN in .env (gitignored) before the first run."
