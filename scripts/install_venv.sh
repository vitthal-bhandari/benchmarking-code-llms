#!/bin/bash
# =============================================================================
# install_venv.sh — Install uv venv + LCB + vLLM
#
# Run INSIDE a gpu-l40s salloc session (vLLM needs CUDA at install time):
#   salloc -A stf -p gpu-l40s -N 1 -c 8 --mem=32G --gpus=1 -t 01:00:00
#   bash scripts/install_venv.sh
# =============================================================================

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PROJECT_DIR="/gscratch/scrubbed/$USER/benchmarking-code-llms"
LCB_DIR="$PROJECT_DIR/LiveCodeBench"

export UV_CACHE_DIR="/gscratch/scrubbed/$USER/.cache/uv"

# LCB recommends Python 3.11
cd "$LCB_DIR"
echo ">>> Creating venv with Python 3.11..."
uv venv --python 3.11
source .venv/bin/activate

echo ">>> Installing LCB dependencies..."
uv pip install -e .

echo ">>> Installing vLLM (CUDA build) via pip --no-cache-dir..."
# Use pip (not uv pip) to avoid a GPFS cross-path copy failure that hits
# flashinfer-cubin's extremely long filenames on Hyak's scrubbed filesystem.
# --no-cache-dir forces pip to extract to a tmp dir and move atomically.
pip install "vllm>=0.8.0" --no-cache-dir

echo ">>> Installing bitsandbytes (for Devstral INT8 fallback)..."
pip install bitsandbytes --no-cache-dir

echo ">>> Upgrading transformers (needed for qwen3_5_moe arch and other new model types)..."
pip install --upgrade transformers --no-cache-dir

echo ">>> Sanity check..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "from lcb_runner.lm_styles import LanguageModelStore; print('LCB models registered:', len(LanguageModelStore))"

echo ""
echo ">>> Setup complete. venv at $LCB_DIR/.venv"
