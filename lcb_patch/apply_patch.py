"""
Apply our model additions to LiveCodeBench's lm_styles.py.

Usage (run from project root after cloning LCB):
    python lcb_patch/apply_patch.py --lcb-dir /gscratch/scrubbed/$USER/benchmarking-code-llms/LiveCodeBench

The script appends our ADDITIONS to LanguageModelList in lm_styles.py.
Safe to run multiple times — it checks for duplicates before inserting.
"""

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# ── BEGIN benchmarking-code-llms additions ──"
ADDITIONS_IMPORT = """

# ── BEGIN benchmarking-code-llms additions ──
# Auto-inserted by lcb_patch/apply_patch.py — do not edit manually.
import sys, os as _os
_patch_dir = _os.path.join(_os.path.dirname(__file__), '..', '..', 'lcb_patch')
sys.path.insert(0, _os.path.abspath(_patch_dir))
from lm_styles_additions import ADDITIONS
LanguageModelList.extend(ADDITIONS)
LanguageModelStore.update({lm.model_name: lm for lm in ADDITIONS})
# ── END benchmarking-code-llms additions ──
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lcb-dir",
        required=True,
        help="Path to the cloned LiveCodeBench repo root",
    )
    args = parser.parse_args()

    lm_styles_path = Path(args.lcb_dir) / "lcb_runner" / "lm_styles.py"
    if not lm_styles_path.exists():
        print(f"ERROR: {lm_styles_path} not found. Is --lcb-dir correct?", file=sys.stderr)
        sys.exit(1)

    content = lm_styles_path.read_text()
    if MARKER in content:
        print("Patch already applied — skipping.")
        return

    content += ADDITIONS_IMPORT
    lm_styles_path.write_text(content)
    print(f"Patch applied to {lm_styles_path}")


if __name__ == "__main__":
    main()
