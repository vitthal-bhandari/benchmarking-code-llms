"""
Pin LiveCodeBench's transformers dependency to the git source build.

Usage (run from project root after cloning LCB, before `uv pip install -e .`):
    python lcb_patch/pin_transformers.py --lcb-dir /gscratch/scrubbed/$USER/benchmarking-code-llms/LiveCodeBench

Why: newer model archs (e.g. qwen3_5_moe / Qwen3.6 FP8) aren't in any released
transformers, so we install transformers from git. LCB's pyproject.toml doesn't
declare transformers directly — it comes in transitively via `vllm`. That means
`uv run` / `uv sync` re-resolves it from vllm's constraint and reverts our source
build back to a released version, re-breaking the arch. Adding transformers as an
explicit top-level dependency pinned to the git URL makes uv treat the source
build as the required version, overriding the transitive resolution. Idempotent —
safe to run multiple times.
"""

import argparse
import re
import sys
from pathlib import Path

GIT_REQ = '    "transformers @ git+https://github.com/huggingface/transformers.git",\n'
DEPS_START_RE = re.compile(r"dependencies\s*=\s*\[")
# Matches an existing quoted transformers dependency (with optional extras/version
# spec), e.g. "transformers", "transformers>=4.51", "transformers[torch]==4.51.3".
# The quote-immediately-before-`transformers` anchor avoids matching e.g.
# "sentence-transformers".
DEP_RE = re.compile(r"""(["'])transformers(\[[^\]]*\])?[^"']*\1""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lcb-dir",
        required=True,
        help="Path to the cloned LiveCodeBench repo root",
    )
    args = parser.parse_args()

    pyproject = Path(args.lcb_dir) / "pyproject.toml"
    if not pyproject.exists():
        print(f"ERROR: {pyproject} not found. Is --lcb-dir correct?", file=sys.stderr)
        sys.exit(1)

    content = pyproject.read_text()

    if "git+https://github.com/huggingface/transformers" in content:
        print("transformers already pinned to git source — skipping.")
        return

    if DEP_RE.search(content):
        # transformers is already declared directly — rewrite it in place.
        def repl(m):
            extras = m.group(2) or ""
            q = m.group(1)
            return f"{q}transformers{extras} @ git+https://github.com/huggingface/transformers.git{q}"

        new_content, n = DEP_RE.subn(repl, content)
        pyproject.write_text(new_content)
        print(f"Pinned transformers to git source in {pyproject} ({n} occurrence(s)).")
        return

    # transformers isn't declared directly (pulled in transitively, e.g. via
    # vllm) — insert it as a new top-level dependency so uv pins it explicitly.
    m = DEPS_START_RE.search(content)
    if not m:
        print(
            "ERROR: no `dependencies = [` array found in pyproject.toml — nothing to pin.",
            file=sys.stderr,
        )
        sys.exit(1)

    insert_at = m.end()
    new_content = content[:insert_at] + "\n" + GIT_REQ + content[insert_at:].lstrip("\n")
    pyproject.write_text(new_content)
    print(f"Added transformers git-source pin as a new dependency in {pyproject}.")


if __name__ == "__main__":
    main()
