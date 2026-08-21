#!/usr/bin/env python3
"""Convert a mini-swe-agent/sb-cli-style preds.json (dict keyed by instance_id,
values holding model_patch/model_name_or_path) into the format the official
SWE-bench harness (swebench.harness.run_evaluation) requires: each value must
also carry its own "instance_id" key. Source file is read-only; writes a new
file so runs/<run>/preds.json stays untouched for sb-cli/other consumers.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Source preds.json (dict-keyed)")
    parser.add_argument("output", type=Path, help="Destination path for harness-compatible preds.json")
    args = parser.parse_args()

    with open(args.input) as f:
        preds = json.load(f)

    if not isinstance(preds, dict):
        raise ValueError(f"{args.input} is not a dict-keyed preds file (got {type(preds)})")

    out = {}
    for instance_id, entry in preds.items():
        out[instance_id] = {**entry, "instance_id": instance_id}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
