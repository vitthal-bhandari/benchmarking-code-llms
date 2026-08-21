#!/usr/bin/env python3
"""Pre-pull SWE-bench eval images with an explicit --platform linux/amd64.

Needed on Apple Silicon: swebench's docker-py client calls `images.get()` then
falls back to `images.pull()` with no platform argument, which fails against
the x86_64-only manifests on Docker Hub ("no matching manifest for
linux/arm64/v8"). Pre-pulling the amd64 variant under the same tag via the
Docker CLI (which does support --platform) makes the harness's images.get()
find it locally and skip its own pull entirely.
"""
import argparse
import json
import subprocess
import sys

from datasets import load_dataset

DATASET_ALIASES = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
    "lite": "SWE-bench/SWE-bench_Lite",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_json", help="Path to a preds.json (dict keyed by instance_id)")
    parser.add_argument("--dataset", default="verified", help="Dataset alias or HF id (default: verified)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.preds_json) as f:
        preds = json.load(f)
    wanted_ids = set(preds.keys())

    dataset_name = DATASET_ALIASES.get(args.dataset, args.dataset)
    ds = load_dataset(dataset_name, split=args.split)

    images = sorted({
        row["image"] for row in ds if row["instance_id"] in wanted_ids
    })
    missing_ids = wanted_ids - {row["instance_id"] for row in ds}
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} instance_ids not found in {dataset_name}/{args.split}: {sorted(missing_ids)}", file=sys.stderr)

    print(f"{len(images)} unique images needed for {len(wanted_ids)} instances")

    local = {
        line for line in subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    }

    for i, image in enumerate(images, 1):
        tag = image if ":" in image.split("/")[-1] else f"{image}:latest"
        if tag in local:
            print(f"[{i}/{len(images)}] already present: {tag}")
            continue
        print(f"[{i}/{len(images)}] pulling {tag} ...")
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", tag], check=True)


if __name__ == "__main__":
    main()
