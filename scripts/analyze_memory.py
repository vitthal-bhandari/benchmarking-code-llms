#!/usr/bin/env python3
"""
Aggregate the memory_trace.json files from a B2 arm (or compare several arms)
into the numbers the advisor asked for: how often each method edits memory, what
the edits do, token efficiency (stored vs used), and — critically for the
untrained-ACM question — how many edits were VOLUNTARY vs FORCED vs HARNESS.

Optionally joins each arm's resolve outcome from logs/apptainer_eval/<run_id>/
(if scored) so you can line up memory behaviour against task success.

Usage:
  # one arm
  python scripts/analyze_memory.py --run-dir runs/mem_summarize_32k
  # compare arms (repeat --run-dir), optional matching eval run-ids
  python scripts/analyze_memory.py \
    --run-dir runs/mem_none_32k --run-dir runs/mem_summarize_32k --run-dir runs/mem_acm_32k \
    --eval logs/apptainer_eval
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path


def load_arm(run_dir: str, eval_root: str | None) -> dict:
    traces = sorted(glob.glob(os.path.join(run_dir, "*", "memory_trace.json")))
    run_id = Path(run_dir).name
    resolved = set()
    if eval_root:
        for rp in glob.glob(os.path.join(eval_root, run_id, "*", "report.json")):
            try:
                if json.load(open(rp)).get("resolved"):
                    resolved.add(Path(rp).parent.name)
            except Exception:
                pass
    rows = []
    for tp in traces:
        iid = Path(tp).parent.name
        try:
            t = json.load(open(tp))
        except Exception:
            continue
        edits = t.get("edits", [])
        triggers = [e.get("trigger", "?") for e in edits]
        rows.append({
            "iid": iid,
            "policy": t.get("policy"),
            "n_edits": len(edits),
            "voluntary": triggers.count("voluntary"),
            "forced": triggers.count("forced"),
            "harness": triggers.count("harness"),
            "peak": t.get("peak_ctx_tokens", 0),
            "avg": t.get("avg_ctx_tokens", 0),
            "compressed": t.get("total_compressed_tokens", 0),
            "archive": t.get("archive_tokens", 0),
            "resolved": iid in resolved if eval_root else None,
        })
    return {"run_id": run_id, "policy": rows[0]["policy"] if rows else "?", "rows": rows, "have_eval": bool(eval_root)}


def summarize(arm: dict) -> dict:
    rows = arm["rows"]
    n = len(rows) or 1
    def col(k): return [r[k] for r in rows]
    edited = [r for r in rows if r["n_edits"] > 0]
    res = [r for r in rows if r["resolved"]] if arm["have_eval"] else []
    return {
        "run_id": arm["run_id"], "policy": arm["policy"], "n": len(rows),
        "instances_edited": len(edited),
        "total_edits": sum(col("n_edits")),
        "voluntary": sum(col("voluntary")), "forced": sum(col("forced")), "harness": sum(col("harness")),
        "mean_edits": round(sum(col("n_edits")) / n, 2),
        "peak_median": int(st.median(col("peak"))) if rows else 0,
        "peak_max": max(col("peak")) if rows else 0,
        "compressed_total": sum(col("compressed")),
        "archive_total": sum(col("archive")),
        "resolved": len(res) if arm["have_eval"] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--eval", default=None, help="logs/apptainer_eval root, to join resolve outcomes")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    arms = [summarize(load_arm(rd, args.eval)) for rd in args.run_dir]

    lines = ["# Memory-arm comparison\n"]
    lines.append("| Arm | policy | n | edited | edits | vol / forced / harness | mean/inst | peak median | compressed | archive | resolved |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for a in arms:
        vfh = f"{a['voluntary']} / {a['forced']} / {a['harness']}"
        res = "—" if a["resolved"] is None else f"{a['resolved']}/{a['n']}"
        lines.append(
            f"| {a['run_id']} | {a['policy']} | {a['n']} | {a['instances_edited']} | {a['total_edits']} | "
            f"{vfh} | {a['mean_edits']} | {a['peak_median']:,} | {a['compressed_total']:,} | {a['archive_total']:,} | {res} |"
        )

    # the untrained-ACM tell
    for a in arms:
        if a["policy"] == "acm" and a["total_edits"]:
            vol_pct = 100 * a["voluntary"] / a["total_edits"]
            lines.append(f"\n**A3 voluntary rate:** {a['voluntary']}/{a['total_edits']} edits "
                         f"({vol_pct:.0f}%) were the model's own choice; the rest were forced at 95% cap. "
                         f"{'The untrained model rarely self-manages — the gap OPD training would fill.' if vol_pct < 33 else 'The untrained model engages the mechanism on its own more than expected.'}")

    out = "\n".join(lines)
    print(out)
    if args.md:
        Path(args.md).write_text(out + "\n")
        print(f"\n>>> {args.md}")


if __name__ == "__main__":
    main()
