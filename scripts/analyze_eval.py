#!/usr/bin/env python3
"""
Collate a scored SWE-bench run into a presentation-ready breakdown that
distinguishes genuine model capability from harness/infra artifacts.

It JOINS two data sources per instance:
  * eval outcome   — logs/apptainer_eval/<run_id>/<iid>/report.json
                     (resolved / patch_successfully_applied, from swebench grading)
  * generation     — runs/**/<iid>/<iid>.traj.json
                     (exit_status, api_calls, submission — why the agent produced
                     what it did)

so every non-resolved instance gets a *reason*, not just a bucket. Two headline
rates are reported: raw (resolved / all submitted) and "fair" (resolved / the
instances that actually applied + ran tests — i.e. excluding empty patches and
patches that never applied, which are infra/harness losses, not the model
getting the fix wrong). 

Usage:
  python scripts/analyze_eval.py --run-id run_qwen_100 [--md report_section.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

# Category -> (label, one-line explanation for the advisor)
EXPLAIN = {
    "resolved":       ("Resolved", "Patch applied cleanly and the hidden FAIL_TO_PASS tests passed — a correct fix."),
    "unresolved":     ("Unresolved (capability)", "Patch applied and tests ran, but the fix didn't satisfy the hidden tests — a genuine, well-formed wrong/incomplete answer. This is the real capability signal."),
    "context_exceeded": ("Empty — context exceeded", "Agent ran out of context window mid-task and submitted nothing. Infra/harness (a no-memory agent re-sends the whole trajectory each step); worst on the hardest instances."),
    "step_limit_loop": ("Empty — step-limit loop", "Agent hit the 250-step cap without submitting — typically stuck repeating an unproductive action. Partly harness (greedy decoding can't escape a loop)."),
    "gen_timeout":    ("Empty — generation timeout", "Generation timed out before a patch was produced. Infra."),
    "empty_other":    ("Empty — other", "No patch produced, generation exit status not one of the above."),
    "prefix_corrupted": ("Apply-failed — prefix corruption", "The diff is real and correct-looking but mini-swe-agent prepended its own log line ('No patch.txt found, using git diff') into the patch body, so it won't apply. A FIXABLE harness bug — silently converts good work into a failure."),
    "malformed_diff": ("Apply-failed — malformed", "Submission isn't a valid unified diff at all (e.g. raw code). A generation/formatting failure."),
    "eval_timeout":   ("Eval — test timeout", "Patch applied but the test suite exceeded the per-instance timeout during scoring. Infra."),
    "apply_failed_other": ("Apply-failed — other", "Patch exists but git apply failed for another reason (e.g. context drift against the base commit)."),
    "no_generation":  ("No generation record", "Eval outcome present but no matching trajectory found — check the run dirs."),
}
CAPABILITY = {"resolved", "unresolved"}           # instances that got a fair shot
HARNESS_INFRA = {"context_exceeded", "step_limit_loop", "gen_timeout", "empty_other",
                 "prefix_corrupted", "malformed_diff", "eval_timeout", "apply_failed_other"}


def find_traj(instance_id: str) -> dict | None:
    hits = glob.glob(f"runs/**/{instance_id}/{instance_id}.traj.json", recursive=True)
    if not hits:
        return None
    # Prefer the most recent if an instance appears in multiple run dirs.
    hits.sort(key=os.path.getmtime, reverse=True)
    return json.load(open(hits[0])).get("info", {})


def categorize(iid: str, ev: dict, gen: dict | None) -> str:
    stage = ev.get("stage")
    if stage == "resolved":
        return "resolved"
    if stage == "unresolved":
        return "unresolved"
    exit_status = (gen or {}).get("exit_status", "")
    submission = (gen or {}).get("submission") or ""
    if stage == "empty_patch":
        return {
            "ContextWindowExceededError": "context_exceeded",
            "LimitsExceeded": "step_limit_loop",
            "Timeout": "gen_timeout",
        }.get(exit_status, "empty_other")
    if stage == "error":
        if ev.get("reason") == "timeout" or "Tests Timed Out" in submission:
            return "eval_timeout"
        if "No patch.txt found" in submission:
            return "prefix_corrupted"
        s = submission.strip()
        if s and not s.startswith("diff") and "diff --git" not in s:
            return "malformed_diff"
        return "apply_failed_other"
    if gen is None:
        return "no_generation"
    return "apply_failed_other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--eval-logs", default=None, help="default: logs/apptainer_eval/<run_id>")
    p.add_argument("--md", default=None, help="also write a markdown section to this path")
    args = p.parse_args()

    eval_dir = Path(args.eval_logs or f"logs/apptainer_eval/{args.run_id}")
    reports = sorted(glob.glob(str(eval_dir / "*" / "report.json")))
    if not reports:
        raise SystemExit(f"No per-instance reports under {eval_dir} — push the eval results first.")

    rows = []
    for rp in reports:
        iid = Path(rp).parent.name
        ev = json.load(open(rp))
        gen = find_traj(iid)
        cat = categorize(iid, ev, gen)
        rows.append({
            "iid": iid, "cat": cat,
            "exit_status": (gen or {}).get("exit_status"),
            "api_calls": (gen or {}).get("model_stats", {}).get("api_calls"),
        })

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r["iid"])

    n = len(rows)
    resolved = len(by_cat.get("resolved", []))
    unresolved = len(by_cat.get("unresolved", []))
    fair_denom = resolved + unresolved

    order = ["resolved", "unresolved", "context_exceeded", "step_limit_loop", "gen_timeout",
             "empty_other", "prefix_corrupted", "malformed_diff", "eval_timeout",
             "apply_failed_other", "no_generation"]

    lines = []
    lines.append(f"# {args.run_id} — resolve breakdown ({n} instances)\n")
    lines.append(f"**Raw resolve rate:  {resolved}/{n} = {resolved/n:.1%}**")
    if fair_denom:
        lines.append(f"**Fair resolve rate: {resolved}/{fair_denom} = {resolved/fair_denom:.1%}**  "
                     f"(of instances that applied + ran tests — excludes empty/apply-failed)\n")
    cap_infra = sum(len(by_cat.get(c, [])) for c in HARNESS_INFRA)
    lines.append(f"Capability-tested: {fair_denom}/{n}  |  Harness/infra losses: {cap_infra}/{n}\n")

    lines.append("| Category | Count | What it means |")
    lines.append("|---|---:|---|")
    for cat in order:
        ids = by_cat.get(cat, [])
        if not ids:
            continue
        label, expl = EXPLAIN[cat]
        lines.append(f"| {label} | {len(ids)} | {expl} |")

    lines.append("\n## Instances per category\n")
    for cat in order:
        ids = by_cat.get(cat, [])
        if not ids:
            continue
        label = EXPLAIN[cat][0]
        lines.append(f"- **{label}** ({len(ids)}): {', '.join(sorted(ids))}")

    out = "\n".join(lines)
    print(out)
    if args.md:
        Path(args.md).write_text(out + "\n")
        print(f"\n>>> wrote {args.md}")


if __name__ == "__main__":
    main()
