#!/usr/bin/env python3
"""
Score a SWE-bench predictions file with the OFFICIAL swebench grading logic,
using Apptainer instead of Docker to run each instance's test suite.

Why this exists (see report.md / plan.md):
  * sb-cli's hosted evaluator has been silently broken for months
    (completed_instances: 0 on every submission -- swe-bench/sb-cli#27,#28,#31).
  * The official `swebench` harness is hard-wired to the Docker Python SDK, and
    Docker is unavailable on UW Hyak/Klone (and any shared HPC cluster). On the
    Mac, Docker's per-instance image + writable-overlay storage also blew past
    the laptop's free disk.
  * Apptainer *is* available on Klone, allows CPU-only jobs, and flattens each
    DockerHub image into a single read-only SIF -- so a pull-and-DELETE pattern
    keeps peak disk to one image at a time, sidestepping the storage wall.

What is and isn't reimplemented:
  * NOT reimplemented -- the correctness logic. We import
    swebench.harness.utils.{load_swebench_dataset, make_test_spec} and
    swebench.harness.grading.get_eval_report and call them directly, so
    resolved/unresolved is decided by SWE-bench's own code. Only the container
    backend (Docker -> Apptainer) is swapped.
  * Reimplemented -- just the "run one container" step: pull image -> apply
    patch + run eval script via one `apptainer exec` -> capture log -> delete
    image. The apply logic mirrors the harness's GIT_APPLY_CMDS fallback chain
    exactly.

Writes: SIFs are read-only, and the images run as root (git apply must modify
root-owned files under /testbed), so each exec uses --fakeroot plus a per-
instance writable --overlay (ext3, sparse, deleted after). --writable-tmpfs is
selectable via OVERLAY_MODE=tmpfs but its size is capped by apptainer.conf and
may be too small for a test suite's writes -- overlay is the safe default.

Resumable: each instance's report is cached under <work_dir>/reports/. A rerun
(e.g. after a checkpoint-partition preemption) skips already-graded instances.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from swebench.harness.utils import load_swebench_dataset, make_test_spec
from swebench.harness.grading import get_eval_report
from swebench.harness.constants import TESTS_TIMEOUT

# The bash run *inside* the container: apply the model patch with swebench's own
# git-apply fallback chain, then run the instance eval script. Apply chatter ->
# apply.log (debug only); eval output -> test_output.log (the graded log). On a
# total apply failure we write the harness's APPLY_PATCH_FAIL marker into the
# graded log so get_eval_report records patch_successfully_applied=False.
WRAPPER_SH = r"""#!/bin/bash
set -uo pipefail
cd /testbed 2>/dev/null || { echo ">>>>> Patch Apply Failed" > /swebench_eval/test_output.log; exit 1; }
PATCH=/swebench_eval/patch.diff

apply_ok=0
attempt=0
{
  for cmd in \
    "git apply --verbose $PATCH" \
    "git apply --verbose --3way $PATCH" \
    "git apply --verbose --reject $PATCH" \
    "patch --batch --forward --fuzz=5 -p1 -i $PATCH"; do
    if [ "$attempt" -gt 0 ]; then
      # a failed --reject leaves partial state that breaks every later attempt;
      # restore a pristine tree first (mirrors the Docker harness)
      git checkout -- . 2>/dev/null
      git clean -fd 2>/dev/null
    fi
    attempt=$((attempt+1))
    if eval "$cmd"; then apply_ok=1; break; fi
  done
} >> /swebench_eval/apply.log 2>&1

if [ "$apply_ok" -eq 0 ]; then
  # the chain can leave the patch fully applied while each command still exits
  # non-zero; treat a clean reverse-check as success
  if git apply --check --reverse "$PATCH" >> /swebench_eval/apply.log 2>&1; then
    apply_ok=1
  fi
fi

if [ "$apply_ok" -eq 0 ]; then
  echo ">>>>> Patch Apply Failed" > /swebench_eval/test_output.log
  exit 1
fi

bash /swebench_eval/eval.sh > /swebench_eval/test_output.log 2>&1
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def sif_name(image: str) -> str:
    """docker image ref -> local .sif filename."""
    return image.split("/")[-1].replace(":", "_") + ".sif"


def run_subprocess(cmd: list[str], timeout: int | None = None) -> tuple[int, str]:
    """Run a command, killing the whole process group on timeout so a hung
    apptainer child tree doesn't leak. Returns (returncode, combined_output)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # own process group -> killpg on timeout
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        return -1, (out or "") + "\n[timed out]"


def pull_image(image: str, sif_path: Path, retries: int = 2) -> tuple[bool, str]:
    """apptainer pull docker://<image> -> sif_path, no cache (we never re-pull
    the same image, so caching only bloats scratch)."""
    for attempt in range(retries + 1):
        rc, out = run_subprocess(
            [
                "apptainer", "pull", "--disable-cache", "--force",
                str(sif_path), f"docker://{image}",
            ],
            timeout=1800,
        )
        if rc == 0 and sif_path.exists():
            return True, out
        if attempt < retries:
            time.sleep(5 * (attempt + 1))
    return False, out


def evaluate_instance(
    instance_id: str,
    pred: dict,
    test_spec,
    args,
) -> dict:
    """Full lifecycle for one instance. Returns the swebench report_map entry
    (dict keyed by instance_id) augmented with a top-level 'stage'/'reason'."""
    reports_dir = Path(args.work_dir) / "reports"
    inst_dir = Path(args.work_dir) / "instances" / instance_id
    report_path = reports_dir / f"{instance_id}.json"

    # Resume: reuse a cached report from an earlier (possibly preempted) run.
    if report_path.exists() and not args.overwrite:
        cached = json.loads(report_path.read_text())
        log(f"[skip] {instance_id}: cached ({cached.get('stage')})")
        return cached

    inst_dir.mkdir(parents=True, exist_ok=True)
    model_patch = pred.get("model_patch") or ""
    pred_for_grading = {
        "instance_id": instance_id,
        "model_name_or_path": pred.get("model_name_or_path", "None"),
        "model_patch": model_patch if model_patch.strip() else None,
    }

    def finalize(report_map: dict, stage: str, reason: str = "") -> dict:
        entry = report_map.get(instance_id, {})
        entry["stage"] = stage
        if reason:
            entry["reason"] = reason
        report_map[instance_id] = entry
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(entry, indent=2))
        if not args.keep_work:
            shutil.rmtree(inst_dir, ignore_errors=True)
        return entry

    # Empty patch: no point pulling a multi-GB image to apply nothing.
    if not model_patch.strip():
        rm = get_eval_report(test_spec, pred_for_grading, os.devnull, True)
        log(f"[empty] {instance_id}: empty patch")
        return finalize(rm, "empty_patch")

    # 1. Pull image -> sif
    sif_path = Path(args.sif_dir) / sif_name(test_spec.image)
    log(f"[pull] {instance_id}: {test_spec.image}")
    ok, pull_out = pull_image(test_spec.image, sif_path)
    if not ok:
        (inst_dir / "pull.log").write_text(pull_out)
        rm = {instance_id: {"patch_exists": True, "resolved": False,
                            "patch_successfully_applied": False, "infra_failure": True}}
        log(f"[ERROR] {instance_id}: image pull failed")
        return finalize(rm, "error", "pull_failed")

    # 2. Stage patch + eval script + wrapper for the bind mount
    (inst_dir / "patch.diff").write_text(model_patch)
    (inst_dir / "eval.sh").write_text(test_spec.eval_script)
    (inst_dir / "run_wrapper.sh").write_text(WRAPPER_SH)
    test_output_path = inst_dir / "test_output.log"

    # 3. Build the exec, with a per-instance writable overlay
    overlay_path = None
    exec_cmd = ["apptainer", "exec", "--fakeroot"]
    if args.overlay_mode == "overlay":
        overlay_path = Path(args.sif_dir) / f"overlay_{instance_id}.img"
        rc, ov_out = run_subprocess(
            ["apptainer", "overlay", "create", "--size", str(args.overlay_mb),
             str(overlay_path)],
            timeout=300,
        )
        if rc != 0:
            (inst_dir / "overlay.log").write_text(ov_out)
            sif_path.unlink(missing_ok=True)
            rm = {instance_id: {"patch_exists": True, "resolved": False,
                                "patch_successfully_applied": False, "infra_failure": True}}
            log(f"[ERROR] {instance_id}: overlay create failed")
            return finalize(rm, "error", "overlay_create_failed")
        exec_cmd += ["--overlay", str(overlay_path)]
    else:  # tmpfs
        exec_cmd += ["--writable-tmpfs"]
    exec_cmd += ["--bind", f"{inst_dir}:/swebench_eval", str(sif_path),
                 "bash", "/swebench_eval/run_wrapper.sh"]

    # 4. Run
    log(f"[run]  {instance_id}: applying patch + running tests")
    rc, run_out = run_subprocess(exec_cmd, timeout=args.timeout)
    (inst_dir / "exec.log").write_text(run_out)

    if rc == -1 and not test_output_path.exists():
        test_output_path.write_text(TESTS_TIMEOUT)

    # 5. Grade with swebench's own function (copy log out before cleanup)
    graded_log = reports_dir / f"{instance_id}.test_output.log"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if test_output_path.exists():
        shutil.copyfile(test_output_path, graded_log)
    else:
        graded_log.write_text(">>>>> Patch Apply Failed")
    report_map = get_eval_report(test_spec, pred_for_grading, str(graded_log), True)

    # 6. Cleanup image + overlay (the whole point -- keep peak disk to ~1 image)
    if not args.keep_sif:
        sif_path.unlink(missing_ok=True)
    if overlay_path is not None:
        overlay_path.unlink(missing_ok=True)

    entry = report_map.get(instance_id, {})
    resolved = entry.get("resolved", False)
    applied = entry.get("patch_successfully_applied", False)
    if resolved:
        stage = "resolved"
    elif applied:
        stage = "unresolved"
    else:
        stage = "error"  # applied-but-no-output / apply-failed / timeout
    log(f"[done] {instance_id}: {stage}")
    return finalize(report_map, stage)


def build_summary(entries: dict[str, dict], run_id: str, total_dataset: int) -> dict:
    buckets = {"resolved": [], "unresolved": [], "empty_patch": [], "error": []}
    for iid, entry in entries.items():
        stage = entry.get("stage", "error")
        if stage == "resolved":
            buckets["resolved"].append(iid)
        elif stage == "unresolved":
            buckets["unresolved"].append(iid)
        elif stage == "empty_patch":
            buckets["empty_patch"].append(iid)
        else:
            buckets["error"].append(iid)
    for v in buckets.values():
        v.sort()
    completed = len(buckets["resolved"]) + len(buckets["unresolved"])
    submitted = len(entries)
    return {
        "run_id": run_id,
        "backend": "apptainer",
        "total_instances": total_dataset,
        "submitted_instances": submitted,
        "completed_instances": completed,
        "resolved_instances": len(buckets["resolved"]),
        "unresolved_instances": len(buckets["unresolved"]),
        "empty_patch_instances": len(buckets["empty_patch"]),
        "error_instances": len(buckets["error"]),
        "resolve_rate_submitted": round(len(buckets["resolved"]) / submitted, 4) if submitted else 0,
        "resolve_rate_completed": round(len(buckets["resolved"]) / completed, 4) if completed else 0,
        "resolved_ids": buckets["resolved"],
        "unresolved_ids": buckets["unresolved"],
        "empty_patch_ids": buckets["empty_patch"],
        "error_ids": buckets["error"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True, help="preds.json (dict keyed by instance_id, or a list)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    p.add_argument("--split", default="test")
    p.add_argument("--instance-ids", default=None, help="comma-separated subset (default: all in preds)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    p.add_argument("--timeout", type=int, default=1800, help="per-instance test timeout (s)")
    p.add_argument("--work-dir", default=os.environ.get("APPTAINER_EVAL_WORKDIR", "apptainer_eval"))
    p.add_argument("--sif-dir", default=os.environ.get("APPTAINER_EVAL_SIFDIR", None),
                   help="where .sif + overlay files live (default: <work-dir>/sif). Point at scratch.")
    p.add_argument("--report-dir", default="local-eval-reports")
    p.add_argument("--overlay-mode", choices=["overlay", "tmpfs"],
                   default=os.environ.get("OVERLAY_MODE", "overlay"))
    p.add_argument("--overlay-mb", type=int, default=int(os.environ.get("OVERLAY_MB", "8192")))
    p.add_argument("--keep-sif", action="store_true", help="don't delete .sif after each instance (debug)")
    p.add_argument("--keep-work", action="store_true", help="don't delete per-instance workdir (debug)")
    p.add_argument("--overwrite", action="store_true", help="ignore cached per-instance reports")
    args = p.parse_args()

    if args.sif_dir is None:
        args.sif_dir = str(Path(args.work_dir) / "sif")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sif_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.work_dir) / "reports").mkdir(parents=True, exist_ok=True)

    preds_raw = json.loads(Path(args.preds).read_text())
    if isinstance(preds_raw, list):
        preds = {p_["instance_id"]: p_ for p_ in preds_raw}
    else:
        preds = preds_raw

    ids = [s.strip() for s in args.instance_ids.split(",")] if args.instance_ids else list(preds.keys())
    ids = [i for i in ids if i in preds]

    log(f">>> Loading dataset {args.dataset} ({args.split}) for {len(ids)} instances")
    dataset = load_swebench_dataset(args.dataset, args.split, ids)
    specs = {row["instance_id"]: make_test_spec(row) for row in dataset}
    total_dataset = len({r["instance_id"] for r in load_swebench_dataset(args.dataset, args.split)})

    missing = [i for i in ids if i not in specs]
    if missing:
        log(f">>> WARNING: {len(missing)} instance(s) not in dataset, skipping: {missing}")
    ids = [i for i in ids if i in specs]

    log(f">>> Evaluating {len(ids)} instances, {args.workers} workers, "
        f"overlay_mode={args.overlay_mode}")

    entries: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(evaluate_instance, iid, preds[iid], specs[iid], args): iid
            for iid in ids
        }
        for fut in concurrent.futures.as_completed(futs):
            iid = futs[fut]
            try:
                entries[iid] = fut.result()
            except Exception as e:  # never let one instance sink the batch
                log(f"[EXC]  {iid}: {e}")
                entries[iid] = {"stage": "error", "reason": f"exception: {e}"}

    summary = build_summary(entries, args.run_id, total_dataset)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.report_dir) / f"{args.run_id}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 60)
    log(f"Run: {args.run_id}  (backend: apptainer)")
    log(f"  submitted:  {summary['submitted_instances']}")
    log(f"  completed:  {summary['completed_instances']} (patch applied + tests ran)")
    log(f"  RESOLVED:   {summary['resolved_instances']}  "
        f"({summary['resolve_rate_submitted']:.1%} of submitted, "
        f"{summary['resolve_rate_completed']:.1%} of completed)")
    log(f"  unresolved: {summary['unresolved_instances']}")
    log(f"  empty:      {summary['empty_patch_instances']}")
    log(f"  errors:     {summary['error_instances']}")
    log(f"Report -> {out_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
