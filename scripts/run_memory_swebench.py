#!/usr/bin/env python3
"""
Drive MemoryAgent over SWE-bench instances — the B2 memory-management arms.

A trimmed copy of mini-swe-agent's own swebench runner that REUSES its helpers
(environment setup, dataset loading, preds writing, slicing) and only swaps the
agent class for scripts/memory_agent.py:MemoryAgent, threading the memory policy
through. Output layout is identical to the stock runner (preds.json +
<iid>/<iid>.traj.json) so the Apptainer scorer and analyze_eval.py work
unchanged; MemoryAgent additionally drops <iid>/memory_trace.json.

Usage (same knobs as serve_and_run, plus the memory ones):
  python scripts/run_memory_swebench.py \
    -m hosted_vllm/Qwen/Qwen3.6-35B-A3B -c swebench.yaml -c configs/api_override_inline_X.yaml \
    --subset verified --split test --slice 0:5 --workers 4 --environment-class singularity \
    --memory-policy summarize --context-cap 32000 --summarize-at 28000 \
    -o runs/mem_a2_summarize
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import traceback
from pathlib import Path

from datasets import load_dataset

from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    filter_instances,
    get_sb_environment,
    update_preds_file,
    remove_from_preds_file,
)
from minisweagent.utils.serialize import recursive_merge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_agent import MemoryAgent, ACM_SYSTEM_ADDENDUM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_memory")


def process_instance(instance: dict, output_dir: Path, config: dict, mem_cfg: dict) -> str:
    iid = instance["instance_id"]
    inst_dir = output_dir / iid
    inst_dir.mkdir(parents=True, exist_ok=True)
    remove_from_preds_file(output_dir / "preds.json", iid)
    (inst_dir / f"{iid}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    agent = None
    exit_status = result = None
    extra_info = {}
    try:
        env = get_sb_environment(config, instance)
        agent = MemoryAgent(model, env, **mem_cfg, **config.get("agent", {}))
        info = agent.run(task)
        exit_status, result = info.get("exit_status"), info.get("submission")
    except Exception as e:  # noqa: BLE001
        log.error(f"[{iid}] {type(e).__name__}: {e}")
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            agent.save(
                inst_dir / f"{iid}.traj.json",
                {"info": {"exit_status": exit_status, "submission": result, **extra_info}, "instance_id": iid},
            )
        update_preds_file(output_dir / "preds.json", iid, model.config.model_name, result or "")
    log.info(f"[{iid}] done: {exit_status}")
    return iid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", dest="slice_spec", default="")
    ap.add_argument("--filter", dest="filter_spec", default="")
    ap.add_argument("--instance-ids", default="", help="comma list; overrides slice/filter")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-w", "--workers", type=int, default=1)
    ap.add_argument("-m", "--model", default=None)
    ap.add_argument("-c", "--config", action="append", default=[])
    ap.add_argument("--environment-class", default="singularity")
    # memory knobs
    ap.add_argument("--memory-policy", choices=["none", "summarize", "acm"], default="none")
    ap.add_argument("--context-cap", type=int, default=32000)
    ap.add_argument("--summarize-at", type=int, default=28000)
    ap.add_argument("--keep-last-k", type=int, default=6)
    ap.add_argument("--redo-existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ds_path = DATASET_MAPPING.get(args.subset, args.subset)
    log.info(f"Loading {ds_path} [{args.split}]")
    instances = list(load_dataset(ds_path, split=args.split))
    if args.instance_ids:
        want = {s.strip() for s in args.instance_ids.split(",")}
        instances = [i for i in instances if i["instance_id"] in want]
    else:
        instances = filter_instances(instances, filter_spec=args.filter_spec, slice_spec=args.slice_spec)

    if not args.redo_existing and (out / "preds.json").exists():
        done = set(json.loads((out / "preds.json").read_text()).keys())
        instances = [i for i in instances if i["instance_id"] not in done]
        log.info(f"Skipping {len(done)} already-done instances")

    # build config
    configs = [get_config_from_spec(spec) for spec in args.config]
    configs.append({"model": {"model_name": args.model} if args.model else {}})
    config = recursive_merge(*configs) if configs else {}
    config.setdefault("agent", {})
    if args.environment_class:
        config.setdefault("environment", {})["environment_class"] = args.environment_class
    # A3: append the ACM memory instructions to the system prompt
    if args.memory_policy == "acm":
        config["agent"]["system_template"] = config["agent"].get("system_template", "") + ACM_SYSTEM_ADDENDUM

    mem_cfg = {
        "memory_policy": args.memory_policy,
        "context_cap": args.context_cap,
        "summarize_at": args.summarize_at,
        "keep_last_k": args.keep_last_k,
    }
    log.info(f"Running {len(instances)} instances | policy={args.memory_policy} cap={args.context_cap} "
             f"summarize_at={args.summarize_at} workers={args.workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_instance, inst, out, config, mem_cfg): inst["instance_id"] for inst in instances}
        for f in concurrent.futures.as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                log.error(f"[{futs[f]}] uncaught: {e}")

    log.info(f"Done. preds -> {out}/preds.json")


if __name__ == "__main__":
    main()
