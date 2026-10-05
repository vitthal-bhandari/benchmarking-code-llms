#!/usr/bin/env python3
"""
Warm the shared singularity/apptainer layer cache with the SWE-bench sandbox
images BEFORE launching the memory arms.

Why: the agent builds each instance's sandbox from docker://swebench/... . With
N arm-jobs launched together, all N hit a COLD shared cache at the same moment
and each pulls the same images, so 9 jobs x 99 instances is ~891 Docker Hub
pulls against a limit of 200/6h even when authenticated. That is exactly what
killed the first 99-instance run: every CalledProcessError was
"TOOMANYREQUESTS ... unauthenticated pull rate limit", clustered in the first
ten minutes of the jobs.

Pulling once, serially, into SINGULARITY_CACHEDIR turns those ~891 pulls into 99.
Run this to completion, then launch the arms — they will all hit a warm cache.

  python3 scripts/prefetch_agent_images.py --instance-file configs/mem_subset_99.txt
  python3 scripts/prefetch_agent_images.py --instance-file ... --workers 3
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def image_for(iid: str) -> str:
    # SWE-bench convention: "__" -> "_1776_", lowercased.
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_').lower()}:latest"

def sif_for(iid: str, sifdir: Path) -> Path:
    return sifdir / (image_for(iid).split("/")[-1].replace(":", "_") + ".sif")


def pull(iid: str, outdir: Path, retries: int = 4) -> tuple[str, bool, str]:
    img = image_for(iid)
    sif = sif_for(iid, outdir)
    if sif.exists() and sif.stat().st_size > 0:
        return iid, True, "cached"
    tool = shutil.which("apptainer") or shutil.which("singularity")
    if not tool:
        return iid, False, "neither apptainer nor singularity on PATH"
    for attempt in range(1, retries + 1):
        p = subprocess.run([tool, "pull", "--force", str(sif), f"docker://{img}"],
                           capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0:
            return iid, True, ""
        if "TOOMANYREQUESTS" in out or "rate limit" in out.lower():
            back = min(300, 20 * 2 ** (attempt - 1))
            print(f"  [{iid}] rate-limited, backing off {back}s (attempt {attempt}/{retries})", flush=True)
            time.sleep(back)
            continue
        return iid, False, out.strip().splitlines()[-1] if out.strip() else f"exit {p.returncode}"
    return iid, False, "rate-limited after all retries"

def default_sif_dir() -> Path:
    """Scratch lives in a different place on each cluster — /gpfs/scrubbed on
    Tillicum, /gscratch/scrubbed on Klone — so probe instead of hardcoding one."""
    user = os.environ.get("USER", "")
    for base in (f"/gpfs/scrubbed/{user}", f"/gscratch/scrubbed/{user}"):
        if Path(base).is_dir():
            return Path(base) / "swebench_sif"
    return Path("swebench_sif").resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-file", default="configs/mem_subset_99.txt")
    ap.add_argument("--sif-dir", default=None,
                    help="persistent dir for the built .sif files; the driver reads these "
                         "instead of docker:// so runs never contact Docker Hub. Defaults to "
                         "scratch on whichever cluster this is.")
    ap.add_argument("--workers", type=int, default=2,
                    help="keep LOW: concurrency is what triggers the rate limit (default 2)")
    a = ap.parse_args()

    if not (os.environ.get("SINGULARITY_DOCKER_USERNAME") or os.environ.get("APPTAINER_DOCKER_USERNAME")):
        print("WARNING: no Docker Hub auth in env. Export DOCKER_USERNAME/DOCKER_TOKEN and\n"
              "         SINGULARITY_DOCKER_USERNAME/PASSWORD first, or you get the 100/6h\n"
              "         anonymous limit and this will not finish.", file=sys.stderr)

    text = Path(a.instance_file).read_text()
    ids = [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]
    cache = os.environ.get("SINGULARITY_CACHEDIR") or os.environ.get("APPTAINER_CACHEDIR") or "(default)"
    print(f">>> prefetching {len(ids)} images into {cache} with {a.workers} workers")

    ok = bad = 0
    outdir = Path(a.sif_dir or os.environ.get("SWEBENCH_SIF_DIR") or default_sif_dir())
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: cannot create sif dir {outdir}: {e}\n"
              f"       pass --sif-dir <a writable path on this cluster>", file=sys.stderr)
        return 2
    print(f">>> sif dir: {outdir}")
    if True:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, (iid, good, err) in enumerate(ex.map(lambda x: pull(x, outdir), ids), 1):
                if good:
                    ok += 1
                else:
                    bad += 1
                    print(f"  [{i}/{len(ids)}] FAILED {iid}: {err}", flush=True)
                if i % 10 == 0:
                    print(f"  ... {i}/{len(ids)} ({ok} ok, {bad} failed)", flush=True)
    print(f">>> done: {ok} built, {bad} failed  ->  {outdir}")
    print(f">>> launch the arms with SWEBENCH_SIF_DIR={outdir}")
    if bad:
        print(">>> re-run to retry the failures before launching the arms")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
