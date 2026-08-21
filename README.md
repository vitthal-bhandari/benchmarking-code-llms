# Benchmarking Coding LLMs

Benchmarking open-source LLM agents on SWE-Bench Verified, self-hosted with vLLM
and run on the UW **Tillicum** (RCC, on-demand H200) cluster via Slurm.
Baselines form a ladder — **B0** zero-shot, **B1** agent without memory, **B2**
agent + memory (the eventual contribution) — so that later memory gains are
attributable to memory rather than to harness sophistication.

## Layout

```
scripts/         Slurm + setup scripts (see below)
configs/         registry.json (litellm cost map) + api_override templates
results/         sb-cli-reports/ — scored eval summaries (resolve rates)
local-eval-reports/  Trustworthy scores from the local Docker harness (see below)
updates/         Advisor / standup write-ups
plan.md          Adaptive plan + running decision log
report.md        Findings log
requirements-working.txt   pip freeze of the known-good vLLM 0.21.0 serving env
requirements-eval.txt      pip freeze of the local Docker-eval venv (eval-venv)

runs/            (gitignored) agent trajectories, preds.json, per-run outputs
logs/            (tracked) Slurm job stdout/stderr — kept in git; Tillicum
                 scratch isn't a reliable single copy
eval-venv/       (gitignored) local Python venv for the SWE-bench Docker harness
local_eval/      (gitignored) harness working dir — build/run logs, per-instance
                 test output; regenerate anytime from runs/*/preds.json
```

Scored summaries (`results/`, `local-eval-reports/`) are tracked; raw
trajectories/preds under `runs/` are heavy and reproducible, so they stay
local.

## Cluster: Tillicum

- Slurm uses **QoS, not partitions** (`--qos=normal`), account `-A stf`, GPUs via
  `--gres=gpu:h200:1`. **No preemption** (no checkpoint queue).
- **Every job must request >=1 GPU — CPU-only jobs are rejected outright.**
  Fixed ratio: max **8 CPUs / 200GB RAM per GPU requested**; sbatch hard-fails
  over that, it's not a soft limit.
- Storage: code + venvs in `/gpfs/projects/stf/$USER/benchmarking-code-llms`
  (backed up); HF/torch/vLLM caches in `/gpfs/scrubbed/$USER/.cache/*` (large,
  purged after 60 days idle — fine for regenerable weights, don't put anything
  else there); `$HOME` is only 10GB, avoid it entirely.
- Toolchain via modules: `module load gcc/11.5.0`.

## Setup (one-time, in a GPU allocation)

```bash
salloc -A stf --qos=normal --gres=gpu:h200:1 -c 8 --mem=64G -t 02:00:00
cd /gpfs/projects/stf/$USER/benchmarking-code-llms
bash scripts/install_venv.sh          # builds .venv (serving) + agent-venv (driver)
```

`.venv` is hand-assembled (vLLM 0.21.0 + transformers-from-git + a self-contained
pip CUDA toolkit). **Do not `uv sync` it** — that reverts vLLM to a version that
breaks Qwen3.6 (see the header in `install_venv.sh`). `requirements-working.txt`
is the exact known-good snapshot.

Default models are **full-weights (BF16)**, not FP8 — H200's 141GB has no need
for the memory-driven quantization Klone's 48GB L40S required, and full weights
avoid DeepGEMM (a from-source CUDA build, the single biggest setup risk on
Klone) entirely. Download weights from the **login node** (no GPU billing) to
`/gpfs/scrubbed`, not home:

```bash
source .venv/bin/activate
export HF_HOME=/gpfs/scrubbed/$USER/.cache/huggingface
hf download Qwen/Qwen3.6-35B-A3B
hf download google/gemma-4-26B-A4B-it
```

## Pipeline (SWE-Bench Verified, B1 agent)

**One job** serves the model and drives the agent together (Tillicum disallows
a separate CPU-only driver job, and a second GPU-billed job just to make HTTP
calls would double cost for nothing):

```bash
sbatch --export=SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_qwen_20 \
  scripts/serve_and_run_swebench.slurm
# actual dir gets the job id appended: runs/run_qwen_20_<jobid>/ (so repeated
# runs stay distinct). The job log prints the resolved path; or: ls -dt runs/*

# Score once it finishes (needs SWEBENCH_API_KEY; sb-cli is in agent-venv).
source agent-venv/bin/activate
RUN=run_qwen_20_<jobid>
sb-cli submit swe-bench_verified test \
  --predictions_path runs/$RUN/preds.json --run_id $RUN
```

Run a second model (e.g. Gemma4) as its own job — it gets its own GPU and runs
in parallel:

```bash
sbatch --export=MODEL_NAME="google/gemma-4-26B-A4B-it",TOOL_CALL_PARSER=gemma4,\
  MAX_NUM_BATCHED_TOKENS=4096,AGENT_MODEL_NAME="hosted_vllm/google/gemma-4-26B-A4B-it",\
  SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_gemma4_20 \
  scripts/serve_and_run_swebench.slurm
```

`serve_vllm.slurm` also still exists standalone, for interactive debugging (a
long-lived server you `curl`/iterate against) — that use case is worth its own
GPU; full scored runs should go through `serve_and_run_swebench.slurm`.

## Scoring (official swebench grading, NOT sb-cli)

**Do not trust `sb-cli`'s hosted evaluator** — as of Aug 2026 it returns
`completed_instances: 0` / `failed_instances: 100%` on every submission
regardless of prediction quality (a known, unresolved outage —
[swe-bench/sb-cli#27](https://github.com/swe-bench/sb-cli/issues/27), #28, #31;
a maintainer confirmed they've stopped accepting submissions). Every
`sb-cli-reports/*.json` result is uninformative, **not** a real 0%. Score with
the official `swebench` grading code instead. Generation is unchanged (Tillicum
GPUs); only scoring moves. There are two backends:

**A) Klone + Apptainer (primary — no Docker, no GPU, real disk).** Klone allows
CPU-only jobs and has Apptainer + `/gscratch/scrubbed`, so this is where full
runs get scored. Each instance pulls its pre-built DockerHub eval image into a
single `.sif`, applies the patch + runs the tests inside it (`--fakeroot` +
per-instance writable overlay), grades with swebench's own `get_eval_report`,
then **deletes the `.sif`** — peak disk stays ~one image, sidestepping the
storage wall that Docker hits. Klone nodes are x86_64, so the images run
natively (no emulation).

```bash
bash scripts/install_klone_eval_venv.sh    # one-time, on a Klone LOGIN node
sbatch --export=PREDS=runs/<run>/preds.json,RUN_ID=<run_id> \
  scripts/run_apptainer_eval.slurm          # CPU-only job (default: ckpt-all)
# -> local-eval-reports/<run_id>.json (tracked). Resumable: rerun to continue
#    after a checkpoint-partition preemption (per-instance reports are cached).
```

**B) Mac + Docker/Colima (fallback — only if you have real free disk).** Same
official harness (`swebench eval`) against Docker on a laptop. Correct, but the
full astropy+django image set (~100GB+ resident) overran a disk-capped Colima
VM; use only for small single-repo subsets.

```bash
bash scripts/install_eval_venv.sh          # one-time: Colima + docker + eval-venv
scripts/run_local_eval.sh runs/<run>/preds.json <run_id> [workers]
```

(The Mac path pre-pulls each image with `--platform linux/amd64` since Docker
Hub has no arm64 manifest for these — not needed on Klone's x86_64 nodes.)

## Scripts

| Script | Purpose |
|---|---|
| `install_venv.sh` | Build both Tillicum venvs (`BUILD_DEEPGEMM=1` only if serving an FP8 checkpoint) |
| `serve_and_run_swebench.slurm` | **Primary generation path**: one GPU job, serves + drives against localhost |
| `serve_vllm.slurm` | Standalone server, for interactive debugging only (see above) |
| `run_swebench_agent.slurm` | Standalone driver — not directly submittable on Tillicum (0-GPU); kept for reference |
| `install_klone_eval_venv.sh` | **Primary scoring setup**: one-time Klone venv (`swebench`) + dataset cache |
| `run_apptainer_eval.slurm` / `run_apptainer_eval.py` | **Primary scoring**: score a run via Apptainer on Klone (CPU-only) |
| `install_eval_venv.sh` | Fallback scoring setup: Colima + Docker + `eval-venv` on a Mac |
| `run_local_eval.sh` | Fallback scoring: official Docker harness on a Mac (small subsets only) |

Cluster-specific details (QoS, GPU ratio, CUDA/JIT toolchain) are documented
inline in each script and in `plan.md`.
