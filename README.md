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
updates/         Advisor / standup write-ups
plan.md          Adaptive plan + running decision log
report.md        Findings log
requirements-working.txt   pip freeze of the known-good vLLM 0.21.0 serving env

runs/            (gitignored) agent trajectories, preds.json, per-run outputs
logs/            (tracked) Slurm job stdout/stderr — kept in git; Tillicum
                 scratch isn't a reliable single copy
```

Scored summaries (`results/`) are tracked; raw trajectories/preds under `runs/`
are heavy and reproducible, so they stay local.

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

## Scripts

| Script | Purpose |
|---|---|
| `install_venv.sh` | Build both venvs (`BUILD_DEEPGEMM=1` only if serving an FP8 checkpoint) |
| `serve_and_run_swebench.slurm` | **Primary path**: one GPU job, serves + drives against localhost |
| `serve_vllm.slurm` | Standalone server, for interactive debugging only (see above) |
| `run_swebench_agent.slurm` | Standalone driver — not directly submittable on Tillicum (0-GPU); kept for reference |

Cluster-specific details (QoS, GPU ratio, CUDA/JIT toolchain) are documented
inline in each script and in `plan.md`.
