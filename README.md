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
- Storage: code + venvs in `/gpfs/projects/stf/$USER/benchmarking-code-llms`;
  HF/torch/vLLM caches in `/gpfs/scrubbed/$USER/.cache/*`; `$HOME` is only 10 GB.
- Toolchain via modules: `module load gcc/11.5.0 cuda/13.0.0`.

## Setup (one-time, in a GPU allocation)

```bash
salloc -A stf --qos=normal --gres=gpu:h200:1 -c 16 --mem=64G -t 02:00:00
cd /gpfs/projects/stf/$USER/benchmarking-code-llms
bash scripts/install_venv.sh          # builds .venv (serving) + agent-venv (driver)
```

`.venv` is hand-assembled (vLLM 0.21.0 + transformers-from-git + a self-contained
pip CUDA toolkit + DeepGEMM for H200 FP8). **Do not `uv sync` it** — that reverts
vLLM to a version that breaks Qwen3.6 (see the header in `install_venv.sh`).
`requirements-working.txt` is the exact known-good snapshot.

## Pipeline (SWE-Bench Verified, B1 agent)

Two jobs: a GPU job serving the model, and a driver job running the agent.

```bash
# 1. Serve the model.
sbatch --export=MAX_MODEL_LEN=262144 scripts/serve_vllm.slurm
squeue --me                                          # note the node
grep "Application startup complete" logs/vllm_serve_<jobid>.out

# 2. Point the agent at that node, then drive it over the benchmark.
cp configs/api_override.example.yaml configs/api_override.yaml   # set api_base -> node
sbatch --export=SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_qwen_20 \
  scripts/run_swebench_agent.slurm

# 3. Score the predictions (needs SWEBENCH_API_KEY; sb-cli is in agent-venv).
source agent-venv/bin/activate
sb-cli submit swe-bench_verified test \
  --predictions_path runs/run_qwen_20/preds.json --run_id run_qwen_20
```

Run a second model (e.g. Gemma4) by serving it with its own
`TOOL_CALL_PARSER`/`api_override_gemma4.yaml` and passing `MODEL_NAME` +
`API_OVERRIDE` to the driver — see the script headers.

## Scripts

| Script | Purpose |
|---|---|
| `install_venv.sh` | Build both venvs (`BUILD_DEEPGEMM=1` default, for H200 FP8) |
| `serve_vllm.slurm` | vLLM OpenAI-compatible server (per-model flags in the header) |
| `run_swebench_agent.slurm` | Drives mini-swe-agent over SWE-Bench Verified against the server |

Cluster-specific details (QoS, GPU flags, CUDA/JIT toolchain, proxy handling)
are documented inline in each script and in `plan.md`.
