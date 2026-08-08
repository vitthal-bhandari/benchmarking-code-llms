# Benchmarking Coding LLMs

Benchmarking open-source LLM agents on coding benchmarks (SWE-Bench Verified,
LiveCodeBench), self-hosted with vLLM and run on the UW Hyak (Klone) HPC cluster
via Slurm. Baselines are organized as a ladder — **B0** zero-shot, **B1** agent
without memory, **B2** agent + memory (the eventual contribution) — so that later
memory gains are attributable to memory rather than to harness sophistication.

## Layout

```
scripts/         Slurm + setup scripts (see below)
configs/         registry.json (litellm cost map) + api_override templates
lcb_patch/       LiveCodeBench patches (registers our models / pins transformers)
results/         sb-cli-reports/ — scored eval summaries (resolve rates)
updates/         Advisor / standup write-ups
plan.md          Adaptive plan + running decision log
report.md        Findings log
requirements-working.txt   pip freeze of the known-good vLLM 0.21.0 env
LiveCodeBench/   Upstream LCB (git submodule)

runs/            (gitignored) agent trajectories, preds.json, per-run logs
logs/            (gitignored) Slurm job stdout/stderr
```

Only scored summaries (`results/`) are tracked; raw trajectories and preds under
`runs/` are heavy, reproducible, and stay local.

## Pipeline (SWE-Bench Verified, B1 agent)

Two Slurm jobs: a GPU job serving the model, and a CPU job driving the agent
against it. Run from the project root on Hyak.

```bash
# 1. Serve the model (OpenAI-compatible vLLM endpoint). H200 shown; see the
#    script header for L40S / quantized / multimodal variants.
sbatch --partition=ckpt-g2 --gpus=h200:1 --export=MAX_MODEL_LEN="262144" scripts/serve_vllm.slurm
squeue --me                                  # note the node, e.g. g3130
grep "Application startup complete" logs/vllm_serve_<jobid>.out

# 2. Point the agent at that node, then drive it over the benchmark.
cp configs/api_override.example.yaml configs/api_override.yaml   # edit api_base -> the node
sbatch --partition=compute --cpus-per-task=8 \
  --export=SLICE="0:20",WORKERS=4,OUTPUT_DIR=runs/run_qwen_20 scripts/run_swebench_agent.slurm

# 3. Score the predictions (needs SWEBENCH_API_KEY; sb-cli lives in agent-venv).
source agent-venv/bin/activate
sb-cli submit swe-bench_verified test \
  --predictions_path runs/run_qwen_20/preds.json --run_id run_qwen_20
```

## Environment

The vLLM/serving env is hand-assembled (vLLM 0.21.0 + transformers-from-git +
CUDA toolkit pieces pip splits out); `scripts/install_venv.sh` is the
reproducible recipe and `requirements-working.txt` is the exact known-good
snapshot. **Do not `uv sync` this env** — it reverts vLLM to a version that
breaks Qwen3.6 support (see the header in `install_venv.sh`).

## Scripts

| Script | Purpose |
|---|---|
| `serve_vllm.slurm` | vLLM OpenAI-compatible server (per-model flags in the header) |
| `run_swebench_agent.slurm` | Drives mini-swe-agent over SWE-Bench Verified against the server |
| `install_venv.sh` | Build the serving venv (`BUILD_DEEPGEMM=1` for H200 FP8) |
| `smoke_test_lcb.slurm` / `eval_lcb_v6.slurm` | LiveCodeBench smoke test / full v6 eval |

Hyak-specific gotchas (partitions, GPU flags, proxy, preemption) are documented
inline in the scripts and in `plan.md`.
