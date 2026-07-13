# Adaptive Plan

_This file tracks our current approach and evolves as we learn more._

---

## Phase: MVP1

### Goal
Benchmark open-source LLM agents on popular coding benchmarks and report results. All runs executed on Hyak (UW Klone HPC cluster) via Slurm jobs.

### Infrastructure
- **Cluster**: Hyak Klone (`klone.hyak.uw.edu`)
- **Account**: `stf`
- **GPU Partition**: `gpu-l40s` (48 GB VRAM, L40S)
- **Project dir**: `/gscratch/scrubbed/$USER/benchmarking-code-llms`
- **Python**: 3.12 via `uv`
- **All caches**: `/gscratch/scrubbed/$USER/.cache/{uv,huggingface,torch}`
- **All runs**: via `sbatch` or `salloc` — never on the login node
- See `HYAK_CHEATSHEET.md` for full reference

### Benchmarks (in planned execution order)

| # | Benchmark | Complexity | Status |
|---|-----------|------------|--------|
| 1 | LiveCodeBench v6 | Low | 🔄 In Progress |
| 2 | SciCode | Low-Medium | Pending |
| 3 | Terminal-Bench v2 | Medium | Pending |
| 4 | Terminal-Bench Hard | Medium | Pending |
| 5 | SWE-Bench Verified | High | Pending |
| 6 | SWE-Bench Pro | High | Pending |
| 7 | SWE-Bench CL | High | Pending |

---

## LiveCodeBench v6 — Setup Plan

### What it is
Competitive programming problems (LeetCode/AtCoder/Codeforces) released after model training cutoffs. v6 has ~1055 problems (May 2023–Apr 2025). Evaluation = code generation → execute against hidden test cases → pass@1.

### Setup Steps
- [ ] Clone LiveCodeBench repo into `/gscratch/scrubbed/$USER/benchmarking-code-llms`
- [ ] Write `pyproject.toml` and set up `uv` env with LiveCodeBench dependencies
- [ ] Decide which models to evaluate
- [ ] Pre-warm model downloads via interactive `salloc` session
- [ ] Write Slurm batch script for evaluation runs
- [ ] Run smoke test on 1–2 problems before full eval

### Models

All models are MoE or large dense — use quantized checkpoints, all fit on L40S (48 GB).

| # | Model | Params | Org | HF Checkpoint | Quant | Est. VRAM |
|---|-------|--------|-----|---------------|-------|-----------|
| 1 | Qwen3.6 | 35B-A3B | Alibaba | `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 (official) | ~35 GB |
| 2 | North Mini Code | 30B-A3B | Cohere | `CohereLabs/North-Mini-Code-1.0-w4a16` | W4A16 (official) | ~20 GB |
| 3 | Devstral Small 2 | 24B Dense | Mistral | `mistralai/Devstral-Small-2-24B-Instruct-2512` | BF16 (load as-is) | ~48 GB ⚠️ tight — may need INT8 |
| 4 | Poolside Laguna XS.2 | 33B-A3B | Poolside | `poolside/Laguna-XS.2-NVFP4` | NVFP4 (official) | ~12 GB |
| 5 | Gemma4 | 26B-A4B | Google | `google/gemma-4-26B-A4B-it` | BF16 or INT4 | TBD |

### Inference Engine
- **vLLM ≥ 0.8** for all models (FP8 + NVFP4 support requires recent version)
- `--n 1 --temperature 0` for greedy pass@1

### LMStyle assignments (in lm_styles.py)
- Qwen3.6 → `QwQ` (thinking-capable Qwen format)
- North Mini Code, Devstral, Poolside, Gemma4 → `LLaMa3` (initial approximation; tune if needed)

### Python version
- **3.11** for LCB venv (LCB's own recommendation; differs from other projects using 3.12)

### Scripts created
| Script | Purpose |
|--------|---------|
| `scripts/setup_lcb.sh` | One-time: clone LCB, dirs, apply patch, create .env |
| `scripts/install_venv.sh` | Run inside `salloc`: create venv, install LCB + vLLM |
| `scripts/prewarm_models.slurm` | Pre-download all 5 model checkpoints |
| `scripts/eval_lcb_v6.slurm` | Parametric eval job (pass `MODEL_NAME` via `--export`) |
| `scripts/submit_all_models.sh` | Submit all 5 eval jobs at once |
| `lcb_patch/lm_styles_additions.py` | Model definitions for LCB |
| `lcb_patch/apply_patch.py` | Inserts our models into LCB's lm_styles.py |

### Setup Steps
- [x] Decide benchmarks and models
- [x] Confirm quantization strategy
- [x] Create LCB lm_styles patch
- [x] Write Slurm scripts
- [x] Run `setup_lcb.sh` on Hyak login node
- [x] Run `install_venv.sh` inside salloc on gpu-l40s
- [ ] Run `prewarm_models.slurm` to cache model weights
- [ ] Smoke test: single model, 5 problems (currently debugging transformers/vLLM version compat for qwen3_5_moe arch)
- [ ] Submit full eval with `submit_all_models.sh`

---

## Baseline Matrix — Next Phase

Goal end-state is an agent + memory contribution. To get there without overshooting,
we climb one rung at a time and only add complexity where a benchmark's own design
makes it informative:

1. **B0 — Zero-shot, no agent, no memory (native harness)**: single prompt → single
   generation → grade. Cheapest, always run first per benchmark.
2. **B1 — Agent (ReAct-style loop), no memory**: model gets tools (execute code,
   run tests, read files) and can iterate within one episode, but starts fresh
   every episode. Isolates "does agentic iteration help" before we ask "does
   memory on top of that help."
3. **B2 — Agent + memory** (the actual contribution, not a baseline): deferred
   until B0/B1 are solid on at least one benchmark.

Not every benchmark supports every rung — don't force a zero-shot mode onto a
benchmark that's inherently agentic (Terminal-Bench), and don't hand-roll a ReAct
loop where the benchmark already ships one (LCB self-repair).

| Benchmark | B0 zero-shot | B1 agent, no-memory | Notes |
|---|---|---|---|
| LiveCodeBench v6 | 🔄 in progress (current work) | Cheap add-on: LCB's built-in `selfrepair` scenario (generate → run tests → fix) — near-zero extra engineering, do this before building any custom agent scaffold | Best place to validate "agent, no-memory" cheaply |
| SciCode | Same harness pattern as LCB, low marginal cost once LCB pipeline works | Defer — low value, redundant with LCB self-repair signal | Not a priority for agent baseline |
| Terminal-Bench v2 | N/A — benchmark is inherently agentic, no meaningful zero-shot mode | Required — use official harness/adapter (e.g. Terminus) | Skip straight to B1 |
| Terminal-Bench Hard | N/A | Reuse same harness validated on v2 | Just a harder task subset |
| SWE-Bench Verified | Direct patch generation (Agentless-style, no tool loop) — cheap | mini-swe-agent or SWE-agent, no memory | Best 2×2 candidate: only benchmark where both B0 and B1 are well-precedented and cheap enough to run both |
| SWE-Bench Pro | Defer until Verified pipeline (both B0 and B1) is proven | Defer, reuse Verified's harness once validated | Don't duplicate engineering effort across SWE-Bench variants |
| SWE-Bench CL | Defer | Defer | Explicitly a continual-learning benchmark — natural home for the eventual B2 (agent + memory) experiment, not a near-term baseline |

### SWE-Bench Verified B1 — ACTIVE TRACK (Jul 2026 wk2)

**LCB is paused** — do not touch its code/scripts until this pipeline works.
The `VLLM_USE_FLASHINFER_SAMPLER=0` fix is already committed to the LCB scripts;
resuming LCB later is just `sbatch scripts/smoke_test_lcb.slurm`.

Tooling decision: **mini-swe-agent** (not AdaMEM's repo). It is the canonical
~100-line no-memory agent for SWE-Bench Verified, runs against any
OpenAI-compatible endpoint via litellm, supports Apptainer/Singularity (Hyak has
no Docker), and is the natural fork point for the later memory variant. AdaMEM's
repo stays a design reference for B2's memory read/write structure only.

Pipeline (prove end-to-end before scaling):
1. `scripts/serve_vllm.slurm`: `vllm serve <model>` on an L40S node → OpenAI API
   (includes `VLLM_USE_FLASHINFER_SAMPLER=0` so no FlashInfer nvcc JIT).
2. Separate small venv (`agent-venv`, py3.12), `pip install mini-swe-agent`.
3. Verify Apptainer/Singularity works on a compute node (`apptainer --version`,
   pull one SWE-Bench instance image) — the known unknown on Hyak.
4. Smoke: mini-extra swebench, **3 instances**, model = served endpoint,
   environment = singularity. Success = 3 patch predictions emitted.
5. Score smoke predictions via `sb-cli` (hosted eval — no local Docker harness).
6. Only then: full Verified run; start with 1 model, expand to top 2–3.

**Sequencing / compute discipline:**
- Finish LCB v6 zero-shot (B0) across all 5 models first — in progress.
- Add LCB `selfrepair` (B1) using the same 5 models — reuses the harness we already
  have running, no new scaffold needed. This is the fastest signal on whether
  agentic iteration matters before investing in SWE-Bench/Terminal-Bench agents.
- Before running any agent baseline (B1) on SWE-Bench or Terminal-Bench, narrow to
  the top 2–3 models from LCB zero-shot results rather than all 5 — 5 models × 2
  baseline types × several benchmarks on a single-GPU L40S queue is more compute
  than we need to make the point. Expand back to all 5 only if time/compute allow.
- SciCode, SWE-Bench Pro, and SWE-Bench CL are explicitly deferred, not dropped —
  revisit once the Verified pipeline (B0 + B1) is proven out.
