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
| 5 | Gemma4 | 26B-A4B | Google | `google/gemma-4-27b-it` *(verify)* | BF16 or INT4 | TBD |

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
- [ ] Run `setup_lcb.sh` on Hyak login node
- [ ] Run `install_venv.sh` inside salloc on gpu-l40s
- [ ] Run `prewarm_models.slurm` to cache model weights
- [ ] Smoke test: single model, 5 problems
- [ ] Submit full eval with `submit_all_models.sh`
