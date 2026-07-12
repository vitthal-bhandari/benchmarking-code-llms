# benchmarking-code-llms — Handoff Context (Cowork → Claude Code)

This file consolidates everything Claude (via Cowork) knows about this project, for
migrating work to Claude Code. Paste/drop this file into the repo (e.g. as
`CLAUDE.md` or a one-time onboarding doc) and Claude Code will have full context.

---

## 1. Repo & Environment

- **Repo**: https://github.com/vitthal-bhandari/benchmarking-code-llms.git
- **Local path (Cowork)**: `/Users/vitthalbhandari/Code/benchmarking coding agents/benchmarking-code-llms`
- **Current branch**: `mvp1` (behind `origin/mvp1` by 1 commit as of last check — `git pull` first)
- **Working tree**: clean as of last check

## 2. Standing Project Instructions (from CLAUDE.md)

> we will keep a record of all our findings in report.md file and our ever-changing
> adaptive plans in plan.md file. i don't want you to create a generic
> ml-project-started-code-template yet. please create appropriate directories and
> subdirectories as and when need arises.

Translated into working rules:
- `report.md` = running log of findings/results (currently empty — "no results yet").
- `plan.md` = adaptive plan, edited in place as strategy evolves (not append-only history).
- Do **not** scaffold a generic ML project template up front. Only create dirs/files
  when a concrete need arises (this is why the repo is still lean).

## 3. User Profile & Preferences

- Vitthal Bhandari (vitthalbhandari98@gmail.com), University of Washington.
- ML/research background; comfortable with HPC, Slurm, Python, Hugging Face ecosystem.
- Runs all experiments on **Hyak Klone HPC** (`stf` account).
- **Strong preference: concise, direct responses.** No unnecessary verbosity or
  padding — this applies to code comments, chat responses, and docs alike.
- Working rule learned from prior sessions: every run must have a corresponding
  Slurm script, even interactive/one-off sessions — nothing ad hoc on the login node.

## 4. Project Goal

Benchmark open-source LLM **coding agents** on popular coding benchmarks, executed
on Hyak Klone via Slurm. Planned benchmark order (by complexity):

| # | Benchmark | Complexity | Status |
|---|-----------|------------|--------|
| 1 | LiveCodeBench v6 | Low | In Progress |
| 2 | SciCode | Low-Medium | Pending |
| 3 | Terminal-Bench v2 | Medium | Pending |
| 4 | Terminal-Bench Hard | Medium | Pending |
| 5 | SWE-Bench Verified | High | Pending |
| 6 | SWE-Bench Pro | High | Pending |
| 7 | SWE-Bench CL | High | Pending |

## 5. Infrastructure Defaults

- Cluster: Hyak Klone (`klone.hyak.uw.edu`)
- Account: `stf`
- GPU partition: `gpu-l40s` (48 GB VRAM, L40S)
- Project dir on cluster: `/gscratch/scrubbed/$USER/benchmarking-code-llms`
- Python: 3.12 via `uv` (project-wide default; **LiveCodeBench itself uses 3.11** — see below)
- All caches redirected off home quota to `/gscratch/scrubbed/$USER/.cache/{uv,huggingface,torch}`
- All runs via `sbatch`/`salloc` — never on the login node
- Full reference: `HYAK_CHEATSHEET.md` in repo root (reusable across ML projects,
  originally written for a different project — Section 11 "Concrete Defaults" needs
  updating to reflect this project instead of `low-resource-asr`/Whisper)

## 6. Current State — LiveCodeBench v6 (from plan.md)

**What it is**: Competitive programming problems (LeetCode/AtCoder/Codeforces)
released after model training cutoffs. v6 has ~1055 problems (May 2023–Apr 2025).
Eval = code generation → execute against hidden tests → pass@1.

**Setup checklist:**
- [x] Decide benchmarks and models
- [x] Confirm quantization strategy
- [x] Create LCB lm_styles patch
- [x] Write Slurm scripts
- [ ] Run `setup_lcb.sh` on Hyak login node
- [ ] Run `install_venv.sh` inside salloc on gpu-l40s
- [ ] Run `prewarm_models.slurm` to cache model weights
- [ ] Smoke test: single model, 5 problems
- [ ] Submit full eval with `submit_all_models.sh`

**Models to evaluate** (all MoE or large dense, quantized to fit L40S 48GB):

| # | Model | Params | Org | HF Checkpoint | Quant | Est. VRAM |
|---|-------|--------|-----|---------------|-------|-----------|
| 1 | Qwen3.6 | 35B-A3B | Alibaba | `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 (official) | ~35 GB |
| 2 | North Mini Code | 30B-A3B | Cohere | `CohereLabs/North-Mini-Code-1.0-w4a16` | W4A16 (official) | ~20 GB |
| 3 | Devstral Small 2 | 24B Dense | Mistral | `mistralai/Devstral-Small-2-24B-Instruct-2512` | BF16 as-is | ~48 GB (tight, may need INT8) |
| 4 | Poolside Laguna XS.2 | 33B-A3B | Poolside | `poolside/Laguna-XS.2-NVFP4` | NVFP4 (official) | ~12 GB |
| 5 | Gemma4 | 26B-A4B | Google | `google/gemma-4-26B-A4B-it` | BF16 or INT4 | TBD |

- Inference engine: **vLLM ≥ 0.8** for all models (FP8 + NVFP4 support needs recent version); `--n 1 --temperature 0` for greedy pass@1.
- LMStyle assignments (`lm_styles.py`): Qwen3.6 → `QwQ` (thinking-capable format); North Mini Code, Devstral, Poolside, Gemma4 → `LLaMa3` (initial approximation — tune if a first run shows formatting issues; Gemma4 in particular may need its own style).
- LCB's own venv uses **Python 3.11** (differs from the 3.12 default elsewhere in the project) — LCB's recommendation.

**Findings so far (report.md)**: none yet — benchmarking hasn't been run.

## 7. Scripts Inventory

All in `scripts/` (repo root), designed to run on Hyak:

| Script | Purpose |
|--------|---------|
| `setup_lcb.sh` | One-time, login node: create dirs, clone LCB, apply patch, write `.env` template |
| `install_venv.sh` | Run inside `salloc` on `gpu-l40s`: create `.venv` (Python 3.11), install LCB + vLLM + bitsandbytes |
| `prewarm_models.slurm` | Pre-download all 5 model checkpoints via `snapshot_download` before eval jobs |
| `prewarm_models_copy.slurm` | Variant of the above targeting `ckpt-all` partition / `--gpus=l40:1` syntax — **check which is canonical before using both** |
| `smoke_test_lcb.slurm` | Fast sanity run on `release_v1` (400 problems, lite) before committing to full v6 |
| `eval_lcb_v6.slurm` | Parametric full v6 eval job; takes `MODEL_NAME` via `--export`, auto-detects quantization from model config, copies results to `results/lcb_v6/<model_slug>/` |
| `submit_all_models.sh` | Submits one `eval_lcb_v6.slurm` job per model in one go |

`lcb_patch/` (patches LiveCodeBench's own `lm_styles.py` to register our 5 models):
- `lm_styles_additions.py` — defines the `ADDITIONS` list of `LanguageModel` entries (see model table above; includes per-model comments on quirks, e.g. Devstral may need INT8/GGUF fallback, Poolside NVFP4 needs vLLM ≥0.8, Gemma4 HF ID still TBD/placeholder).
- `apply_patch.py` — idempotent patcher; appends a marked block to `lcb_runner/lm_styles.py` that imports and extends `LanguageModelList`/`LanguageModelStore`. Usage: `python lcb_patch/apply_patch.py --lcb-dir <path-to-cloned-LCB>`.

`.env.example` — template for `HF_TOKEN` (required) and `WANDB_API_KEY` (optional).

## 8. Known Gotchas / Open Issues to Carry Forward

- **Two prewarm scripts exist** (`prewarm_models.slurm` targets `gpu-l40s`/`--gpus=1`;
  `prewarm_models_copy.slurm` targets `ckpt-all`/`--gpus=l40:1`). Decide which is
  canonical and delete/rename the other before relying on it.
- **Gemma4 HF checkpoint ID is a placeholder** (`google/gemma-4-26B-A4B-it`) — verify
  it exists before prewarming/evaluating.
- **Devstral Small 2 BF16 is ~48GB**, right at the L40S ceiling — may need INT8
  (bitsandbytes, already installed by `install_venv.sh`) or a GGUF Q8 build.
- **LMStyle assignments for North Mini Code, Devstral, Poolside, Gemma4 are all
  approximated as `LLaMa3`** — tune after the first real run confirms prompt
  formatting looks right per model.
- Nothing in this repo has actually been executed on Hyak yet — the checklist in
  Section 6 is still mostly unchecked. Next concrete step is running `setup_lcb.sh`.
- `HYAK_CHEATSHEET.md` was copy-forwarded from an older project (`low-resource-asr`,
  Whisper fine-tuning) — most of it (storage layout, Slurm basics, `uv` setup,
  gotchas) is generically reusable, but Section 11 "This Project's Concrete
  Defaults" still names the old project and should be corrected.

## 9. Full Reference Files (verbatim, for completeness)

### `HYAK_CHEATSHEET.md`
Full Hyak/Slurm/uv operating guide — partitions, storage quotas (`/gscratch/scrubbed`
purges after 21 days), `uv` + Python pinning, Slurm templates (single job + array),
module loading, HF cache pre-warming pattern, and a "Persistent Gotchas" list (venvs
are partition-specific and must be rebuilt when switching partitions; `PYTHONPATH`
sometimes stripped by `uv run`; `pyctcdecode`/Whisper-specific notes not relevant to
this project). See file directly in repo root — reproduced in full in this project's
git history, not duplicated here to keep this handoff focused.

---

*Generated for migration from Claude (Cowork) to Claude Code. Source files: CLAUDE.md,
plan.md, report.md, README.md, HYAK_CHEATSHEET.md, scripts/*, lcb_patch/*, plus
prior-session memory (user profile, project state).*
