# Findings Report

_This file tracks findings, observations, and results as the project evolves._

---

## Phase: MVP1

_No benchmark scores yet — infrastructure/pipeline findings below._

### SWE-Bench Verified B1 (mini-swe-agent, no memory) — smoke test, Jul 13 2026

Pipeline: vLLM-served `Qwen/Qwen3.6-35B-A3B-FP8` (single L40S, `--max-model-len
65536`) + `mini-swe-agent` v2.4.5 via `hosted_vllm/` litellm provider +
Singularity/Apptainer sandboxing (Hyak has no Docker). 3-instance smoke test
(`smoke_run10/`, `astropy__astropy-{12907,13033,13236}`): **3/3 `Submitted`**
— first fully clean run of this pipeline end-to-end. (Task-level correctness of
the submitted patches not yet scored — see below.)

**Finding — vLLM tool-call-parser must match the model's actual tag format,
not just "a parser for the model family."** Qwen3.6 emits tool calls as
`<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`.
The `hermes` parser (common default recommendation for Qwen-family models)
expects a JSON-blob format instead and silently fails closed: it leaves the
API response's `tool_calls` field `null` and dumps the raw tags into `content`
rather than erroring loudly. mini-swe-agent correctly detects "no tool call
found," retries, and gives up after `max_consecutive_format_errors` (3) with
`RepeatedFormatError` — a generic-looking failure that is actually a specific,
diagnosable parser mismatch. Confirmed root cause by diffing a failing
trajectory's raw model output against vLLM's tool-parser source; fixed by
switching to `qwen3_coder` (matches the exact `<tool_call>`/`<function=`
sentinel tokens). Lesson: when an agent framework reports "no tool calls" or
similar format errors against a self-hosted vLLM model, check the raw
`content` field of the API response for an unparsed tool-call block before
assuming it's a prompting/model-capability problem — it may just be the wrong
`--tool-call-parser`.

**Finding — vLLM's `serve --help` no longer lists individual flags by
default** (this vLLM version groups them; use `vllm serve --help=<flag-name>`
or `--help=<ConfigGroup>`, e.g. `--help=tool-call-parser`, to see valid
choices for a specific flag). Also requires a GPU to build (fails on Hyak
login nodes with "Failed to infer device type") — run `--help` from an
allocated compute node, not the login node.

---

## Phase: MVP2 (Tillicum migration)

### First scored SWE-Bench Verified B1 result on Tillicum — Aug 9 2026

`run_qwen_20_216375`: full-weights `Qwen/Qwen3.6-35B-A3B` (BF16, native
context) on a Tillicum H200, `mini-swe-agent` at temp=0, SWE-Bench Verified
slice `0:20` (all astropy). **Scored 0/20 resolved**
(`sb-cli-reports/swe-bench_verified__test__run_qwen_20_216375.json`).

**The honest denominator is 0/12, not 0/20.** Only 12 of the 20 were a fair
capability test; the other 8 failed for non-capability reasons:

| Bucket | Count | Cause |
|---|---|---|
| Clean patch (fair test) | 12 | 0 resolved — the capability signal |
| ContextWindowExceededError | 6 | The stale `--max-model-len 65536` default (this run predated the native-context fix); produced empty patches |
| Malformed patch | 1 (14369) | Agent emitted raw Python, not a diff |
| Prefix-corrupted patch | 1 (14508) | A real, correct-looking diff prefixed with mini-swe-agent's literal `"No patch.txt found, using git diff\n"` log line → un-appliable. **Harness bug worth fixing** — it silently converts good patches into failures. |

**This is now a genuine capability result, not an artifact.** The *same* 20-
instance slice has returned ~0 resolved across three independent configs —
Klone/FP8/65K, Klone/native, and now Tillicum/BF16/native — ruling out
quantization and hardware. Trajectory analysis shows the mechanism:
**"plausible but not exact"** — the agent reliably localizes to the right file
and writes a targeted fix for the *described* symptom, but misses the exact
behavior the withheld tests check (canonical: `7166` patches `isfunction` →
`isfunction or property` but not the `staticmethod`/`classmethod` paths the
gold patch also covers).

**Caveats that keep this from meaning "the model is weak":** (1) n=12 is tiny;
(2) all 20 instances are astropy — a single repo, unrepresentative of full
Verified (django/sympy/matplotlib/sklearn/…); (3) the harness is deliberately
minimal (single-shot, no memory/self-repair/retrieval) — this is the **B1
floor** the eventual B2 (memory) must beat, and vendor's ~73% comes from far
more engineered scaffolds. Near-0 on hard single-repo instances under a
minimal harness is expected and, for the research design, a clean baseline.

**`sb-cli` report schema is uninformative** (again): `resolved`/`unresolved`/
`error` all empty, everything dumped into `failed_ids`, so it cannot
distinguish "patch didn't apply" from "tests ran and failed". Patch
applicability has to be inferred from the local `preds.json` instead.

### Tillicum serving — the fix chain (Aug 9 2026)

Standing up vLLM 0.21.0 for full-weights Qwen3.6 on a Tillicum H200 required a
sequence of fixes, each masking the next (all now baked into
`scripts/serve_vllm.slurm` + `serve_and_run_swebench.slurm`):
- **Multimodal encoder warmup** hangs/crashes on a text-only workload. Qwen3.6
  is a vision-language model; the encoder-profiling step hung 28 min on image,
  and capping only image moved it to video, which segfaulted. Fix:
  `--limit-mm-per-prompt '{"image":0,"video":0}'` (cap every modality).
- **MoE backend segfault (the real blocker).** vLLM auto-selects `FlashInfer
  CUTLASS` for unquantized/BF16 MoE on Hopper, and that TensorRT-LLM SM90
  CUTLASS kernel segfaults at the native level during the startup profiling
  forward pass. Fix: `--kernel-config.moe_backend triton` (found by reading
  `vllm/model_executor/layers/fused_moe/oracle/unquantized.py`, not guessing).
- **`module load gcc` is a no-op on Tillicum** — the system `/usr/bin/g++` is
  already 11.5.0, so the earlier "old gcc" theory was wrong; gcc was never the
  cause of the startup crashes.
- Full-weights BF16 (dropping the FP8 checkpoint) removes the need for
  DeepGEMM entirely on H200's 141GB — the FP8 choice was only ever a Klone
  48GB-L40S memory workaround.
