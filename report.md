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

### CORRECTION — every 0% result above was a broken scorer, not the model (Aug 20 2026)

**Retracted: "This is now a genuine capability result, not an artifact"** (the
line above, re: `run_qwen_20_216375`). It was wrong. Advisor review (skepticism
about empty-prediction-file / scoring-path bugs) prompted a re-check of the raw
`sb-cli` report JSON, not just the headline resolved count, across all three
runs scored to date (`run_qwen_20_216375`, `run_qwen_100`, `run_qwen_temp1_b_222103`):

```
completed_instances: 0
failed_instances:    <submitted_instances>   (100%, every run)
resolved_instances:  0
unresolved_instances: 0
```

`completed_instances: 0` is the tell: in sb-cli's schema `failed` ≠
`unresolved` — `unresolved` means the harness ran the tests and the patch
didn't fix the bug (a real signal); `failed` means the evaluation job itself
never completed. **100% of our submissions across three independent runs
never finished evaluating at all**, including 77 well-formed clean patches
from `run_qwen_100`.

This turned out to be a known, currently-open, unresolved outage in sb-cli's
hosted evaluator: [swe-bench/sb-cli#27](https://github.com/swe-bench/sb-cli/issues/27),
[#28](https://github.com/swe-bench/sb-cli/issues/28),
[#31](https://github.com/swe-bench/sb-cli/issues/31) — multiple independent
users report identical `completed_instances: 0` on both SWE-bench Lite and
Verified since May 2026, with the *same* predictions files scoring correctly
against the official local harness. A SWE-bench maintainer confirmed on #27
(2026-06-23): *"we don't accept submissions anymore to any of our
leaderboards, not via the experiments repo and not via sb-cli... not sure
when/if we'll get to this issue."* Our first submission was Aug 9 — over a
month after the service was already dead.

**Fix: switched scoring to the official local `swebench` harness, run via
Docker on a Mac (Colima), CPU-only** — see the new "Scoring (local Docker
harness)" section of `README.md` and `scripts/install_eval_venv.sh` /
`scripts/run_local_eval.sh`. Two Apple-Silicon-specific gaps found and fixed
along the way: (1) the harness's dict-format predictions loader requires each
entry to carry its own `instance_id` key, which sb-cli's more lenient parser
didn't need (`scripts/prepare_harness_preds.py` converts); (2) SWE-bench's
Docker Hub images are x86_64-only with no arm64 manifest, so swebench's own
(platform-less) `docker pull` call 404s on Apple Silicon — fixed by
pre-pulling each image with `--platform linux/amd64` first
(`scripts/prefetch_harness_images.py`), so the harness's own `images.get()`
finds it locally and never calls its broken pull path.

**Re-scored `run_qwen_20_216375` (the only run re-scored so far): 10/20
resolved (50%), not 0/20.** Against the "ran at all" denominator (13 patches
that were well-formed enough to apply and execute — 6 were empty from the
pre-native-context `65536` cap, 1 was the malformed `14369` diff) that's
**10/13 (77%)**. `astropy__astropy-7166` — the instance previously singled out
in the "plausible but not exact" narrative below, based on reading the diff by
eye — is confirmed **resolved**, not almost-right. The prefix-corruption bug
(`14508`, `"No patch.txt found, using git diff\n"` leaking into the patch
body) also needs a correction: it *did* apply under the real harness (ended up
`unresolved`, not `error`) — so it cost us nothing in this run, though it's
still worth fixing since it's fragile by luck, not by design.

The rest of the "plausible but not exact" mechanism description below, and the
`run_qwen_100`/temp=1.0 comparison numbers, are pending re-score with the same
local harness — do not cite the old 0% figures for those until updated here.

### First trustworthy multi-repo B1 result — Qwen3.6-35B-A3B, 99 instances (Aug 25 2026)

Scored via the Apptainer harness on Klone (`scripts/run_apptainer_eval.slurm`);
cross-validated against Mac/Docker on the 20-subset (both 10/20, instance-for-
instance), so the backend is trusted. Breakdown produced by
`scripts/analyze_eval.py` (joins eval outcome × generation trajectory).

**Qwen3.6-35B-A3B · SWE-bench Verified · B1 (mini-swe-agent, no memory, temp=0)
· 99 instances (astropy + django):**
- **Raw resolve rate: 58/99 = 58.6%**
- **Fair resolve rate: 58/78 = 74.4%** (of instances that applied + ran tests)
- Consistent across repos: astropy 13/21 (62%), django 45/78 (58%) — not a
  single-repo artifact.

This **retracts the entire "~0% Pass@1" narrative** — that was the broken sb-cli
scorer (see the Aug 20 correction above), never the model. 58.6% raw is squarely
in the expected band for a strong ~30B MoE on a deliberately minimal ReAct
harness, and is the **B1 floor** the eventual memory contribution (B2) must beat.

Where the 41 non-resolved went (harness/infra vs. genuine capability):

| Category | N | Kind |
|---|---:|---|
| Unresolved (applied + tests ran, fix wrong/incomplete) | 20 | **Genuine capability miss** — the honest B1 ceiling |
| Empty — step-limit loop (250-step cap, no patch) | 12 | Harness/infra — **largest recoverable bucket**; consistent with temp=0 greedy-decoding loops (the temp=1.0 experiment targets exactly this) |
| Apply-failed — malformed (non-diff output) | 4 | Generation/format failure |
| Empty — context-exceeded / gen-timeout / other | 4 | Infra |
| Apply-failed — other (git apply drift) | 1 | Infra |

Key framing: **21 of 41 non-resolved are harness/infra losses, not the model
being wrong** — so the fair 74.4% is the honest capability number, and the
12 step-limit loops are a concrete, testable improvement lever. Note the
`14508` prefix-corruption case now lands in *unresolved* (it applied via the
`--3way`/`--reject` fallback chain this run) rather than erroring — the bug is
fragile-by-luck, still worth fixing but it didn't cost a point here.

---

**Local Docker scoring paused (disk) — `run_qwen_20_216375` is the only run
re-scored so far.** SWE-bench's per-instance eval images are large (~4GB
shared base per repo, but *read-only image layers alone* for the full
astropy+django `run_qwen_100` set consumed the entire 28GB Colima VM disk cap
— before even accounting for the writable container overlay each running
instance additionally needs, which is what actually failed first: 46
already-cached instances all errored with "no space left on device" trying to
*start*, 0 completed). Growing the VM disk further would have cut the Mac's
free space toward single-digit GB, so we stopped and tore the VM down instead
(host disk fully restored to baseline, ~41GB free). `eval-venv/` and the
`scripts/*eval*` tooling are kept (small, no disk risk) since they're proven
correct on the astropy case — resuming just needs more disk than this laptop
should spend, e.g. a cloud VM with real storage. `run_qwen_100` and
`run_qwen_temp1_b_222103` remain unscored by any trustworthy method as of this
writing.

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
