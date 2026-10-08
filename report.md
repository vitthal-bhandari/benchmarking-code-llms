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

### 3-MODEL COMPARISON — SWE-Bench Verified, 125 instances, B1 (Aug 31 2026)

First full head-to-head. Generation on Tillicum H200s (mini-swe-agent, no
memory, **default sampling** — no imposed temperature), scored on Klone via the
Apptainer harness. Config is uniform across models except necessary per-model
serving accommodations (tool-call parser; Gemma4/North on vLLM 0.28 via
`.venv-new`; Gemma4 alone needs `MAX_TOKENS=8192` to not hang — see
serving-recipes). Slice `0:125` (astropy + django).

| Model | Raw resolve | Fair resolve (applied + ran) | Got to a fair test |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | **62.4%** (78/125) | **79.6%** (78/98) | 98/125 |
| Gemma4-26B-A4B | 37.6% (47/125) | 59.5% (47/79) | 79/125 |
| North-Mini-Code-30B-A3B | 32.0% (40/125) | 59.7% (40/67) | 67/125 |

**Ranking: Qwen ≫ Gemma4 ≈ North.** Qwen wins on both metrics by a wide margin.
Gemma4 and North are nearly tied on *fair* rate (~59-60%) — comparable when they
produce a runnable patch — but their *raw* rates crater because they far more
often fail to produce one at all (the "got to a fair test" column: 98 vs 79 vs 67).

**Default sampling helped Qwen** vs. the earlier temp=0 run (58.6% → 62.4% raw),
consistent with the greedy-decoding-loop hypothesis. (Not a strict A/B — different
N and instance set — but directional.)

#### Failure taxonomy (Qwen & Gemma4), and what's fixable

Joining each eval outcome with its generation exit status:

**Qwen — 47 non-resolved:**
- **20 unresolved (capability)** — patch applied, tests ran, fix was wrong/
  incomplete. The genuine model ceiling; only B2 (memory)/a better model/pass@k
  moves this.
- **15 preds.json corruption (HARNESS BUG — recoverable!)** — the model produced
  a *clean, valid diff* (present in `info.submission` in the trajectory), but
  mini-swe-agent's patch extraction wrote **garbage into `preds.json`** — a
  human-readable "=== Final Patch Summary ===", or raw file contents, instead of
  the diff. `git apply` sees non-diff text → "unrecognized input" → scored as a
  failure that is **not the model's fault**. The `"No patch.txt found, using git
  diff"` class of bug, at scale. **Rebuild `preds.json` from the trajectories'
  `info.submission` and re-score → recovers up to 15 instances** (Qwen's true raw
  rate is therefore meaningfully higher than 62.4%, plausibly ~70-74%).
- **1 patch-applied-but-test-errored** (`django-12304`: patch introduced a
  Py-version-incompatible `boundary=` enum kwarg → import crash). Borderline
  capability.
- **11 empty** (8 RepeatedFormatError, 2 empty-Submitted, 1 LimitsExceeded) —
  edge cases; Qwen otherwise submits cleanly (125/125 generation-Submitted).

**Gemma4 — 78 non-resolved:**
- **32 unresolved (capability)** — model ceiling, as above.
- **27 empty — RepeatedFormatError (the dominant loss)** — Gemma4 rambles (leaks
  its `<|channel>thought` reasoning tokens), can't emit a clean tool call within
  `max_consecutive_format_errors`, and gives up with no patch. Fixes: (a) a
  **vLLM `--reasoning-parser`** for its channel format so the thinking is stripped
  and the tool call parses — the clean fix, serving-side; (b) **raise `MAX_TOKENS`**
  (16384) so it can finish rambling and still reach the tool call — a real but
  imperfect lever (trades against the slowness/hang the cap prevents).
- **9 LimitsExceeded** (250-step cap) + **7 ContextWindowExceeded** — its
  verbosity: long rambling trajectories hit step/context limits. A reasoning
  parser shrinks trajectories (helps both); trajectory summarization (B2) helps
  context; raising the step cap rarely helps an unproductive loop.
- **3 other** (2 apply-fail, 1 empty-Submitted).

**Fix priority:** (1) the `preds.json` rebuild is the highest-value, near-free
fix — recovers ~15 real Qwen wins with no re-generation; (2) a Gemma4 reasoning
parser is the biggest Gemma4 lever; (3) the capability misses (Qwen 20, Gemma4
32) are the honest ceiling and the target of the memory (B2) contribution.

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

---

## B2 — Memory management: first full 3-arm run (42 instances, cap 24k)

Setup: Qwen3.6-35B-A3B, SWE-Bench Verified `mem_subset` (42 context-stressing
instances), hard cap 24,000 tok, identical model + sampling (temp 0.6 / top_p
0.95 / top_k 20 / seed 0). One variable: the memory policy. Jobs 332110 (none),
332111 (summarize), 332112 (acm). Token counting = exact Qwen tokenizer
(calibration ratio **1.000** vs vLLM prompt_tokens on every arm).

| Arm | Submitted | Ctx deaths | Loopers | Total edits | Edits/inst | Voluntary | Peak median |
|---|--:|--:|--:|--:|--:|--:|--:|
| A1 none | 0 | **42** | 0 | 0 | 0.0 | — | 20,292 |
| A2 summarize | 25 | **0** | 15 | 347 | 8.26 | 0 (harness) | 18,117 |
| A3 acm | 25 | **12** | 5 | 195 | 4.64 | **0 / 195 (0%)** | 19,647 |

("Submitted" = produced a patch; correctness scoring on Klone still pending.)

### Findings
1. **A1 confirms the subset is genuinely cap-stressing:** 42/42 die at the cap
   with zero progress — the "cost of doing nothing" baseline.
2. **ACM Base behaviour reproduces at scale:** A3's voluntary rate is **0/195
   (0%)** — the untrained model never self-invokes `manage_context`; every edit
   was the 95% forced fallback. This is exactly the paper's Figure-4 claim
   ("even strong models lack the proactivity to manage their own context without
   dedicated training"), now shown for untrained Qwen.
3. **The two policies fail differently.** A2 (harness compresses directly,
   budget-aware, before every over-threshold call) **never overflows** (0 ctx
   deaths) but works harder (347 edits vs 195) and leaves 15 instances looping to
   the step limit. A3 (nudge-only, model-executed, fires at 95%) **dies 12×** —
   3 never compressed (ignored the nudge), 9 compressed but a single large
   observation still blew the budget. → open faithfulness question below.
4. **More edits ↔ lower success**, monotonic in both arms (advisor's hypothesis
   confirmed). Success by edit tercile: A2 14/14 → 9/14 → 2/14; A3 11/14 → 11/14
   → 3/14. Correlation(edits, success): A2 **−0.50**, A3 **−0.58**. Interpretation:
   edits are largely a *symptom* of a hard/long task, and each lossy compression
   also risks dropping needed context — both drive success down. (Correlation,
   not proven causation — the cap-sweep would separate the two.)

### Open faithfulness question (blocks the A3-death conclusion)
A3's 12 deaths depend on our choice that the 95% fallback only *nudges* (the
model must execute `manage_context`). If ACM's real `runner.py:930` instead
*hard-compresses* at the limit, A3 should never die from raw overflow and we
should add a harness backstop so A3 deaths reflect genuine non-proactivity, not
missing plumbing. Verify against ACM source before writing this up as a result.

### Deliverables
- `scripts/render_trajectory_html.py` now charts the real per-step ctx series
  (true sawtooth + cap line). Rendered set: `site/all/{none,summarize,acm}/`;
  advisor landing page + 4 hand-picked traces: `site/advisor/index.html`.
- `report_memory.md` (via `scripts/analyze_memory.py`) — the arm-comparison table.

### Next
- Klone Apptainer scoring → resolve rate (submitted ≠ resolved).
- Resolve the ACM 95% faithfulness question; re-run A3 if a backstop is needed.
- Re-derive `mem_subset` with the exact tokenizer (was chars/4); then cap sweep
  (16k/24k/32k) for the degradation-curve figure.

---

## B2 — 3 models x 3 arms, 99 instances, 64k budget (runs 3446xx)

Qwen3.5-9B, MiMo-V2.6-Distill-Qwen-9B, Gemma-4-12B-it. Identical model, sampling
and 64k budget within each model; the only variable is the memory policy. Token
calibration 1.000 on every run; zero `CalledProcessError` (local-SIF fix).

| Model | Arm | Sub. | OOC | Edits/inst | Voluntary | Peak | Steps | Ovh. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen3.5-9B | none | 30 | **66** | 0 | — | 57K | 123 | 0 |
| | summarize | 46 | **0** | 1.12 | 0 | 54K | 176 | 0.8% |
| | acm | 48 | 5 | 1.00 | **0** | 54K | 182 | 0.7% |
| MiMo-9B | none | 20 | **59** | 0 | — | 57K | 130 | 0 |
| | summarize | 30 | **0** | 1.54 | 0 | 54K | 191 | 0.8% |
| | acm | 37 | 2 | 0.70 | **0** | 54K | 183 | 0.6% |
| Gemma-4-12B-it | none | 62 | 27 | 0 | — | 35K | 71 | 0 |
| | summarize | 85 | **0** | 0.58 | 0 | 32K | 88 | 0.9% |
| | acm | 82 | 3 | 0.48 | **0** | 39K | 94 | 0.8% |

### Findings (behavioural — these are solid)
1. **Zero voluntary compressions across all three models.** Every one of the 216
   edits was harness-triggered (A2) or forced at the 95% threshold (A3), and
   `query_memory` was invoked once in the entire run. This is ACM's Figure-4
   claim reproduced at scale, and it now holds with the tools registered as real
   function calls — an earlier build asked models to type `manage_context` into a
   bash tool, which they ignored, and that artefact made forced compliance look
   like 0/6 when it is really 4/4.
2. **Memory management eliminates context death.** A1 pins at the ceiling (median
   peak 57,431 against a 57,344 usable budget) and loses 27-66 instances per
   model; A2 loses none and A3 loses 2-5. Submissions rise 60-85%.
3. **The failure mode shifts rather than disappearing.** `LimitsExceeded` climbs
   sharply in the memory arms (Qwen 3 -> 45, MiMo 18 -> 56): agents that would
   have died of context now exhaust the step limit instead. Memory buys
   exploration, not completion.
4. **Memory is nearly free** — summariser and retrieval calls cost 0.6-0.9% of
   agent tokens.
5. **Gemma is a different regime**: 35K peak vs 57K, far fewer deaths, highest
   submission rate; it solves in shorter trajectories and feels less pressure.

### Pass@1 (clean scoring round)
The first scoring round was void: the overlay directory was keyed by instance_id
inside the SHARED sif dir, so two runs grading the same instance mounted the same
writable overlay. 195 of 277 apply-failures (70%) were sandbox collisions, and
cached verdicts from that round could not be trusted either, since a contaminated
/testbed can yield a wrong verdict rather than a crash. After moving the overlay
under the per-run work_dir and re-grading all 891 with OVERWRITE=1, every job
reports "no sandbox failures detected" and apply rates rose from 25-80% to
67-100%.

| Model | A1 none | A2 summarize | A3 acm (Base) |
|---|--:|--:|--:|
| Qwen3.5-9B | 0.172 | **0.374** | 0.303 |
| MiMo-V2.6-Distill-Qwen-9B | 0.121 | 0.212 | **0.263** |
| Gemma-4-12B-it | 0.293 | 0.273 | **0.384** |
| *ACM paper, Qwen3.5-9B @128k* | *0.489* | *--* | *0.508* |

1. **Memory management improves resolve rate on every model.** Best memory arm vs
   A1: Qwen +118%, MiMo +117%, Gemma +31%. A3 beats A1 on all three.
2. **A3 beats A2 on two of three models** (MiMo, Gemma); A2 wins on Qwen. With both
   arms now firing at the identical trigger, that difference is mechanism only.
3. **The effect is far larger than ACM report.** Their ReAct->Base gap is
   0.489->0.508, just +3.9%, because at a 128K window SWE-Bench rarely applies
   context pressure at all (median peak ~23k). At our imposed 64k budget the cap
   actually binds -- A1 pins at the ceiling and loses 27-66 instances -- and the
   benefit grows to +31-118%. **The value of context management is a function of
   how tight the budget is**, which their single operating point cannot show.
4. Absolute numbers sit below theirs (0.172 vs 0.489 for ReAct) for three known
   reasons: a bash-only scaffold against their three tools (execute_bash,
   str_replace_editor, submit_patch), a 64k budget against 128k, and a
   99-instance subset chosen for context stress, i.e. harder than average.
