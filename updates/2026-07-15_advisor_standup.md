# Advisor Standup — Jul 15, 2026

## Headline
B1 (agent, no-memory) baseline pipeline for SWE-Bench Verified is now proven end-to-end on Hyak — vLLM serving, tool-calling, Singularity sandboxing (no Docker needed), and hosted scoring via `sb-cli` all confirmed working. Currently scaling from a 3-instance smoke test to 20-instance runs on two models in parallel.

## This week's progress
- Stood up a full agentic SWE-Bench Verified pipeline: self-hosted vLLM (OpenAI-compatible API) + `mini-swe-agent` (canonical ~100-line ReAct-style agent, no memory) + Apptainer/Singularity sandboxing + `sb-cli` hosted scoring.
- Root-caused and fixed the main blocker: the tool-call parser must match a model's *exact* native tool-call tag format, not just "a parser for the model family" — confirmed by diffing raw model output against vLLM's parser source, not by guessing.
- Smoke test (3 instances, Qwen3.6-35B-A3B): 3/3 completed cleanly, 0/3 resolved. Not a red flag at this sample size — patches were genuine, targeted, plausible fixes that just didn't pass the hidden test suite.
- **In progress, results by call time:** 20 instances × 2 models running in parallel — Qwen3.6-35B-A3B-FP8 and Gemma4-26B-A4B (FP8-quantized on load; native checkpoint is ~49GB, doesn't fit a 48GB GPU otherwise).
- *[Fill in before the call: final resolved/failed counts for both models from `sb-cli-reports/`]*

## Eval setup

**Models evaluated**
- `Qwen/Qwen3.6-35B-A3B-FP8` — Alibaba, 35B total / ~3B active params (MoE), officially pre-quantized FP8 checkpoint.
- `google/gemma-4-26B-A4B-it` — Google, 26B total / ~4B active params (MoE), instruction-tuned. Native checkpoint is unquantized (~49GB on disk); served with on-the-fly FP8 quantization (vLLM `--quantization fp8`) since it doesn't fit a 48GB GPU at native precision.

**Benchmark subset**
- Dataset: `princeton-nlp/SWE-Bench_Verified`, `test` split (the only split this dataset has).
- Subset: first 20 instances in dataset order (slice `0:20` of 500 total) — **not a random sample**.
- Agent tier: B1 — ReAct-style agent (`mini-swe-agent` v2.4.5), no memory, 4 concurrent workers per model.
- Environment: Apptainer/Singularity sandboxing (no Docker available on Hyak).
- Scoring: hosted `sb-cli submit swe-bench_verified test` (Princeton's official scoring API).

**Known limitations**
- **Sample size & selection**: n=20 per model is not statistically robust for a resolve-rate estimate, and the 20 instances are the *first* 20 in dataset order, not randomly sampled — possible selection bias if the dataset has any non-random structure (e.g. grouped by repo).
- **Context length capped at 65,536 tokens** for both models (vs. Qwen3.6's native 262,144), driven by GPU memory, not model capability — after loading Qwen's 35B FP8 weights on a single 48GB L40S, only ~4.6GB remains for KV cache. Several instances in both runs hit `ContextWindowExceededError` from this cap — notably the *same* instance (`astropy__astropy-13579`) failed this way in both the Qwen and Gemma4 logs, suggesting it's driven by that instance's difficulty (more agent steps → more accumulated context, since our no-memory agent keeps the full trajectory in every prompt) rather than a per-model quirk. For the full 500-instance run, the plan is to serve from an H200 node (141GB vs. 48GB, already confirmed available on Hyak) to raise the cap substantially, and/or investigate a per-episode step limit in `mini-swe-agent`'s config to bound context growth directly. Worth flagging as a talking point: this failure mode — a no-memory agent's context growing unboundedly over a long trajectory — is precisely the problem AdaMEM and the Proactive Memory Agent paper (below) are designed to solve, which makes it a concrete, first-hand motivating example for the B2 work rather than just an infra footnote.
- **Asymmetric quantization between the two models**: Qwen runs on its official, pre-quantized FP8 checkpoint; Gemma4 runs on vLLM's on-the-fly dynamic FP8 conversion of a native BF16 checkpoint (forced by the same 48GB single-GPU constraint). These aren't necessarily equivalent quantization processes — any score gap between the two models could partly reflect this asymmetry rather than a pure model-capability difference.
- **Single-GPU serving throughout**, no tensor parallelism — every model had to fit (quantized if necessary) on one 48GB L40S, which constrained which models/configs were even viable to test tonight.
- **Hyak scheduling constraints**: preemptible partitions (`ckpt-all`/`ckpt-g2`) can kill a job mid-run without warning; owned partitions have limited, shared capacity; account-wide CPU/GPU quotas capped how many parallel jobs could run at once. Didn't affect final results (preempted jobs were just restarted) but did constrain iteration speed tonight.
- **Model selection wasn't data-driven**: the plan was to rank all 5 candidate models via LiveCodeBench zero-shot first; that ranking run is still paused mid-debug, so Qwen3.6 + Gemma4 was a pragmatic choice under tonight's deadline, not an empirically justified one.
- **No baseline comparison yet**: only the B1 (agent, no memory) tier has been run — no B0 (zero-shot) or B2 (memory) numbers exist yet on the same instances, so we can't yet say whether agentic iteration itself is even helping relative to a single-shot patch attempt.

## Next steps
- Score both 20-instance runs, compare resolved rates.
- Diagnose the `smoke_run10` 0/3 result at larger n (was it small-sample noise or a real pattern?).
- Resume the paused LiveCodeBench zero-shot (B0) run to properly rank the remaining candidate models before further B1 scale-up.
- Longer term: B2 (agent + memory) is the actual research contribution — deferred until B0/B1 are solid on at least one benchmark.

---

## This week's readings

**[AdaMEM: Test-Time Adaptive Memory for Language Agents](https://yunx-z.github.io/AdaMEM/)**
- Problem: static, episode-start-only memory retrieval doesn't adapt as a long task evolves.
- Method: hybrid memory — long-term raw trajectory store + a short-term "strategy" memory synthesized fresh at each decision step (no weight updates). Two modes trade off adaptivity vs. inference cost (regenerate every step vs. persistent-with-refresh).
- Trains the strategy-generation policy with STEP-MFT: rejection sampling that keeps only examples where the retrieved strategy actually changed the agent's chosen action on a successful trajectory.
- Results: +13% relative on ALFWorld, +11% on WebShop, consistent gains on HotpotQA agentic search, favorable performance-vs-token-cost Pareto frontier.
- Relevance: already our design reference for B2's memory read/write structure (per `plan.md`).

**[Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents](https://github.com/yifannnwu/proactive-memory-agent)**
- Problem: "behavioral state decay" — critical context (requirements, prior attempts, environment facts) gets buried or pushed out of the context window over a long trajectory.
- Method: a *second*, plug-and-play memory agent runs alongside an unmodified action agent — periodically reads recent trajectory steps, maintains a structured memory bank, and decides whether to inject a reminder or stay silent.
- Evaluated on Terminal-Bench 2.0 (also in our own benchmark matrix) — Sonnet 4.5 action agent + Opus 4.6 memory agent vs. Sonnet 4.5 alone.
- Relevance: a concrete, benchmark-matched architectural pattern (separate observer/memory agent vs. baking memory into the main loop) worth weighing against AdaMEM's approach when we design B2.
