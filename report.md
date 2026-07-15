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
