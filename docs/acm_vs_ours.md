# ACM: official repo and paper vs our implementation

Written Nov 18 2026, before the dumpling seed runs. Sources:

- **Repo:** github.com/lixiaochuan2020/agentic-context-management @ `f06f90e` (main):
  `src/runner.py`, `src/history.py`, `src/prompts.py`, `src/tools.py`, `src/client.py`,
  `src/configs/*.py`, `configs/default.yaml`, `scripts/*.sh`.
- **Paper:** arXiv 2607.23809 (24 pp.): §3 (method), Table 2, §5.1, §5.3 / Fig. 4,
  App. A (baselines), App. B (prompts and tool schemas).
- **Ours:** `scripts/memory_agent.py`, `scripts/run_memory_swebench.py`,
  `scripts/serve_and_run_swebench.slurm`, `configs/models/qwen35-9b*.env`.

The repo only implements BrowseComp-Plus / DeepSearchQA / BrowseComp. The paper reports
SWE-bench Verified, but none of the SWE-bench code is released. Apart from the tool schemas
in App. B.2, everything SWE-specific below is our adaptation.

Status key: **=** faithful · **≈** adapted, same intent · **≠** differs · **+** we add something ACM lacks · **−** ACM has it, we don't.

## 1. The 95% threshold and how episodes end

| | ACM repo | ACM paper | ours (v1, all runs so far) | v2 |
|---|---|---|---|---|
| ACM agent at 95% | `elif token_count >= 0.95·window` → STOP prompt, one more generation, episode ends | not described; "ends when finish action or context reaches the limit" (§3) | **≠** up to 3 compress nudges, then STOP | **=** `acm`: STOP directly. Old behaviour kept as `acm_nudge` |
| Compress nudge | only when `min_turns>0 and turn<min_turns` or resuming from `initial_history` (scaling / teacher-guided modes); up to 10×; text says "you have not reached the minimum number of research turns" | not mentioned | ≠ always on (3×) | `acm_nudge` only |
| Harness compresses for the model | never | never ("agent-initiated", §3) | = never | = never |
| ReAct baseline at 95% | **STOP** (the branch is not gated on `use_memory_tools`) | "ends only when the model emits a final answer or hits the 128K context cap" (App. A) | **≠** no STOP; `ContextWindowExceededError`, empty patch (97/97 at 32k, 65/65 at 64k) | **=** STOP → submit current diff |
| Context overflow | status `max_context_length`; answer salvaged from the last assistant message if it has one | — | dies, no patch | STOP fires first; if the model ignores it, the harness submits `git diff` itself |
| Repo quirk | 95% check uses the raw count without the 16,384 output reserve, so with local vLLM (`max-model-len 131072`) the prompt is rejected at 114,688 (87.5%) before the 95% STOP can fire. `count_tokens` = `litellm.token_counter` ignoring tool schemas, which undercounts further | — | our trigger uses projected count (raw + reserve) and the exact tokenizer, so STOP always has room | same as ours |

## 2. What the model is told (system prompt)

| | ACM repo | ACM paper (App. B.1) | ours v1 | v2 |
|---|---|---|---|---|
| Context-window disclosure, **both** modes: "Your context window is {N} tokens… once context is full the conversation ends immediately. You must provide your answer… before reaching the context limit." | yes | yes | **−** never told the cap, so a marker of 54,000 is uninterpretable | **=** all arms (SWE wording: submit your fix) |
| Memory header (short-/long-term memory, manage_context semantics, query_memory) | yes | yes | ≈ shorter paraphrase | = repo text, search wording → code wording |
| Marker paragraph ("the system appends [CURRENT CONTEXT TOKEN: N]… do NOT emit it yourself") | yes | **omitted** in the printed prompt | = | = |
| Strategy bullets (search more if unsure; *watch the marker, compress dead ends*; query_memory when a prior summary is relevant; mc/qm do not end the task) | yes (4 bullets) | 3 bullets (no marker bullet) | ≈ one "Guidance" sentence | = repo's 4 bullets, adapted to code |
| Task-specific strategy / answer format | per benchmark | BCP / DSQA only; SWE not printed | mini-swe-agent's SWE template | unchanged |

## 3. The token marker

| | ACM repo | ours v1 | v2 |
|---|---|---|---|
| Where | appended to the **latest** tool message only, in a copy at request time; never stored | **≠** appended to **every** observation, stored permanently | **=** latest tool message, transient |
| Value | raw count + `max_new_tokens` (16,384) | = raw count + `max_tokens` (8,192) | = |
| Baseline gets it too | yes (`token_hint` passed regardless of mode, unexplained in the baseline prompt) | no | flag `MARKER_ALL_ARMS` (default off, **pending decision**) |
| Counting | `litellm.token_counter`, no tool schemas (undercounts) | + exact tokenizer incl. tools (ratio 1.000 vs vLLM) | ours |

## 4. `manage_context`

| | ACM repo | ACM paper | ours v1 | v2 |
|---|---|---|---|---|
| Range | `[last_boundary, calling turn]`; `mc_drop_call_message=True` deletes the calling assistant turn too | "everything since your last call up to (but not including) the message that issued this tool call" (tool schema) | ≈ excludes the calling turn | = repo default (drops it) |
| Recent turns kept verbatim | none | none | = none | = |
| Earlier summaries | stack (boundary moves after the summary) | — | = stack | = |
| What the model gets back | tool result = `[summary_id: N] <inner memory text>` | "a fresh summary… is returned as the tool result" | **≠** summary inserted as a user message *before* the call; tool result "compressed N messages, freed ~X tokens. Continue working." | **=** tool result is the summary |
| Behavioural consequence (ours) | — | — | the model's next turn restarts ("Let me start by…") after 95–98% of compressions | to be measured |
| Empty range | "Error: nothing to compress…" | — | ≈ | = ACM text |

## 5. Summarizer

| | ACM repo | ACM paper | ours v1 | v2 |
|---|---|---|---|---|
| Model | `configs/default.yaml`: same model (self-summarisation); dataclass default `gpt-5.4-mini` | Qwen3.5-9B is "the summarizer and querier" (§5.1) | = same model | = |
| Prompt shape | one user message: question + conversation + instruction | same (App. B.3) | ≈ system + user | = one user message |
| Instruction | Knowledge state / Thoughts, ≤4096 tok, preserve identifiers, start with `<memory>` | same | ≈ SWE adaptation | ≈ re-adapted from the repo text |
| Input formatting | `[role] content`: tool calls **dropped**; reasoning **kept** (vLLM has no reasoning parser, so `<think>` text sits in `content`) | — | **≠** `[role] content`, and our vLLM uses `--reasoning-parser qwen3`, so tool-call turns are `content=None`. Summarizer sees **neither reasoning nor commands** | **+** reasoning **and** commands included |
| Output parsing | extract inner `<memory>`; parse_ok needs ≥50 chars; **retry once** | — | **−** raw text kept, no retry | = extract + retry once |
| Thinking stripped | `_strip_thinking` | — | n/a (parser separates it) | = |

## 6. `query_memory`

| | ACM repo | ACM paper | ours v1 | v2 |
|---|---|---|---|---|
| Mechanism | loads `summary_N.json`, LLM extracts what matches the query | same | = in-memory archive, LLM extraction | = |
| Prompt | `QUERY_MEMORY_PROMPT`: bullets in up to 3 sections (Relevant findings / Dead ends / Open), identifiers verbatim, exact sentinel when nothing relevant | same (App. B.4) | ≈ free-form "extract only what answers" | = repo prompt, identifiers → file paths, line numbers, test ids, errors |
| Input formatting | `[role] content` (no tool calls) | — | same as summarizer v1 | + reasoning and commands |
| Return | `[query_memory: summary_id=N]\n\n{result}` | — | ≈ `[summary_id: N] retrieved for "q": …` | = ACM format |
| Tool description guard | — | "DO NOT CALL THIS TOOL until at least one manage_context call has produced a summary_id" | − | = paper |
| Nudges toward it | BCP search results for docs seen before a compression carry "[seen earlier but compressed by manage_context… Call query_memory(summary_id, query)…]" | described for search tool (App. B.2) | **−** no SWE analog | **not in v2**: benchmark-specific affordance, no evidence ACM's SWE build had one. Candidate for the proactivity screen (re-read hint when a compressed file is reopened) |
| Usage observed | Qwen3.5-9B Base: 0.6 calls/question on BCP (Fig. 4) | | 1 call in ~300 A3 episodes | |

## 7. Agent scaffold and environment (SWE-bench)

| | ACM paper (App. B.2) | ours | v2 |
|---|---|---|---|
| Tools | `execute_bash` (300 s timeout, long output truncated), `str_replace_editor` (view/create/str_replace/insert/undo_edit), `submit_patch` (harness runs `git diff`) | **≠** mini-swe-agent: one `bash` tool; submission by echoing a sentinel then `git diff` | unchanged. **Largest remaining gap**: changing it breaks comparability with every result so far, and is a separate decision |
| Sandbox | Modal | Docker / Apptainer | — |
| Context window | 128K | 32K / 64K / 128K (imposed) | 64K, 128K |
| Turn cap | BCP `max_iterations=100`; SWE not stated | 250 / 500 / 1000 | 1000 |
| max_new_tokens | 16,384 | 8,192 | 8,192 (budget arithmetic at small caps) |
| Sampling | no sampling params sent; vLLM / generation_config defaults | temp 0.6, top_p 0.95, top_k 20 (Qwen card) | unchanged |
| Tool-call parser | `qwen3_xml` | `qwen3_coder` | unchanged |
| Reasoning in history | kept inline in `content` (no reasoning parser) | separate `reasoning_content` field; whether the chat template re-renders it for older turns is **unverified** | check during smoke |
| Truncation retry (`finish_reason=length`, no tool call) | injects "Your previous response was truncated…" | mini-swe-agent's FormatError re-prompt | unchanged |
| Trials | 4 rollout trials × 150 (BCP, Fig. 4) | 1 per arm | 3 seeds |

## 8. Baselines

| | ACM paper | ours | note |
|---|---|---|---|
| ReAct | full history, ends on answer or 128K cap; prompt states the window | A1: same, but no window disclosure and no STOP | v2 adds both |
| Summary Agent (ReSum / ACON) | external trigger at a fixed threshold (Fig. 1: 90%); "the agent discards **all** prior messages and continues over H′ = {s, o_sum}" (§3) | A2: trigger 95%, keeps the **last ≤3 turns verbatim**, re-summarises the previous summary each time | **pending decision**: keep A2 as is (more continuity), or make it the paper's summary agent (`A2_KEEP_LAST_K=0`) |

## 9. Corrections to our own earlier claims

1. *"A3's 0% voluntary reproduces ACM's Figure-4 claim."* Only half right. Fig. 4's
   near-zero is **GPT-5.5**. Untrained **Qwen3.5-9B calls manage_context 2.1×/question
   and query_memory 0.6×** on BCP under ACM's harness, without nudges (their nudge is
   scaling-mode only). We see ~0 on SWE-bench. Candidate explanations, all fixed or
   testable in v2: no context-window disclosure; one-line guidance instead of the strategy
   block; markers on every observation; summaries that restart the model; task domain
   (search vs code); bash-only scaffold.
2. *"A3 = ACM's untrained baseline."* Not as run. v1 A3 is ACM plus a scaling-mode nudge,
   which produced nearly all of its compressions.
3. *"Memory is worth 18× at 32k."* Measured against an A1 that cannot submit once it
   overflows. ACM's ReAct gets a STOP. The v2 A1 will show how much of the gap remains.

## 10. v2 change list

Approved (Nov 18):
1. A1 STOP at 95% → submit (harness submits `git diff` if the model ignores it).
2. Summarizer and querier see reasoning and commands.
3. Marker on the latest tool message only, transient.
4. `manage_context` returns the summary as its tool result (drops the calling turn).
5. A3 in two variants: `acm` (faithful: STOP at 95%) and `acm_nudge` (v1 behaviour), both at 64k.

Recommended for faithfulness and folded into v2 (flag them if you disagree):
6. Context-window disclosure line in every arm's system prompt.
7. ACM's memory header and 4-bullet strategy (repo text, code wording).
8. ACM's summarizer shape (single user message), `<memory>` extraction, one retry.
9. ACM's `QUERY_MEMORY_PROMPT`, return format, and the paper's tool descriptions
   (incl. the "do not call until…" guard).

Pending decisions (flags, current behaviour by default):
- `MARKER_ALL_ARMS`: give A1/A2 the marker as ACM's baseline gets it.
- `A2_KEEP_LAST_K`: 3 (ours) vs 0 (paper's summary agent).
- Tool scaffold (`str_replace_editor` / `submit_patch`): not planned now.
