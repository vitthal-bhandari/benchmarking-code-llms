#!/usr/bin/env python3
"""
MemoryAgent — a mini-swe-agent DefaultAgent subclass that adds a pluggable
memory-management policy and per-step telemetry, for the B2 study.

Three policies (one swapped variable; everything else identical):
  none       A1 — no memory management. Pure instrumentation. If the live
                  message list exceeds the served context cap, the API errors →
                  mini-swe-agent's ContextWindowExceededError exit (the "dies"
                  baseline).
  summarize  A2 — harness-triggered. When live context >= summarize_at, compress
                  the settled middle (messages[2 : last-K turns]) into one
                  summary via a tool-free summarizer call, archive the originals,
                  continue. Lossy (no retrieval).
  acm        A3 — the ACM scaffold on an UNTRAINED model (= ACM's own stage-1
                  baseline). Agent-decided: after each observation the agent sees
                  "[CURRENT CONTEXT TOKEN: N]" and may issue `manage_context`
                  (compress since last call, archived losslessly) or
                  `query_memory <id>` (retrieve an archived summary's originals).
                  Implemented by intercepting those two BASH COMMANDS — faithful
                  to the mechanism; mini-swe-agent's tool parser only allows the
                  `bash` tool, so structured tools aren't an option.

Every run also writes `memory_trace.json` next to the trajectory: a per-turn
time series of live-context tokens + every memory edit (span, tokens
before/after, summary size), which feeds scripts/analyze_memory.py and lights up
the memory panels in scripts/render_trajectory_html.py.
"""
from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

import litellm
import requests

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

# ── Summarizer prompt (SWE-bench adaptation of ACM's _SUMMARY_INSTRUCTION_NOQUERY) ──
SUMMARIZER_SYSTEM = (
    "You compress an AI coding agent's working context into a compact memory "
    "entry that replaces the older messages. Output ONLY a <memory>...</memory> "
    "block, no preamble."
)
SUMMARIZER_INSTRUCTION = """\
Compress the conversation above into a working-memory entry that will replace
those messages in the agent's context.

<memory>
## Knowledge state
- Files/functions inspected and what was found (paths + line refs, verbatim).
- The bug's root cause as currently understood, with the evidence.
- Edits already made (file + what changed) and their test results.
- Approaches already ruled out, and why.
- Open sub-questions / what is still unverified.

## Thoughts
- The fix direction converged on and why.
- The concrete next step (specific file to edit / test to run) — not "continue".
</memory>

Preserve identifiers verbatim: file paths, function/class names, test IDs, error
messages, line numbers. Keep it under 4096 tokens. Your reply must start with
`<memory>` and end with `</memory>`."""

RETRIEVAL_SYSTEM = (
    "You retrieve detail from an AI coding agent's archived (compressed) messages. "
    "Answer ONLY from the archived text provided. Quote identifiers verbatim: file "
    "paths, function and class names, test ids, error messages, line numbers. Be "
    "concise — your reply is injected back into a context that is already tight. If "
    "the archive does not contain the answer, say exactly that."
)

# ── ACM system-prompt addendum (A3 only; appended by the driver) ──
ACM_SYSTEM_ADDENDUM = """

## Managing your context memory
Your in-context conversation is short-term memory; compressed segments live in
long-term memory on disk. After each tool result the system appends
"[CURRENT CONTEXT TOKEN: N]" — your current context usage. Read it; never write
it yourself.

You have two extra tools beyond `bash`:
- The `manage_context` tool takes no arguments. When you call it, the system
  compresses everything in your conversation since your previous manage_context
  call (or since the start of the task if this is your first call) up to — but
  not including — the message that issued the call. The system prompt and the
  original task are always preserved. A summary tagged "[summary_id: N]" is
  returned and the original messages are saved to disk. This does NOT end the
  task — keep working after it.
- Use the `query_memory` tool to retrieve detailed information from any prior
  summary's original messages by referencing its summary_id, together with a
  query describing exactly what you need.

Guidance: when [CURRENT CONTEXT TOKEN: N] climbs and your recent turns contain
dead ends or duplicated exploration, call manage_context to free space. Make sure
your work so far is captured before you do. Do not compress away the exact edit
you are mid-way through."""


# ── Forced-compression nudge (A3), adapted from ACM's BCP_COMPRESS_PROMPT ──
# ACM injects this at 95% of the context window (runner.py:930) so the agent
# doesn't just die when it fails to compress voluntarily. It's still the *model*
# that executes the compression — we only nudge; we never compress for it in A3.
# Mirrors ACM's BCP_COMPRESS_PROMPT, which frames this as a TOOL CALL
# ("You MUST immediately call manage_context()"), not a shell command.
FORCE_COMPRESS_PROMPT = (
    "WARNING: Your context is nearly full, but your task is not yet complete.\n\n"
    "You MUST immediately call manage_context() to compress everything since your "
    "previous manage_context call into a single summary, freeing space. Do NOT "
    "submit yet, and do NOT run any other tool first. Call manage_context now, "
    "then continue working."
)

# ACM caps how many times it will re-ask for a compression, and resets that
# counter once the context drops back under 90% (runner.py: the
# `compress_prompt_count < MAX_COMPRESS_RETRIES` guard and the
# `if token_count < int(context_window * 0.90): compress_prompt_count = 0` reset).
MAX_COMPRESS_RETRIES = 3
RESET_FRAC = 0.90
# Tokens of raw-count headroom reserved for the STOP_PROMPT exchange itself (the
# prompt plus the model's final submit command). The retry budget must not eat
# this: a nudge the model ignores still costs a full turn, so without a
# budget-based escalation the context blows past the hard limit while we are
# still politely asking it to compress (observed: MiMo took 2 nudges, compressed
# nothing, and died before the 3rd nudge let STOP_PROMPT fire).
# Headroom budgets, in ABSOLUTE tokens measured back from the hard prompt wall.
#
# ACM triggers at 0.95*context_window, which assumes a large window: at their
# ~128k that leaves ~6.4k tokens between "start nudging" and "rejected". At the
# 24-28k budgets this study imposes, 5% is only ~1.2-1.4k — less than one tool
# observation — so the nudge and the stop escalation collide and neither gets to
# work (observed: all three models made ZERO edits, and Gemma skipped straight to
# STOP without ever nudging). We therefore keep ACM's rule but take whichever
# fires FIRST, so the mechanism retains the same absolute room ACM enjoys:
#   nudge  when within NUDGE_RESERVE of the wall (or at ACM's 0.95, if earlier)
#   STOP   when within STOP_RESERVE  of the wall
# This is a documented deviation forced by the small-window regime, not a change
# to ACM's semantics.
NUDGE_RESERVE = 3000
STOP_RESERVE = 1000

# ACM's escalation after the compress nudges are exhausted: rather than letting
# the episode die on overflow, it injects a STOP_PROMPT, takes ONE more response
# and extracts a final answer from it. SWE-bench adaptation: the "final answer"
# is the submission, which mini-swe-agent detects via this sentinel in a bash
# command, so we spell the command out verbatim.
STOP_PROMPT = (
    "STOP. Your context is full and you cannot continue investigating. Do NOT run "
    "any more exploration commands. Submit whatever work you have RIGHT NOW as your "
    "final answer, by running exactly this as the entire command:\n"
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cd /testbed && git diff\n"
    "If you have made no edits yet, run it anyway so an empty patch is recorded."
)


# ── ACM's two memory tools, registered as REAL function-calling tools ──
# Schemas mirror src/tools.py in the ACM repo: manage_context is parameterless,
# query_memory takes (summary_id, query) and SEARCHES the archived originals.
# Registering them matters: mini-swe-agent exposes only a bash tool, so our
# earlier build made the model type "manage_context" as a shell command — an
# off-distribution affordance that models post-trained to call tools largely
# ignored (Qwen and MiMo refused 3 explicit nudges each; Gemma complied once).
MANAGE_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "manage_context",
        "description": (
            "Compress your working memory. Call this when context is filling up with dead "
            "ends, duplicates, or detail you no longer need verbatim. The system "
            "automatically selects the compression range from your last manage_context call "
            "(or the start of the investigation) up to the current message, preserving the "
            "system prompt and the original question. A summary prefixed with "
            "\"[summary_id: N]\" is returned and the original messages are saved to disk."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
QUERY_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": (
            "Retrieve detailed information from a previously compressed summary. Each "
            "manage_context call returns a summary prefixed with \"[summary_id: N]\"; pass "
            "that N here together with what you need, and the original (uncompressed) "
            "messages behind it will be searched."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary_id": {
                    "type": "integer",
                    "description": (
                        "The summary_id (as shown in the [summary_id: N] prefix of a prior "
                        "manage_context summary) whose original content should be searched."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "What specific information to extract from the original messages of "
                        "that summary."
                    ),
                },
            },
            "required": ["summary_id", "query"],
        },
    },
}
MEM_TOOLS = [MANAGE_CONTEXT_TOOL, QUERY_MEMORY_TOOL]
MEM_TOOL_NAMES = {"manage_context", "query_memory"}


# ══════════════════════════════════════════════════════════════════════════
# Harness v2 (Nov 18 2026): faithful to the ACM repo @ f06f90e and the paper's
# App. B. See docs/acm_vs_ours.md for every difference and why. v1 objects
# above are kept unchanged so the scored v1 runs stay reproducible.
# ══════════════════════════════════════════════════════════════════════════

# _CONTEXT_WINDOW_LINE (prompts.py), given to EVERY arm, ReAct baseline included.
CONTEXT_WINDOW_LINE_V2 = (
    "Your context window is {context_window} tokens. Plan your work accordingly — once context "
    "is full the conversation ends immediately. You must submit your fix before reaching the "
    "context limit."
)

# _HEADER_WITH_MEMORY (prompts.py), task wording changed from search to code.
ACM_HEADER_V2 = """\
You need to solve the task by interacting with the repository through the shell and managing your context memory. Your in-context information serves as short-term memory; previously compressed segments live in long-term memory.

After each tool result, the system appends "[CURRENT CONTEXT TOKEN: N]" at the end of the tool response — this tells you your current context usage. Read it; do NOT emit it yourself.

The "manage_context" tool takes no arguments. When you call it, the system compresses everything in your conversation since your previous manage_context call (or since the start of the task if this is your first call) up to (but not including) the message that issued the call. The system prompt and the original task are always preserved. The original messages in the compressed range are saved to disk; the tool returns a summary of paths explored, reasoning, and conclusions, prefixed with "[summary_id: N]".

Use the "query_memory" tool to retrieve detailed information from any prior summary's original messages by referencing the summary_id."""

# _MEMORY_MANAGEMENT_STRATEGY (prompts.py), 4 bullets, search -> code wording.
ACM_STRATEGY_V2 = """\
Strategy:
- If you are not confident in your diagnosis or fix, investigate more — read the relevant code and run tests to corroborate or contradict your hypothesis. Do not submit a low-confidence fix when more investigation is still cheap.
- Watch the [CURRENT CONTEXT TOKEN: N] marker at the end of each tool result. When it climbs and your recent steps contain dead ends or duplicates, call "manage_context" to compress them into a [summary_id: N] entry, freeing space. Do not write this marker in your own responses; it is an environmental signal you consume, not produce.
- If a prior summary looks relevant to a new direction, call the "query_memory" tool to pull detailed content out of that summary's original messages.
- Calling the "manage_context" tool or the "query_memory" tool does not end the task. After these tools return, continue working — they exist to make room for and surface more evidence, not to wrap up. Only submit when you are confident in your fix."""


def build_system_addendum(memory_policy: str, harness: str, context_cap: int) -> str:
    """Text appended to mini-swe-agent's system template. v1: ACM addendum on
    A3 only. v2: ACM's layout — memory header (A3), context-window line (all
    arms), strategy block (A3)."""
    is_acm = memory_policy in ("acm", "acm_nudge")
    if harness != "v2":
        return ACM_SYSTEM_ADDENDUM if is_acm else ""
    parts = []
    if is_acm:
        parts.append(ACM_HEADER_V2)
    parts.append(CONTEXT_WINDOW_LINE_V2.format(context_window=context_cap))
    if is_acm:
        parts.append(ACM_STRATEGY_V2)
    return "\n\n" + "\n\n".join(parts)


# _SUMMARY_INSTRUCTION / _SUMMARY_INSTRUCTION_NOQUERY (prompts.py), adapted to
# code. Sent as ONE user message, no system prompt, like call_summarizer.
_SUMMARY_INSTRUCTION_V2 = """\
Original task: {question}

Conversation to compress:

{conversation}

Compress the conversation above into a working-memory entry. This entry
replaces the archived messages in the agent's working context{query_clause}.

Coverage:
- Knowledge state — facts established about the codebase and the bug, with
  file paths and line numbers; candidate root causes and fixes with the
  evidence supporting or contradicting each; edits already made and their
  test results; hypotheses already ruled out and what eliminated them;
  remaining open sub-questions.
- Thoughts — a concise distillation of the most recent reasoning chain
  from the latest assistant turns (what direction the agent has converged
  on and why), AND the concrete next step the agent should take after
  this compression (specific file to read or edit / test to run{query_step} —
  not a generic "continue investigating").

Output exactly one <memory>...</memory> block with the two sections in
this order:

<memory>
## Knowledge state
<facts, candidates, eliminated hypotheses, open sub-questions, in short prose>

## Thoughts
<a few sentences distilling the latest reasoning thread, followed by 1-2
sentences naming the concrete next step>
</memory>

- ≤ 4096 tokens total inside the block, so try to be concise but still cover all the important information.
- Preserve identifiers verbatim — file paths, function and class names, line numbers, test ids, error messages.
- Do not enumerate every command you ran.
- After closing the chat-template-opened `</think>`, your very next token
  must be `<memory>`. Do NOT emit any preamble between `</think>` and
  `<memory>` — no `Thinking Process:` heading, no numbered planning list,
  no `Analysis:` / `Plan:` / `Let me ...` lead-in, no re-stated rules,
  no re-quoted task. Plan inside `<think>` if you need to plan;
  the visible output starts directly with `<memory>` and ends with
  `</memory>`."""

# QUERY_MEMORY_PROMPT (prompts.py), identifiers switched to code ones.
QUERY_MEMORY_PROMPT_V2 = """\
Saved messages under summary_id={summary_id}:

{history}

Recall request — extract content relevant to: {query}

Output format (strict):
- Put any internal reasoning inside <think>...</think> — these will be stripped.
- After </think>, write the recall as compact bullets only. No prose preamble. No restatement of the query. No address to the reader (no "the user", "the agent", "you").
- Each bullet must carry concrete identifiers (file paths, line numbers, function and class names, test ids, error messages) verbatim — never paraphrase code or error text.
- Group into up to three sections (omit any that are empty):
  - **Relevant findings:** facts that bear on the query, with supporting file paths / line numbers.
  - **Dead ends:** commands / files / hypotheses tried that produced nothing.
  - **Open / unresolved:** most promising direction still to verify.
- If nothing in the saved messages is relevant, output exactly: `(nothing relevant under summary_id={summary_id})`."""

# STOP_PROMPT (prompts.py) adapted: SWE's "final answer" is the patch.
STOP_PROMPT_V2 = (
    "The task must now conclude. You have reached the context window limit.\n\n"
    "IMPORTANT: Do NOT run any more exploration or test commands. Submit your current work "
    "RIGHT NOW by running exactly this as the entire command:\n"
    "cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff\n"
    "If you have made no edits, run it anyway so an empty patch is recorded."
)
# If the model ignores the STOP, the harness submits the tracked-file diff itself
# (ACM ends the episode after one post-STOP generation regardless).
FORCED_SUBMIT_CMD = "cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"

# Tool schemas exactly as the paper prints them (App. B.2), "investigation" -> "task".
MANAGE_CONTEXT_TOOL_V2 = {
    "type": "function",
    "function": {
        "name": "manage_context",
        "description": (
            "Compress your working memory. Call this when context is filling up with dead ends, "
            "duplicates, or detail you no longer need verbatim. The system automatically picks the "
            "range to compress: everything since your last manage_context call (or since the start "
            "of the task if this is your first call) up to (but not including) the message that "
            "issued this tool call. The system prompt and the original task are always preserved. "
            "The original messages in the compressed range are saved to disk as "
            "summary_{summary_id}.json (retrievable later with query_memory(summary_id, query)), "
            "and a fresh summary --- focused on paths explored, reasoning, and conclusions --- is "
            "returned as the tool result. The summary text is prefixed with \"[summary_id: N]\" so "
            "you can refer to it directly in subsequent reasoning and pull the raw content back via "
            "query_memory(summary_id=N, ...). This tool takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
QUERY_MEMORY_TOOL_V2 = {
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": (
            "Retrieve detailed information from a previously compressed summary. Each "
            "manage_context call returns a summary prefixed with [summary_id: N] and saves the "
            "original messages to disk as summary_{N}.json. This tool loads "
            "summary_{summary_id}.json and uses an LLM to extract information matching your query "
            "from the original (uncompressed) content. DO NOT CALL THIS TOOL until at least one "
            "manage_context call has produced a summary_id."
        ),
        "parameters": QUERY_MEMORY_TOOL["function"]["parameters"],
    },
}
MEM_TOOLS_V2 = [MANAGE_CONTEXT_TOOL_V2, QUERY_MEMORY_TOOL_V2]

_MEMORY_RE = re.compile(r"<memory>(.*?)</memory>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_memory(text: str) -> tuple[str, bool]:
    """runner.py _extract_memory: inner <memory> text, parse_ok iff a bounded
    block of >=50 chars with no nested <memory> / </think>."""
    if not text:
        return "", False
    m = _MEMORY_RE.search(text)
    if m:
        inner = m.group(1).strip()
        if len(inner) >= 50 and "<memory>" not in inner and "</think>" not in inner:
            return inner, True
    return text.strip(), False


def _install_memory_tools(model, mem_tools=None):
    """Rebind `model` to a subclass that advertises the two memory tools and
    accepts their tool calls. mini-swe-agent hardcodes tools=[BASH_TOOL] in
    _query and parse_toolcall_actions raises FormatError on any other name, so
    both have to be overridden; bash keeps going through the stock parser so its
    error semantics are unchanged."""
    from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions

    tools = [BASH_TOOL, *(mem_tools or MEM_TOOLS)]
    base = type(model)
    if getattr(base, "_acm_memory_tools", False):
        return model

    class _WithMemoryTools(base):
        _acm_memory_tools = True

        def _query(self, messages, **kwargs):
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                **(self.config.model_kwargs | kwargs),
            )

        def _parse_actions(self, response) -> list[dict]:
            tool_calls = response.choices[0].message.tool_calls or []
            tmpl_kwargs = {"finish_reason": response.choices[0].finish_reason}
            if not tool_calls:  # delegate so the canonical FormatError is raised
                return parse_toolcall_actions(
                    [], format_error_template=self.config.format_error_template,
                    template_kwargs=tmpl_kwargs)
            actions = []
            for tc in tool_calls:  # preserve the model's ordering
                if tc.function.name in MEM_TOOL_NAMES:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    actions.append({"command": tc.function.name, "tool_call_id": tc.id,
                                    "mem_tool": tc.function.name, "mem_args": args})
                else:
                    actions += parse_toolcall_actions(
                        [tc], format_error_template=self.config.format_error_template,
                        template_kwargs=tmpl_kwargs)
            return actions

    model.__class__ = _WithMemoryTools
    return model


def _memory_marker(n: int) -> str:
    return f"\n[CURRENT CONTEXT TOKEN: {n}]"


class MemoryAgent(DefaultAgent):
    def __init__(
        self,
        model,
        env,
        *,
        memory_policy: str = "none",
        context_cap: int = 32000,
        summarize_at: int = 28000,
        keep_last_k: int = 3,
        force_frac: float = 0.95,   # ACM's threshold (runner.py:930)
        harness: str = "v1",        # v1 = every run up to Nov 2026; v2 = docs/acm_vs_ours.md
        marker_all_arms: bool = False,  # v2: also show A1/A2 the token marker (ACM's ReAct gets it)
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)  # memory_* are captured here, not passed to AgentConfig
        # acm_nudge = ACM + the scaling-mode compress nudge (what v1 called "acm").
        self.is_acm = memory_policy in ("acm", "acm_nudge")
        self.harness = harness
        self.v2 = harness == "v2"
        self.nudge = memory_policy == "acm_nudge" or (memory_policy == "acm" and not self.v2)
        self.marker_on = self.is_acm or (self.v2 and marker_all_arms)
        self._mem_tools = MEM_TOOLS_V2 if self.v2 else MEM_TOOLS
        self.memory_policy = memory_policy
        if self.is_acm:
            _install_memory_tools(model, self._mem_tools)
        self.context_cap = context_cap
        self.summarize_at = summarize_at
        self.keep_last_k = keep_last_k
        self.force_frac = force_frac  # A3 forced-nudge threshold as a fraction of the cap
        self._task = ""
        # Count tokens with the SAME tokenizer vLLM uses to enforce the cap
        # (litellm.token_counter undercounts it by ~15%, which made A3's forced
        # nudge unreachable). Fall back to litellm -> chars/4 if the endpoint is
        # unavailable.
        api_base = (getattr(model.config, "model_kwargs", {}) or {}).get("api_base", "")
        self._tokenize_url = (api_base.rstrip("/").removesuffix("/v1") + "/tokenize") if api_base else None
        mn = model.config.model_name
        self._served_model = mn.split("/", 1)[1] if mn.startswith("hosted_vllm/") else mn
        # ACM (runner.py:850) projects the NEXT prompt size as count + the output
        # reservation vLLM sets aside, so the trigger compares like-for-like
        # against the cap. Without this, a threshold on the raw count fires only
        # AFTER vLLM has already rejected the prompt (cap - max_tokens).
        self._max_tokens_reserve = int((getattr(model.config, "model_kwargs", {}) or {}).get("max_tokens") or 4096)
        self.mem_steps: list[dict] = []   # per-turn: {step, ctx_tokens, n_messages, edit}
        self.archive: list[dict] = []     # [{summary_id, messages:[...]}]
        self.n_edits = 0
        self.n_retrievals = 0
        self._t0 = time.time()
        self._tool_calls = Counter()      # bash / manage_context / query_memory
        self._usage = []                  # (prompt_tokens, completion_tokens) per agent call
        self._last_usage = None
        self._obs_tokens = []             # size of every observation fed back
        self._mem_op_usage = [0, 0]       # tokens the summarizer + retrieval calls cost
        self._mem_op_calls = 0
        self._pending_edit: dict | None = None
        self._last_mc_end = 2             # A3: start of the next compressible range
        self._calibrated = False          # log token-count vs vLLM prompt_tokens once
        self._compress_prompt_count = 0   # A3: ACM's bounded compress-nudge retries
        self._compress_prompts_total = 0  # cumulative (the above resets under 90%)
        self._stop_prompted = False       # A3: STOP_PROMPT is injected at most once
        self._stop_call = None            # v2: n_calls of the one post-STOP generation
        self._boundary_after_obs = False  # v2: move the mc boundary after the summary tool msg
        self._summary_parse_fail = 0      # v2: summaries that fell back to raw text
        self._forced_submit = False       # v2: STOP ignored, harness submitted the diff
        # Append-only mirror of every message ever added, never compressed — ACM's
        # HistoryManager.raw_messages ("for traj dumps"). self.messages loses the
        # compressed spans, so without this the full trajectory is unrecoverable.
        self.raw_messages: list[dict] = []

    def add_messages(self, *messages):
        """Mirror into the append-only raw history before the live list can be
        compressed. Everything mini-swe-agent adds (system, task, assistant turns,
        observations) flows through here."""
        for m in messages:
            try:
                self.raw_messages.append(copy.deepcopy(m))
            except Exception:
                self.raw_messages.append(dict(m) if isinstance(m, dict) else {"raw": str(m)})
        return super().add_messages(*messages)

    # ── token accounting ────────────────────────────────────────────────
    # ACM's finding (client.py LocalClient.count_tokens): count with the MODEL's
    # own tokenizer via apply_chat_template — litellm.token_counter uses a generic
    # tokenizer and undercounts Qwen by ~15%, which is what broke A3. Cascade:
    #   1) local Qwen tokenizer (apply_chat_template)  — exact, matches vLLM
    #   2) the serving vLLM's /tokenize endpoint        — exact, no local deps
    #   3) litellm.token_counter                         — undercounts
    #   4) chars/4
    def _get_tokenizer(self):
        if getattr(self, "_tok", "unset") == "unset":
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self._served_model)
            except Exception as e:
                self.logger.warning(f"local tokenizer unavailable ({e}); using /tokenize or litellm")
                self._tok = None
        return self._tok

    def _count_tokens(self, messages: list[dict]) -> int:
        # Count what the model ACTUALLY sends: full messages (minus our internal
        # "extra" field) INCLUDING tool_calls, plus the bash tool schema. Counting
        # only {role, content} strips every assistant turn's tool_call and omits
        # the tools, undercounting vLLM by ~5k — which silently defeats the
        # thresholds (compression never fires before vLLM rejects the prompt).
        api_msgs = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        tok = self._get_tokenizer()
        if tok is not None:
            try:
                # Qwen's template iterates tool_call.arguments as a mapping; vLLM
                # json-decodes the string before rendering, so do the same here or
                # every history with a tool call falls through to /tokenize.
                tpl_msgs = []
                for m in api_msgs:
                    if m.get("tool_calls"):
                        m = dict(m)
                        tcs = []
                        for tc in m["tool_calls"]:
                            tc = json.loads(json.dumps(tc, default=str))
                            fn = tc.get("function") or {}
                            if isinstance(fn.get("arguments"), str):
                                try:
                                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                                except Exception:
                                    pass
                            tcs.append(tc)
                        m["tool_calls"] = tcs
                    tpl_msgs.append(m)
                ids = tok.apply_chat_template(tpl_msgs, tools=self._tools_for_counting(),
                                              add_generation_prompt=True, tokenize=True)
                # transformers>=5 returns a BatchEncoding {input_ids, attention_mask};
                # len() of it is 2, which silently disabled every threshold.
                if hasattr(ids, "keys") and "input_ids" in ids:
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                return len(ids)
            except Exception:
                pass
        if self._tokenize_url:
            try:
                r = requests.post(
                    self._tokenize_url,
                    json={"model": self._served_model, "messages": api_msgs,
                          "tools": self._tools_for_counting(), "add_generation_prompt": True},
                    timeout=20,
                )
                if r.ok:
                    return int(r.json()["count"])
            except Exception:
                pass
        try:
            return int(litellm.token_counter(model=self.model.config.model_name, messages=api_msgs))
        except Exception:
            return sum(len(str(m.get("content", ""))) for m in messages) // 4

    def _tools_for_counting(self) -> list[dict]:
        """The tool schemas count toward the prompt, so A3 must include its two
        extra tools or every threshold is computed against the wrong budget."""
        return [BASH_TOOL, *self._mem_tools] if self.is_acm else [BASH_TOOL]

    def _projected(self, messages: list[dict]) -> int:
        """count + output reservation = a conservative upper bound on the next
        prompt size, comparable directly against the cap (ACM's N)."""
        return self._count_tokens(messages) + self._max_tokens_reserve

    def _turn_start_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m.get("role") == "assistant"]

    # ── summarization (shared by A2 and A3) ─────────────────────────────
    # ── v2 span formatting: what the summarizer / querier reads ──────────
    @staticmethod
    def _render_msg(m: dict, with_reasoning: bool = True, max_obs_chars: int | None = None) -> str:
        """[role] + reasoning + content + tool calls. v1 (and ACM) send only
        `content`, which is None on every tool-call turn under our reasoning
        parser, so the summarizer saw neither the agent's thinking nor its
        commands."""
        role = m.get("role", "?")
        parts = []
        if role == "assistant":
            rc = (m.get("reasoning_content") or "").strip()
            if with_reasoning and rc:
                parts.append(f"<think>\n{rc}\n</think>")
            c = m.get("content")
            if c not in (None, "", "None"):
                parts.append(str(c).strip())
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name, args = fn.get("name", "?"), fn.get("arguments") or "{}"
                try:
                    a = json.loads(args) if isinstance(args, str) else (args or {})
                except Exception:
                    a = {"raw": str(args)}
                if name == "bash" and isinstance(a, dict) and "command" in a:
                    parts.append(f"$ {a['command']}")
                else:
                    parts.append(f"→ {name}({json.dumps(a)})")
        else:
            c = "" if m.get("content") is None else str(m.get("content"))
            if max_obs_chars and len(c) > max_obs_chars:
                h = max_obs_chars // 2
                c = c[:h] + f"\n…[{len(c) - max_obs_chars} chars elided]…\n" + c[-h:]
            parts.append(c)
        return f"[{role}]\n" + "\n".join(p for p in parts if p)

    def _span_text(self, msgs: list[dict], budget_tokens: int) -> str:
        """Render a span within a token budget so the summarizer prompt never
        exceeds the vLLM cap. Degrades in steps: full → reasoning only on the
        last 5 assistant turns → observations middle-truncated to 2k chars →
        keep the most recent text."""
        tok = self._get_tokenizer()
        n_tok = (lambda s: len(tok.encode(s))) if tok is not None else (lambda s: len(s) // 3)
        asst_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        recent = set(asst_idx[-5:])
        attempts = [
            lambda i, m: self._render_msg(m),
            lambda i, m: self._render_msg(m, with_reasoning=i in recent),
            lambda i, m: self._render_msg(m, with_reasoning=i in recent, max_obs_chars=2000),
        ]
        text = ""
        for render in attempts:
            text = "\n\n".join(render(i, m) for i, m in enumerate(msgs))
            if n_tok(text) <= budget_tokens:
                return text
        # last resort: keep the tail (most recent work matters most for "Thoughts")
        keep = budget_tokens * 3
        return "…[earlier messages elided]…\n" + text[-keep:]

    def _summarizer_budget(self) -> int:
        """Prompt room left for the conversation text: cap − output reserve −
        task − instruction − margin."""
        task_tok = len(self._task) // 3
        return max(4000, self.context_cap - self._max_tokens_reserve - task_tok - 1500)

    def _mem_llm(self, prompt: str) -> str:
        """One plain (tool-free) completion, as ACM's summarizer / querier:
        a single user message, same model and sampling as the agent."""
        kwargs = dict(self.model.config.model_kwargs)
        kwargs.pop("parallel_tool_calls", None)
        resp = litellm.completion(model=self.model.config.model_name,
                                  messages=[{"role": "user", "content": prompt}], **kwargs)
        self._note_mem_op(resp)
        return _THINK_RE.sub("", resp.choices[0].message.content or "").strip()

    def _summarize_v2(self, span_messages: list[dict]) -> str:
        """call_summarizer: ACM instruction, <memory> extraction, one retry.
        Returns the INNER memory text."""
        conv = self._span_text(span_messages, self._summarizer_budget())
        prompt = _SUMMARY_INSTRUCTION_V2.format(
            question=self._task, conversation=conv,
            query_clause="; future calls to query_memory(summary_id, query) will search this text"
                         if self.is_acm else "",
            query_step=" / memory query" if self.is_acm else "")
        last = ""
        for attempt in range(2):
            try:
                inner, ok = _extract_memory(self._mem_llm(prompt))
            except Exception as e:
                self.logger.warning(f"summarizer call failed (attempt {attempt}): {e}")
                inner, ok = f"(summary unavailable: {e})", False
            last = inner
            if ok:
                return inner
        self._summary_parse_fail += 1
        return last

    def _summarize(self, span_messages: list[dict]) -> str:
        if self.v2:
            return f"<memory>\n{self._summarize_v2(span_messages)}\n</memory>"
        convo = "\n\n".join(f"[{m.get('role')}]\n{str(m.get('content',''))}" for m in span_messages)
        user = f"Original task:\n{self._task}\n\nConversation to compress:\n\n{convo}\n\n{SUMMARIZER_INSTRUCTION}"
        kwargs = dict(self.model.config.model_kwargs)  # api_base, temperature, drop_params, etc.
        kwargs.pop("parallel_tool_calls", None)
        try:
            resp = litellm.completion(
                model=self.model.config.model_name,
                messages=[{"role": "system", "content": SUMMARIZER_SYSTEM}, {"role": "user", "content": user}],
                **kwargs,  # NB: no tools — plain completion
            )
            self._note_mem_op(resp)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            self.logger.warning(f"summarizer call failed: {e}")
            return f"<memory>\n(summary unavailable: {e})\n</memory>"

    def _compress_range(self, start: int, end: int, trigger: str = "harness") -> dict | None:
        """Replace messages[start:end] with one summary message; archive originals.
        `end` must sit on a turn boundary (an assistant index) so no tool_call/tool
        pair is split. `trigger` records WHY this edit happened (harness / voluntary
        / forced) — the key metric for the untrained-ACM question. Returns an edit
        record, or None if nothing to compress."""
        if end <= start:
            return None
        original = self.messages[start:end]
        before = self._count_tokens(self.messages)
        summary_text = self._summarize(original)
        sid = self.n_edits + 1
        summary_msg = {
            "role": "user",
            "content": f"[summary_id: {sid}] {len(original)} earlier messages were compressed to free context:\n{summary_text}",
            "extra": {"is_summary": True, "summary_id": sid, "span": [start, end], "n_compressed": len(original)},
        }
        self.messages[start:end] = [summary_msg]
        after = self._count_tokens(self.messages)
        self.archive.append({"summary_id": sid, "messages": original})
        self.n_edits += 1
        edit = {
            "type": "summarize", "trigger": trigger, "span": [start, end],
            "tokens_before": before, "tokens_after": after,
            "summary_tokens": self._count_tokens([summary_msg]), "summary_id": sid, "n_compressed": len(original),
        }
        summary_msg["extra"]["tokens_before"] = before
        summary_msg["extra"]["summary_tokens"] = edit["summary_tokens"]
        self._pending_edit = edit
        # after a compression the whole tail before the summary is gone; the next
        # compressible range for A3 starts right after the summary
        self._last_mc_end = start + 1
        return edit

    def _compressible_tail_boundary(self, since: int) -> int:
        """Turn-aligned boundary: compress messages[since:boundary], keep the tail.
        Keeps at most keep_last_k recent turns, and further caps the protected
        tail to ~0.55*cap so anchors + summary + tail fit under the context limit
        even when recent observations are large (the bug that made A2 die at a
        tight cap despite compressing)."""
        starts = [i for i in self._turn_start_indices() if i >= since]
        if len(starts) <= 1:
            return -1
        # Reserve room for what we DON'T compress: the anchors (system+task, kept
        # at [:2]), the summary we're about to write (<=4096), and the model's
        # next response (~2000). The protected tail must fit in what's left.
        anchors = self._count_tokens(self.messages[:2])
        budget = max(2000, self.context_cap - anchors - 6000)
        boundary = starts[-1]  # keep at least the most recent turn
        for s in reversed(starts):
            if self._count_tokens(self.messages[s:]) > budget:
                break
            boundary = s
        if len(starts) >= self.keep_last_k:  # never keep MORE than keep_last_k turns
            boundary = max(boundary, starts[-self.keep_last_k])
        return boundary if boundary > since else -1

    # ── pre-call hooks: A2 harness compression, A3 forced-compression nudge ──
    def _inject_stop(self, why: str) -> None:
        """runner.py's STOP branch: one STOP message, one more generation, end."""
        if self._stop_prompted:
            return
        self.logger.warning(f"[{self.harness}] {why} — escalating to STOP_PROMPT")
        self.add_messages({"role": "user", "content": STOP_PROMPT_V2 if self.v2 else STOP_PROMPT,
                           "extra": {"stop_prompt": True}})
        self._stop_prompted = True
        self._stop_call = self.n_calls + 1   # the generation that answers the STOP

    def _force_submit(self):
        """The post-STOP generation did not submit: submit the tracked-file diff
        ourselves. ACM ends the episode after that one generation either way."""
        from minisweagent.exceptions import Submitted
        self.logger.warning(f"[{self.harness}] STOP ignored — harness submits git diff")
        self._forced_submit = True
        self.env.execute({"command": FORCED_SUBMIT_CMD})  # raises Submitted on success
        raise Submitted({"role": "exit", "content": "",
                         "extra": {"exit_status": "Submitted", "submission": ""}})

    def query(self) -> dict:
        if self._stop_call is not None and self.n_calls >= self._stop_call:
            self._force_submit()
        if self.memory_policy == "summarize" and self._projected(self.messages) >= self.summarize_at:
            end = self._compressible_tail_boundary(since=2)
            if end > 2:
                self._compress_range(2, end, trigger="harness")
            # v2 backstop: compression could not get back under the wall
            if self.v2 and self._projected(self.messages) >= self.context_cap - STOP_RESERVE:
                self._inject_stop("still over the wall after compression")
        elif self.v2 and not self.nudge:
            # A1 and ACM-faithful A3: STOP at 95%, no compress nudge (runner.py's
            # default path; the nudge exists only in min_turns / continue modes)
            if self._projected(self.messages) >= self.summarize_at:
                self._inject_stop("context at 95%")
        elif self.is_acm:
            # ACM's per-turn context check (runner.py), in its order:
            #   1. reset the compress-attempt counter once we drop back under 90%
            #   2. at >=force_frac, nudge the model to compress — but only while
            #      attempts remain (MAX_COMPRESS_RETRIES)
            #   3. once they're exhausted, escalate to STOP_PROMPT so the episode
            #      ends with a submission instead of dying on overflow
            # _projected already adds the output reservation, so these compare a
            # true upper bound on the next prompt against the cap.
            proj = self._projected(self.messages)
            count = proj - self._max_tokens_reserve
            # the real wall: vLLM rejects any prompt above this raw count
            hard_limit = self.context_cap - self._max_tokens_reserve
            if proj < int(self.context_cap * RESET_FRAC):
                self._compress_prompt_count = 0
            nudge_at = min(int(self.context_cap * self.force_frac) - self._max_tokens_reserve,
                           hard_limit - NUDGE_RESERVE)
            if count >= nudge_at:
                last_extra = self.messages[-1].get("extra", {}) if self.messages else {}
                # Escalate on exhausted retries or exhausted headroom, whichever
                # comes first — but never before the model has had at least one
                # chance to compress, or the mechanism is never exercised at all.
                out_of_room = (count >= hard_limit - STOP_RESERVE
                               and self._compress_prompt_count > 0)
                if self._compress_prompt_count < MAX_COMPRESS_RETRIES and not out_of_room:
                    if not last_extra.get("force_compress"):
                        self.add_messages({"role": "user", "content": FORCE_COMPRESS_PROMPT,
                                           "extra": {"force_compress": True}})
                        self._compress_prompt_count += 1
                        self._compress_prompts_total += 1
                elif not self._stop_prompted:
                    why = "out of headroom" if out_of_room else "nudges exhausted"
                    self._inject_stop(f"{why} (nudges={self._compress_prompt_count}, count={count}, "
                                      f"hard_limit={hard_limit})")
                    if not self.v2:
                        self._stop_call = None  # v1 had no forced submit
        # one-time calibration: does our token count match vLLM's real prompt?
        pre_count = self._count_tokens(self.messages) if not self._calibrated else None
        # v2 marker: history.get_messages_for_model(token_hint=N) — appended to
        # the LATEST tool message of the outgoing copy only, never stored
        marked, orig = None, None
        if self.v2 and self.marker_on:
            proj = self._projected(self.messages)
            marked = next((m for m in reversed(self.messages) if m.get("role") == "tool"), None)
            if marked is not None and isinstance(marked.get("content"), str):
                orig = marked["content"]
                marked["content"] = orig + f"\n\n[CURRENT CONTEXT TOKEN: {proj}]"
            else:
                marked = None
        try:
            msg = super().query()
        finally:
            if marked is not None:
                marked["content"] = orig
        usage = ((msg.get("extra", {}).get("response") or {}).get("usage") or {})
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if pt is not None:
            self._usage.append((pt, ct or 0))
            self._last_usage = (pt, ct or 0)
        if pre_count is not None:
            actual = ((msg.get("extra", {}).get("response") or {}).get("usage") or {}).get("prompt_tokens")
            if actual:
                ratio = actual / max(1, pre_count)
                self.logger.warning(f"[token-calib] my_count={pre_count} vLLM_prompt_tokens={actual} "
                                    f"ratio={ratio:.3f}")
                if not 0.9 <= ratio <= 1.1:
                    # every threshold (compression, nudge, STOP) is computed from
                    # this count; a wrong counter silently disables them all
                    self.logger.error(f"[token-calib] TOKEN COUNT MISMATCH ratio={ratio:.3f}: "
                                      f"thresholds will fire at the wrong context size")
            self._calibrated = True
        return msg

    # ── A3 hook: intercept manage_context / query_memory bash commands ───
    def execute_actions(self, message: dict) -> list[dict]:
        acts = message.get("extra", {}).get("actions", []) or []
        for a in acts:
            self._tool_calls[a.get("mem_tool") or
                             ("manage_context" if (a.get("command") or "").strip() == "manage_context"
                              else "query_memory" if (a.get("command") or "").startswith("query_memory")
                              else "bash")] += 1
        if not self.is_acm:
            out = super().execute_actions(message)
            self._record_obs(out)
            return out
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            cmd = (action.get("command") or "").strip()
            tool = action.get("mem_tool")          # set when it arrived as a real tool call
            args = action.get("mem_args") or {}
            if tool == "manage_context" or cmd == "manage_context":
                outputs.append(self._handle_manage_context_v2() if self.v2 else self._handle_manage_context())
            elif tool == "query_memory" or cmd.startswith("query_memory"):
                outputs.append(self._handle_query_memory(cmd, args))
            else:
                out = self.env.execute(action)
                out = dict(out)
                if not self.v2:  # v2 marks only the latest tool message, at query time
                    out["output"] = (out.get("output") or "") + _memory_marker(self._projected(self.messages))
                outputs.append(out)
        msgs = self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
        if self._boundary_after_obs:
            # runner.py: new_last_boundary_pos = len(hm.messages) after the summary tool msg
            self._last_mc_end = len(self.messages)
            self._boundary_after_obs = False
        self._record_obs(msgs)
        return msgs

    def _obs(self, text: str, returncode: int = 0) -> dict:
        """Synthetic observation matching env.execute's shape. Must carry an
        exception_info key: swebench.yaml's observation_template does
        `{% if output.exception_info %}` under StrictUndefined, which raises
        UndefinedError if the key is absent."""
        marker = "" if self.v2 else _memory_marker(self._projected(self.messages))
        return {"output": text + marker, "returncode": returncode, "exception_info": None}

    def _handle_manage_context_v2(self) -> dict:
        """runner.py _handle_manage_context with mc_drop_call_message=True:
        compress [last boundary, calling assistant turn] inclusive, delete it,
        and return the summary itself as the tool result. Earlier summaries
        (tool messages before the boundary) stack untouched."""
        start = self._last_mc_end
        end = len(self.messages)  # the calling assistant turn is the last message
        if end <= start:
            return self._obs("Error: nothing to compress. There are no messages between your "
                             "previous manage_context call and this one.")
        forced = any(m.get("extra", {}).get("force_compress") for m in self.messages[max(0, end - 3):end])
        original = copy.deepcopy(self.messages[start:end])
        before = self._count_tokens(self.messages)
        inner = self._summarize_v2(original)
        sid = self.n_edits + 1
        feedback = f"[summary_id: {sid}] {inner}"
        del self.messages[start:end]
        self.archive.append({"summary_id": sid, "messages": original})
        self.n_edits += 1
        summary_tokens = self._count_tokens([{"role": "user", "content": feedback}])
        self._pending_edit = {
            "type": "summarize", "trigger": "forced" if forced else "voluntary",
            "span": [start, end], "tokens_before": before,
            "tokens_after": self._count_tokens(self.messages) + summary_tokens,
            "summary_tokens": summary_tokens, "summary_id": sid, "n_compressed": len(original),
        }
        self._boundary_after_obs = True
        return self._obs(feedback)

    def _handle_manage_context(self) -> dict:
        # compress everything since the last manage_context up to (not incl.) the
        # message that issued this command (which is the current last assistant turn)
        starts = self._turn_start_indices()
        end = starts[-1] if starts else len(self.messages)  # the manage_context turn itself
        end = max(end, self._last_mc_end)
        # voluntary unless the model was nudged by the 95% forced prompt this turn
        forced = any(m.get("extra", {}).get("force_compress") for m in self.messages[-3:])
        edit = self._compress_range(self._last_mc_end, end, trigger="forced" if forced else "voluntary")
        if edit is None:
            return self._obs("manage_context: nothing new to compress yet.")
        freed = edit["tokens_before"] - edit["tokens_after"]
        return self._obs(f"[summary_id: {edit['summary_id']}] compressed {edit['n_compressed']} messages, "
                         f"freed ~{freed} tokens. Continue working.")

    def _handle_query_memory(self, cmd: str, args: dict | None = None) -> dict:
        """ACM loads summary_{N}.json and uses an LLM to extract only what the
        query asks for. Dumping the whole archived span back into the context —
        which an earlier build did — defeats the purpose: it re-inflates the very
        context the compression freed."""
        args = args or {}
        sid, query = args.get("summary_id"), (args.get("query") or "").strip()
        if sid is None:  # legacy bash form: "query_memory <id> [query...]"
            parts = cmd.split()
            for tok in parts[1:]:
                try:
                    sid = int(tok.strip("[]#")); break
                except ValueError:
                    continue
            query = query or " ".join(parts[2:])
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            return self._obs("query_memory: summary_id must be an integer.", returncode=1)
        entry = next((a for a in self.archive if a["summary_id"] == sid), None)
        if self.v2:  # runner.py _handle_query_memory, error texts verbatim
            if not query:
                return self._obs("Error: missing required parameters (summary_id, query).", returncode=1)
            if entry is None:
                return self._obs(f"Error: summary_id {sid} not found.", returncode=1)
            history = self._span_text(entry["messages"], self._summarizer_budget())
            try:
                found = self._mem_llm(QUERY_MEMORY_PROMPT_V2.format(
                    summary_id=sid, history=history, query=query))
            except Exception as e:
                self.logger.warning(f"query_memory extraction failed: {e}")
                found = f"(retrieval unavailable: {e})"
            self.n_retrievals += 1
            return self._obs(f"[query_memory: summary_id={sid}]\n\n{found}")
        if entry is None:
            have = sorted(a["summary_id"] for a in self.archive)
            return self._obs(f"query_memory: no summary_id={sid} (have {have}).", returncode=1)
        if not query:
            return self._obs("query_memory: a query is required.", returncode=1)
        convo = "\n\n".join(f"[{m.get('role')}]\n{str(m.get('content',''))}" for m in entry["messages"])
        try:
            kwargs = dict(self.model.config.model_kwargs)
            kwargs.pop("parallel_tool_calls", None)
            resp = litellm.completion(
                model=self.model.config.model_name,
                messages=[
                    {"role": "system", "content": RETRIEVAL_SYSTEM},
                    {"role": "user", "content": f"Archived messages (summary_id {sid}):\n\n{convo}"
                                                f"\n\nExtract only what answers: {query}"},
                ],
                **kwargs,  # NB: no tools — plain completion
            )
            self._note_mem_op(resp)
            found = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            self.logger.warning(f"query_memory extraction failed: {e}")
            found = f"(retrieval unavailable: {e})"
        self.n_retrievals += 1
        return self._obs(f"[summary_id: {sid}] retrieved for \"{query}\":\n{found}")

    # ── per-turn telemetry ──────────────────────────────────────────────
    def _record_obs(self, msgs) -> None:
        """Size of what each tool result puts back into context. Observation size
        is the main driver of context growth and differs sharply between
        benchmarks (test output vs file dumps vs terminal scrollback), so it is
        the key quantity for comparing token behaviour ACROSS datasets."""
        for m in (msgs or []):
            try:
                self._obs_tokens.append(self._count_tokens([m]))
            except Exception:
                pass

    def _note_mem_op(self, resp) -> None:
        """Summarizer and retrieval calls are real token cost that never appears
        in the agent's own trajectory — A2/A3 pay it and A1 does not, so it has
        to be counted separately for any honest efficiency comparison."""
        self._mem_op_calls += 1
        try:
            u = resp.usage
            self._mem_op_usage[0] += int(getattr(u, "prompt_tokens", 0) or 0)
            self._mem_op_usage[1] += int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            pass

    def step(self) -> list[dict]:
        result = super().step()
        pt, ct = self._last_usage or (None, None)
        self._last_usage = None
        self.mem_steps.append({
            "step": self.n_calls,
            "ctx_tokens": self._count_tokens(self.messages),
            "n_messages": len(self.messages),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "edit": self._pending_edit,
        })
        self._pending_edit = None
        return result

    # ── persistence: dump memory_trace.json alongside the trajectory ────
    def save(self, path=None, *args, **kwargs):
        super().save(path, *args, **kwargs)
        if not path:
            return
        edits = [s["edit"] for s in self.mem_steps if s["edit"]]
        ctx = [s["ctx_tokens"] for s in self.mem_steps] or [0]
        archive_tokens = sum(self._count_tokens(a["messages"]) for a in self.archive) if self.archive else 0
        trace = {
            "policy": self.memory_policy,
            "context_cap": self.context_cap,
            "summarize_at": self.summarize_at,
            "keep_last_k": self.keep_last_k,
            "n_edits": len(edits),
            "n_retrievals": self.n_retrievals,
            "peak_ctx_tokens": max(ctx),
            "avg_ctx_tokens": sum(ctx) // len(ctx),
            "final_ctx_tokens": ctx[-1],
            "total_compressed_tokens": sum(e["tokens_before"] - e["tokens_after"] for e in edits),
            "archive_tokens": archive_tokens,
            "steps": self.mem_steps,
            "edits": edits,
            # ── trajectory / cost metrics ────────────────────────────────
            "n_steps": len(self.mem_steps),
            "wall_seconds": round(time.time() - self._t0, 1),
            "tool_calls": dict(self._tool_calls),
            "n_tool_calls": sum(self._tool_calls.values()),
            "total_prompt_tokens": sum(u[0] for u in self._usage),
            "total_completion_tokens": sum(u[1] for u in self._usage),
            "mem_op_calls": self._mem_op_calls,
            "mem_op_prompt_tokens": self._mem_op_usage[0],
            "mem_op_completion_tokens": self._mem_op_usage[1],
            "obs_tokens_mean": int(sum(self._obs_tokens) / len(self._obs_tokens)) if self._obs_tokens else 0,
            "obs_tokens_p90": (sorted(self._obs_tokens)[int(len(self._obs_tokens) * 0.9)]
                               if self._obs_tokens else 0),
            "obs_tokens_max": max(self._obs_tokens) if self._obs_tokens else 0,
            "n_observations": len(self._obs_tokens),
            # ── benchmark-invariant ratios (comparable across datasets/caps) ──
            # ctx_utilization: how much of the budget the run actually used
            # growth_per_step: how fast this benchmark fills context
            # compression_ratio: how much a summary retains of what it replaced
            # mem_overhead: memory-op tokens as a share of agent tokens
            "ctx_utilization": round(max(ctx) / max(1, self.context_cap), 3),
            "growth_per_step": int(max(ctx) / max(1, len(self.mem_steps))),
            "compression_ratio": (round(sum(e["summary_tokens"] for e in edits) /
                                        max(1, sum(e["tokens_before"] - e["tokens_after"] for e in edits)), 3)
                                  if edits else None),
            "mem_overhead": (round(sum(self._mem_op_usage) /
                                   max(1, sum(u[0] + u[1] for u in self._usage)), 4)
                             if self._usage else None),
            "compress_prompts": self._compress_prompts_total,
            "stop_prompted": self._stop_prompted,
            "harness": self.harness,
            "nudge": self.nudge,
            "marker_on": self.marker_on,
            "forced_submit": self._forced_submit,
            "summary_parse_fail": self._summary_parse_fail,
            "n_raw_messages": len(self.raw_messages),
            "saved_at": time.time(),
        }
        try:
            (Path(path).parent / "memory_trace.json").write_text(json.dumps(trace, indent=2))
        except Exception as e:
            self.logger.warning(f"could not write memory_trace.json: {e}")
        # The append-only history: the full step-by-step path including every span
        # that compression removed from self.messages. Kept in its own file so
        # memory_trace.json stays small enough to load for aggregate analysis.
        try:
            (Path(path).parent / "raw_messages.json").write_text(
                json.dumps({"instance_messages": self.raw_messages}, indent=2, default=str))
        except Exception as e:
            self.logger.warning(f"could not write raw_messages.json: {e}")
        if self.memory_policy == "none" and len(self.raw_messages) != len(self.messages):
            # with no compression the mirror must equal the live list; a mismatch
            # means some code path appends to self.messages without add_messages
            self.logger.warning(f"[raw-mirror] policy=none but raw={len(self.raw_messages)} "
                                f"live={len(self.messages)} — mirror is missing messages")
