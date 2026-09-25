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

import json
import time
from pathlib import Path

import litellm
import requests

from minisweagent.agents.default import DefaultAgent

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

# ── ACM system-prompt addendum (A3 only; appended by the driver) ──
ACM_SYSTEM_ADDENDUM = """

## Managing your context memory
Your in-context conversation is short-term memory; compressed segments live in
long-term memory on disk. After each command result the system appends
"[CURRENT CONTEXT TOKEN: N]" — your current context usage. Read it; never write
it yourself.

Two extra commands are available through the bash tool:
- `manage_context` — run this (as the entire command) to compress everything
  since your previous manage_context call into a single on-disk summary, freeing
  context. The system picks the range; the summary is returned with a
  [summary_id: N] tag. This does NOT end the task — keep working after it.
- `query_memory <summary_id>` — retrieve the original messages behind a prior
  summary when you need detail you compressed away.

Guidance: when [CURRENT CONTEXT TOKEN: N] climbs and your recent turns contain
dead ends or duplicated exploration, call `manage_context` to free space. Make
sure your work so far is captured before you do. Do not compress away the exact
edit you are mid-way through."""


# ── Forced-compression nudge (A3), adapted from ACM's BCP_COMPRESS_PROMPT ──
# ACM injects this at 95% of the context window (runner.py:930) so the agent
# doesn't just die when it fails to compress voluntarily. It's still the *model*
# that executes the compression — we only nudge; we never compress for it in A3.
FORCE_COMPRESS_PROMPT = (
    "WARNING: your context is nearly full and the task is not finished. You MUST "
    "immediately issue `manage_context` (as the entire command) to compress "
    "everything since your last manage_context call and free space. Do NOT submit "
    "yet. Run `manage_context` now, then continue working."
)


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
        keep_last_k: int = 6,
        force_frac: float = 0.85,
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)  # memory_* are captured here, not passed to AgentConfig
        self.memory_policy = memory_policy
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
        self.mem_steps: list[dict] = []   # per-turn: {step, ctx_tokens, n_messages, edit}
        self.archive: list[dict] = []     # [{summary_id, messages:[...]}]
        self.n_edits = 0
        self._pending_edit: dict | None = None
        self._last_mc_end = 2             # A3: start of the next compressible range

    # ── token accounting ────────────────────────────────────────────────
    def _count_tokens(self, messages: list[dict]) -> int:
        slim = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages]
        # 1) exact: ask the serving vLLM to tokenize with its own chat template
        if self._tokenize_url:
            try:
                r = requests.post(
                    self._tokenize_url,
                    json={"model": self._served_model, "messages": slim, "add_generation_prompt": True},
                    timeout=20,
                )
                if r.ok:
                    return int(r.json()["count"])
            except Exception:
                pass
        # 2) fallback: litellm (undercounts vLLM), then chars/4
        try:
            return int(litellm.token_counter(model=self.model.config.model_name, messages=slim))
        except Exception:
            return sum(len(str(m.get("content", ""))) for m in messages) // 4

    def _turn_start_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m.get("role") == "assistant"]

    # ── summarization (shared by A2 and A3) ─────────────────────────────
    def _summarize(self, span_messages: list[dict]) -> str:
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
        """Largest turn-aligned end index that keeps the last keep_last_k turns."""
        starts = [i for i in self._turn_start_indices() if i >= since]
        if len(starts) <= self.keep_last_k:
            return -1
        return starts[-self.keep_last_k]  # keep the last K turns intact

    # ── pre-call hooks: A2 harness compression, A3 forced-compression nudge ──
    def query(self) -> dict:
        if self.memory_policy == "summarize" and self._count_tokens(self.messages) >= self.summarize_at:
            end = self._compressible_tail_boundary(since=2)
            if end > 2:
                self._compress_range(2, end, trigger="harness")
        elif self.memory_policy == "acm" and self._count_tokens(self.messages) >= int(self.context_cap * self.force_frac):
            # ACM-style forced nudge — fires at force_frac of the cap (0.85 by
            # default: below 1.0 so the prompt + tools + the model's response
            # still fit under max_model_len). Inject once per crossing.
            if not (self.messages and self.messages[-1].get("extra", {}).get("force_compress")):
                self.add_messages({"role": "user", "content": FORCE_COMPRESS_PROMPT,
                                   "extra": {"force_compress": True}})
        return super().query()

    # ── A3 hook: intercept manage_context / query_memory bash commands ───
    def execute_actions(self, message: dict) -> list[dict]:
        if self.memory_policy != "acm":
            return super().execute_actions(message)
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            cmd = (action.get("command") or "").strip()
            if cmd == "manage_context":
                outputs.append(self._handle_manage_context())
            elif cmd.startswith("query_memory"):
                outputs.append(self._handle_query_memory(cmd))
            else:
                out = self.env.execute(action)
                out = dict(out)
                out["output"] = (out.get("output") or "") + _memory_marker(self._count_tokens(self.messages))
                outputs.append(out)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

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
            return {"output": "manage_context: nothing new to compress yet." + _memory_marker(self._count_tokens(self.messages)),
                    "returncode": 0}
        freed = edit["tokens_before"] - edit["tokens_after"]
        return {
            "output": f"[summary_id: {edit['summary_id']}] compressed {edit['n_compressed']} messages, "
                      f"freed ~{freed} tokens. Continue working." + _memory_marker(self._count_tokens(self.messages)),
            "returncode": 0,
        }

    def _handle_query_memory(self, cmd: str) -> dict:
        parts = cmd.split()
        sid = None
        for p in parts[1:]:
            try:
                sid = int(p.strip("[]#")); break
            except ValueError:
                continue
        entry = next((a for a in self.archive if a["summary_id"] == sid), None)
        if entry is None:
            return {"output": f"query_memory: no summary_id={sid}." + _memory_marker(self._count_tokens(self.messages)),
                    "returncode": 1}
        dump = "\n\n".join(f"[{m.get('role')}]\n{str(m.get('content',''))}" for m in entry["messages"])
        return {"output": f"Retrieved original messages for summary_id={sid}:\n{dump}"
                          + _memory_marker(self._count_tokens(self.messages)),
                "returncode": 0}

    # ── per-turn telemetry ──────────────────────────────────────────────
    def step(self) -> list[dict]:
        result = super().step()
        self.mem_steps.append({
            "step": self.n_calls,
            "ctx_tokens": self._count_tokens(self.messages),
            "n_messages": len(self.messages),
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
            "peak_ctx_tokens": max(ctx),
            "avg_ctx_tokens": sum(ctx) // len(ctx),
            "final_ctx_tokens": ctx[-1],
            "total_compressed_tokens": sum(e["tokens_before"] - e["tokens_after"] for e in edits),
            "archive_tokens": archive_tokens,
            "steps": self.mem_steps,
            "edits": edits,
            "saved_at": time.time(),
        }
        try:
            (Path(path).parent / "memory_trace.json").write_text(json.dumps(trace, indent=2))
        except Exception as e:
            self.logger.warning(f"could not write memory_trace.json: {e}")
