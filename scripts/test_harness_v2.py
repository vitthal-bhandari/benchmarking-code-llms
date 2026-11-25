#!/usr/bin/env python3
"""Offline test of MemoryAgent v1/v2 message handling. No GPU, no vLLM, no SWE-bench
image: mini-swe-agent's real agent loop + a LocalEnvironment over a scratch git
repo + a scripted fake LLM patched in for litellm.completion.

  python scripts/test_harness_v2.py        # needs mini-swe-agent 2.4.x, transformers

Checks, per arm, the behaviours docs/acm_vs_ours.md lists for v2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litellm  # noqa: E402
from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models import get_model  # noqa: E402

import memory_agent as ma  # noqa: E402

CAP, MAXTOK = 9000, 1000
SUMMARIZE_AT = int(0.95 * CAP)
MEMORY_BLOCK = ("<memory>\n## Knowledge state\n- /repo/a.py line 3 holds the bug; edited it.\n\n"
                "## Thoughts\nNext: run the test in /repo/test_a.py.\n</memory>")


def resp(content=None, tool_calls=None, reasoning=None):
    msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return litellm.ModelResponse(
        choices=[{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": msg}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, model="fake")


def call(name, args, i=[0]):
    i[0] += 1
    return [{"id": f"call_{i[0]}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]


def bash(cmd):
    return call("bash", {"command": cmd})


class FakeLLM:
    """Plays an agent. `script` maps the agent-turn index to a response; memory
    calls (no tools) get a canned <memory>; STOP / nudge get scenario replies."""

    def __init__(self, script, on_stop="ignore", on_nudge="comply"):
        self.script, self.on_stop, self.on_nudge = script, on_stop, on_nudge
        self.agent_requests, self.mem_prompts, self.turn = [], [], 0

    def __call__(self, model=None, messages=None, tools=None, **kw):
        if not tools:  # summarizer / querier
            self.mem_prompts.append(messages)
            text = messages[-1]["content"]
            if "Recall request" in text:
                return resp(content="<think>scan</think>- **Relevant findings:** /repo/a.py line 3")
            return resp(content=MEMORY_BLOCK)
        self.agent_requests.append({"messages": json.loads(json.dumps(messages, default=str)),
                                    "tools": [t["function"]["name"] for t in tools]})
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), {})
        lu = str(last_user.get("content", ""))
        last = messages[-1]
        if last["role"] == "user" and "must now conclude" in lu or "STOP." in lu:
            if self.on_stop == "submit":
                return resp(tool_calls=bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"))
            return resp(tool_calls=bash("ls"), reasoning="ignoring the stop")
        if last["role"] == "user" and "MUST immediately call manage_context" in lu:
            return resp(tool_calls=call("manage_context", {}), reasoning="compress now")
        self.turn += 1
        r = self.script.get(self.turn)
        if r is None:  # default: grow the context with a big observation
            return resp(tool_calls=bash("python3 -c \"print('lorem ipsum dolor ' * 220)\""),
                        reasoning=f"thinking at turn {self.turn}")
        return r


def make_repo():
    d = Path(tempfile.mkdtemp(prefix="v2test_"))
    subprocess.run("git init -q && echo 'x = 1' > a.py && git add a.py && "
                   "git -c user.email=t@t -c user.name=t commit -qm init", shell=True, cwd=d, check=True)
    return d


def run_arm(policy, harness, script, on_stop="ignore", on_nudge="comply", step_limit=40):
    repo = make_repo()
    ma.FORCED_SUBMIT_CMD = f"cd {repo} && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"
    fake = FakeLLM(script, on_stop, on_nudge)
    litellm.completion = fake
    ma.litellm.completion = fake
    cfg = get_config_from_spec("swebench.yaml")
    cfg["agent"]["step_limit"] = step_limit
    cfg["agent"]["system_template"] += ma.build_system_addendum(policy, harness, CAP)
    cfg["model"]["model_name"] = "hosted_vllm/Qwen/Qwen3.5-9B"
    cfg["model"].setdefault("model_kwargs", {})["max_tokens"] = MAXTOK
    model = get_model(config=cfg["model"])
    env = LocalEnvironment(cwd=str(repo), timeout=30)
    agent = ma.MemoryAgent(model, env, memory_policy=policy, context_cap=CAP, summarize_at=SUMMARIZE_AT,
                           keep_last_k=3, harness=harness, **cfg["agent"])
    info = agent.run("Fix the bug in a.py.")
    out = Path(tempfile.mkdtemp()) / "t.traj.json"
    agent.save(out)
    trace = json.loads((out.parent / "memory_trace.json").read_text())
    return agent, fake, info, trace


FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


MARKER_RE = __import__("re").compile(r"\[CURRENT CONTEXT TOKEN: \d+\]")


def markers_in(msgs):
    """Real markers only (digits), never the system prompt's own description of them."""
    return sum(len(MARKER_RE.findall(str(m.get("content")))) for m in msgs if m.get("role") != "system")


def main():
    edit = resp(tool_calls=bash("sed -i.bak 's/x = 1/x = 2/' a.py && rm a.py.bak && cat a.py"),
                reasoning="edit the bug")

    print("\n[A3 v2 faithful] voluntary mc, query_memory, STOP ignored -> forced submit")
    agent, fake, info, tr = run_arm("acm", "v2", {
        1: edit,
        3: resp(tool_calls=call("manage_context", {}), reasoning="free space"),
        4: resp(tool_calls=call("query_memory", {"summary_id": 1, "query": "which file has the bug"})),
    })
    sysmsg = fake.agent_requests[0]["messages"][0]["content"]
    check(f"Your context window is {CAP} tokens" in sysmsg, "system prompt states the context window")
    check("Watch the [CURRENT CONTEXT TOKEN: N] marker" in sysmsg, "ACM strategy block present")
    check(fake.agent_requests[0]["tools"] == ["bash", "manage_context", "query_memory"], "3 tools offered")
    check("DO NOT CALL THIS TOOL until" in json.dumps(ma.MEM_TOOLS_V2), "paper's query_memory guard")
    check(all(markers_in(r["messages"]) <= 1 for r in fake.agent_requests),
          "outgoing requests carry at most ONE marker")
    later = [r for r in fake.agent_requests if any(m["role"] == "tool" for m in r["messages"])]
    check(all(str([m for m in r["messages"] if m["role"] == "tool"][-1]["content"]).rstrip().endswith("]")
              and "[CURRENT CONTEXT TOKEN" in str([m for m in r["messages"] if m["role"] == "tool"][-1]["content"])
              for r in later), "the marker sits on the latest tool message")
    check(markers_in(agent.messages) == 0 and markers_in(agent.raw_messages) == 0, "no marker stored in history")
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    summ = [m for m in tool_msgs if str(m["content"]).find("[summary_id: 1]") >= 0]
    check(len(summ) == 1 and "Knowledge state" in summ[0]["content"] and "<memory>" not in summ[0]["content"],
          "manage_context returns the inner summary as its tool result")
    mc_turns = [m for m in agent.messages if m["role"] == "assistant" and
                any((tc.get("function") or {}).get("name") == "manage_context" for tc in (m.get("tool_calls") or []))]
    check(not mc_turns, "calling assistant turn dropped (mc_drop_call_message)")
    i_sum = agent.messages.index(summ[0])
    check(i_sum == 2, f"summary tool msg sits right after system+task (index {i_sum})")
    sp = fake.mem_prompts[0][0]["content"]
    check(len(fake.mem_prompts[0]) == 1 and fake.mem_prompts[0][0]["role"] == "user",
          "summarizer gets one user message (no system)")
    check("$ sed -i.bak" in sp and "<think>\nedit the bug\n</think>" in sp,
          "summarizer input includes commands and reasoning")
    check("future calls to query_memory" in sp, "A3 summary instruction mentions query_memory")
    qm = [m for m in tool_msgs if "[query_memory: summary_id=1]" in str(m["content"])]
    check(len(qm) == 1 and "<think>" not in qm[0]["content"], "query_memory returns ACM format, thinking stripped")
    check("Saved messages under summary_id=1" in fake.mem_prompts[1][0]["content"], "ACM recall prompt used")
    check(tr["compress_prompts"] == 0, "no compress nudge in faithful A3")
    check(tr["stop_prompted"] and tr["forced_submit"], "STOP at 95% then forced submit when ignored")
    check(info.get("exit_status") == "Submitted" and "x = 2" in info.get("submission", ""),
          "forced submission carries the git diff")
    check([e["trigger"] for e in tr["edits"]] == ["voluntary"], "edit logged as voluntary")
    check(agent._last_mc_end == agent.messages.index(summ[0]) + 1 or agent._last_mc_end > 2,
          "boundary moved past the summary")

    print("\n[A1 v2] STOP at 95% -> model submits itself")
    agent, fake, info, tr = run_arm("none", "v2", {1: edit}, on_stop="submit")
    sysmsg = fake.agent_requests[0]["messages"][0]["content"]
    check(f"Your context window is {CAP} tokens" in sysmsg and "manage_context" not in sysmsg,
          "A1 prompt: window line, no memory text")
    check(fake.agent_requests[0]["tools"] == ["bash"], "A1 has bash only")
    check(all(markers_in(r["messages"]) == 0 for r in fake.agent_requests), "A1 sees no marker (default)")
    check(tr["stop_prompted"] and not tr["forced_submit"], "STOP fired, model submitted on its own")
    check(info.get("exit_status") == "Submitted" and "x = 2" in info.get("submission", ""), "A1 patch kept")
    check(max(s["ctx_tokens"] for s in tr["steps"]) < CAP, "never exceeded the cap")

    print("\n[A3 v2 acm_nudge] nudge at 95% -> forced compression")
    agent, fake, info, tr = run_arm("acm_nudge", "v2", {1: edit}, step_limit=14)
    check(tr["compress_prompts"] >= 1 and tr["n_edits"] >= 1, "nudged and compressed")
    check(all(e["trigger"] == "forced" for e in tr["edits"]), "edits logged as forced")
    check(all(markers_in(r["messages"]) <= 1 for r in fake.agent_requests), "single transient marker")

    print("\n[A2 v2] harness compression keeps last 3 turns, window line, no marker")
    agent, fake, info, tr = run_arm("summarize", "v2", {1: edit}, step_limit=14)
    check(tr["n_edits"] >= 1 and all(e["trigger"] == "harness" for e in tr["edits"]), "harness compressions")
    s_msgs = [m for m in agent.messages if (m.get("extra") or {}).get("is_summary")]
    check(s_msgs and "<memory>" in s_msgs[-1]["content"] and "Knowledge state" in s_msgs[-1]["content"],
          "A2 summary message carries the extracted <memory>")
    check("future calls to query_memory" not in fake.mem_prompts[0][0]["content"], "A2 instruction has no query clause")
    check(all(markers_in(r["messages"]) == 0 for r in fake.agent_requests), "A2 sees no marker (default)")
    check(f"Your context window is {CAP} tokens" in fake.agent_requests[0]["messages"][0]["content"], "A2 window line")

    print("\n[A3 v1 regression] marker on every observation, nudges, no forced submit")
    agent, fake, info, tr = run_arm("acm", "v1", {1: edit}, step_limit=14)
    obs = [m for m in agent.messages if m["role"] == "tool"]
    check(sum("[CURRENT CONTEXT TOKEN" in str(m["content"]) for m in obs) >= len(obs) - 1,
          "v1 stamps the marker into stored observations")
    check(tr["compress_prompts"] >= 1, "v1 nudges")
    check("Your context window is" not in fake.agent_requests[0]["messages"][0]["content"], "v1 prompt unchanged")
    check(not tr["forced_submit"], "v1 never force-submits")

    print("\n[chat template] Qwen3.5 renders a tool message whose calling turn was dropped")
    tok = agent._get_tokenizer()
    if tok is None:
        print("  skip (no local tokenizer)")
    else:
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                {"role": "tool", "content": "[summary_id: 1] ...", "tool_call_id": "call_9"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "call_10", "type": "function",
                                 "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
                {"role": "tool", "content": "a.py", "tool_call_id": "call_10"}]
        try:
            txt = tok.apply_chat_template(msgs, tools=[ma.BASH_TOOL], tokenize=False, add_generation_prompt=True)
            check("<tool_response>\n[summary_id: 1]" in txt, "orphan summary renders as a tool_response")
        except Exception as e:
            check(False, f"chat template rejected the orphan tool message: {e}")
        n = agent._count_tokens([{"role": "system", "content": "s"}, {"role": "user", "content": "t"},
                                 {"role": "assistant", "content": None, "tool_calls": bash("ls -la")},
                                 {"role": "tool", "content": "a.py", "tool_call_id": "x"}])
        check(n > 30, f"local tokenizer counts a tool-call history ({n} tokens), no /tokenize fallback")

    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
