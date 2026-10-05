#!/usr/bin/env python3
"""
Build the results table across (model x arm) runs, in ACL booktabs LaTeX and
markdown. Reads memory_trace.json + traj.json, optionally joins Klone resolve
outcomes.

  python3 scripts/make_table.py --run-dir runs/qwen35-9b_none_* --run-dir ... \
      [--eval logs/apptainer_eval] [--tex table.tex] [--md table.md]

Every column is either a count or a ratio that is comparable across benchmarks,
so the same table works for SWE-bench, Terminal-Bench or frontier SWE.
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path

def load(run_dir: str, eval_root: str | None) -> dict | None:
    rows = []
    resolved = set()
    rid = Path(run_dir).name
    if eval_root:
        for rp in glob.glob(f"{eval_root}/{rid}/*/report.json"):
            try:
                if json.load(open(rp)).get("resolved"): resolved.add(Path(rp).parent.name)
            except Exception: pass
    for tp in glob.glob(f"{run_dir}/*/memory_trace.json"):
        iid = Path(tp).parent.name
        try: t = json.load(open(tp))
        except Exception: continue
        status = "?"
        tj = Path(tp).parent / f"{iid}.traj.json"
        if tj.exists():
            try: status = json.load(open(tj)).get("info", {}).get("exit_status", "?")
            except Exception: pass
        t["_iid"], t["_status"], t["_resolved"] = iid, status, iid in resolved
        rows.append(t)
    if not rows: return None
    g = lambda k, d=0: [r.get(k) if r.get(k) is not None else d for r in rows]
    n = len(rows)
    edits = sum(g("n_edits"))
    trig = [e.get("trigger") for r in rows for e in r.get("edits", [])]
    tc = {}
    for r in rows:
        for k, v in (r.get("tool_calls") or {}).items(): tc[k] = tc.get(k, 0) + v
    agent_tok = sum(g("total_prompt_tokens")) + sum(g("total_completion_tokens"))
    mem_tok = sum(g("mem_op_prompt_tokens")) + sum(g("mem_op_completion_tokens"))
    return {
        "run": rid, "policy": rows[0].get("policy", "?"), "n": n,
        "submitted": sum(1 for r in rows if r["_status"] == "Submitted"),
        "ctx_death": sum(1 for r in rows if "ContextWindow" in str(r["_status"])),
        "resolved": sum(1 for r in rows if r["_resolved"]) if eval_root else None,
        "steps": round(sum(g("n_steps")) / n, 1),
        "tool_calls": round(sum(g("n_tool_calls")) / n, 1),
        "mc": tc.get("manage_context", 0), "qm": tc.get("query_memory", 0),
        "edits": edits, "edits_per": round(edits / n, 2),
        "vol": trig.count("voluntary"), "forced": trig.count("forced"), "harness": trig.count("harness"),
        "peak": int(st.median(g("peak_ctx_tokens"))),
        "util": round(st.mean([r.get("ctx_utilization") or 0 for r in rows]), 3),
        "growth": int(st.mean(g("growth_per_step"))),
        "obs_mean": int(st.mean(g("obs_tokens_mean"))), "obs_p90": int(st.median(g("obs_tokens_p90"))),
        "agent_ktok": round(agent_tok / n / 1000, 1),
        "mem_ktok": round(mem_tok / n / 1000, 1),
        "overhead": round(mem_tok / max(1, agent_tok), 4),
    }

# ACM Table 2 reports Pass@1 / Tools / Peak Tok.; we mirror those three and add
# the memory-specific columns their table has no column for.
COLS = [("arm","Method",None),("resolved_rate","Pass@1","p"),("tool_calls","Tools","f"),
        ("peak","Peak Tok.","K"),("edits_per","Edits","f"),("vol","Vol.","d"),
        ("util","Util.","f"),("overhead","Ovh.","f"),("steps","Steps","f"),
        ("submitted","Sub.","d"),("ctx_death","OOC","d")]

ARM_LABEL = {"none": "A1 ReAct (no memory)", "summarize": "A2 Summarize-on-threshold",
             "acm": "A3 ACM Base (untrained)"}

# Reference numbers from ACM (Table 2, SWE-Bench Verified, Qwen3.5-9B base).
# Their scaffold is a THREE-tool surface (execute_bash, str_replace_editor,
# submit_patch); ours is mini-swe-agent's single bash tool, so Tools and
# Peak Tok. are not directly comparable — flagged in the caption.
ACM_REF = [("\\textit{ReAct (ACM paper)}", 0.489, 74.7, 59000),
           ("\\textit{ACM Base (ACM paper)}", 0.508, 77.6, 46000),
           ("\\textit{ACM Post-Trained (paper)}", 0.530, 79.3, 50000)]


def fmt(v, kind):
    if v is None: return "--"
    if kind == "p": return f"{v:.3f}"
    if kind == "K": return f"{v/1000:.0f}K"
    if kind == "d": return str(int(v))
    if kind == "f": return f"{v:g}"
    return str(v)


def model_of(run: str) -> str:
    for m in ("qwen35-9b", "mimo-9b", "gemma4-12b-it"):
        if run.startswith(m): return m
    return run.split("_")[0]


PRETTY = {"qwen35-9b": "Qwen3.5-9B", "mimo-9b": "MiMo-V2.6-Distill-Qwen-9B",
          "gemma4-12b-it": "Gemma-4-12B-it"}
ARM_ORDER = {"none": 0, "summarize": 1, "acm": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--eval"); ap.add_argument("--tex"); ap.add_argument("--md")
    ap.add_argument("--changelog", action="store_true",
                    help="also emit the since-last-meeting changelog table")
    a = ap.parse_args()
    dirs = [d for pat in a.run_dir for d in sorted(glob.glob(pat))]
    rows = [r for r in (load(d, a.eval) for d in dirs) if r]
    if not rows:
        print("no runs found"); return
    for r in rows:
        r["arm"] = ARM_LABEL.get(r["policy"], r["policy"])
        r["resolved_rate"] = (r["resolved"] / r["n"]) if r.get("resolved") is not None else None
        r["_model"] = model_of(r["run"])
    rows.sort(key=lambda r: (r["_model"], ARM_ORDER.get(r["policy"], 9)))

    body, cur = [], None
    for r in rows:
        if r["_model"] != cur:
            cur = r["_model"]
            body.append(("GROUP", PRETTY.get(cur, cur)))
        body.append(("ROW", [fmt(r.get(k), t) for k, _, t in COLS]))
    body.append(("GROUP", "Reference: ACM paper, Qwen3.5-9B base"))
    for name, pa, tl, pk in ACM_REF:
        body.append(("ROW", [name, f"{pa:.3f}", f"{tl:g}", f"{pk/1000:.0f}K"]
                            + ["--"] * (len(COLS) - 4)))

    md = ["| " + " | ".join(c[1] for c in COLS) + " |",
          "|" + "|".join(["---"] * len(COLS)) + "|"]
    for kind, v in body:
        md.append(f"| **{v}** |" + " |" * (len(COLS) - 1) if kind == "GROUP"
                  else "| " + " | ".join(v) + " |")
    md = "\n".join(md)

    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\begin{tabular}{l" + "r" * (len(COLS) - 1) + "}", r"\toprule",
           r"\multicolumn{%d}{c}{\textbf{SWE-Bench Verified}} \\" % len(COLS),
           r"\cmidrule(lr){1-%d}" % len(COLS),
           " & ".join(c[1] for c in COLS) + r" \\", r"\midrule"]
    first = True
    for kind, v in body:
        if kind == "GROUP":
            if not first: tex.append(r"\midrule")
            tex.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(COLS), v))
            first = False
        else:
            tex.append(" & ".join(v) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Memory-management arms on SWE-Bench Verified. "
            r"\textbf{Pass@1} is the resolved rate; \textbf{Tools} the mean tool calls per "
            r"episode; \textbf{Peak Tok.} the mean peak context, following ACM's Table~2. "
            r"\textbf{Edits} is mean memory compressions per instance, \textbf{Vol.} those the "
            r"model issued unprompted, \textbf{Util.} peak context as a fraction of the budget, "
            r"\textbf{Ovh.} memory-operation tokens as a fraction of agent tokens, "
            r"\textbf{Sub.} submitted patches and \textbf{OOC} out-of-context failures. "
            r"All arms share one model, sampling configuration and 64K budget; the only "
            r"variable is the memory policy. ACM reference rows use a three-tool scaffold "
            r"(\texttt{execute\_bash}, \texttt{str\_replace\_editor}, \texttt{submit\_patch}) "
            r"whereas ours exposes a single bash tool, so Tools and Peak Tok. are not directly "
            r"comparable across that boundary.}",
            r"\label{tab:memory-arms}", r"\end{table*}"]
    tex = "\n".join(tex)

    if a.changelog:
        tex += "\n\n" + CHANGELOG_TEX

    print(md)
    if a.md: Path(a.md).write_text(md + "\n"); print(f"\n>>> {a.md}")
    if a.tex: Path(a.tex).write_text(tex + "\n"); print(f">>> {a.tex}")


CHANGELOG_TEX = r"""\begin{table}[t]
\centering\small
\begin{tabular}{@{}lp{0.72\linewidth}@{}}
\toprule
\textbf{Area} & \textbf{Change} \\
\midrule
Models & Qwen3.6-35B-A3B $\rightarrow$ three 9--12B models: Qwen3.5-9B, MiMo-V2.6-Distill-Qwen-9B (same base, isolating post-training), Gemma-4-12B-it. \\
\addlinespace
ACM fidelity & Audited the reference implementation; added the STOP\_PROMPT escalation, bounded compress retries, and append-only trajectory capture. Registered \texttt{manage\_context}/\texttt{query\_memory} as real function-calling tools: previously issued as bash strings and ignored, forced compliance rose $0/6 \rightarrow 4/4$. \\
\addlinespace
Context budget & $24$K $\rightarrow$ $64$K. ACM runs at $128$K; at $24$K their $0.95$ trigger leaves under one observation of headroom and A3 performed no compressions. \\
\addlinespace
Configuration & A2 now fires at A3's exact trigger, leaving the mechanism as the sole variable; evaluation set $42 \rightarrow 99$ instances. \\
\addlinespace
Infrastructure & Fixed silently dropped \texttt{top\_k} sampling, a Gemma audio-tower startup crash, \texttt{--export} comma truncation, and a Docker Hub pull stampede; added step/token/tool-call instrumentation. \\
\bottomrule
\end{tabular}
\caption{Changes since the previous meeting.}
\label{tab:changelog}
\end{table}
"""

if __name__ == "__main__":
    main()
