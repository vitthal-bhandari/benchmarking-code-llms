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

COLS = [("run","Run",None),("policy","Arm",None),("n","$N$","d"),("resolved","Res.","d"),
        ("submitted","Sub.","d"),("ctx_death","OOC","d"),("steps","Steps","f"),
        ("tool_calls","Tools","f"),("edits_per","Edits","f"),("vol","Vol.","d"),
        ("peak","Peak","k"),("util","Util.","f"),("growth","Gr/st","d"),
        ("obs_mean","Obs","d"),("agent_ktok","Tok(k)","f"),("overhead","Ovh.","f")]

def fmt(v, kind):
    if v is None: return "--"
    if kind == "k": return f"{v:,}"
    if kind == "d": return str(int(v))
    if kind == "f": return f"{v:g}"
    return str(v).replace("_", r"\_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--eval"); ap.add_argument("--tex"); ap.add_argument("--md")
    a = ap.parse_args()
    dirs = [d for pat in a.run_dir for d in sorted(glob.glob(pat))]
    rows = [r for r in (load(d, a.eval) for d in dirs) if r]
    if not rows: print("no runs found"); return

    md = ["| " + " | ".join(c[1] for c in COLS) + " |",
          "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(fmt(r.get(k), t) for k, _, t in COLS) + " |")
    md = "\n".join(md)

    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\begin{tabular}{ll" + "r" * (len(COLS) - 2) + "}", r"\toprule",
           " & ".join(c[1] for c in COLS) + r" \\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join(fmt(r.get(k), t) for k, _, t in COLS) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Memory-management arms. \textbf{Res.}=resolved, \textbf{Sub.}=submitted, "
            r"\textbf{OOC}=out-of-context failures, \textbf{Vol.}=voluntary compressions, "
            r"\textbf{Util.}=peak context as a fraction of the budget, \textbf{Gr/st}=context "
            r"tokens added per step, \textbf{Obs}=mean observation size, \textbf{Tok(k)}=agent "
            r"tokens per instance, \textbf{Ovh.}=memory-op tokens as a fraction of agent tokens. "
            r"Ratio columns are benchmark-invariant.}",
           r"\label{tab:memory-arms}", r"\end{table*}"]
    tex = "\n".join(tex)

    print(md)
    if a.md: Path(a.md).write_text(md + "\n"); print(f"\n>>> {a.md}")
    if a.tex: Path(a.tex).write_text(tex + "\n"); print(f">>> {a.tex}")

if __name__ == "__main__":
    main()
