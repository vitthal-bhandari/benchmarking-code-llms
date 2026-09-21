#!/usr/bin/env python3
"""
Render a mini-swe-agent trajectory (traj.json) as a standalone, shareable HTML
page — for eyeballing traces and presenting them to an advisor.

Works on ANY existing run (no memory data needed): it shows the header stats, a
context-growth chart (approx tokens of the live message list per step — the
monotonic climb that motivates memory management), and every step as a card.

If a sibling `memory_trace.json` (from the A2/A3 MemoryAgent) is present, it
overlays compression events on the chart and badges the summary messages.

Usage:
  python scripts/render_trajectory_html.py runs/run_qwen_100_a_219591/astropy__astropy-12907/astropy__astropy-12907.traj.json
  python scripts/render_trajectory_html.py --run-dir runs/run_qwen_100_a_219591 -o site/traj/
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from pathlib import Path


def approx_tokens(text) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text)
    return len(text) // 4  # rough chars/4; fine for a growth curve, not billing


def msg_text(m: dict) -> str:
    c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c, indent=2)


def command_of(m: dict) -> str | None:
    """Best-effort extract the bash command an assistant turn issued."""
    ex = m.get("extra", {})
    acts = ex.get("actions")
    if acts:
        if isinstance(acts, list):
            return "\n".join(a if isinstance(a, str) else json.dumps(a) for a in acts)
        return str(acts)
    for tc in m.get("tool_calls") or []:
        fn = (tc or {}).get("function", {})
        args = fn.get("arguments")
        if args:
            try:
                return json.loads(args).get("command", args) if isinstance(args, str) else str(args)
            except Exception:
                return str(args)
    return None


def svg_growth_chart(step_tokens: list[int], edit_steps: list[int], width=760, height=140) -> str:
    """Inline SVG line of live-context tokens per step; red ticks at compressions."""
    if not step_tokens:
        return ""
    n = len(step_tokens)
    mx = max(step_tokens) or 1
    pad = 6
    def x(i): return pad + (i / max(1, n - 1)) * (width - 2 * pad)
    def y(v): return (height - pad) - (v / mx) * (height - 2 * pad)
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(step_tokens))
    area = f"{pad},{height-pad} " + pts + f" {x(n-1):.1f},{height-pad}"
    ticks = "".join(
        f'<line x1="{x(s):.1f}" y1="{pad}" x2="{x(s):.1f}" y2="{height-pad}" class="edit-tick"/>'
        for s in edit_steps if 0 <= s < n
    )
    return f"""<svg viewBox="0 0 {width} {height}" class="chart" preserveAspectRatio="none" role="img"
      aria-label="context tokens per step, peak {mx:,}">
      <polygon points="{area}" class="area"/>
      <polyline points="{pts}" class="line"/>
      {ticks}
    </svg>"""


ROLE_META = {
    "system": ("System", "role-system"),
    "user": ("Task", "role-user"),
    "assistant": ("Agent", "role-assistant"),
    "tool": ("Observation", "role-tool"),
    "exit": ("Exit", "role-exit"),
}


def render(traj_path: Path) -> str:
    t = json.loads(traj_path.read_text())
    info = t.get("info", {})
    msgs = t.get("messages", [])
    iid = t.get("instance_id", traj_path.stem)
    cfg = info.get("config", {}) or {}
    model = cfg.get("model_name", "unknown")
    exit_status = info.get("exit_status", "?")
    api_calls = info.get("model_stats", {}).get("api_calls", "?")

    mem_path = traj_path.with_name("memory_trace.json")
    mem = json.loads(mem_path.read_text()) if mem_path.exists() else None
    edit_steps = [i for i, r in enumerate(mem.get("steps", []))
                  if r.get("edit")] if mem else []
    summary_spans = {}  # (compressed msg index) -> edit info, if memory trace present

    # context-token growth: cumulative live-context size after each message
    running, series = 0, []
    for m in msgs:
        running += approx_tokens(msg_text(m)) + approx_tokens(command_of(m))
        series.append(running)
    peak = max(series) if series else 0

    # step cards
    cards = []
    step_no = 0
    for m in msgs:
        role = m.get("role", "?")
        label, cls = ROLE_META.get(role, (role.title(), "role-other"))
        body_parts = []
        if role == "assistant":
            step_no += 1
            label = f"Agent · step {step_no}"
            txt = msg_text(m).strip()
            if txt:
                body_parts.append(f'<div class="prose">{html.escape(txt)}</div>')
            cmd = command_of(m)
            if cmd:
                body_parts.append(f'<pre class="cmd"><span class="lbl">$ command</span>{html.escape(cmd)}</pre>')
            if m.get("extra", {}).get("is_summary"):
                cls = "role-summary"
                e = m["extra"]
                label = f"Memory · compressed {e.get('span')} (−{e.get('tokens_before',0)-e.get('summary_tokens',0):,} tok)"
        elif role == "exit":
            e = m.get("extra", {})
            sub = e.get("submission", "") or ""
            body_parts.append(f'<div class="pill">{html.escape(str(e.get("exit_status","")))}</div>')
            if sub.strip():
                body_parts.append(f'<pre class="patch"><span class="lbl">submission (patch)</span>{html.escape(sub)}</pre>')
        else:
            txt = msg_text(m)
            long = len(txt) > 1400
            shown = html.escape(txt[:1400]) + ("…" if long else "")
            pre_cls = "obs" if role == "tool" else "prose"
            body_parts.append(f'<pre class="{pre_cls}">{shown}</pre>' if role == "tool"
                              else f'<div class="{pre_cls}">{shown}</div>')
        tok = approx_tokens(msg_text(m)) + approx_tokens(command_of(m))
        cards.append(
            f'<div class="card {cls}"><div class="chead"><span class="rlabel">{html.escape(label)}</span>'
            f'<span class="toks">~{tok:,} tok</span></div>{"".join(body_parts)}</div>'
        )

    chart = svg_growth_chart(series, edit_steps)
    mem_note = ""
    if mem:
        nedits = len(edit_steps)
        mem_note = f'<span class="stat"><b>{nedits}</b> memory edit{"s" if nedits!=1 else ""}</span>'

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(iid)} · trajectory</title>
<style>
:root {{ --bg:#f7f8fa; --surface:#fff; --ink:#1a1d24; --muted:#5b6270; --faint:#868d9b;
  --hair:#e3e6ec; --accent:#3a5bd9; --sys:#7a8699; --task:#2f7d9e; --agent:#3a5bd9;
  --obs:#5b6270; --exit:#2f9e6f; --summary:#c9761d; --code-bg:#f1f3f7; }}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme=light]) {{
  --bg:#14171d; --surface:#1c2028; --ink:#e8ebf1; --muted:#9aa2b1; --faint:#79808f;
  --hair:#2b303b; --accent:#8098f2; --sys:#8a95a8; --task:#5fb3d4; --agent:#8098f2;
  --obs:#9aa2b1; --exit:#45b98a; --summary:#e0a24d; --code-bg:#22262f; }} }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,system-ui,sans-serif;line-height:1.55}}
.wrap{{max-width:900px;margin:0 auto;padding:28px 20px 80px}}
.mono,pre{{font-family:"IBM Plex Mono",ui-monospace,"SF Mono",monospace}}
h1{{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:18px}}
.stats{{display:flex;flex-wrap:wrap;gap:8px 20px;padding:14px 16px;background:var(--surface);
  border:1px solid var(--hair);border-radius:10px;margin-bottom:14px;font-size:13px}}
.stat b{{font-variant-numeric:tabular-nums}}
.stat .k{{color:var(--faint)}}
.chartwrap{{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
  padding:14px 16px 8px;margin-bottom:22px}}
.chartwrap .cap{{font-size:12px;color:var(--faint);margin-bottom:6px}}
.chart{{width:100%;height:140px;display:block}}
.chart .area{{fill:color-mix(in srgb,var(--accent) 12%,transparent);stroke:none}}
.chart .line{{fill:none;stroke:var(--accent);stroke-width:1.6}}
.chart .edit-tick{{stroke:#d1495b;stroke-width:1.4;stroke-dasharray:3 2}}
.card{{background:var(--surface);border:1px solid var(--hair);border-left-width:3px;
  border-radius:9px;padding:12px 14px;margin-bottom:10px}}
.chead{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}}
.rlabel{{font-size:11.5px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}}
.toks{{font-size:11px;color:var(--faint);font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.role-system{{border-left-color:var(--sys)}} .role-system .rlabel{{color:var(--sys)}}
.role-user{{border-left-color:var(--task)}} .role-user .rlabel{{color:var(--task)}}
.role-assistant{{border-left-color:var(--agent)}} .role-assistant .rlabel{{color:var(--agent)}}
.role-tool{{border-left-color:var(--obs)}} .role-tool .rlabel{{color:var(--obs)}}
.role-exit{{border-left-color:var(--exit)}} .role-exit .rlabel{{color:var(--exit)}}
.role-summary{{border-left-color:var(--summary);background:color-mix(in srgb,var(--summary) 7%,var(--surface))}}
.role-summary .rlabel{{color:var(--summary)}}
.prose{{white-space:pre-wrap;font-size:13.5px}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--code-bg);
  border-radius:6px;padding:10px 12px;margin:6px 0 0;overflow-x:auto}}
pre .lbl{{display:block;font-size:10px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);margin-bottom:6px}}
.cmd{{border-left:2px solid var(--agent)}}
.patch{{border-left:2px solid var(--exit)}}
.pill{{display:inline-block;font-size:11px;font-weight:600;font-family:"IBM Plex Mono",monospace;
  padding:2px 8px;border-radius:20px;background:color-mix(in srgb,var(--exit) 15%,transparent);color:var(--exit)}}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
</head><body><div class="wrap">
  <h1>{html.escape(iid)}</h1>
  <div class="sub mono">{html.escape(str(model))}</div>
  <div class="stats">
    <span class="stat"><span class="k">exit</span> <b>{html.escape(str(exit_status))}</b></span>
    <span class="stat"><span class="k">model calls</span> <b>{api_calls}</b></span>
    <span class="stat"><span class="k">peak context</span> <b>~{peak:,}</b> tok</span>
    <span class="stat"><span class="k">messages</span> <b>{len(msgs)}</b></span>
    {mem_note}
  </div>
  <div class="chartwrap">
    <div class="cap">Live-context size per message (~tokens){' · red = compression' if edit_steps else ' · no memory management → grows unbounded'}</div>
    {chart}
  </div>
  {"".join(cards)}
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", nargs="?", help="path to a *.traj.json")
    ap.add_argument("--run-dir", help="render every *.traj.json under this dir")
    ap.add_argument("-o", "--out", default=None, help="output .html file, or dir with --run-dir")
    args = ap.parse_args()

    if args.run_dir:
        out_dir = Path(args.out or "site/traj")
        out_dir.mkdir(parents=True, exist_ok=True)
        trajs = sorted(glob.glob(os.path.join(args.run_dir, "**", "*.traj.json"), recursive=True))
        for tp in trajs:
            tp = Path(tp)
            (out_dir / f"{tp.stem}.html").write_text(render(tp))
        print(f">>> rendered {len(trajs)} trajectories -> {out_dir}/")
    else:
        if not args.traj:
            ap.error("give a traj.json path or --run-dir")
        tp = Path(args.traj)
        out = Path(args.out) if args.out else tp.with_suffix(".html")
        out.write_text(render(tp))
        print(f">>> {out}")


if __name__ == "__main__":
    main()
