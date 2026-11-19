#!/usr/bin/env python3
"""A2 (summarize) vs A3 (acm): is the Pass@1 gap a harness effect or noise?

Three questions, answered from runs/ and logs/apptainer_eval/ only (no GPU):

1. Noise floor. Qwen3.5-9B at 64k was run twice per arm with the same config
   (344xx: step_limit 250, 350xx: step_limit 500). Two runs of the SAME arm
   disagree on some instances; that disagreement is the noise floor. If A2-vs-A3
   disagreement is no larger, the arm gap is indistinguishable from re-rolling.
2. Decomposition. An instance where neither arm ever reached the compression
   trigger saw NO memory mechanism at all; A2 and A3 differ there only in the
   prompt (system addendum, two tool schemas, [CURRENT CONTEXT TOKEN] marker).
   Splitting the gap into "never compressed" vs "compressed" instances separates
   prompt effects from mechanism effects.
3. Divergence. How early do A2 and A3 trajectories for the same instance part
   ways, compared with two runs of the same arm?

Usage: python scripts/analyze_a2_a3.py [--json out.json]
"""
import argparse, glob, json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "logs" / "apptainer_eval"

# (label, A2 run, A3 run[, A1 run]) — only cells that have been scored
CELLS = {
    "qwen 64k s250":  ("qwen35-9b_summarize_344614", "qwen35-9b_acm_344615", "qwen35-9b_none_344613"),
    "qwen 64k s500":  ("qwen35-9b_summarize_350423", "qwen35-9b_acm_350424", "qwen35-9b_none_350422"),
    "qwen 32k s500":  ("qwen35-9b-32k_summarize_350420", "qwen35-9b-32k_acm_350421", "qwen35-9b-32k_none_350419"),
    "qwen 128k s500": ("qwen35-9b-128k_summarize_350426", "qwen35-9b-128k_acm_350427", "qwen35-9b-128k_none_350425"),
    "mimo 64k s250":  ("mimo-9b_summarize_344617", "mimo-9b_acm_344618", "mimo-9b_none_344616"),
    "gemma 64k s250": ("gemma4-12b-it_summarize_344620", "gemma4-12b-it_acm_344621", "gemma4-12b-it_none_344619"),
}
# same arm, same budget, run twice (step limit 250 vs 500)
REPLICATES = {
    "A1 none":      ("qwen35-9b_none_344613", "qwen35-9b_none_350422"),
    "A2 summarize": ("qwen35-9b_summarize_344614", "qwen35-9b_summarize_350423"),
    "A3 acm":       ("qwen35-9b_acm_344615", "qwen35-9b_acm_350424"),
}


def instances(run):
    return sorted(p.parent.name for p in (ROOT / "runs" / run).glob("*/memory_trace.json"))


def resolved(run):
    out = set()
    for rp in glob.glob(str(EVAL / run / "*" / "report.json")):
        try:
            d = json.load(open(rp))
        except Exception:
            continue
        iid = Path(rp).parent.name
        v = d.get(iid, d)
        if isinstance(v, dict) and v.get("resolved"):
            out.add(iid)
    return out


def trace(run, iid):
    try:
        return json.load(open(ROOT / "runs" / run / iid / "memory_trace.json"))
    except Exception:
        return None


def exit_status(run, iid):
    try:
        return json.load(open(ROOT / "runs" / run / iid / f"{iid}.traj.json")).get("info", {}).get("exit_status", "?")
    except Exception:
        return "?"


def assistant_actions(run, iid, n=60):
    """First n bash commands the agent issued (from the uncompressed history)."""
    p = ROOT / "runs" / run / iid / "raw_messages.json"
    try:
        msgs = json.load(open(p))
        if isinstance(msgs, dict):
            msgs = msgs.get("instance_messages", [])
    except Exception:
        return None
    acts = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            f = tc.get("function", {}) if isinstance(tc, dict) else {}
            acts.append((f.get("name"), f.get("arguments")))
        if len(acts) >= n:
            break
    return acts[:n]


def first_divergence(a, b):
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def flips(r1, r2, ids):
    a = {i for i in ids if i in r1 and i not in r2}
    b = {i for i in ids if i in r2 and i not in r1}
    return a, b


def mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(r1, r2, ids, B=10000, seed=0):
    rng = random.Random(seed)
    ids = list(ids)
    d = [(i in r1) - (i in r2) for i in ids]
    stats = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(B))
    return stats[int(0.025 * B)], stats[int(0.975 * B)]


def compressed(run, iid):
    t = trace(run, iid)
    return bool(t and t.get("n_edits", 0) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--no-divergence", action="store_true")
    args = ap.parse_args()
    out = {}

    print("## 1. Noise floor: same arm run twice (Qwen3.5-9B, 64k, step 250 vs 500)\n")
    print("| arm | Pass@1 run1 | Pass@1 run2 | flips 1->0 | flips 0->1 | discordant | McNemar p |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    out["replicates"] = {}
    for arm, (r1, r2) in REPLICATES.items():
        ids = sorted(set(instances(r1)) & set(instances(r2)))
        R1, R2 = resolved(r1), resolved(r2)
        a, b = flips(R1, R2, ids)
        p = mcnemar_p(len(a), len(b))
        print(f"| {arm} | {len(R1 & set(ids))/len(ids):.3f} | {len(R2 & set(ids))/len(ids):.3f} "
              f"| {len(a)} | {len(b)} | {len(a)+len(b)} ({(len(a)+len(b))/len(ids):.0%}) | {p:.2f} |")
        out["replicates"][arm] = dict(n=len(ids), only1=sorted(a), only2=sorted(b), p=p)

    print("\n## 2. A2 vs A3 per cell, split by whether the memory mechanism ever fired\n")
    print("| cell | n | A2 | A3 | A3-A2 | 95% CI | McNemar p | discordant | "
          "never-compressed: n, A2, A3 | compressed: n, A2, A3 |")
    print("|---|--:|--:|--:|--:|---|--:|--:|---|---|")
    out["cells"] = {}
    for cell, (a2, a3, *_) in CELLS.items():
        ids = sorted(set(instances(a2)) & set(instances(a3)))
        if not ids:
            continue
        R2, R3 = resolved(a2), resolved(a3)
        if not R2 and not R3:
            print(f"| {cell} | (unscored) |")
            continue
        only2, only3 = flips(R2, R3, ids)
        lo, hi = boot_ci(R3, R2, ids)
        neither = [i for i in ids if not compressed(a2, i) and not compressed(a3, i)]
        either = [i for i in ids if i not in set(neither)]
        f = lambda R, S: sum(1 for i in S if i in R)
        n2, n3 = f(R2, ids), f(R3, ids)
        print(f"| {cell} | {len(ids)} | {n2} | {n3} | {(n3-n2)/len(ids):+.3f} | [{lo:+.3f}, {hi:+.3f}] "
              f"| {mcnemar_p(len(only2), len(only3)):.2f} | {len(only2)+len(only3)} "
              f"| {len(neither)}: {f(R2,neither)} vs {f(R3,neither)} "
              f"| {len(either)}: {f(R2,either)} vs {f(R3,either)} |")
        out["cells"][cell] = dict(n=len(ids), a2=n2, a3=n3, only_a2=sorted(only2), only_a3=sorted(only3),
                                  neither=len(neither), neither_a2=f(R2, neither), neither_a3=f(R3, neither),
                                  either=len(either), either_a2=f(R2, either), either_a3=f(R3, either))

    print("\n## 3. A3-only events on instances where the arms disagree (Qwen 64k s500)\n")
    a2, a3 = CELLS["qwen 64k s500"][:2]
    R2, R3 = resolved(a2), resolved(a3)
    ids = sorted(set(instances(a2)) & set(instances(a3)))
    print("| instance | won by | A2 exit | A2 edits | A3 exit | A3 edits | A3 nudges | A3 stop | A3 tools |")
    print("|---|---|---|--:|---|--:|--:|---|---|")
    for i in ids:
        if (i in R2) == (i in R3):
            continue
        t2, t3 = trace(a2, i) or {}, trace(a3, i) or {}
        print(f"| {i} | {'A2' if i in R2 else 'A3'} | {exit_status(a2, i)} | {t2.get('n_edits')} "
              f"| {exit_status(a3, i)} | {t3.get('n_edits')} | {t3.get('compress_prompts')} "
              f"| {t3.get('stop_prompted')} | {t3.get('tool_calls')} |")

    if not args.no_divergence:
        print("\n## 4. Step at which two trajectories first issue a different command\n")
        print("| pair | n | median first-divergence step | identical first action | identical first 5 |")
        print("|---|--:|--:|--:|--:|")
        pairs = {
            "A2 vs A2 (rerun)": REPLICATES["A2 summarize"],
            "A3 vs A3 (rerun)": REPLICATES["A3 acm"],
            "A1 vs A1 (rerun)": REPLICATES["A1 none"],
            "A2 vs A3 (s500)": CELLS["qwen 64k s500"][:2],
            "A1 vs A2 (s500)": (CELLS["qwen 64k s500"][2], CELLS["qwen 64k s500"][0]),
        }
        out["divergence"] = {}
        for name, (x, y) in pairs.items():
            ids = sorted(set(instances(x)) & set(instances(y)))
            ds = [first_divergence(assistant_actions(x, i), assistant_actions(y, i)) for i in ids]
            ds = [d for d in ds if d is not None]
            if not ds:
                continue
            ds_sorted = sorted(ds)
            med = ds_sorted[len(ds) // 2]
            print(f"| {name} | {len(ds)} | {med} | {sum(d >= 1 for d in ds)/len(ds):.0%} "
                  f"| {sum(d >= 5 for d in ds)/len(ds):.0%} |")
            out["divergence"][name] = ds

    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
