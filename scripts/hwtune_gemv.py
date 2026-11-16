"""Stage 1 of data/hwtune/PROTOCOL.md: heuristic vs autotuning vs hardware-informed rewrite on int4 GEMV.

  python scripts/hwtune_gemv.py          # writes data/hwtune/gemv.json and gemv.txt
"""
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def noise(us):
    return 0.079 if us < 8 else 0.012 if us <= 20 else 0.015


def load():
    suite, rows = None, []
    for line in (ROOT / "docs/optimize_benchmark.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"## suite (\w)", line)
        if m:
            suite = m[1]
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if suite and len(c) == 9 and "/" in c[0] and re.match(r"[\d.]+$", c[1]):
            t = float(c[1])
            blind = float(c[3]) if c[3] else 1.0
            rows.append(dict(suite=suite, id=c[0], shape=c[0].split("/")[0], cfg="/".join(c[0].split("/")[1:]),
                             t=t, t_hw=t / blind))
    g = defaultdict(list)
    for r in rows:
        g[(r["suite"], r["shape"])].append(r)
    return g


def heuristic(cs):
    lc = [c for c in cs if "/lane_contig/" in "/" + c["cfg"] + "/"]
    if lc:
        return max(lc, key=lambda c: int(re.match(r"R(\d+)", c["cfg"])[1]))
    return cs[0]


def autotune_k(cs, k):
    ts = [c["t"] for c in cs]
    return statistics.mean(min(s) for s in itertools.combinations(ts, k))


def main():
    G = load()
    out, lines = {}, []
    for (s, shape), cs in sorted(G.items()):
        h = heuristic(cs)
        best = min(cs, key=lambda c: c["t"])
        reach = min(cs, key=lambda c: min(c["t"], c["t_hw"]))
        t_reach = min(reach["t"], reach["t_hw"])
        g = best["t"] / t_reach
        rec = dict(suite=s, shape=shape, n=len(cs), heuristic=h["cfg"], t_heur=h["t"], t_heur_hw=h["t_hw"],
                   autotune_best=best["cfg"], t_auto=best["t"], hw_reach=reach["cfg"] + (
                       "+rewrite" if reach["t_hw"] < reach["t"] else ""), t_reach=t_reach,
                   g_reach_vs_auto=g, win=g - 1 > noise(best["t"]),
                   auto_vs_heur=h["t"] / best["t"], reach_vs_heur=h["t"] / t_reach,
                   auto_k={k: autotune_k(cs, k) for k in range(1, len(cs) + 1)})
        out[f"{s}:{shape}"] = rec
    lines.append(f"{'suite:shape':14s} {'n':>2s} {'heuristic':22s} {'t_heur':>7s} {'autotune best':24s} {'t_auto':>7s} "
                 f"{'hw reach':32s} {'t_hw':>7s} {'hw/auto':>7s} win")
    for k, r in out.items():
        lines.append(f"{k:14s} {r['n']:2d} {r['heuristic']:22s} {r['t_heur']:7.2f} {r['autotune_best']:24s} "
                     f"{r['t_auto']:7.2f} {r['hw_reach']:32s} {r['t_reach']:7.2f} {r['g_reach_vs_auto']:7.3f} "
                     f"{'WIN' if r['win'] else ''}")
    lines.append("")
    gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))  # noqa: E731
    summ = {}
    for name, suites in [("A (post hoc)", "A"), ("B (held out)", "B"), ("C (held out)", "C"), ("B+C", "BC")]:
        rs = [r for r in out.values() if r["suite"] in suites]
        d = dict(shapes=len(rs), hw_vs_auto=gm([r["g_reach_vs_auto"] for r in rs]),
                 wins=sum(r["win"] for r in rs), auto_vs_heur=gm([r["auto_vs_heur"] for r in rs]),
                 hw_vs_heur=gm([r["reach_vs_heur"] for r in rs]),
                 heur_plus_rewrite_vs_heur=gm([r["t_heur"] / min(r["t_heur"], r["t_heur_hw"]) for r in rs]),
                 auto2_vs_heur=gm([r["t_heur"] / r["auto_k"][2] for r in rs]))
        summ[name] = d
        lines.append(f"{name:13s} shapes {d['shapes']:2d}  autotune(all) vs heuristic {d['auto_vs_heur']:.3f}  "
                     f"autotune(2) vs heuristic {d['auto2_vs_heur']:.3f}  heuristic+rewrite vs heuristic "
                     f"{d['heur_plus_rewrite_vs_heur']:.3f}  hw_reach vs heuristic {d['hw_vs_heur']:.3f}  "
                     f"hw_reach vs autotune(all) {d['hw_vs_auto']:.3f}  wins {d['wins']}/{d['shapes']}")
    bc = summ["B+C"]
    verdict = bc["hw_vs_auto"] >= 1.02 and bc["wins"] * 2 >= bc["shapes"]
    lines.append(f"\nH1 on held-out GEMV (geomean >= 1.02 and wins >= half): {'SUPPORTED' if verdict else 'REFUTED'}")
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "data/hwtune/gemv.txt").write_text(txt + "\n")
    (ROOT / "data/hwtune/gemv.json").write_text(json.dumps({"shapes": out, "summary": summ, "h1": verdict}, indent=1))


if __name__ == "__main__":
    main()
