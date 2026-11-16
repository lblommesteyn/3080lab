"""Addendum 1 of data/hwtune/PROTOCOL.md: can cost model v3 pick the GEMV configuration without measuring?

  python scripts/hwtune_h2.py      # data/hwtune/h2.txt, h2.json
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hwtune_gemv import autotune_k, heuristic, load  # noqa: E402


def spearman(a, b):
    ra = {i: r for r, i in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    rb = {i: r for r, i in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    n = len(a)
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def main():
    G = load()
    P = json.loads((ROOT / "data/hwtune/preds_v3.json").read_text())
    seen = {}
    for (s, _), cs in G.items():
        for c in cs:
            assert c["id"] not in seen or seen[c["id"]] == s, f"id collision {c['id']}"
            seen[c["id"]] = s
    rows, lines = {}, []
    for (s, shape), cs in sorted(G.items()):
        best = min(c["t"] for c in cs)
        order = sorted(cs, key=lambda c: P[c["id"]]["t_ns"])
        r = dict(suite=s, n=len(cs),
                 heuristic=heuristic(cs)["t"] / best, model1=order[0]["t"] / best,
                 model2=min(order[0]["t"], order[1]["t"]) / best,
                 auto1=autotune_k(cs, 1) / best, auto2=autotune_k(cs, 2) / best,
                 rho=spearman([P[c["id"]]["t_ns"] for c in cs], [c["t"] for c in cs]),
                 pick=order[0]["cfg"], best=min(cs, key=lambda c: c["t"])["cfg"],
                 pred_err=statistics.median(abs(P[c["id"]]["t_ns"] / 1e3 / c["t"] - 1) for c in cs))
        rows[f"{s}:{shape}"] = r
        lines.append(f"{s}:{shape:6s} n={r['n']} heur {r['heuristic']:.3f} model1 {r['model1']:.3f} "
                     f"model2 {r['model2']:.3f} auto1 {r['auto1']:.3f} auto2 {r['auto2']:.3f} rho {r['rho']:+.2f} "
                     f"pred_err {r['pred_err']:.0%}  model picks {r['pick']}, best {r['best']}")
    gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))  # noqa: E731
    summ = {}
    lines.append("")
    lines.append("regret = t(policy) / t(autotune(all)), geomean over shapes (1.000 = always the best)")
    for name, ss in [("A (post hoc)", "A"), ("B (held out)", "B"), ("C (held out)", "C"), ("B+C", "BC")]:
        rs = [r for r in rows.values() if r["suite"] in ss]
        d = {k: gm([r[k] for r in rs]) for k in ("heuristic", "model1", "model2", "auto1", "auto2")}
        d.update(shapes=len(rs), rho=statistics.mean(r["rho"] for r in rs),
                 model1_exact=sum(r["model1"] == 1.0 for r in rs))
        summ[name] = d
        lines.append(f"{name:13s} heuristic {d['heuristic']:.3f}  model(1) {d['model1']:.3f}  model(2) {d['model2']:.3f}  "
                     f"autotune(1) {d['auto1']:.3f}  autotune(2) {d['auto2']:.3f}  mean rho {d['rho']:+.2f}  "
                     f"model(1) exact {d['model1_exact']}/{d['shapes']}")
    bc = summ["B+C"]
    ok = bc["model1"] < bc["heuristic"] and bc["model1"] < bc["auto2"] and bc["model2"] <= 1.01
    lines.append(f"\nH2 on held-out GEMV: {'SUPPORTED' if ok else 'REFUTED'}")
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "data/hwtune/h2.txt").write_text(txt + "\n")
    (ROOT / "data/hwtune/h2.json").write_text(json.dumps({"shapes": rows, "summary": summ, "h2": ok}, indent=1))


if __name__ == "__main__":
    main()
