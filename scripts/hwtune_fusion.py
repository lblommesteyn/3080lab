"""Stage 2 of data/hwtune/PROTOCOL.md on launch structure: fixed rules vs autotuning vs the launch-cost model.

Each PolyBench program picks one of {base, graph, fused, fused_graph}. y = log(t_base / t_variant).
  python scripts/hwtune_fusion.py   # data/hwtune/fusion.txt, fusion.json
"""
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def groups(path):
    g = defaultdict(dict)
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            g[r["group"]][r["transform"]] = (r["y"], r["features"]["hw_pred"])
    for v in g.values():
        v["base"] = (0.0, 0.0)
    return g


def evaluate(g, label):
    gm = lambda xs: math.exp(statistics.mean(xs))  # noqa: E731 (xs are log speedups)
    V = ["base", "graph", "fused", "fused_graph"]
    best = {p: max(v[0] for v in d.values()) for p, d in g.items()}
    res = {}
    for f in V[1:]:
        res[f"always {f}"] = gm([d[f][0] if f in d else 0.0 for d in g.values()])
    for k in (1, 2, 3, 4):
        res[f"autotune({k})"] = gm([statistics.mean(max(d[v][0] for v in s)
                                                    for s in itertools.combinations([v for v in V if v in d], k))
                                    for d in g.values()])
    for k in (1, 2):
        vals = []
        for d in g.values():
            order = sorted([v for v in V if v in d], key=lambda v: -d[v][1])
            vals.append(max(d[v][0] for v in order[:k]))
        res[f"hw_model top-{k}"] = gm(vals)
    res["oracle"] = gm(list(best.values()))
    lines = [f"## {label}: {len(g)} programs; speedup over base, geomean (oracle = autotune(4))"]
    lines += [f"  {k:22s} {v:.3f}" for k, v in res.items()]
    return res, lines


def main():
    out, lines = {}, []
    out["polybench_fusion (held out)"], l1 = evaluate(groups(ROOT / "data/optsel/corpus_polybench_fusion.jsonl"),
                                                      "polybench_fusion, full size (held-out family)")
    lines += l1
    d = json.loads((ROOT / "data/optsel/demo_eval.json").read_text())
    g = defaultdict(dict)
    for r in d["rows"]:
        g[r["group"]][r["transform"]] = (r["y"], r["features"]["hw_pred"])
    for v in g.values():
        v["base"] = (0.0, 0.0)
    out["demo 1/4 size (never measured)"], l2 = evaluate(g, "PolyBench at 1/4 size (unseen size, job 2222)")
    lines += [""] + l2
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "data/hwtune/fusion.txt").write_text(txt + "\n")
    (ROOT / "data/hwtune/fusion.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
