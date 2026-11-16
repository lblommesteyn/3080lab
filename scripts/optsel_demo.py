"""Guided decision demo (PROTOCOL.md addendum): pick a configuration per PolyBench program at 1/4 size.

Predictors are fit on all five benchmark families (corpus_*.jsonl); this run's measurements are used only
for the base time (an input) and for scoring the choices afterwards.

  XK_KDIR=vendor/ext_kernels_s4 CUDA_VISIBLE_DEVICES=-1 python scripts/optsel_demo.py [in=data/optsel/demo_small.json]
"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _site  # noqa: E402,F401  (venv packages under a bare python)
import torch  # noqa: E402,F401
os.environ.setdefault("XK_KDIR", str(ROOT / "vendor" / "ext_kernels_s4"))
import optsel_eval as E  # noqa: E402
import xkernel_bench as X  # noqa: E402
from lab import optsel, xkernel  # noqa: E402
from optsel_build import gap_constants  # noqa: E402

OPTS = ["graph", "fused", "fused_graph"]


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    d = json.loads((ROOT / kv.get("in", "data/optsel/demo_small.json")).read_text())
    gap = gap_constants(json.loads((ROOT / "data/optsel/launch_gap.json").read_text()))
    progs = X.programs()
    train = [r for r in E.load() if r["transform"] != "aa"]
    rows, aa = [], {}
    for p, x in d.items():
        vb = x["variants"]["base"]
        if "us" not in vb:
            continue
        tb = statistics.median(vb["us"])
        if "us" in x["variants"].get("base_dup", {}):
            aa[p] = math.log(tb / statistics.median(x["variants"]["base_dup"]["us"]))
        args = [set(a for a, (_, _, isp) in zip(l.args, xkernel.analyze(l.src).params) if isp)
                for l in progs[p]["launches"]]
        tot = sum(len(a) for a in args)
        shared = sum(len(a & set().union(*(args[:i] + args[i + 1:]))) for i, a in enumerate(args))
        for v in OPTS:
            vc = x["variants"].get(v, {})
            if "us" not in vc:
                continue
            lb, lc = x["launches"]["base"], x["launches"][v]
            f = optsel.fusion_decision(tb, lb, lc, v.endswith("graph"), gap, shared / max(tot, 1) if lc < lb else 0.0)
            rows.append({"family": "demo", "group": p, "transform": v, "y": math.log(tb / statistics.median(vc["us"])),
                         "t_base_us": tb, "features": f})
    preds, lam = E.predictors(train, rows)
    preds["always_fused_graph"] = (np.array([1.0 if r["transform"] == "fused_graph" else -1.0 for r in rows]),
                                   np.full(len(rows), 0.5))
    res = {"aa": aa, "lambda": lam, "gap": gap, "rows": rows, "predictors": {}}
    print(f"A/A (base vs base_dup) per program: " + " ".join(f"{p}:{v:+.3f}" for p, v in aa.items()))
    print(f"{'predictor':20s} {'realised':>8s} {'regret':>7s} {'slower':>6s}  choices")
    for name, (s, pr) in preds.items():
        app = s > E.TAU
        ch, rg, groups = E.group_choice(rows, s, app)
        picks = {}
        for g in groups:
            idx = [i for i, r in enumerate(rows) if r["group"] == g and app[i]]
            picks[g] = rows[max(idx, key=lambda i: s[i])]["transform"] if idx else "base"
        slower = int(sum(c < -E.TAU for c in ch))
        res["predictors"][name] = {"realised_geomean": math.exp(ch.mean()), "mean_regret": float(rg.mean()),
                                   "slower_than_base": slower, "picks": picks, "scores": [float(z) for z in s]}
        print(f"{name:20s} {math.exp(ch.mean()):8.4f} {rg.mean():7.4f} {slower:6d}  " +
              " ".join(f"{g}:{picks[g]}" for g in groups))
    print("measured y: " + " ".join(f"{r['group']}/{r['transform']}:{r['y']:+.3f}(hw {r['features']['hw_pred']:+.3f})" for r in rows))
    (ROOT / "data/optsel/demo_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
