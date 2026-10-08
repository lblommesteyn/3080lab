"""Score predictor v0 against every saved measurement (latest record per experiment).

Usage: python scripts/eval_model.py [experiment-prefix ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from lab import model

RESULTS = Path(__file__).resolve().parents[1] / "results"
SKIP = ("l1_replacement", "chase_l1_carveout")  # need a capacity/carveout model, not v0

latest = {}
for f in sorted(RESULTS.glob("*/record.json")):
    rec = json.loads(f.read_text())
    latest[rec["experiment"]] = (f.parent, rec)

args = sys.argv[1:]
version = 1
if args and args[0].startswith("--v"):
    version = int(args.pop(0)[3:])
prefixes = args
rows = []
for name, (d, rec) in sorted(latest.items()):
    if name.startswith(SKIP) or (prefixes and not name.startswith(tuple(prefixes))):
        continue
    for label, s in rec["summary"].items():
        art = s.get("kernel_artifact")
        if not art or not (d / f"{art}.sass.json").exists():
            continue
        p = s["params"]
        # Illegal programs (patched schedules that compute wrong answers, or that
        # corrupted their own clock registers) are not timing ground truth.
        if s.get("correctness") is False or s["metrics"]["cycles"]["median"] <= 0:
            continue
        if any(t.get("correct") is False for t in rec["trials"].get(label, [])):
            continue
        body = model.body_from_listing(d / f"{art}.sass.json")
        if not body:
            continue
        pred = model.simulate(body, warps=p.get("warps", 1), iters=p["iters"],
                              mem_level=model.mem_level_for(p, name), version=version)
        meas = s["metrics"]["cycles"]["median"]
        rows.append((name, label, meas, pred["cycles"], (pred["cycles"] - meas) / meas))

by_exp = defaultdict(list)
for r in rows:
    by_exp[r[0]].append(r)
print(f"{'experiment':<22} {'n':>3} {'median|err|':>11} {'max|err|':>9}")
for name, rs in by_exp.items():
    e = np.abs([r[4] for r in rs])
    print(f"{name:<22} {len(rs):>3} {np.median(e):>10.1%} {e.max():>9.1%}")
allerr = np.abs([r[4] for r in rows])
print(f"\nALL: n={len(rows)}  median |err| {np.median(allerr):.1%}  within 5%: {np.mean(allerr < 0.05):.0%}  "
      f"within 10%: {np.mean(allerr < 0.10):.0%}")
print("\nworst 15:")
for r in sorted(rows, key=lambda r: -abs(r[4]))[:15]:
    print(f"  {r[0]:<22} {r[1]:<14} meas {r[2]:>14,.0f}  pred {r[3]:>14,.0f}  {r[4]:+.1%}")
