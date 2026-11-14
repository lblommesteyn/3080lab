"""Per-kernel speedups with bootstrap 95% confidence intervals from a suite_run JSON.

A kernel counts as improved only if the whole interval is above 1.01 (the suite's run-to-run noise
is about 1%), regressed only if it is entirely below 0.99; everything else is a tie.

usage: python scripts/ra_report.py data/suite/ra_suite.json [base=ptxas] [boot=2000]
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys


def ci_ratio(base: list, arm: list, boot: int, rng: random.Random) -> tuple[float, float, float]:
    """median(base)/median(arm) and a percentile bootstrap interval, resampling rounds independently."""
    r = statistics.median(base) / statistics.median(arm)
    bs = []
    for _ in range(boot):
        b = [rng.choice(base) for _ in base]
        a = [rng.choice(arm) for _ in arm]
        bs.append(statistics.median(b) / statistics.median(a))
    bs.sort()
    return r, bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1]


def main():
    path = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    base, boot = kv.get("base", "ptxas"), int(kv.get("boot", 2000))
    d = json.load(open(path))
    rng = random.Random(0)
    arms = [v for v in d["variants"] if v != base]
    summary = {v: {"ratios": [], "improved": [], "regressed": []} for v in arms}
    print(f"{'case':14s} " + " ".join(f"{v:>26s}" for v in arms))
    for name, c in d["cases"].items():
        vb = c["variants"].get(base, {})
        if "us" not in vb:
            continue
        cells = []
        for v in arms:
            va = c["variants"].get(v, {})
            if "us" not in va:
                cells.append(f"{va.get('status', '-'):>26s}")
                continue
            r, lo, hi = ci_ratio(vb["us"], va["us"], boot, rng)
            s = summary[v]
            s["ratios"].append(r)
            if lo > 1.01:
                s["improved"].append((name, round(r, 3)))
            if hi < 0.99:
                s["regressed"].append((name, round(r, 3)))
            cells.append(f"{r:7.3f} [{lo:.3f}, {hi:.3f}]".rjust(26))
        print(f"{name:14s} " + " ".join(cells))
    for v, s in summary.items():
        if not s["ratios"]:
            continue
        g = math.exp(statistics.fmean(map(math.log, s["ratios"])))
        print(f"{v}: geomean {g:.4f} over {len(s['ratios'])}, worst {min(s['ratios']):.3f}, "
              f"improved (CI > 1.01) {len(s['improved'])} {s['improved']}, regressed (CI < 0.99) {len(s['regressed'])} {s['regressed']}")


if __name__ == "__main__":
    main()
