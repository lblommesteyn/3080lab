"""Static headroom probe: can a different legal order and/or register coloring cut register-file read
cycles below ptxas's? (CPU only.)

Hot-loop RF read cycles per warp (lab/regalloc._loop_cost: two banks, one read per bank per cycle,
(bank, slot) reuse cache) for:
  ptxas            ptxas order, ptxas reuse flags
  identity_late    ptxas order, our reuse model (lab/resched._reuse_block)
  sb_late          scoreboard-aware order, our reuse model
  random[k]        k random legal orders (resched policy="random"), our reuse model
each also after annealed web recoloring (lab/regalloc.optimize). Random orders ignore latency, so
this bounds the RF side only; a schedule must still meet latencies.

usage: python scripts/ra_reuse_probe.py seed1008 seed1010 ... [orders=30] [out=data/ra/reuse_probe.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _site  # noqa: E402,F401  (venv packages under a bare python)

import torch  # noqa: E402,F401  (loads the nvrtc DLLs the toolchain needs)
from lab import regalloc, resched, sass, toolchain  # noqa: E402
from lab.experiments.randk import ReschedRandom  # noqa: E402


def rf_cost(cub: bytes, steps: int) -> tuple[int, int, int]:
    g = regalloc.build(cub, "k")
    W = regalloc.webs(g)
    body = sass.loop_body(g["ins"])
    base = regalloc._loop_cost(g, body, W["occ_web"], {w: w[1] for w in W["members"]})
    best = regalloc.optimize(cub, "k", steps=steps)["best_cost"] if steps else base
    return len(body), base, best


def main():
    args = sys.argv[1:]
    kv = dict(a.split("=", 1) for a in args if "=" in a)
    want = [a for a in args if "=" not in a]
    n_orders, steps = int(kv.get("orders", 30)), int(kv.get("steps", 3000))
    out = ROOT / kv.get("out", "data/ra/reuse_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    exp = ReschedRandom()
    rows = []
    for v in exp.variants({"n": 40}):
        s = v.label.split("/")[0]
        if v.params["policy"] != "ptxas" or s not in want:
            continue
        cub = toolchain.build(exp.source(v)).cubin
        row = {"name": s}
        n, b, r = rf_cost(cub, steps)
        row.update(n=n, ptxas=b, ptxas_recolor=r)
        for name, pol in (("identity_late", "identity"), ("sb_late", "sb")):
            c = resched.reschedule(cub, "k", policy=pol, late_entry=True, sticky_waits=True)[0]
            _, b, r = rf_cost(c, steps)
            row[name], row[name + "_recolor"] = b, r
        rnd = []
        for k in range(n_orders):
            try:
                c = resched.reschedule(cub, "k", policy="random", seed=k, late_entry=True, sticky_waits=True)[0]
            except Exception as e:                  # an order the encoder refuses: skip it
                rnd.append({"seed": k, "error": repr(e)[:80]})
                continue
            _, b, _ = rf_cost(c, 0)
            rnd.append({"seed": k, "rf": b})
        ok = [x for x in rnd if "rf" in x]
        row["random_min"] = min(x["rf"] for x in ok) if ok else None
        row["random_median"] = sorted(x["rf"] for x in ok)[len(ok) // 2] if ok else None
        if ok:                                      # recolor the best random order too
            k = min(ok, key=lambda x: x["rf"])["seed"]
            c = resched.reschedule(cub, "k", policy="random", seed=k, late_entry=True, sticky_waits=True)[0]
            row["random_min_recolor"] = rf_cost(c, steps)[2]
        row["random"] = rnd
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "random"}), flush=True)
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
