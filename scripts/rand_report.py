"""Summarize a resched_rand* log: per-policy geomean of ptxas/policy cycles, and which kernels
genuinely improve or regress.

Each kernel's timing line gives cycles/op with its p05/p95 over trials. A kernel counts as improved
only if the policy's p95 is below 0.99x ptxas's p05 (the intervals separate by more than 1%), as
regressed only if its p05 is above 1.01x ptxas's p95. Everything else is a tie.

usage: python scripts/rand_report.py <log> [vs=identity_late] [out=...json]
"""
from __future__ import annotations

import json
import math
import re
import sys


def parse(txt: str) -> dict:
    d = {}
    for m in re.finditer(r"^\[(seed\d+)/(\w+)\]\n((?:  .*\n)*)", txt, re.M):
        c = re.search(r"cycles/op:\s*([\d.]+)\s*\[p05 ([\d.]+), p95 ([\d.]+)", m.group(3))
        t = re.search(r"throughput:\s*([\d.]+)", m.group(3))
        w = re.search(r"= (\d+) warps/SM", m.group(3))
        if c:
            d[(m.group(1), m.group(2))] = {"cyc": float(c[1]), "p05": float(c[2]), "p95": float(c[3]),
                                           "ipc": float(t[1]) if t else None, "warps_sm": int(w[1]) if w else None}
    return d


def main():
    path = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    txt = open(path, encoding="utf-8", errors="replace").read()
    d = parse(txt)
    seeds = sorted({s for s, _ in d})
    pols = sorted({p for _, p in d if p != "ptxas"})
    out = {"log": path, "policies": {}}
    for ref in ["ptxas"] + ([kv["vs"]] if "vs" in kv else []):
        for p in pols:
            if p == ref:
                continue
            rs, imp, reg = [], [], []
            for s in seeds:
                a, b = d.get((s, ref)), d.get((s, p))
                if not a or not b:
                    continue
                rs.append(a["cyc"] / b["cyc"])
                if b["p95"] < 0.99 * a["p05"]:
                    imp.append((s, round(a["cyc"] / b["cyc"], 3)))
                if b["p05"] > 1.01 * a["p95"]:
                    reg.append((s, round(a["cyc"] / b["cyc"], 3)))
            if not rs:
                continue
            g = math.exp(sum(map(math.log, rs)) / len(rs))
            print(f"{ref:>13s} / {p:20s} n={len(rs):2d} geomean {g:.4f} worst {min(rs):.3f} best {max(rs):.3f}  "
                  f"improved {len(imp)} regressed {len(reg)}")
            if imp or reg:
                print(f"{'':16s}improved {imp}\n{'':16s}regressed {reg}")
            out["policies"][f"{ref}/{p}"] = {"n": len(rs), "geomean": g, "worst": min(rs), "best": max(rs),
                                             "improved": imp, "regressed": reg}
    mism = re.findall(r"(?i)mismatch|wrong|FAIL", txt)
    print("mismatch/fail lines:", len(mism))
    out["mismatch_lines"] = len(mism)
    if "out" in kv:
        open(kv["out"], "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
