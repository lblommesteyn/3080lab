"""Static headroom for joint register allocation + scheduling (CPU only, no GPU).

For every hot loop in the corpus (40 random kernels + the 19/20-case suite):
  bank   register-read cycles of one warp through the loop (two banks, (bank, slot) reuse cache,
         lab/regalloc._loop_cost) for ptxas's allocation vs the best annealed web recoloring.
         Lower bound = instruction count (one read cycle each).
  deps   single-warp critical path of the loop body with every dependence (RAW + WAR + WAW) vs
         RAW only, using lab/resched.latency for RAW and 1 cycle for WAR/WAW. The gap is what
         renaming could buy a scheduler on one warp.
  sim    lab/model.simulate (v1: pipes, per-bank RF ports, reuse cache) of ptxas vs the recolored
         loop at the measured occupancy, as a predicted speedup.

usage: python scripts/ra_headroom.py [randk] [suite] [out=data/ra/headroom.json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _site  # noqa: E402,F401  (venv packages under a bare python)

import torch  # noqa: E402,F401  (loads the nvrtc DLLs the toolchain needs)
from lab import model, regalloc, resched, sass, toolchain  # noqa: E402
from lab import operands as opnd  # noqa: E402


def occupancy_from_log(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="replace")
    occ = {}
    for m in re.finditer(r"^\[(seed\d+)/ptxas\]\n((?:  .*\n)*)", txt, re.M):
        o = re.search(r"= (\d+) warps/SM", m.group(2))
        if o:
            occ[m.group(1)] = int(o.group(1))
    return occ


def crit_paths(body: list, cache: dict) -> tuple[int, int]:
    table = resched.fixed_latency_table(body, cache)
    res = [resched._resources(x, cache) for x in body]
    n = len(body)
    full, raw = [0] * n, [0] * n
    last_def: dict = {}
    readers: dict = {}
    for v in range(n):
        if res[v] is None:                         # unmodelled: serialize behind everything
            full[v] = raw[v] = max(full + [0])
            last_def.clear(); readers.clear()
            continue
        d, u = res[v]
        f = r = 0
        if resched._memory(body[v]):                # memory order: no alias facts, so a load stays
            for p in range(v):                      # behind earlier stores and a store behind every
                if resched._memory(body[p]) and (resched._writes_mem(body[p]) or resched._writes_mem(body[v])):
                    f = max(f, full[p] + 1); r = max(r, raw[p] + 1)   # earlier memory op
        for x in u:
            if x in last_def:
                p = last_def[x]
                lat = resched.var_latency(body[p]) if body[p].wbar != 7 else resched.latency(body[p], table, body[v])
                f = max(f, full[p] + lat); r = max(r, raw[p] + lat)
        for x in d:
            if x in last_def:                       # WAW
                f = max(f, full[last_def[x]] + 1)
            for p in readers.get(x, ()):            # WAR
                f = max(f, full[p] + 1)
        full[v], raw[v] = f, r
        for x in u:
            readers.setdefault(x, []).append(v)
        for x in d:
            last_def[x] = v
            readers[x] = []
    return max(full, default=0) + 1, max(raw, default=0) + 1


def analyze(name: str, cub: bytes, warps_sm: int | None, steps: int) -> dict:
    ins = sass.parse(toolchain.disassemble(cub))
    body = sass.loop_body(ins)
    row = {"name": name, "n_loop": len(body), "warps_sm": warps_sm}
    if not body:
        row["skip"] = "no loop"
        return row
    cache = opnd._load_cache()
    cpf, cpr = crit_paths(body, cache)
    row.update(cp_all=cpf, cp_raw=cpr, stall_sum=sum(x.stall or 32 for x in body))
    try:
        r = regalloc.optimize(cub, "k", steps=steps)
        row.update(rf_ptxas=r["base_cost"], rf_best=r["best_cost"], hot_webs=r["hot_webs"])
        if warps_sm and r["best_cost"] < r["base_cost"]:
            new = regalloc.apply(cub, r, "k")
            b2 = sass.loop_body(sass.parse(toolchain.disassemble(new)))
            p0 = model.simulate(body, warps=warps_sm, iters=1000, sim_iters=4)["per_iter"]
            p1 = model.simulate(b2, warps=warps_sm, iters=1000, sim_iters=4)["per_iter"]
            row["sim_speedup"] = p0 / p1
    except Exception as e:                          # refuse, record why
        row["ra_error"] = repr(e)[:160]
    return row


def main():
    args = sys.argv[1:]
    kv = dict(a.split("=", 1) for a in args if "=" in a)
    sets = [a for a in args if "=" not in a] or ["randk", "suite"]
    out = ROOT / kv.get("out", "data/ra/headroom.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = int(kv.get("steps", 6000))
    rows = []
    if "randk" in sets:
        from lab.experiments.randk import ReschedRandom
        occ = occupancy_from_log(ROOT / "data/ablation/job2054_exit_cycles.txt")
        exp = ReschedRandom()
        for v in exp.variants({"n": 40}):
            s = v.label.split("/")[0]
            if v.params["policy"] != "ptxas":
                continue
            rows.append(analyze(s, toolchain.build(exp.source(v)).cubin, occ.get(s), steps))
            print(json.dumps(rows[-1]), flush=True)
    if "suite" in sets:
        from lab import suite
        for c in suite.CASES:
            threads = 1
            for d in c.block:
                threads *= d
            # one block per SM is the conservative occupancy (suite kernels are mostly memory bound)
            rows.append(analyze("suite:" + c.name, toolchain.build(c.src).cubin, max(1, threads // 32), steps))
            print(json.dumps(rows[-1]), flush=True)
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
