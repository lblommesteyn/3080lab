"""Optimization-selection features (Nov 14 benchmark, see data/optsel/PROTOCOL.md).

A decision is (baseline binary, candidate binary, launch context). Features describe the baseline and
the candidate-minus-baseline change, from static SASS plus the launch configuration; nothing here reads
a measured time of the candidate. `hw_pred` is the existing measured hardware model's predicted log
speedup (lab/model.simulate on the hot loop at the launch's warps/SM), used both as a feature and as a
baseline predictor on its own.
"""
from __future__ import annotations

import math

from . import model, resched, sass, toolchain
from . import operands as opnd

MEM_OPS = ("LDG", "STG", "LDS", "STS", "LD", "ST", "ATOM", "RED", "LDGSTS", "LDL", "STL")
_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = opnd._load_cache()
    return _CACHE


def loop_stats(body: list) -> dict:
    """Static description of one hot loop body (or the whole kernel when there is no loop)."""
    n = len(body)
    ops = [i.opcode.split(".")[0] for i in body]
    st = {"n": n, "stall_sum": sum(i.stall for i in body), "waits": sum(bin(i.wait).count("1") for i in body),
          "f_mem": sum(o in MEM_OPS for o in ops) / max(n, 1),
          "f_mufu": sum(o == "MUFU" for o in ops) / max(n, 1),
          "f_fp": sum(o in ("FFMA", "FADD", "FMUL", "HFMA2", "DFMA") for o in ops) / max(n, 1),
          "f_int": sum(o in ("IMAD", "IADD3", "LOP3", "SHF", "LEA", "ISETP", "IMNMX") for o in ops) / max(n, 1),
          "f_sync": sum(o in ("BAR", "SHFL", "MEMBAR") for o in ops) / max(n, 1),
          "reuse": sum(bin(i.reuse).count("1") for i in body)}
    st["depth"] = crit_depth(body)
    return st


def crit_depth(body: list) -> int:
    """Dependence-DAG critical path of one pass through `body` (RAW at resched latencies, WAR/WAW 1,
    memory order kept), the same model as scripts/swp_headroom.py cp_iter."""
    cache = _cache()
    try:
        table = resched.fixed_latency_table(body, cache)
        res = [resched._resources(x, cache) for x in body]
    except Exception:  # noqa: BLE001
        return -1
    if any(r is None for r in res):
        return -1
    full, last_def, readers = [0] * len(body), {}, {}
    for v, (d, u) in enumerate(res):
        f = 0
        if resched._memory(body[v]):
            for p in range(v):
                if resched._memory(body[p]) and (resched._writes_mem(body[p]) or resched._writes_mem(body[v])):
                    f = max(f, full[p] + 1)
        for x in u:
            if x in last_def:
                p = last_def[x]
                ip = body[p]
                L = resched.var_latency(ip) if ip.wbar != 7 else resched.latency(ip, table, body[v])
                f = max(f, full[p] + L)
        for x in d:
            if x in last_def:
                f = max(f, full[last_def[x]] + 1)
            for p in readers.get(x, ()):
                f = max(f, full[p] + 1)
        full[v] = f
        for x in u:
            readers.setdefault(x, []).append(v)
        for x in d:
            last_def[x], readers[x] = v, []
    return max(full, default=0) + 1


def hot(cubin: bytes) -> list:
    ins = sass.parse(toolchain.disassemble(cubin))
    return sass.loop_body(ins) or ins


def sim(body, warps: int) -> float | None:
    try:
        return model.simulate(body, warps=max(1, min(int(warps), 48)), iters=1000, sim_iters=4)["per_iter"]
    except Exception:  # noqa: BLE001
        return None


def sass_decision(base: bytes, cand: bytes, *, warps_sm: int, regs_base: int, regs_cand: int,
                  occ_base: int, occ_cand: int, ctx: dict, kernel: str = "k") -> dict:
    """Features for a SASS-level transformation (same launch, different binary)."""
    hb, hc = hot(base), hot(cand)
    sb, sc = loop_stats(hb), loop_stats(hc)
    pb, pc = sim(hb, warps_sm), sim(hc, warps_sm)
    f = {"kind_sass": 1, "kind_fusion": 0, "kind_tile": 0,
         "occ": occ_base / 48, "d_occ": (occ_cand - occ_base) / 48,
         "regs": regs_base / 255, "d_regs": (regs_cand - regs_base) / 255,
         "f_mem": sb["f_mem"], "f_mufu": sb["f_mufu"], "f_fp": sb["f_fp"], "f_int": sb["f_int"], "f_sync": sb["f_sync"],
         "d_n": _lr(sc["n"], sb["n"]), "d_stall": _lr(sc["stall_sum"], sb["stall_sum"]),
         "d_waits": (sc["waits"] - sb["waits"]) / max(sb["n"], 1),
         "d_depth": _lr(sc["depth"], sb["depth"]) if sb["depth"] > 0 and sc["depth"] > 0 else 0.0,
         "d_reuse": (sc["reuse"] - sb["reuse"]) / max(sb["n"], 1),
         "changed": int(base != cand),
         "d_launch_frac": 0.0, "bw_frac": ctx.get("bw_frac", 0.0),
         "hw_pred": math.log(pb / pc) if pb and pc else 0.0}
    return f


def resident_warps(regs: int, threads: int, smem: int, grid_blocks: int, sms: int = 68) -> int:
    """Warps per SM actually resident: occupancy limit, capped by how many blocks the grid gives each SM."""
    from .model_gpu import resident_warps_per_sm
    wpb = (threads + 31) // 32
    lim = resident_warps_per_sm(regs, threads, smem)
    return max(1, min(lim, -(-grid_blocks // sms) * wpb))


def fusion_pred(t_base_us: float, launches_base: int, launches_cand: int, graph: bool, gap: dict) -> float:
    """Launch-cost hardware model for fusion / CUDA graphs: each launch removed saves one stream launch
    gap; a graph turns every remaining stream gap into a graph gap. gap = {"stream": us, "graph": us}
    from scripts/launch_gap.py (dependent back-to-back launches, same Python launch path)."""
    saved = (launches_base - launches_cand) * gap["stream"]
    if graph:
        saved += launches_cand * (gap["stream"] - gap["graph"])
    t = max(t_base_us - saved, 0.05 * t_base_us)
    return math.log(t_base_us / t)


def fusion_decision(t_base_us, launches_base, launches_cand, graph, gap, shared_frac):
    f = {k: 0.0 for k in FEATURES}
    f.update(kind_fusion=1, kind_sass=0, changed=1,
             d_launch_frac=(launches_base - launches_cand) / launches_base,
             d_n=math.log(max(launches_cand, 1) / launches_base),
             hw_pred=fusion_pred(t_base_us, launches_base, launches_cand, graph, gap),
             bw_frac=shared_frac)
    return f


def _lr(a, b):
    return math.log(max(a, 1e-9) / max(b, 1e-9))


FEATURES = ["occ", "d_occ", "regs", "d_regs", "f_mem", "f_mufu", "f_fp", "f_int", "f_sync", "d_n", "d_stall",
            "d_waits", "d_depth", "d_reuse", "changed", "d_launch_frac", "bw_frac", "hw_pred",
            "kind_sass", "kind_fusion", "kind_tile"]
