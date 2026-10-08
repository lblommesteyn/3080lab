"""Predictor v0: cycles for a SASS loop on one SM, from measured machine parameters.

The control bits already encode every dependency (ptxas did the dataflow
analysis and wrote it down as stall counts and scoreboard waits), so the
model does not track registers. It simulates:

  * per-warp in-order issue: next issue >= this issue + stall
  * scoreboards: a producer with wbar=b releases b after its op latency;
    an instruction whose wait mask includes b cannot issue before that
  * structural ports: each SM partition (warp w -> partition w % 4) has a
    dispatch port that FP32 ops hold 1 cycle and half-rate ops hold 2;
    MUFU, SHFL and FP64 have their own (partly SM-wide) ports
  * one issue per partition per cycle, greedy-then-oldest warp selection

All parameters are from 3080lab measurements (README). Steady state is
found by simulating a few loop iterations and extrapolating.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sass import Instr

# ---- machine parameters (cycles) -------------------------------------------------
BRANCH_TAKEN_PENALTY = 7      # fitted: dependent_ffma loop = sum(stalls) + 7
STALL0_CYCLES = 32            # stall field 0 behaves like ~32 (stall probes: 32.8/op)
READ_BARRIER = 2
RF_MODEL = "shared"         # "per_bank": each bank has its own read port; "shared": one port per partition
REUSE_SCOPE = "warp"          # "warp": each warp keeps its own reuse slots; "partition": any other warp's issue clears them

MEM_LATENCY = {"L1": 35, "L2": 238, "DRAM": 498, "SMEM": 23}
SB_LATENCY = {  # issue -> dependent may issue, for scoreboarded producers
    "DFMA": 55, "MUFU": 17, "MUFU.SIN": 19, "SHFL": 26, "LDS": 23,
}

# opcode-prefix -> (partition port, cost, sm-wide port, sm cost)
PORTS = [
    ("FFMA", "disp", 1, None, 0), ("FADD", "disp", 1, None, 0), ("FMUL", "disp", 1, None, 0),
    ("FSETP", "disp", 1, None, 0), ("FMNMX", "disp", 1, None, 0), ("FSEL", "disp", 1, None, 0),
    ("HFMA2", "disp", 2, None, 0), ("HADD2", "disp", 2, None, 0), ("HMUL2", "disp", 2, None, 0),
    ("IMAD", "disp", 2, None, 0), ("IADD3", "disp", 2, None, 0), ("LOP3", "disp", 2, None, 0),
    ("SHF", "disp", 2, None, 0), ("ISETP", "disp", 2, None, 0), ("LEA", "disp", 2, None, 0),
    ("SEL", "disp", 2, None, 0), ("PLOP3", "disp", 2, None, 0), ("MOV", "disp", 2, None, 0),
    ("MUFU", "mufu", 8, None, 0),
    ("SHFL", "shfl", 4, "shfl", 2),
    ("DFMA", "fp64", 16, "fp64", 16), ("DADD", "fp64", 16, "fp64", 16), ("DMUL", "fp64", 16, "fp64", 16),
    ("LDG", "lsu", 1, None, 0), ("LD", "lsu", 1, None, 0), ("LDS", "lsu", 1, None, 0),
    ("STG", "lsu", 1, None, 0), ("ST", "lsu", 1, None, 0), ("STS", "lsu", 1, None, 0),
]


# v1: FFMA-class ops can use either FMA pipe (pooled: cost 1 per warp-instr);
# IMAD/HFMA2 are half rate on the FMA pool; integer ALU ops have their own pipe.
PORTS_V1 = [
    ("FFMA", "fma", 1), ("FADD", "fma", 1), ("FMUL", "fma", 1), ("HFMA2", "fma", 2), ("HADD2", "fma", 2),
    ("HMUL2", "fma", 2), ("IMAD", "fma", 2),
    ("IADD3", "alu", 2), ("LOP3", "alu", 2), ("SHF", "alu", 2), ("ISETP", "alu", 2), ("LEA", "alu", 2),
    ("SEL", "alu", 2), ("PLOP3", "alu", 2), ("MOV", "alu", 2), ("FSETP", "alu", 2), ("FMNMX", "alu", 2),
    ("FSEL", "alu", 2), ("IMNMX", "alu", 2),
]


def port_of_v1(op: str):
    for prefix, port, cost in PORTS_V1:
        if op == prefix or op.startswith(prefix + "."):
            return port, cost, None, 0
    return port_of(op)


def bank_reads(ops, reuse_cache: dict, wid: int) -> list[int]:
    """Reads needed from [bank0, bank1]. Reuse cache: one entry per (bank, operand slot)
    holding (warp, reg); a hit costs no read."""
    if ops is None or ops.wide or not ops.complete:
        return [0, 0]
    banks = [0, 0]
    for slot, reg in ops.srcs:
        if reg == 255 or reuse_cache.get((reg & 1, slot)) == (wid, reg):
            continue
        banks[reg & 1] += 1
    return banks


def update_reuse(ops, ins, reuse_cache: dict, wid: int):
    """After an instruction: each slot it READ either keeps the value (reuse flag set and
    the warp does not yield) or evicts that (bank, slot) entry. Slots it does not read
    (RZ, immediates, constants) leave the cache alone (Huerta et al.: an entry is
    invalidated when a read arrives for the same bank and operand position)."""
    if ops is None or ops.wide or not ops.complete:
        reuse_cache.clear()
        return
    for slot, reg in ops.srcs:
        if reg == 255:
            continue
        key = (reg & 1, slot)
        if ins.yield_ and ins.reuse >> "abc".index(slot) & 1:
            reuse_cache[key] = (wid, reg)
        else:
            reuse_cache.pop(key, None)


def rf_cycles(ops, reuse_cache: dict, wid: int) -> int:
    return max(1, max(bank_reads(ops, reuse_cache, wid)))


def port_of(op: str):
    for prefix, port, cost, smport, smcost in PORTS:
        if op == prefix or op.startswith(prefix + "."):
            return port, cost, smport, smcost
    return "misc", 1, None, 0   # branches, CS2R, uniform ops, NOP ...


def sb_latency(ins: Instr, mem_level: str) -> int:
    op = ins.opcode
    if op.startswith(("LDG", "LD.")) or op == "LD":
        return MEM_LATENCY[mem_level]
    if op.startswith("LDS"):
        return SB_LATENCY["LDS"]
    if op.startswith("MUFU.SIN") or op.startswith("MUFU.COS"):
        return SB_LATENCY["MUFU.SIN"]
    for k in ("DFMA", "DADD", "DMUL", "MUFU", "SHFL"):
        if op.startswith(k):
            return SB_LATENCY[k]
    return 20  # unknown variable-latency producer


@dataclass
class Warp:
    wid: int
    pc: int = 0
    iter: int = 0
    ready_at: int = 0
    bar_release: list = field(default_factory=lambda: [0] * 6)
    iter_end: list = field(default_factory=list)


def simulate(body: list[Instr], warps: int, iters: int, mem_level: str = "L1",
             sim_iters: int = 6, version: int = 1) -> dict:
    """Predict cycles for `iters` trips through `body` with `warps` warps in one block.
    version 0: one shared dispatch port, no register file. version 1: fma/alu pipes,
    register-bank read ports and the operand reuse cache."""
    n_sim = min(iters, sim_iters)
    ops_of = [None] * len(body)
    if version >= 1:
        from . import operands as _opnd
        cache = _opnd._load_cache()
        ops_of = [_opnd.operands(i, cache) for i in body]
    reuse = [dict() for _ in range(4)]          # partition scope
    reuse_w = [dict() for _ in range(warps)]    # warp scope
    pof = port_of_v1 if version >= 1 else port_of
    ws = [Warp(w) for w in range(warps)]
    part_port: list[dict] = [dict() for _ in range(4)]
    sm_port: dict = {}
    last_issued = [None] * 4
    t = 0
    alive = list(ws)
    nb = len(body)
    while alive:
        issued = False
        for p in range(4):
            cands = [w for w in alive if w.wid % 4 == p and w.ready_at <= t]
            if not cands:
                continue
            # greedy-then-oldest: keep issuing from the last warp if it can go
            cands.sort(key=lambda w: (w is not last_issued[p], w.wid))
            for w in cands:
                ins = body[w.pc]
                if any(ins.wait >> b & 1 and w.bar_release[b] > t for b in range(6)):
                    continue
                port, cost, smp, smc = pof(ins.opcode)
                if part_port[p].get(port, 0) > t or (smp and sm_port.get(smp, 0) > t):
                    continue
                if version >= 1:
                    if RF_MODEL == "shared" and part_port[p].get("rf", 0) > t:
                        continue
                    rcache = reuse_w[w.wid] if REUSE_SCOPE == "warp" else reuse[p]
                    if RF_MODEL == "per_bank":
                        br = bank_reads(ops_of[w.pc], rcache, w.wid)
                        if any(br[b] and part_port[p].get(f"bank{b}", 0) > t for b in (0, 1)):
                            continue
                        for b in (0, 1):
                            if br[b]:
                                part_port[p][f"bank{b}"] = t + br[b]
                    else:
                        part_port[p]["rf"] = t + rf_cycles(ops_of[w.pc], rcache, w.wid)
                    update_reuse(ops_of[w.pc], ins, rcache, w.wid)
                part_port[p][port] = t + cost
                if smp:
                    sm_port[smp] = t + smc
                if ins.wbar != 7:
                    w.bar_release[ins.wbar] = max(w.bar_release[ins.wbar], t + sb_latency(ins, mem_level))
                if ins.rbar != 7:
                    w.bar_release[ins.rbar] = max(w.bar_release[ins.rbar], t + READ_BARRIER)
                stall = ins.stall if ins.stall else STALL0_CYCLES
                w.ready_at = t + stall
                w.pc += 1
                if w.pc == nb:
                    w.pc = 0
                    w.iter += 1
                    w.ready_at += BRANCH_TAKEN_PENALTY
                    w.iter_end.append(w.ready_at)
                    if w.iter >= n_sim:
                        alive.remove(w)
                last_issued[p] = w
                issued = True
                break
        if issued or not alive:
            t += 1
        else:  # skip idle cycles: nothing can issue before the earliest warp is ready
            t = max(t + 1, min(w.ready_at for w in alive))
    # Throughput view: greedy scheduling lets one warp run ahead during a short
    # simulation, so per-warp slopes are misleading. Use the time for ALL warps to
    # finish n_sim iterations; the measured kernel time is likewise the slowest warp.
    t_all = max(w.iter_end[-1] for w in ws)
    pi = t_all / n_sim
    return {"per_iter": pi, "cycles": pi * iters, "sim_iters": n_sim}


def mem_level_for(params: dict, exp_name: str) -> str:
    if exp_name.startswith("chase_shared"):
        return "SMEM"
    if not exp_name.startswith("chase"):
        return "L1"
    size = params.get("bytes", 0)
    if "_cg" not in exp_name and size <= 96 * 1024:
        return "L1"
    return "L2" if size <= 4_500_000 else "DRAM"


def body_from_listing(path) -> list[Instr]:
    """Rebuild Instr objects from a saved *.sass.json artifact (largest loop)."""
    import json
    from .sass import loop_body
    rows = json.loads(open(path).read())
    ins = []
    for r in rows:
        raw = int(r["raw"], 16)
        ins.append(Instr(r["offset"], r["text"], raw & (2**64 - 1), raw >> 64, r.get("label")))
    return loop_body(ins)
