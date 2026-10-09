"""Predicted runtime of a streaming kernel from its SASS, including split-sector re-fetches.

    T_trip = max(T_sm, T_mem)          (one loop trip of every warp of the grid)

  T_mem = warps_total x (unique streamed bytes + re-fetched bytes) per warp-trip / BW(inflight)
          BW(q) = min(peak, q / L), fitted to mem_mlp (model_gpu.MemFit)
  re-fetched bytes = sum over split pairs of  overlap x bytes(second half) x m(D)
          D = streaming bytes the warp requests between the two halves x concurrent warps GPU-wide
          m(D) = DRAM re-fetch fraction of the later half, interpolated from mem_split_mech
                 (measured: halves G loads apart, m = peak / split_bandwidth - 1)
  T_sm  = SM simulator (lab/model.py v1) on the loop body with DRAM-latency loads, at the
          occupancy the kernel's register count allows

Only microbenchmark data enters the constants. Kernels it is evaluated on are never used to fit.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import sass, sectors, toolchain
from .model_gpu import F_CLK_GHZ, RESULTS, SMS, fit_mlp, resident_warps_per_sm

WARPS_IN_MECH = 68 * 2 * 8          # mem_split_mech launch: 136 blocks x 8 warps, all resident


# bytes one first-half warp-load of mem_split_mech puts in L2: 32 lanes x 32 B stride x 16 B wide
# touches 32 whole sectors = 1024 B. v1/v2 used the 512 useful bytes, while the kernel side counts
# sector footprint (sectors.warp_bytes): a 2x unit mismatch that shifted the curve (fixed in v3).
MECH_BYTES_PER_LOAD = {1: 512, 2: 512, 3: 1024, 4: 1024}


@lru_cache(maxsize=None)
def refetch_curve(cache: str = "nc", version: int = 3) -> tuple:
    """((D bytes, m), ...) from the latest mem_split_mech run."""
    f = sorted(RESULTS.glob("*_mem_split_mech/record.json"))[-1]
    r = json.loads(f.read_text())["summary"]
    pts = []
    for G in (1, 2, 4, 8, 16, 32):
        split = r[f"{cache}/G{G}/split"]["metrics"]["useful_gbps"]["median"]
        contig = r[f"{cache}/G{G}/contig"]["metrics"]["useful_gbps"]["median"]
        m = min(1.0, max(0.0, contig / split - 1.0))
        pts.append(((G - 1) * MECH_BYTES_PER_LOAD[version] * WARPS_IN_MECH, m))
    return tuple(pts)


def refetch(D: float, cache: str = "nc", version: int = 3) -> float:
    pts = refetch_curve(cache, version)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if D <= xs[0]:
        return ys[0]
    if D >= xs[-1]:
        return ys[-1]
    return float(np.interp(D, xs, ys))


@lru_cache(maxsize=None)
def _mem():
    return fit_mlp()


@lru_cache(maxsize=None)
def _dram_lat():
    """DRAM latency (cycles) at which the SM simulator reproduces mem_mlp; cached per record."""
    from .model_gpu import calibrate_dram_latency
    rec = sorted(RESULTS.glob("*_mem_mlp/record.json"))[-1]
    cache = RESULTS.parent / "data" / "dram_latency.json"
    try:
        c = json.loads(cache.read_text())
        if c.get("record") == rec.parent.name:
            return c["cycles"]
    except (OSError, ValueError):
        pass
    lat = calibrate_dram_latency()
    cache.write_text(json.dumps({"record": rec.parent.name, "cycles": lat}))
    return lat


def regcount(text: str) -> int:
    m = re.search(r"SHI_REGISTERS=(\d+)", text)
    return int(m.group(1)) if m else 255


@dataclass(frozen=True)
class Launch:
    block: int = 128
    grid: int | None = None              # None: enough blocks for 4 full waves
    trips: float | None = None           # loop trips per warp (None: steady state, fixed cost ignored)


MODEL_VERSION = 3          # v4 (post hoc on suite C, not held-out validated) is opt-in


@lru_cache(maxsize=None)
def fixed_overhead_us() -> float:
    """Fill/drain + launch cost of a GPU-wide streaming kernel, from the pure-read roofline
    microbenchmark (model_gpu.calibrate_fixed_overhead)."""
    from .model_gpu import calibrate_fixed_overhead
    return calibrate_fixed_overhead(_mem())


def predict(cubin: bytes, launch: Launch = Launch(), kernel: str = "k", version: int | None = None,
            sm_reference: bytes | None = None) -> dict:
    """v3 (after suite A refuted two v1 assumptions, see README):
      * the SM-simulator term is taken from `sm_reference` (the ORIGINAL kernel) for a rewrite: the
        rewrite executes the same instructions, and the simulator's predicted latency-hiding gain
        from moved loads was not real (o15/dn15: all predicted gain came from it; measured 1.00x);
      * re-fetch curve in sector-footprint units (MECH_BYTES_PER_LOAD);
      * whole-kernel time = fixed overhead + trips x per-trip time when launch.trips is known."""
    """version 1 charged re-fetched bytes through the in-flight (Little's law) bandwidth too, so a
    latency-bound kernel looked slower with more DRAM bytes. Refuted by the first optimize
    benchmark (o15/dn15 R4U2: predicted 1.21x, measured 1.00x): the later half is requested in
    both versions, only where it is served from changes, so re-fetches cost time only at the DRAM
    bandwidth roof. version 2: T_mem = max(unique / BW(inflight), (unique + refetched) / peak)."""
    version = version or MODEL_VERSION
    from . import model
    text = toolchain.disassemble(cubin)
    ins = sass.parse(text)
    regs = regcount(text)
    wpb = (launch.block + 31) // 32
    wres = resident_warps_per_sm(regs, launch.block)
    grid = launch.grid or (wres // wpb) * SMS * 4
    warps_total = grid * wpb
    concurrent = min(warps_total, wres * SMS)
    version = version or MODEL_VERSION
    a = sectors.analyze(ins=ins)
    loop = [l for l in a.loads if l.in_loop] or a.loads
    stream = [l for l in loop if sectors.streams(l)]
    idx = {l.offset: k for k, l in enumerate(loop)}
    unique, refetched, pairs = 0.0, 0.0, []
    shared = {}
    for p in a.pairs:
        if p.second.offset in idx and sectors.streams(p.second):
            shared[p.second.offset] = p
    for l in stream:
        wb = sectors.warp_bytes(l)
        p = shared.get(l.offset)
        if p is None:
            unique += wb
            continue
        unique += wb * (1 - p.overlap)
        i, j = idx[p.first.offset], idx[p.second.offset]
        between = sum(sectors.warp_bytes(x) for x in loop[i + 1:j] if sectors.streams(x))
        D = between * concurrent
        if version >= 4 and launch.trips and launch.trips < 1:
            D *= launch.trips                          # only that fraction of lanes requests anything
        m = refetch(D, l.cache, version)
        refetched += wb * p.overlap * m
        pairs.append({"first": p.first.offset, "second": p.second.offset, "overlap": round(p.overlap, 3),
                      "between_bytes": between, "D_mb": round(D / 2**20, 2), "m": round(m, 3)})
    mem = _mem()
    inflight = concurrent * unique
    # (tried after suite C: inflight = concurrent x barrier-tracked OUTSTANDING bytes x active lanes.
    #  Refuted: suite A/B median error 4% -> 24%, r_major wins predicted as losses. Not used;
    #  outstanding_bytes() is kept for the record.)
    bw = mem.bw(inflight)                                  # GB/s == bytes/ns
    if version == 1:
        t_mem = warps_total * (unique + refetched) / bw        # ns per GPU trip
    else:
        t_mem = max(warps_total * unique / bw, warps_total * (unique + refetched) / mem.peak_gbps)
    if version >= 4 and sm_reference is not None:
        # v4: the reference instruction stream, at THIS cubin's occupancy (renaming can cost warps)
        t_sm = _t_sm(sm_reference, launch, regs)
    elif version >= 3 and sm_reference is not None:
        t_sm = _t_sm(sm_reference, launch)
    else:
        t_sm = _t_sm(cubin, launch)
    t = max(t_mem, t_sm)
    if version >= 4 and launch.trips:
        # a partial trip (K shorter than one lane sweep) still pays a full trip of latency/issue but
        # moves only its share of bytes
        t = fixed_overhead_us() * 1e3 + max(math.ceil(launch.trips) * t_sm, launch.trips * t_mem)
    elif version >= 3 and launch.trips:
        t = fixed_overhead_us() * 1e3 + launch.trips * t
    return {"t_ns": t, "regs": regs, "resident_warps_sm": wres, "concurrent_warps": concurrent,
            "unique_bytes_per_warp_trip": unique, "refetched_bytes_per_warp_trip": refetched,
            "bw_gbps": bw, "t_mem_ns": t_mem, "t_sm_ns": t_sm,
            "bound": "mem" if t_mem >= t_sm else "sm", "pairs": pairs}


@lru_cache(maxsize=64)
def _t_sm(cubin: bytes, launch: Launch, regs: int | None = None) -> float:
    """SM-simulator time per GPU trip (ns): every warp of the grid does one loop trip.
    regs overrides the occupancy (v4: the original's stream at a rewrite's register count)."""
    from . import model
    text = toolchain.disassemble(cubin)
    ins = sass.parse(text)
    regs = regs or regcount(text)
    wpb = (launch.block + 31) // 32
    wres = resident_warps_per_sm(regs, launch.block)
    grid = launch.grid or (wres // wpb) * SMS * 4
    warps_total = grid * wpb
    body = sass.loop_body(ins) or ins
    model.MEM_LATENCY["DRAM"] = _dram_lat()
    w_sim = max(1, min(wres, -(-warps_total // SMS)))
    sim = model.simulate(body, warps=w_sim, iters=6, mem_level="DRAM", sim_iters=6, version=1)
    return sim["per_iter"] * ((warps_total / SMS) / w_sim) / F_CLK_GHZ


def outstanding_bytes(ins, a) -> float:
    """Time-averaged streaming bytes one warp has in flight in the loop body (barrier tracked)."""
    body = sass.loop_body(ins) or ins
    wb = {l.offset: (sectors.warp_bytes(l) if sectors.streams(l) else 0) for l in a.loads}
    out = {b: 0 for b in range(6)}
    tot = tw = 0
    for pass_ in range(2):                             # second pass: loop-carried steady state
        for x in body:
            for b in range(6):
                if x.wait >> b & 1:
                    out[b] = 0
            if x.opcode.startswith("LDG") and x.wbar != 7:
                out[x.wbar] += wb.get(x.offset, 0)
            if pass_:
                w = x.stall or 1
                tot += sum(out.values()) * w
                tw += w
    return tot / tw if tw else 0.0
