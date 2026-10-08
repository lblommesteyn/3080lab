"""Whole-GPU model for streaming (memory-bound-ish) kernels such as decode GEMVs.

  T = max(T_mem, T_issue) + T_launch
  T_mem   = bytes / BW(inflight),  BW(q) = min(BW_peak, q / L)      (Little's law)
            q = bytes in flight GPU-wide = resident warps x bytes each warp issues
                per loop iteration before first use (all of an iteration's loads issue together)
  T_issue = v1 SM simulator on the real SASS loop body, resident warps per SM,
            trips per warp, memory latency hidden (pure issue/dependency time)

Constants come only from microbenchmarks: BW_peak and L are fitted to the
mem_mlp sweep (DRAM bandwidth vs SMs x warps), T_launch from graph_overhead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
SMS = 68
MAX_WARPS_PER_SM = 48
REGS_PER_SM = 65536
LAUNCH_US = 3.0   # graph ramp + tail for a GPU-wide kernel (graph_overhead: ~1 us empty + ramp)


def latest(exp: str) -> dict:
    f = sorted(RESULTS.glob(f"*_{exp}/record.json"))[-1]
    return json.loads(f.read_text())


@dataclass
class MemFit:
    peak_gbps: float
    lat_ns: float

    def bw(self, inflight_bytes: float) -> float:
        return min(self.peak_gbps, inflight_bytes / self.lat_ns)   # bytes/ns == GB/s


def loads_per_iter(sass_json: Path) -> tuple[int, int]:
    """(number of global loads, bytes per lane) in the SASS loop body, i.e. what one warp has
    in flight per trip when all of an iteration's loads issue before the first use."""
    from . import model
    body = model.body_from_listing(sass_json)
    lds = [i for i in body if i.opcode.startswith(("LDG", "LD."))]
    width = lambda op: 16 if ".128" in op else 8 if ".64" in op else 4  # noqa: E731
    return len(lds), sum(width(i.opcode) for i in lds)


def fit_mlp() -> MemFit:
    """Fit BW = min(peak, q/L) to mem_mlp, with q = SMs x warps x 32 lanes x bytes-per-lane in
    flight per loop trip (read from the kernel's SASS: ptxas unrolls the loop)."""
    f = sorted(RESULTS.glob("*_mem_mlp/record.json"))[-1]
    r = json.loads(f.read_text())
    q, bw = [], []
    for lab, s in r["summary"].items():
        p = s["params"]
        _, per_lane = loads_per_iter(f.parent / f"{s['kernel_artifact']}.sass.json")
        q.append(p["sms"] * p["wps"] * 32 * per_lane)
        bw.append(s["metrics"]["gbps"]["median"])
    q, bw = np.array(q, float), np.array(bw, float)
    peak = float(np.percentile(bw, 98))
    lat_ns = float(np.median((q / bw)[bw < 0.6 * peak]))   # latency-bound points
    return MemFit(peak, lat_ns)


def resident_warps_per_sm(regs: int, threads_per_block: int, smem: int = 0) -> int:
    wpb = (threads_per_block + 31) // 32
    per_warp = ((regs * 32 + 255) // 256) * 256
    by_regs = REGS_PER_SM // (per_warp * wpb)
    by_warps = MAX_WARPS_PER_SM // wpb
    by_smem = (100 * 1024) // smem if smem else 16
    return max(1, min(16, by_regs, by_warps, by_smem)) * wpb


def predict_stream(*, total_bytes: float, warps_total: int, bytes_per_warp_iter: float,
                   trips_per_warp: float, warps_per_sm_resident: int, issue_ns_per_trip: float,
                   mem: MemFit) -> dict:
    concurrent = min(warps_total, warps_per_sm_resident * SMS)
    inflight = concurrent * bytes_per_warp_iter
    t_mem = total_bytes / mem.bw(inflight)
    waves = warps_total / concurrent
    t_issue = waves * trips_per_warp * issue_ns_per_trip
    return {"us": (max(t_mem, t_issue)) / 1e3 + LAUNCH_US, "t_mem_us": t_mem / 1e3, "t_issue_us": t_issue / 1e3,
            "inflight_kb": inflight / 1024, "bw_gbps": mem.bw(inflight), "bound": "mem" if t_mem >= t_issue else "issue"}


F_CLK_GHZ = 1.95


def calibrate_dram_latency() -> int:
    """Pick the DRAM load latency (cycles) at which the SM simulator reproduces the measured
    1-SM x 1-warp mem_mlp bandwidth. Only microbenchmark data is used."""
    from . import model
    f = sorted(RESULTS.glob("*_mem_mlp/record.json"))[-1]
    r = json.loads(f.read_text())
    s = r["summary"]["sms1/w1"]
    body = model.body_from_listing(f.parent / f"{s['kernel_artifact']}.sass.json")
    _, per_lane = loads_per_iter(f.parent / f"{s['kernel_artifact']}.sass.json")
    target = s["metrics"]["gbps"]["median"]                  # GB/s == bytes/ns
    best = None
    for lat in range(200, 1600, 10):
        model.MEM_LATENCY["DRAM"] = lat
        pi = model.simulate(body, warps=1, iters=8, mem_level="DRAM", sim_iters=6)["per_iter"]
        bw = 32 * per_lane / (pi / F_CLK_GHZ)
        if best is None or abs(bw - target) < best[1]:
            best = (lat, abs(bw - target), bw)
    model.MEM_LATENCY["DRAM"] = best[0]
    return best[0]


SPLIT_SECTOR_EFF = 441 / 723   # mem_split_gap: halves of a sector requested >= 8 loads apart


def straight_parts(instrs: list) -> tuple[list, list]:
    """(prologue, epilogue) around the largest loop: what a warp executes once."""
    from . import sass as _s
    body = _s.loop_body(instrs)
    if not body:
        return instrs, []
    offs = [x.offset for x in instrs]
    i0 = offs.index(body[0].offset)
    i1 = offs.index(body[-1].offset) + 1
    post = []
    for x in instrs[i1:]:
        post.append(x)
        if x.opcode == "EXIT" and not x.text.strip().startswith("@"):
            break
    return instrs[:i0], post


def predict_sim(body, *, total_bytes: float, warps_total: int, threads_per_block: int, regs: int,
                trips_per_warp: int, mem: MemFit, dram_lat: int, l1_lines: dict | None = None,
                split_sector: bool = False, straight: tuple | None = None) -> dict:
    """T = max(SM simulation with DRAM-latency loads, total_bytes / effective peak).
    split_sector: weight halves of each sector requested far apart (bandwidth x SPLIT_SECTOR_EFF).
    straight: (prologue, epilogue) instruction lists simulated once per warp wave and added."""
    from . import model
    model.MEM_LATENCY["DRAM"] = dram_lat
    wres = resident_warps_per_sm(regs, threads_per_block)
    per_sm = -(-warps_total // SMS)
    w_sim = max(1, min(wres, per_sm))
    # throughput-bound: time scales with total work, so a partial last wave counts fractionally
    waves = per_sm / w_sim
    sim = model.simulate(body, warps=w_sim, iters=max(1, trips_per_warp), mem_level="DRAM",
                         sim_iters=min(6, max(1, trips_per_warp)), version=1, l1_lines=l1_lines)
    t_sm = waves * sim["cycles"] / F_CLK_GHZ          # ns
    if straight:
        once = [x for part in straight for x in part]
        if once:
            s2 = model.simulate(once, warps=w_sim, iters=1, mem_level="DRAM", sim_iters=1, version=1)
            # prologue/epilogue of later waves overlap other warps' loops: only the first
            # fill and the last drain are exposed, i.e. once per kernel
            t_sm += s2["cycles"] / F_CLK_GHZ
    t_bw = total_bytes / (mem.peak_gbps * (SPLIT_SECTOR_EFF if split_sector else 1.0))   # ns
    return {"us": max(t_sm, t_bw) / 1e3, "t_sm_us": t_sm / 1e3, "t_bw_us": t_bw / 1e3,
            "bound": "sm" if t_sm >= t_bw else "bw", "w_sim": w_sim, "waves": waves}


def calibrate_fixed_overhead(mem: MemFit) -> float:
    """Fixed fill/drain cost (us) of a GPU-wide streaming kernel, from the pure-read roofline
    microbenchmarks (gemv_int4_v3 */roofline): measured - bytes/peak, median over sizes."""
    r = latest("gemv_int4_v3")
    oh = []
    for lab, s in r["summary"].items():
        if s["params"].get("roofline"):
            p = s["params"]
            bytes_ = p["N"] * p["K"] / 2
            oh.append(s["metrics"]["us"]["median"] - bytes_ / mem.peak_gbps / 1e3)
    return float(np.median(oh))
