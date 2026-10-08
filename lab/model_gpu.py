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


def fit_mlp() -> MemFit:
    """Fit BW = min(peak, q/L) to mem_mlp (q = sms * warps * 32 lanes * 16 B)."""
    r = latest("mem_mlp")
    q, bw = [], []
    for lab, s in r["summary"].items():
        p = s["params"]
        q.append(p["sms"] * p["wps"] * 32 * 16)
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
