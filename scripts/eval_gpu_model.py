"""Score the whole-GPU streaming model (lab/model_gpu.py) on every measured GEMV variant.

Constants come from microbenchmarks only (mem_mlp fit, SM simulator); GEMV
timings are never used to fit anything. Compared against in-kernel spans
(globaltimer first-start..last-end), so no launch overhead is added.
"""
import json
import math
from pathlib import Path

import numpy as np

from lab import model, model_gpu as G

R = Path(__file__).resolve().parents[1] / "results"
F_CLK_GHZ = 1.95


def gemv_spec(exp: str, p: dict):
    """-> (warps_total, threads_per_block, trips_per_warp, dram_bytes_per_warp_trip, total_bytes)"""
    N, K = p["N"], p["K"]
    if exp == "gemv_int4_v3":
        R_, U = p["R"], p["U"]
        warps = math.ceil(N / R_)
        trips = math.ceil((K / 32) / (32 * U))
        per = R_ * U * 32 * (16 + 4 / 4)          # uint4 weights + one fp32 scale per 4 uint4 (group 128)
        tpb = p["T"]
        total = N * K / 2 + N * K / 128 * 4
    elif exp in ("gemv_int4_v4", "gemv_int4_v4_q15"):
        if p.get("mode") != "splitk":
            return None
        S_ = p["S"]
        warps = math.ceil(N / 4) * S_
        trips = math.ceil(math.ceil((K / 32) / S_) / 32)
        per = 4 * 32 * (16 + 1)
        tpb = 32 * S_
        total = N * K / 2 + N * K / 128 * 4
    elif exp == "gemv_q4_0_blocks":
        S_, Gr = p["S"], p["G"]
        warps = math.ceil(N / 4) * S_
        trips = math.ceil(math.ceil((K / 32) / S_) / 32)
        per = 4 * 32 * (16 + 2)                      # fp16 scale per uint4 (group 32)
        tpb = 32 * S_ * Gr
        total = N * K / 2 + N * K / 32 * 2
    else:
        return None
    return warps, tpb, trips, per, total


def main():
    mem = G.fit_mlp()
    print(f"mem fit from mem_mlp: peak {mem.peak_gbps:.0f} GB/s, loaded latency {mem.lat_ns:.0f} ns")
    rows = []
    for exp in ("gemv_int4_v3", "gemv_int4_v4", "gemv_int4_v4_q15", "gemv_q4_0_blocks"):
        f = sorted(R.glob(f"*_{exp}/record.json"))
        if not f:
            continue
        d = f[-1].parent
        rec = json.loads(f[-1].read_text())
        for lab, s in rec["summary"].items():
            p = s["params"]
            if p.get("roofline"):
                continue
            spec = gemv_spec(exp, p)
            if spec is None:
                continue
            warps, tpb, trips, per, total = spec
            regs = s["resources"]["registers"]
            wres = G.resident_warps_per_sm(regs, tpb)
            body = model.body_from_listing(d / f"{s['kernel_artifact']}.sass.json")
            sim = model.simulate(body, warps=min(wres, 32), iters=4, mem_level="L1", sim_iters=4, version=1)
            issue_ns = sim["per_iter"] / F_CLK_GHZ * (wres / min(wres, 32))
            pr = G.predict_stream(total_bytes=total, warps_total=warps, bytes_per_warp_iter=per, trips_per_warp=trips,
                                  warps_per_sm_resident=wres, issue_ns_per_trip=issue_ns, mem=mem)
            pred = pr["us"] - G.LAUNCH_US
            meas = s["metrics"]["us"]["median"]
            rows.append((exp, lab, meas, pred, (pred - meas) / meas, pr["bound"], pr["inflight_kb"]))
    e = np.abs([r[4] for r in rows])
    print(f"\nGEMV variants: n={len(rows)}  median |err| {np.median(e):.1%}  within 10%: {np.mean(e < .10):.0%}  "
          f"within 20%: {np.mean(e < .20):.0%}")
    for exp in sorted({r[0] for r in rows}):
        ee = np.abs([r[4] for r in rows if r[0] == exp])
        print(f"  {exp:<20} n={len(ee):3d} median {np.median(ee):.1%}")
    print("\nworst 12:")
    for r in sorted(rows, key=lambda r: -abs(r[4]))[:12]:
        print(f"  {r[0]:<18} {r[1]:<22} meas {r[2]:7.1f} us  pred {r[3]:7.1f} us  {r[4]:+.0%}  [{r[5]}, {r[6]:.0f} KB in flight]")


if __name__ == "__main__":
    main()
