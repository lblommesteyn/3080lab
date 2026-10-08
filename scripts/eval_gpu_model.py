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
import sys as _sys
USE_L1 = "--no-l1" not in _sys.argv
USE_SPLIT = "--no-split" not in _sys.argv
USE_STRAIGHT = "--no-straight" not in _sys.argv


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


# lines (128 B) touched per load instruction, per pointer role, from each family's access pattern
LINES = {
    "gemv_int4_v3": {"W": 4, "X": 32, "S": 1},      # fp32 x: each lane reads its own 128 B chunk
    "gemv_int4_v4": {"W": 4, "X": 32, "S": 1},
    "gemv_int4_v4_q15": {"W": 4, "X": 32, "S": 1},
    "gemv_q4_0_blocks": {"W": 4, "X": 16, "S": 1},  # bf16 x: 64 B per lane
}


def l1_map(cubin: bytes, exp: str, active: float = 1.0, U: int = 1) -> dict:
    """active: average fraction of lanes that issue loads (short split-K chunks idle lanes).
    U: uint4 loads per row per lane per trip; lanes are then 16*U bytes apart, so each weight
    load instruction spans U times more lines (gemv_l1_pollution showed the cost)."""
    prov = model.load_provenance(cubin)
    out = {}
    for off, params in prov.items():
        role = "W" if 0 in params else "X" if 2 in params else "S" if 1 in params else None
        if role:
            lines = min(32, LINES[exp][role] * (U if role == "W" else 1))
            out[off] = max(1, round(lines * active))
    return out


def active_fraction(exp: str, p: dict) -> float:
    if exp in ("gemv_int4_v4", "gemv_int4_v4_q15", "gemv_q4_0_blocks") and p.get("S"):
        chunk = math.ceil((p["K"] / 32) / p["S"])
        trips = math.ceil(chunk / 32)
        return chunk / (32 * trips)
    return 1.0


def main():
    mem = G.fit_mlp()
    lat = G.calibrate_dram_latency()
    fixed = G.calibrate_fixed_overhead(mem)
    print(f"fixed streaming-kernel overhead from roofline microbenchmarks: {fixed:.2f} us")
    print(f"peak {mem.peak_gbps:.0f} GB/s (mem_mlp); DRAM load latency calibrated on the 1-SM/1-warp point: "
          f"{lat} cycles = {lat / G.F_CLK_GHZ:.0f} ns")
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
            body = model.body_from_listing(d / f"{s['kernel_artifact']}.sass.json")
            l1 = (l1_map((d / f"{s['kernel_artifact']}.cubin").read_bytes(), exp, active_fraction(exp, p),
                         p.get("U", 1)) if USE_L1 else None)
            allins = model.body_from_listing.__globals__["Instr"]  # noqa: F841 (type only)
            rows_json = json.loads((d / f"{s['kernel_artifact']}.sass.json").read_text())
            from lab.sass import Instr as _I
            full = [_I(r_["offset"], r_["text"], int(r_["raw"], 16) & (2**64 - 1), int(r_["raw"], 16) >> 64, r_.get("label"))
                    for r_ in rows_json]
            split = exp == "gemv_int4_v3" and p.get("U", 1) > 1 and p.get("worder") != "lane_contig"
            pr = G.predict_sim(body, total_bytes=total, warps_total=warps, threads_per_block=tpb,
                               regs=s["resources"]["registers"], trips_per_warp=trips, mem=mem, dram_lat=lat,
                               l1_lines=l1, split_sector=split and USE_SPLIT,
                               straight=G.straight_parts(full) if USE_STRAIGHT else None)
            meas = s["metrics"]["us"]["median"]
            pr["us"] += 0.0 if USE_STRAIGHT else fixed   # straight-line simulation replaces the constant
            rows.append((exp, lab, meas, pr["us"], (pr["us"] - meas) / meas, pr["bound"], pr["waves"]))
    e = np.abs([r[4] for r in rows])
    print()
    print(f"GEMV variants: n={len(rows)}  median |err| {np.median(e):.1%}  within 10%: {np.mean(e < .10):.0%}  "
          f"within 20%: {np.mean(e < .20):.0%}")
    for exp in sorted({r[0] for r in rows}):
        ee = np.abs([r[4] for r in rows if r[0] == exp])
        print(f"  {exp:<20} n={len(ee):3d} median {np.median(ee):.1%}")
    print()
    print("worst 12:")
    for r in sorted(rows, key=lambda r: -abs(r[4]))[:12]:
        print(f"  {r[0]:<18} {r[1]:<22} meas {r[2]:7.1f} us  pred {r[3]:7.1f} us  {r[4]:+.0%}  [{r[5]}-bound, {r[6]:.1f} waves]")
    (R / "gpu_model_eval.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
