"""Evaluate lab/gemm_model.py (multistage GEMM occupancy model) against measured prefill GEMM times.

  python scripts/eval_gemm_model.py [only=A,B]        (CPU only: compiles, never launches)

Measured: scripts/prefill_gemm_bench.py P=512 MS=1, Oct 10 2026 (median of the runs listed; none of
these numbers were used to set a model constant).
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import gemm_model as GM  # noqa: E402

SHAPES = [("qkv", 2048, 1536), ("o", 1536, 1536), ("gu", 17920, 1536), ("down", 1536, 8960)]
P = 512
# name: (Tile, {gemm: [measured us, ...]})
MEASURED = {
    "A 64x64 4w KB2 S3 (default o/down)": (GM.Tile(2, 2, 2, 4, 2, 3, 3), {
        "qkv": [78.7, 75.7, 74.9], "o": [45.4, 43.1, 43.5], "gu": [399.5, 404.9, 403.3], "down": [192.1, 195.8, 191.6]}),
    "A-noepi": (GM.Tile(2, 2, 2, 4, 2, 3, 3, "noepi"), {"qkv": [73.0], "o": [40.0], "gu": [383.1], "down": [183.0]}),
    "A-noimma": (GM.Tile(2, 2, 2, 4, 2, 3, 3, "noimma"), {"qkv": [75.7], "o": [45.3], "gu": [400.0], "down": [193.7]}),
    "A MINB=4": (GM.Tile(2, 2, 2, 4, 2, 3, 4), {"qkv": [77.2], "o": [44.5], "gu": [404.3], "down": [194.0]}),
    "G 64x64 4w KB2 S4 MINB2": (GM.Tile(2, 2, 2, 4, 2, 4, 2), {"qkv": [72.8], "o": [62.2], "gu": [547.6], "down": [249.8]}),
    "H 64x64 4w KB4 S2 MINB2": (GM.Tile(2, 2, 2, 4, 4, 2, 2), {"qkv": [66.2], "o": [57.7], "gu": [465.4], "down": [224.6]}),
    "B 128x64 8w KB2 S3": (GM.Tile(4, 2, 2, 4, 2, 3, 2), {"qkv": [56.0], "o": [50.8], "gu": [361.6], "down": [210.4]}),
    "C 128x64 8w KB2 S2": (GM.Tile(4, 2, 2, 4, 2, 2, 2), {"qkv": [52.6], "o": [49.7], "gu": [378.3], "down": [212.9]}),
    "D 128x64 8w KB4 S2 (default qkv/gu)": (GM.Tile(4, 2, 2, 4, 4, 2, 2), {
        "qkv": [54.7], "o": [49.9], "gu": [357.3], "down": [198.8]}),
    "E 128x64 4w MT4": (GM.Tile(2, 2, 4, 4, 2, 3, 2), {"qkv": [62.4], "o": [55.3], "gu": [431.7], "down": [198.1]}),
    "F 128x64 4w NT8": (GM.Tile(4, 1, 2, 8, 2, 3, 2), {"qkv": [65.3], "o": [53.2], "gu": [455.7], "down": [198.5]}),
}


def main():
    only = None
    for a in sys.argv[1:]:
        if a.startswith("only="):
            only = a[5:].split(",")
    errs, rows = [], []
    for name, (tile, meas) in MEASURED.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        bld = GM.build(tile)
        occ = GM.blocks_per_sm(bld["regs"], tile.nthr, bld["smem"])
        print(f"== {name}: {bld['regs']}r smem {bld['smem']} -> {occ['blocks']} blocks/SM ({occ['limit']}), "
              f"{occ['blocks'] * tile.nthr // 32} warps/SM", flush=True)
        for g, N, K in SHAPES:
            pr = GM.predict(tile, N, K, P, bld)
            m = statistics.median(meas[g])
            e = pr["us"] / m - 1
            errs.append(abs(e))
            rows.append((name, g, m, pr["us"]))
            w = pr["waves"][0]
            print(f"  {g:5s} measured {m:6.1f} us  predicted {pr['us']:6.1f} us  ({e:+.0%})  "
                  f"{len(pr['waves'])} waves, {w['per_sm']} blk/SM, copy {w['copy_cyc']} stage {w['stage_cyc']} cyc",
                  flush=True)
    if errs:
        print(f"median |error| {statistics.median(errs):.1%}, within 20%: {sum(e < .2 for e in errs)}/{len(errs)}")
    # ranking: per GEMM, does the model pick the measured-best config?
    for g, _, _ in SHAPES:
        rs = [r for r in rows if r[1] == g and "-no" not in r[0]]
        if rs:
            bm = min(rs, key=lambda r: r[2])
            bp = min(rs, key=lambda r: r[3])
            print(f"{g:5s}: measured best {bm[0].split()[0]} ({bm[2]:.1f}), model picks {bp[0].split()[0]} "
                  f"(measured {bp[2]:.1f})")


if __name__ == "__main__":
    main()
