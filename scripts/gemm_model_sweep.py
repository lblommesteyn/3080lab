"""Predict every multistage int8 GEMM tile config with lab/gemm_model.py (CPU only) and rank them per
Qwen2.5-1.5B prefill GEMM at P = 512 (bench epilogue: resid for all).

  python scripts/gemm_model_sweep.py [out=preds.json] [j=8]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import gemm_model as GM  # noqa: E402

SHAPES = [("qkv", 2048, 1536), ("o", 1536, 1536), ("gu", 17920, 1536), ("down", 1536, 8960)]
P = 512


def configs():
    for WM, WN, MT, NT in [(2, 2, 2, 4), (4, 2, 2, 4), (8, 2, 2, 4), (4, 4, 2, 4), (2, 4, 2, 4), (4, 1, 2, 8),
                           (2, 2, 4, 4), (4, 2, 1, 4), (2, 2, 2, 8), (8, 1, 2, 4), (4, 2, 2, 2)]:
        for KB, NSTG in [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3)]:
            for MINB in (1, 2, 3):
                yield GM.Tile(WM, WN, MT, NT, KB, NSTG, MINB)


def one(t):
    try:
        b = GM.build(t)
    except (AssertionError, RuntimeError):              # smem > 48 KB etc.
        return None
    if b.get("spills"):
        return None
    occ = GM.blocks_per_sm(b["regs"], t.nthr, b["smem"])
    row = {"tile": [t.WM, t.WN, t.MT, t.NT, t.KB, t.NSTG, t.MINB], "BM": t.BM, "BN": t.BN, "regs": b["regs"],
           "smem": b["smem"], "blocks_sm": occ["blocks"], "warps_sm": occ["blocks"] * t.nthr // 32}
    for g, N, K in SHAPES:
        row[g] = round(GM.predict(t, N, K, P, b)["us"], 1)
    row["layer"] = round(sum(row[g] for g, _, _ in SHAPES), 1)
    print(json.dumps(row), flush=True)
    return row


def main():
    out = "preds.json"
    for a in sys.argv[1:]:
        if a.startswith("out="):
            out = a[4:]
    from multiprocessing import Pool
    todo = [t for t in configs() if not (1536 % t.BM or t.nthr > 1024)]
    with Pool(int(dict(a.split("=") for a in sys.argv[1:] if "=" in a).get("j", 8))) as pool:
        res = [r for r in pool.imap_unordered(one, todo) if r]
    Path(out).write_text(json.dumps(res, indent=1))
    for g, _, _ in SHAPES + [("layer", 0, 0)]:
        best = sorted(res, key=lambda r: r[g])[:5]
        print(g, [(r["tile"], r[g]) for r in best])


if __name__ == "__main__":
    main()
