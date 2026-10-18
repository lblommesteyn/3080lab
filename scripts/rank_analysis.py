"""Why does a GEMM model with ~6-10% median error rank tiles badly? (Experiment B, Oct 2026)

CPU only. Uses the 22 measured tiles (11 in-sample, 11 held-out) x 4 prefill GEMMs and the current
model (lab/gemm_model.py: cap 500 lanes in flight, latency 570). Reports:

1. ranking quality per GEMM: Spearman, top-1 regret, pairwise order accuracy by measured gap;
2. how the log error splits into a per-tile part (same sign on every GEMM), a per-GEMM part and a
   residual: a per-tile error that is as large as the spread between good tiles misranks them even
   when the median error is small;
3. which tile features the per-tile error follows (warps/SM, blocks/SM, stages, tile shape, regs).

  python scripts/rank_analysis.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402,F401  (puts the CUDA DLLs nvrtc needs on the search path)
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
from eval_gemm_model import MEASURED, P, SHAPES  # noqa: E402

from lab import gemm_model as GM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tiles_and_times():
    held = json.loads((ROOT / "data" / "gemm_model_heldout_preds.json").read_text())["measured"]["rows"]
    rows = [(t, {g: statistics.median(v) for g, v in m.items()}, "in") for t, m in MEASURED.values() if not t.ABL]
    rows += [(GM.Tile(*r["tile"]), {g: r[g] for g, _, _ in SHAPES}, "held") for r in held]
    return rows


def main():
    kv = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    if "queue" in kv:
        GM.CP_QUEUE = None if kv["queue"] == "none" else float(kv["queue"])
    if "cap" in kv:
        GM.CP_INFLIGHT = None if kv["cap"] == "none" else float(kv["cap"])
    print(f"model: CP_QUEUE {GM.CP_QUEUE}, CP_INFLIGHT {GM.CP_INFLIGHT}")
    rows = tiles_and_times()
    built = {t: GM.build(t) for t, _, _ in rows}
    gs = [g for g, _, _ in SHAPES]
    M = np.array([[m[g] for g in gs] for _, m, _ in rows])
    Pr = np.array([[GM.predict(t, N, K, P, built[t])["us"] for g, N, K in SHAPES] for t, _, _ in rows])
    E = np.log(Pr / M)                                             # log error, tiles x gemms
    n = len(rows)
    print(f"{n} tiles x {len(gs)} GEMMs; median |error| {np.median(np.abs(Pr / M - 1)) * 100:.1f}%\n")

    print("1. ranking per GEMM")
    for j, g in enumerate(gs):
        m, p = M[:, j], Pr[:, j]
        rho = np.corrcoef(np.argsort(np.argsort(m)), np.argsort(np.argsort(p)))[0, 1]
        regret = m[np.argmin(p)] / m.min() - 1
        spread_top = np.sort(m)[:5] / m.min() - 1
        print(f"  {g:5s} spearman {rho:5.2f}  top-1 regret {regret * 100:5.1f}%  "
              f"best 5 measured within {spread_top[-1] * 100:4.1f}% of the best")
    bins = [(0, .05), (.05, .10), (.10, .20), (.20, 9)]
    cnt = {b: [0, 0] for b in bins}
    for j in range(len(gs)):
        for a in range(n):
            for b in range(a + 1, n):
                gap = abs(math.log(M[a, j] / M[b, j]))
                ok = (M[a, j] < M[b, j]) == (Pr[a, j] < Pr[b, j])
                for bb in bins:
                    if bb[0] <= gap < bb[1]:
                        cnt[bb][0] += ok
                        cnt[bb][1] += 1
    print("  pairwise order accuracy by measured gap: " +
          "  ".join(f"{int(b[0] * 100)}-{int(b[1] * 100) if b[1] < 9 else 'inf'}%: {c[0]}/{c[1]} = {c[0] / c[1]:.2f}"
                    for b, c in cnt.items()))

    print("\n2. log-error decomposition (two-way, tiles x GEMMs)")
    mu = E.mean()
    te = E.mean(1) - mu
    ge = E.mean(0) - mu
    res = E - mu - te[:, None] - ge[None, :]
    tot = ((E - mu) ** 2).sum()
    print(f"  per-tile {((te ** 2).sum() * len(gs)) / tot * 100:4.1f}%  per-GEMM {((ge ** 2).sum() * n) / tot * 100:4.1f}%  "
          f"residual {(res ** 2).sum() / tot * 100:4.1f}%   (sd: tile {te.std() * 100:.1f}%, residual {res.std() * 100:.1f}%)")
    meas_sd = np.log(M).std(0)
    good = [np.log(np.sort(M[:, j])[:6]).std() for j in range(len(gs))]
    print(f"  measured log-sd across tiles per GEMM {np.round(meas_sd * 100, 1)}%; among the best 6 {np.round(np.array(good) * 100, 1)}%")

    print("\n3. per-tile error vs features (Pearson r; positive = model too slow for larger feature)")
    feats = {}
    for i, (t, _, _) in enumerate(rows):
        occ = GM.blocks_per_sm(built[t]["regs"], t.nthr, built[t]["smem"])
        f = {"warps/block": t.nthr // 32, "blocks/SM": occ["blocks"], "warps/SM": occ["blocks"] * t.nthr // 32,
             "stages": t.NSTG, "KB": t.KB, "BM": t.BM, "BN": t.BN, "regs": built[t]["regs"],
             "smem KB": built[t]["smem"] / 1024, "stage lanes/warp": t.stage_lanes() / (t.nthr // 32)}
        for k, v in f.items():
            feats.setdefault(k, []).append(v)
    for k, v in feats.items():
        v = np.array(v, float)
        r = np.corrcoef(v, te)[0, 1] if v.std() > 0 else float("nan")
        print(f"  {k:18s} r = {r:+.2f}")
    print("\n  per tile: tile, held?, warps/SM, blocks/SM, per-tile error %, per-GEMM errors %")
    order = np.argsort(te)
    for i in order:
        t, _, split = rows[i]
        print(f"  {str((t.WM, t.WN, t.MT, t.NT, t.KB, t.NSTG, t.MINB)):24s} {split:4s} {feats['warps/SM'][i]:3d} "
              f"{feats['blocks/SM'][i]:2d}  {te[i] * 100:+6.1f}   " + " ".join(f"{(Pr[i, j] / M[i, j] - 1) * 100:+6.1f}" for j in range(len(gs))))
    out = {"queue": GM.CP_QUEUE, "cap": GM.CP_INFLIGHT, "tiles": [[*(rows[i][0].WM, rows[i][0].WN, rows[i][0].MT, rows[i][0].NT, rows[i][0].KB, rows[i][0].NSTG, rows[i][0].MINB)]
                     for i in range(n)], "split": [r[2] for r in rows], "measured": M.tolist(), "pred": Pr.tolist(),
           "feats": {k: list(map(float, v)) for k, v in feats.items()}}
    if "out" in kv:
        (ROOT / "data" / kv["out"]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
