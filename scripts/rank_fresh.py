"""Blind tile-ranking test for lab/gemm_model.py (Experiment B, Oct 2026).

  python scripts/rank_fresh.py predict [j=8]      CPU: predict every tile config with two models, pick a
                                                  fresh test set, write data/rank_fresh/preds.json
  pcslurm submit -- python scripts/rank_fresh.py measure [rounds=3]
                                                  GPU: time the test set (interleaved rounds), write measured.json
  python scripts/rank_fresh.py score              compare: Spearman, top-1 regret, pairwise accuracy

Models: "old" is the Oct 11 model (copy cap 500 lanes, latency 570). "queue" adds the copy-issue
throttle (an LDGSTS stalls its warp while more than CP_QUEUE = 128 lanes wait for the copy port),
chosen on the 88 old timings (scripts/rank_analysis.py). The test set contains no tile timed before
Oct 18 (the 20 of scripts/rank_analysis.py), and includes tiles over 48 KB of shared memory that
could not be built before Oct 15. preds.json is committed before `measure` runs.
"""
import ctypes
import json
import math
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "rank_fresh"
SHAPES = [("qkv", 2048, 1536), ("o", 1536, 1536), ("gu", 17920, 1536), ("down", 1536, 8960)]
P = 512
MODELS = {"old": None, "queue": 128.0}
WARPS = [(2, 2, 2, 4), (4, 2, 2, 4), (8, 2, 2, 4), (4, 4, 2, 4), (2, 4, 2, 4), (4, 1, 2, 8), (2, 2, 4, 4),
         (4, 2, 1, 4), (2, 2, 2, 8), (8, 1, 2, 4), (4, 2, 2, 2), (2, 4, 2, 2), (8, 2, 1, 4), (4, 4, 1, 4),
         (2, 2, 4, 2), (4, 2, 4, 2), (8, 1, 2, 2)]
STAGING = [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (2, 5), (4, 4)]


def seen():
    import rank_analysis as RA
    return {(t.WM, t.WN, t.MT, t.NT, t.KB, t.NSTG, t.MINB) for t, _, _ in RA.tiles_and_times()}


def one(cfg):
    import torch  # noqa: F401  (CUDA DLLs for nvrtc)
    from lab import gemm_model as GM
    t = GM.Tile(*cfg)
    try:
        b = GM.build(t)
    except (AssertionError, RuntimeError, KeyError, ValueError):
        return None
    if b.get("spills") or not b["body"]:
        return None
    occ = GM.blocks_per_sm(b["regs"], t.nthr, b["smem"])
    row = {"tile": list(cfg), "regs": b["regs"], "smem": b["smem"], "blocks_sm": occ["blocks"],
           "warps_sm": occ["blocks"] * t.nthr // 32}
    for m, q in MODELS.items():
        GM.CP_QUEUE = q
        row[m] = {g: round(GM.predict(t, N, K, P, b)["us"], 2) for g, N, K in SHAPES}
    print(json.dumps(row), flush=True)
    return row


def predict(j):
    old = seen()
    cfgs = sorted({(*w, kb, ns, mb) for w in WARPS for kb, ns in STAGING for mb in (1, 2, 3)} - old)
    print(f"{len(cfgs)} unseen configs", flush=True)
    with Pool(j) as pool:
        rows = [r for r in pool.map(one, cfgs) if r]
    # test set: per model the best 6 by layer time, then random fill from either model's top 30, to 18
    layer = lambda r, m: sum(r[m].values())  # noqa: E731
    pick = []
    for m in MODELS:
        for r in sorted(rows, key=lambda r: layer(r, m))[:6]:
            if r not in pick:
                pick.append(r)
    pool_ = [r for m in MODELS for r in sorted(rows, key=lambda r: layer(r, m))[:30] if r not in pick]
    rnd = random.Random(18)
    uniq = []
    for r in pool_:
        if r not in uniq:
            uniq.append(r)
    rnd.shuffle(uniq)
    pick += uniq[:max(0, 18 - len(pick))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds.json").write_text(json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "models": MODELS, "all": rows,
        "test": [r["tile"] for r in pick],
        "anchors": [[2, 2, 2, 4, 2, 3, 3], [4, 2, 2, 4, 4, 2, 2], [4, 2, 2, 2, 4, 2, 1]]}, indent=1))
    print(f"{len(rows)} built; test set {len(pick)} tiles", flush=True)


def measure(rounds):
    import torch
    import batched_gemv_bench as BB
    from qwen_e2e import check, cu
    from lab import qwen_batched as QB
    d = json.loads((OUT / "preds.json").read_text())
    tiles = [tuple(t) for t in d["test"] + d["anchors"]]
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    torch.manual_seed(0)
    data = {}
    for g, N, K in SHAPES:
        W = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int64, device="cuda").to(torch.int32)
        Sc = (torch.rand(N, K // 32, device="cuda") * 0.02).half()
        Wp, Sp = QB.pack_q4_i8(W, Sc)
        Xq = torch.randint(-127, 128, (P, K), dtype=torch.int8, device="cuda")
        Xd = torch.rand(P, K // 32, device="cuda") * 0.01
        Xs = torch.randint(-2**20, 2**20, (2 * P, K // 32), dtype=torch.int32, device="cuda")
        Y = torch.zeros(P, N, device="cuda")
        data[g] = (N, K, [Wp.data_ptr(), Sp.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(), Y.data_ptr(), 0,
                          ctypes.c_int32(N), ctypes.c_int32(K), ctypes.c_int32(P)], (Wp, Sp, Xq, Xd, Xs, Y))
    kerns = {}
    for t in tiles:
        WM, WN, MT, NT, KB, NSTG, MINB = t
        kerns[t] = BB.Kern(QB.gemm_q4i8_ms_source(WM, WN, MT, NT, "resid", KB, NSTG, MINB))
    times = {t: {g: [] for g, _, _ in SHAPES} for t in tiles}
    for rnd in range(rounds):
        order = list(tiles)
        random.Random(rnd).shuffle(order)
        for t in order:
            WM, WN, MT, NT, KB, NSTG, MINB = t
            BM, BN, nthr = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
            for g, (N, K, args, _) in data.items():
                fn = lambda: kerns[t].launch((P + BN - 1) // BN, nthr, args, N // BM)  # noqa: E731
                times[t][g].append(BB.timeit(fn, 20))
        print(f"round {rnd + 1}/{rounds} done", flush=True)
    rows = [{"tile": list(t), **{g: round(min(v), 2) for g, v in times[t].items()},
             "all": {g: [round(x, 2) for x in v] for g, v in times[t].items()}} for t in tiles]
    for r in rows:
        print(json.dumps({k: r[k] for k in ("tile", "qkv", "o", "gu", "down")}), flush=True)
    (OUT / "measured.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "rounds": rounds,
                                                   "rows": rows}, indent=1))


def score():
    d = json.loads((OUT / "preds.json").read_text())
    m = {tuple(r["tile"]): r for r in json.loads((OUT / "measured.json").read_text())["rows"]}
    pr = {tuple(r["tile"]): r for r in d["all"]}
    test = [tuple(t) for t in d["test"]]

    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for k, i in enumerate(o):
            r[i] = k
        return r
    for name in MODELS:
        print(f"\nmodel {name}")
        acc = {"near": [0, 0], "all": [0, 0]}
        for g, _, _ in SHAPES:
            meas = [m[t][g] for t in test]
            pred = [pr[t][name][g] for t in test]
            ra, rb = rank(meas), rank(pred)
            n = len(test)
            rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(ra, rb)) / (n * (n * n - 1))
            best = min(meas)
            regret = meas[pred.index(min(pred))] / best - 1
            near = [i for i in range(n) if meas[i] <= best * 1.10]
            for a in range(n):
                for b in range(a + 1, n):
                    ok = (meas[a] < meas[b]) == (pred[a] < pred[b])
                    acc["all"][0] += ok
                    acc["all"][1] += 1
                    if a in near and b in near:
                        acc["near"][0] += ok
                        acc["near"][1] += 1
            err = statistics.median(abs(p / q - 1) for p, q in zip(pred, meas))
            print(f"  {g:5s} spearman {rho:5.2f}  top-1 regret {regret * 100:5.1f}%  median |err| {err * 100:4.1f}%  "
                  f"{len(near)} tiles within 10% of the best")
        print(f"  pairwise accuracy: all {acc['all'][0]}/{acc['all'][1]} = {acc['all'][0] / acc['all'][1]:.2f}, "
              f"near-optimal pairs {acc['near'][0]}/{acc['near'][1]} = {acc['near'][0] / max(1, acc['near'][1]):.2f}")


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    cmd = sys.argv[1]
    if cmd == "predict":
        predict(int(kv.get("j", 8)))
    elif cmd == "measure":
        measure(int(kv.get("rounds", 3)))
    else:
        score()
