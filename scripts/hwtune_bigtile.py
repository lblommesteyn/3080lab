"""Addendum 3 of data/hwtune/PROTOCOL.md: the whole prefill GEMM tile space on unseen (Llama-3-8B) shapes.

  CUDA_VISIBLE_DEVICES=-1 python scripts/hwtune_bigtile.py predict [j=8]   CPU: build + predict every tile
  pcslurm submit -- python scripts/hwtune_bigtile.py measure [rounds=3]      GPU: time every built tile
  python scripts/hwtune_bigtile.py score                                    budgeted search comparison

The tile space and both cost models are those of scripts/rank_fresh.py (lab/gemm_model.py, constants
fitted on Qwen2.5-1.5B timings only). Shapes here are Llama-3-8B's four linear layers at a 512-token
prompt, never timed in this repo. preds.json is committed before `measure` runs.
"""
import ctypes
import itertools
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
import _site  # noqa: E402,F401
OUT = ROOT / "data" / "hwtune" / "bigtile"
SHAPES = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gu", 28672, 4096), ("down", 4096, 14336)]
P = 512
MODELS = {"old": None, "queue": 128.0}
WARPS = [(2, 2, 2, 4), (4, 2, 2, 4), (8, 2, 2, 4), (4, 4, 2, 4), (2, 4, 2, 4), (4, 1, 2, 8), (2, 2, 4, 4),
         (4, 2, 1, 4), (2, 2, 2, 8), (8, 1, 2, 4), (4, 2, 2, 2), (2, 4, 2, 2), (8, 2, 1, 4), (4, 4, 1, 4),
         (2, 2, 4, 2), (4, 2, 4, 2), (8, 1, 2, 2)]
STAGING = [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (2, 5), (4, 4)]
DEFAULT = (2, 2, 2, 4, 2, 3, 3)
KS = (1, 2, 4, 8, 16)


def one(cfg):
    import torch  # noqa: F401  (CUDA DLLs for nvrtc)
    from lab import gemm_model as GM
    t = GM.Tile(*cfg)
    if any(N % t.BM or K % (32 * t.KB) for _, N, K in SHAPES):
        return None
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
    return row


def predict(j):
    cfgs = sorted({(*w, kb, ns, mb) for w in WARPS for kb, ns in STAGING for mb in (1, 2, 3)})
    with Pool(j) as pool:
        rows = [r for r in pool.map(one, cfgs) if r]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "shapes": SHAPES,
                                                "P": P, "models": MODELS, "all": rows}, indent=1))
    print(f"{len(cfgs)} configs, {len(rows)} built and predicted", flush=True)


def measure(rounds):
    import torch
    import batched_gemv_bench as BB
    from qwen_e2e import check, cu
    from lab import qwen_batched as QB
    d = json.loads((OUT / "preds.json").read_text())
    tiles = [tuple(r["tile"]) for r in d["all"]]
    if DEFAULT not in tiles:
        tiles.append(DEFAULT)
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
        del W, Sc
    kerns = {}
    for i, t in enumerate(tiles):
        WM, WN, MT, NT, KB, NSTG, MINB = t
        kerns[t] = BB.Kern(QB.gemm_q4i8_ms_source(WM, WN, MT, NT, "resid", KB, NSTG, MINB))
        if i % 25 == 0:
            print(f"compiled {i + 1}/{len(tiles)}", flush=True)
    # correctness: every tile's output on each shape must match the default tile's (same inputs)
    bad = []
    for g, (N, K, args, bufs) in data.items():
        outs = {}
        for t in (DEFAULT,) + tuple(x for x in tiles if x != DEFAULT):
            WM, WN, MT, NT, KB, NSTG, MINB = t
            BM, BN, nthr = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
            bufs[5].zero_()
            kerns[t].launch((P + BN - 1) // BN, nthr, args, N // BM)
            torch.cuda.synchronize()
            y = bufs[5].clone()
            if t == DEFAULT:
                ref = y
            elif not torch.allclose(y, ref, rtol=1e-3, atol=1e-2):
                bad.append((list(t), g, float((y - ref).abs().max())))
        print(f"checked {g}: {len(bad)} mismatches so far", flush=True)
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
    rows = [{"tile": list(t), "all": {g: [round(x, 3) for x in v] for g, v in times[t].items()}} for t in tiles]
    (OUT / "measured.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "rounds": rounds,
                                                   "mismatches": bad, "rows": rows}, indent=1))
    print(f"wrote measured.json: {len(rows)} tiles, {len(bad)} output mismatches", flush=True)


def score():
    d = json.loads((OUT / "preds.json").read_text())
    m = json.loads((OUT / "measured.json").read_text())
    P_ = {tuple(r["tile"]): r for r in d["all"]}
    bad = {tuple(b[0]) for b in m["mismatches"]}
    M = {tuple(r["tile"]): {g: statistics.median(v) for g, v in r["all"].items()} for r in m["rows"]}
    tiles = [t for t in P_ if t in M and t not in bad]
    rng = random.Random(0)
    gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))  # noqa: E731
    shapes = [g for g, _, _ in SHAPES]
    res = {g: {} for g in shapes}
    occ = max(tiles, key=lambda t: (P_[t]["warps_sm"], P_[t]["blocks_sm"]))
    for g in shapes:
        t = {x: M[x][g] for x in tiles}
        best = min(t.values())
        ts = list(t.values())
        r = res[g]
        r["default"] = M[DEFAULT][g] / best
        r["max_occupancy"] = t[occ] / best
        for k in KS:
            r[f"random({k})"] = statistics.mean(min(rng.sample(ts, k)) for _ in range(20000)) / best
            for mo in MODELS:
                order = sorted(tiles, key=lambda x: P_[x][mo][g])
                r[f"{mo}({k})"] = min(t[x] for x in order[:k]) / best
        # round-to-round noise of the measured best tile: max/min of its rounds
        bt = min(tiles, key=lambda x: t[x])
        allr = next(rr["all"][g] for rr in m["rows"] if tuple(rr["tile"]) == bt)
        r["noise_best"] = max(allr) / min(allr)
        mt = [P_[x]["old"][g] for x in tiles]
        r["spearman_old"] = _spearman(mt, [t[x] for x in tiles])
        r["spearman_queue"] = _spearman([P_[x]["queue"][g] for x in tiles], [t[x] for x in tiles])
    keys = [k for k in res[shapes[0]] if not k.startswith(("noise", "spearman"))]
    summ = {k: gm([res[g][k] for g in shapes]) for k in keys}
    lines = [f"{len(tiles)} tiles x {len(shapes)} Llama-3-8B shapes (P = {P}); regret = t / t(best tile), "
             f"medians of {m['rounds']} rounds; output mismatches: {len(bad)}; max_occupancy = {list(occ)}",
             f"{'policy':14s} " + " ".join(f"{g:>7s}" for g in shapes) + "  geomean"]
    for k in keys:
        lines.append(f"{k:14s} " + " ".join(f"{res[g][k]:7.3f}" for g in shapes) + f"  {summ[k]:7.3f}")
    lines.append("noise (best tile max/min over rounds): " + " ".join(f"{g} {res[g]['noise_best']:.3f}" for g in shapes))
    lines.append("spearman old: " + " ".join(f"{g} {res[g]['spearman_old']:+.2f}" for g in shapes)
                 + "   queue: " + " ".join(f"{g} {res[g]['spearman_queue']:+.2f}" for g in shapes))
    verdicts = {}
    for mo in MODELS:
        beats_all_k = all(summ[f"{mo}({k})"] < summ[f"random({k})"] for k in (1, 2, 4, 8))
        shapes_k4 = sum(res[g][f"{mo}(4)"] < res[g]["random(4)"] for g in shapes)
        beats_heur = summ[f"{mo}(1)"] < min(summ["default"], summ["max_occupancy"])
        need = next((k for k in (1, 2, 4, 8, 16) if summ[f"random({k})"] <= summ[f"{mo}(4)"]), None)
        verdicts[mo] = dict(beats_random_all_k=beats_all_k, shapes_won_k4=shapes_k4, beats_heuristics_k1=beats_heur,
                            random_budget_to_match_model4=need,
                            supported=beats_all_k and shapes_k4 >= 3 and beats_heur)
        lines.append(f"{mo}: beats random at k=1,2,4,8 {beats_all_k}; shapes won at k=4 {shapes_k4}/4; "
                     f"k=1 beats both heuristics {beats_heur}; random needs k={need} to match {mo}(4) "
                     f"-> {'SUPPORTED' if verdicts[mo]['supported'] else 'REFUTED'}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "score.txt").write_text(txt + "\n")
    (OUT / "score.json").write_text(json.dumps({"shapes": res, "summary": summ, "verdicts": verdicts}, indent=1))


def _spearman(a, b):
    n = len(a)
    ra = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: a[i]))}
    rb = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: b[i]))}
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    cmd = sys.argv[1]
    if cmd == "predict":
        predict(int(kv.get("j", 8)))
    elif cmd == "measure":
        measure(int(kv.get("rounds", 3)))
    else:
        score()
