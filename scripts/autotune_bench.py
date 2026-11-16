"""Budgeted autotuning benchmark for prefill GEMM tiles (docs/reports/autotune-benchmark.md).

  CUDA_VISIBLE_DEVICES=-1 python scripts/autotune_bench.py predict [j=8]  CPU: build every tile, predict
                                                                          every (tile, shape); frozen
  pcslurm submit -- python scripts/autotune_bench.py measure [rounds=3]     GPU: correctness check of every
                                                                          candidate, then interleaved timing
  python scripts/autotune_sim.py                                          CPU: search strategies on the table

Frozen protocol: data/autotune_bench/PROTOCOL.md. The cost model is lab/gemm_model.py with
CP_QUEUE = None (the "copy-cap" model; constants fitted on Qwen2.5-1.5B timings only); it is not
refitted here. The tile grid is scripts/rank_fresh.py's (17 warp layouts x 7 stagings x 3 MINB).
"""
import ctypes
import hashlib
import json
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

OUT = ROOT / "data" / "autotune_bench"
# (family, layer, N, K): fused qkv, attention out, fused gate+up, down. None of these was ever timed
# in this repo; the model's constants come from Qwen2.5-1.5B (N, K in {1536, 2048, 8960, 17920}).
LAYERS = [
    ("qwen2.5-7b", "qkv", 4608, 3584), ("qwen2.5-7b", "o", 3584, 3584),
    ("qwen2.5-7b", "gu", 37888, 3584), ("qwen2.5-7b", "down", 3584, 18944),
    ("gemma-2-9b", "qkv", 8192, 3584), ("gemma-2-9b", "o", 3584, 4096),
    ("gemma-2-9b", "gu", 28672, 3584), ("gemma-2-9b", "down", 3584, 14336),
    ("phi-3-mini", "qkv", 9216, 3072), ("phi-3-mini", "o", 3072, 3072),
    ("phi-3-mini", "gu", 16384, 3072), ("phi-3-mini", "down", 3072, 8192),
    ("qwen2.5-14b", "qkv", 7168, 5120), ("qwen2.5-14b", "o", 5120, 5120),
    ("qwen2.5-14b", "gu", 27648, 5120), ("qwen2.5-14b", "down", 5120, 13824),
]
PS = (256, 1024)
SHAPES = [(f"{f}/{l}/P{p}", N, K, p) for f, l, N, K in LAYERS for p in PS]
WARPS = [(2, 2, 2, 4), (4, 2, 2, 4), (8, 2, 2, 4), (4, 4, 2, 4), (2, 4, 2, 4), (4, 1, 2, 8), (2, 2, 4, 4),
         (4, 2, 1, 4), (2, 2, 2, 8), (8, 1, 2, 4), (4, 2, 2, 2), (2, 4, 2, 2), (8, 2, 1, 4), (4, 4, 1, 4),
         (2, 2, 4, 2), (4, 2, 4, 2), (8, 1, 2, 2)]
STAGING = [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (2, 5), (4, 4)]
DEFAULT = (2, 2, 2, 4, 2, 3, 3)
REL_TOL = 1e-2


def valid(t, N, K):
    WM, WN, MT, NT, KB = t[:5]
    return N % (16 * MT * WM) == 0 and K % (32 * KB) == 0


def one(cfg):
    import torch  # noqa: F401  (CUDA DLLs for nvrtc)
    from lab import gemm_model as GM
    t = GM.Tile(*cfg)
    t0 = time.perf_counter()
    try:
        b = GM.build(t)
    except (AssertionError, RuntimeError, KeyError, ValueError) as e:
        return {"tile": list(cfg), "built": False, "why": type(e).__name__}
    t_build = time.perf_counter() - t0
    if b.get("spills") or not b["body"]:
        return {"tile": list(cfg), "built": False, "why": "spills" if b.get("spills") else "empty"}
    occ = GM.blocks_per_sm(b["regs"], t.nthr, b["smem"])
    GM.CP_QUEUE = None                               # copy-cap model, as frozen
    t0 = time.perf_counter()
    pred = {s: round(GM.predict(t, N, K, P, b)["us"], 3) for s, N, K, P in SHAPES if valid(cfg, N, K)}
    t_pred = time.perf_counter() - t0
    return {"tile": list(cfg), "built": True, "regs": b["regs"], "smem": b["smem"], "blocks_sm": occ["blocks"],
            "warps_sm": occ["blocks"] * t.nthr // 32, "build_s": round(t_build, 3),
            "predict_s": round(t_pred, 4), "pred": pred}


def predict(j):
    cfgs = sorted({(*w, kb, ns, mb) for w in WARPS for kb, ns in STAGING for mb in (1, 2, 3)})
    t0 = time.perf_counter()
    with Pool(j) as pool:
        rows = pool.map(one, cfgs)
    built = [r for r in rows if r["built"]]
    per_shape = {s: sum(s in r["pred"] for r in built) for s, *_ in SHAPES}
    src = (ROOT / "lab" / "gemm_model.py").read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds.json").write_text(json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "model": "copy-cap (lab/gemm_model.py, CP_QUEUE=None)",
        "gemm_model_sha256": hashlib.sha256(src).hexdigest(), "shapes": SHAPES, "default": DEFAULT,
        "grid": len(cfgs), "built": len(built), "candidates_per_shape": per_shape,
        "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}, indent=1))
    print(f"{len(cfgs)} configs, {len(built)} built; candidates per shape "
          f"min {min(per_shape.values())} max {max(per_shape.values())}", flush=True)


def measure(rounds):
    import torch
    import batched_gemv_bench as BB
    from qwen_e2e import check, cu
    from lab import qwen_batched as QB
    d = json.loads((OUT / "preds.json").read_text())
    rows = {tuple(r["tile"]): r for r in d["rows"] if r["built"]}
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    torch.manual_seed(0)
    quant = BB.Kern(QB.QUANT_Q8)
    kerns, t0 = {}, time.perf_counter()
    for i, t in enumerate(rows):
        kerns[t] = BB.Kern(QB.gemm_q4i8_ms_source(*t[:4], "resid", *t[4:]))
        if i % 50 == 0:
            print(f"compiled {i + 1}/{len(rows)}", flush=True)
    kerns["AA"] = BB.Kern(QB.gemm_q4i8_ms_source(*DEFAULT[:4], "resid", *DEFAULT[4:]))   # A/A twin of DEFAULT
    print(f"compiled all in {time.perf_counter() - t0:.0f} s", flush=True)

    # inputs: one weight per layer, one correctly quantized activation per (layer, P)
    data, check_rows = {}, []
    weights = {}
    for f, l, N, K in LAYERS:
        W = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int64, device="cuda").to(torch.int32)
        Sc = (torch.rand(N, K // 32, device="cuda") * 0.02).half()
        Wp, Sp = QB.pack_q4_i8(W, Sc)
        weights[(f, l)] = (W, Sc, Wp, Sp)
    for s, N, K, P in SHAPES:
        f, l, _ = s.split("/")
        W, Sc, Wp, Sp = weights[(f, l)]
        X = torch.randn(P, K, device="cuda").to(torch.bfloat16)
        Xq = torch.zeros(P, K, dtype=torch.int8, device="cuda")
        Xd = torch.zeros(P, K // 32, device="cuda")
        Xs = torch.zeros(2 * P, K // 32, dtype=torch.int32, device="cuda")
        quant.launch((P * K // 32 + 7) // 8, 256, [X.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(),
                                                   ctypes.c_int32(P), ctypes.c_int32(K)])
        parts = []
        for r0 in range(0, N, 1024):
            w = W[r0:r0 + 1024].to(torch.int64) & 0xFFFFFFFF
            q = torch.stack([(w >> (4 * i)) & 15 for i in range(8)], -1).reshape(-1, K).float() - 8
            parts.append(X.float() @ (q * Sc[r0:r0 + 1024].float().repeat_interleave(32, 1)).t())
        ref = torch.cat(parts, 1)
        del parts, w, q, X
        Y = torch.zeros(P, N, device="cuda")
        args = [Wp.data_ptr(), Sp.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(), Y.data_ptr(), 0,
                ctypes.c_int32(N), ctypes.c_int32(K), ctypes.c_int32(P)]
        cands = [t for t in rows if s in rows[t]["pred"]] + ["AA"]
        ok = []
        for t in cands:
            tt = DEFAULT if t == "AA" else t
            WM, WN, MT, NT = tt[:4]
            Y.zero_()
            try:
                kerns[t].launch((P + 8 * NT * WN - 1) // (8 * NT * WN), 32 * WM * WN, args, N // (16 * MT * WM))
                torch.cuda.synchronize()
                e = float((Y - ref).abs().max() / ref.abs().max())
            except Exception as ex:  # noqa: BLE001  (record and exclude)
                e = None
                print(f"launch failed {s} {t}: {ex}", flush=True)
            passed = e is not None and e < REL_TOL
            check_rows.append({"shape": s, "tile": t if t == "AA" else list(t), "rel_err": e, "pass": passed})
            if passed:
                ok.append(t)
        del ref
        data[s] = (N, K, P, args, (Xq, Xd, Xs, Y), ok)
        print(f"checked {s}: {len(ok) - ('AA' in ok)} / {len(cands) - 1} candidates pass", flush=True)
    (OUT / "check.json").write_text(json.dumps({"rel_tol": REL_TOL, "rows": check_rows}, indent=1))

    # timing: every round times every (shape, passing tile) once, all pairs in a fresh random order
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)
    for rnd in range(rounds):
        f_out = raw / f"round{rnd}.json"
        if f_out.exists():
            print(f"round {rnd} exists, skipping", flush=True)
            continue
        pairs = [(s, t) for s in data for t in data[s][5]]
        random.Random(1000 + rnd).shuffle(pairs)
        res, t_r = [], time.perf_counter()
        for i, (s, t) in enumerate(pairs):
            N, K, P, args = data[s][:4]
            tt = DEFAULT if t == "AA" else t
            WM, WN, MT, NT = tt[:4]
            fn = lambda: kerns[t].launch((P + 8 * NT * WN - 1) // (8 * NT * WN), 32 * WM * WN, args,  # noqa: E731
                                         N // (16 * MT * WM))
            w0 = time.perf_counter()
            try:
                us = BB.timeit(fn, 20)
            except Exception as ex:  # noqa: BLE001
                us = None
                print(f"timing failed {s} {t}: {ex}", flush=True)
            res.append({"shape": s, "tile": t if t == "AA" else list(t), "us": us,
                        "wall_s": round(time.perf_counter() - w0, 3)})
            if i % 2000 == 0:
                print(f"round {rnd}: {i}/{len(pairs)} ({time.perf_counter() - t_r:.0f} s)", flush=True)
        f_out.write_text(json.dumps({"round": rnd, "seed": 1000 + rnd, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "rows": res}))
        print(f"round {rnd + 1}/{rounds} done in {time.perf_counter() - t_r:.0f} s", flush=True)
    print("measure done", flush=True)


def profile():
    """Where does one measurement's wall time go? (timing-harness diagnosis, no results recorded)"""
    import torch
    import batched_gemv_bench as BB
    from qwen_e2e import check, cu
    from lab import qwen_batched as QB
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    k = BB.Kern(QB.gemm_q4i8_ms_source(*DEFAULT[:4], "resid", *DEFAULT[4:]))
    for s, N, K, P in (SHAPES[2], SHAPES[5]):
        W = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int64, device="cuda").to(torch.int32)
        Sc = (torch.rand(N, K // 32, device="cuda") * 0.02).half()
        Wp, Sp = QB.pack_q4_i8(W, Sc)
        Xq = torch.zeros(P, K, dtype=torch.int8, device="cuda"); Xd = torch.zeros(P, K // 32, device="cuda")
        Xs = torch.zeros(2 * P, K // 32, dtype=torch.int32, device="cuda"); Y = torch.zeros(P, N, device="cuda")
        args = [Wp.data_ptr(), Sp.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(), Y.data_ptr(), 0,
                ctypes.c_int32(N), ctypes.c_int32(K), ctypes.c_int32(P)]
        fn = lambda: k.launch((P + 63) // 64, 128, args, N // 64)  # noqa: E731
        for _ in range(3):
            t0 = time.perf_counter(); us = BB.timeit(fn, 20); t1 = time.perf_counter()
            fn(); torch.cuda.synchronize(); t2 = time.perf_counter()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    fn()
            torch.cuda.synchronize(); t3 = time.perf_counter()
            print(f"{s}: kernel {us:.1f} us; timeit wall {t1 - t0:.3f} s; one launch+sync {t2 - t1:.4f} s; "
                  f"graph capture {t3 - t2:.3f} s", flush=True)


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    cmd = sys.argv[1]
    if cmd == "predict":
        predict(int(kv.get("j", 8)))
    elif cmd == "profile":
        profile()
    elif cmd == "measure":
        measure(int(kv.get("rounds", 3)))
