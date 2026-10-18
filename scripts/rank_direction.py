"""Single-variable direction test for lab/gemm_model.py (Experiment B, Oct 2026).

  python scripts/rank_direction.py predict        CPU: build every variant, write the models' predicted
                                                  direction to data/rank_direction/preds.json
  pcslurm submit -- python scripts/rank_direction.py measure [rounds=3]
  python scripts/rank_direction.py score          predicted vs measured direction, per factor

Each variant changes ONE thing relative to a base tile:
  occ   the same cubin, launched with padding dynamic shared memory so the SM holds k blocks instead of
        its natural count (occupancy alone: code, registers, copies and stages are identical)
  minb  __launch_bounds__ min blocks (register cap; ptxas may recompile the loop around it)
  nstg  cp.async ring depth +-1 (copy/compute overlap and shared memory per block)
  kb    k-chunk per stage (shared memory per stage, stage count, copy granularity)
A change is "flat" when it moves the time by less than FLAT (2%, about twice the harness CV); direction
accuracy counts only pairs the measurement calls non-flat, and separately how often the model calls a
measured-flat change flat. Bases: the three anchors of scripts/rank_fresh.py plus the six best tiles
of the "queue" model in data/rank_fresh/preds.json. preds.json is committed before `measure` runs.
"""
import ctypes
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "rank_direction"
SHAPES = [("qkv", 2048, 1536), ("o", 1536, 1536), ("gu", 17920, 1536), ("down", 1536, 8960)]
P = 512
MODELS = {"old": None, "queue": 128.0}
FLAT = 0.02
SMEM_BLOCK_MAX = 101376          # sm_86 max shared memory per block (static + dynamic)


def bases():
    d = json.loads((ROOT / "data" / "rank_fresh" / "preds.json").read_text())
    out = [tuple(t) for t in d["anchors"]]
    best = sorted(d["all"], key=lambda r: sum(r["queue"].values()))
    for r in best:
        if len(out) >= 9:
            break
        if tuple(r["tile"]) not in out:
            out.append(tuple(r["tile"]))
    return out


def neighbours(t):
    WM, WN, MT, NT, KB, NSTG, MINB = t
    out = []
    for m in (1, 2, 3):
        if m != MINB:
            out.append(("minb", (WM, WN, MT, NT, KB, NSTG, m)))
    for s in (NSTG - 1, NSTG + 1):
        if 2 <= s <= 5:
            out.append(("nstg", (WM, WN, MT, NT, KB, s, MINB)))
    for k in (2, 4):
        if k != KB:
            out.append(("kb", (WM, WN, MT, NT, k, NSTG, MINB)))
    return out


def occ_pad(GM, b, nthr, k):
    """Total shared memory per block that leaves exactly k blocks per SM (None if impossible)."""
    tot = GM.SMEM_PER_SM // k - GM.SMEM_RESERVED
    tot = min(tot, SMEM_BLOCK_MAX)
    if tot < b["smem"] or GM.blocks_per_sm(b["regs"], nthr, tot)["blocks"] != k:
        return None
    return tot


def predict():
    import torch  # noqa: F401  (CUDA DLLs for nvrtc)
    from lab import gemm_model as GM
    built, rows = {}, []

    def get(t):
        if t not in built:
            try:
                b = GM.build(GM.Tile(*t))
                built[t] = None if (b.get("spills") or not b["body"]) else b
            except (AssertionError, RuntimeError, KeyError, ValueError):
                built[t] = None
        return built[t]

    def pred(t, b, smem=None):
        bb = dict(b, smem=smem if smem is not None else b["smem"])
        res = {}
        for m, q in MODELS.items():
            GM.CP_QUEUE = q
            res[m] = {g: round(GM.predict(GM.Tile(*t), N, K, P, bb)["us"], 2) for g, N, K in SHAPES}
        return res

    for base in bases():
        b = get(base)
        if b is None:
            print("base does not build:", base, flush=True)
            continue
        nthr = GM.Tile(*base).nthr
        nat = GM.blocks_per_sm(b["regs"], nthr, b["smem"])["blocks"]
        rows.append({"base": list(base), "factor": "base", "tile": list(base), "pad_smem": 0,
                     "regs": b["regs"], "smem": b["smem"], "blocks_sm": nat, **pred(base, b)})
        for k in range(1, nat):
            tot = occ_pad(GM, b, nthr, k)
            if tot is None:
                continue
            rows.append({"base": list(base), "factor": "occ", "tile": list(base), "pad_smem": tot - b["smem"],
                         "regs": b["regs"], "smem": tot, "blocks_sm": k, **pred(base, b, tot)})
        for f, t in neighbours(base):
            bt = get(t)
            if bt is None:
                continue
            rows.append({"base": list(base), "factor": f, "tile": list(t), "pad_smem": 0, "regs": bt["regs"],
                         "smem": bt["smem"], "blocks_sm": GM.blocks_per_sm(bt["regs"], nthr, bt["smem"])["blocks"],
                         **pred(t, bt)})
        print(f"{base}: {sum(r['base'] == list(base) for r in rows)} rows", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "models": MODELS,
                                                "flat": FLAT, "rows": rows}, indent=1))
    print(f"{len(rows)} rows", flush=True)


def measure(rounds):
    import torch
    import batched_gemv_bench as BB
    from qwen_e2e import check, cu
    from lab import qwen_batched as QB
    rows = json.loads((OUT / "preds.json").read_text())["rows"]
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
    for i, r in enumerate(rows):
        t = tuple(r["tile"])
        if t not in kerns:
            kerns[t] = BB.Kern(QB.gemm_q4i8_ms_source(*t[:4], "resid", *t[4:]))
    times = {i: {g: [] for g, _, _ in SHAPES} for i in range(len(rows))}
    for rnd in range(rounds):
        order = list(range(len(rows)))
        random.Random(rnd).shuffle(order)
        for i in order:
            r = rows[i]
            WM, WN, MT, NT, KB, NSTG, MINB = r["tile"]
            BM, BN, nthr = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
            kern = kerns[tuple(r["tile"])]
            dyn = kern.smem + r["pad_smem"]
            check(cu.cuFuncSetAttribute(kern.fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        dyn))
            for g, (N, K, args, _) in data.items():
                def fn(kern=kern, dyn=dyn, BN=BN, nthr=nthr, args=args, BM=BM, N=N):
                    old = kern.smem
                    kern.smem = dyn
                    try:
                        kern.launch((P + BN - 1) // BN, nthr, args, N // BM)
                    finally:
                        kern.smem = old
                times[i][g].append(BB.timeit(fn, 20))
        print(f"round {rnd + 1}/{rounds} done", flush=True)
    out = [{**{k: r[k] for k in ("base", "factor", "tile", "pad_smem", "blocks_sm")},
            **{g: round(min(v), 2) for g, v in times[i].items()},
            "all": {g: [round(x, 2) for x in v] for g, v in times[i].items()}} for i, r in enumerate(rows)]
    for r in out:
        print(json.dumps({k: r[k] for k in ("base", "factor", "tile", "blocks_sm", "qkv", "o", "gu", "down")}), flush=True)
    (OUT / "measured.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "rounds": rounds,
                                                   "rows": out}, indent=1))


def _sign(x):
    return 0 if abs(x) < FLAT else (1 if x > 0 else -1)


def score():
    pr = json.loads((OUT / "preds.json").read_text())["rows"]
    me = json.loads((OUT / "measured.json").read_text())["rows"]
    key = lambda r: (tuple(r["base"]), r["factor"], tuple(r["tile"]), r["blocks_sm"] if r["factor"] == "occ" else 0)  # noqa: E731
    mm = {key(r): r for r in me}
    for name in MODELS:
        print(f"\nmodel {name}   (change relative to the base tile; flat = under {FLAT:.0%})")
        tally = {}
        for r in pr:
            if r["factor"] == "base":
                continue
            b = next(x for x in pr if x["factor"] == "base" and x["base"] == r["base"])
            mb, mr = mm[key(b)], mm[key(r)]
            for g, _, _ in SHAPES:
                dp = r[name][g] / b[name][g] - 1
                dm = mr[g] / mb[g] - 1
                t = tally.setdefault(r["factor"], {"n": 0, "ok": 0, "flat_n": 0, "flat_ok": 0, "wrong": []})
                if _sign(dm) == 0:
                    t["flat_n"] += 1
                    t["flat_ok"] += _sign(dp) == 0
                else:
                    t["n"] += 1
                    ok = _sign(dp) == _sign(dm)
                    t["ok"] += ok
                    if not ok:
                        t["wrong"].append((tuple(r["base"]), tuple(r["tile"]), r["blocks_sm"], g,
                                           round(dp * 100, 1), round(dm * 100, 1)))
        for f, t in tally.items():
            print(f"  {f:5s} direction right {t['ok']}/{t['n']}   measured-flat called flat {t['flat_ok']}/{t['flat_n']}")
            for w in t["wrong"][:8]:
                print(f"        wrong: base {w[0]} -> {w[1]} blocks/SM {w[2]} {w[3]}: predicted {w[4]:+}%, measured {w[5]:+}%")


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    cmd = sys.argv[1]
    if cmd == "predict":
        predict()
    elif cmd == "measure":
        measure(int(kv.get("rounds", 3)))
    else:
        score()
