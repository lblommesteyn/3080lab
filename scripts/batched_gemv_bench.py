"""Batched Q4_0 GEMV vs B launches of the single-vector kernel, on Qwen2.5-1.5B decode shapes.

  pcslurm submit -- python scripts/batched_gemv_bench.py [S]

Correctness: each batched output column must match the single-vector kernel on that vector
(relative error, fp32 accumulation in a different order). Timing: CUDA graph of 20 launches,
warmed clocks, median of 7 replays.
"""
import ctypes
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_e2e import check, cu  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402
from lab import qwen_kernels as KS  # noqa: E402
from lab import toolchain  # noqa: E402

SHAPES = [("qkv", 2048, 1536, "biasf"), ("o", 1536, 1536, "resid"), ("gu", 17920, 1536, "swiglu"),
          ("down", 1536, 8960, "resid")]


class Kern:
    def __init__(self, src):
        b = toolchain.build(src)
        self.regs = toolchain.ptxas_resources(b.ptxas_log)["registers"]
        self.fn = check(cu.cuModuleGetFunction(check(cu.cuModuleLoadData(b.cubin)), b"k"))
        self.keep = []

    def launch(self, grid, block, args, gy=1):
        a = [x if isinstance(x, (ctypes.c_uint64, ctypes.c_int32)) else ctypes.c_uint64(x) for x in args]
        p = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
        self.keep.append((a, p))
        st_ = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(self.fn, grid, gy, 1, block, 1, 1, 0, st_, ctypes.addressof(p), 0))


def timeit(fn, reps=20):
    s0 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    t_end = time.perf_counter() + 0.3
    while time.perf_counter() < t_end:
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / reps * 1e3)
    return st.median(out)


VARIANT = {}
MMA = False
BIG = False
V2 = False
V3 = False


def main():
    global VARIANT
    S_ = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    global MMA, BIG, V2, V3
    for a in sys.argv[2:]:
        if a == "mma":
            MMA = True
            continue
        if a == "v2":              # S is ignored; VARIANT = WM, WK, MT, PF
            MMA = V2 = True
            continue
        if a == "v3":              # VARIANT = WM, WK, MT, NS, KS; fp16 X
            MMA = V2 = V3 = True
            continue
        if a == "big":
            BIG = True
            continue
        k, v = a.split("=")
        VARIANT[k] = int(v)
    print("variant", S_, VARIANT, flush=True)
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    dev = "cuda"
    torch.manual_seed(0)
    WS = torch.zeros(16 * 32 * 17920, device=dev)
    CNT = torch.zeros(4096, dtype=torch.int32, device=dev)
    for name, N, K, epi in SHAPES:
        W = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int64, device=dev).to(torch.int32)
        Sc = (torch.rand(N, K // 32, device=dev) * 0.02).half()
        aux = torch.randn(N, device=dev)
        single = Kern(KS.gemv_source(S_, epi, "g32f16"))
        nout = N // 2 if epi == "swiglu" else N
        ytype = torch.float32 if epi == "resid" else torch.bfloat16
        row = [f"{name:5s} N={N:5d} K={K:5d}"]
        for B in (((1, 2, 4, 8, 16, 32) if not BIG else (8, 16, 32)) if MMA else (1, 2, 4, 8)):
            X = torch.randn(B, K, device=dev).to(torch.bfloat16).contiguous()
            # reference: the single-vector kernel on each vector
            Yref = torch.zeros(B, nout, dtype=ytype, device=dev)
            for b in range(B):
                single.launch((N + 3) // 4, 32 * S_, [W.data_ptr(), Sc.data_ptr(), X[b].data_ptr(),
                                                       Yref[b].data_ptr(), aux.data_ptr(), ctypes.c_int32(N),
                                                       ctypes.c_int32(K)])
            Wb, Sb, blk, extra, gy = W, Sc, 32 * S_, [], 1
            if V2:
                v = {"WM": 4, "WK": 2, "MT": 2, "PF": 1, "KS": 1, "ACC16": 0, **VARIANT}
                if V3:
                    batched = Kern(QB.gemv_v3_source(v["WM"], v["WK"], v["MT"], B, epi, VARIANT.get("NS", 3), v["KS"]))
                else:
                    batched = Kern(QB.gemv_v2_source(v["WM"], v["WK"], v["MT"], B, epi, v["PF"], v["KS"], bool(v["ACC16"])))
                extra = [WS.data_ptr(), CNT.data_ptr()]
                gy = v["KS"]
                R = 16 * v["MT"] * v["WM"]
                grid_b, blk = (N + R - 1) // R, 32 * v["WM"] * v["WK"]
                Wb, Sb = QB.pack_q4_mma(W, Sc)
            else:
                batched = Kern(QB.gemv_mma_source(S_, B, epi, **VARIANT) if MMA else QB.gemv_batched_source(S_, B, epi, **VARIANT))
                grid_b = (N + 16 * VARIANT.get("MT", 1) - 1) // (16 * VARIANT.get("MT", 1)) if MMA else (N + 3) // 4
            Y = torch.zeros(B, nout, dtype=ytype, device=dev)
            Xin = X.half() if V3 else X
            batched.launch(grid_b, blk, [Wb.data_ptr(), Sb.data_ptr(), Xin.data_ptr(), Y.data_ptr(),
                                         aux.data_ptr(), ctypes.c_int32(N), ctypes.c_int32(K), *extra], gy)
            torch.cuda.synchronize()
            err = float((Y.float() - Yref.float()).abs().max() / Yref.float().abs().max().clamp_min(1e-6))

            def run_single():
                for b in range(B):
                    single.launch((N + 3) // 4, 32 * S_, [W.data_ptr(), Sc.data_ptr(), X[b].data_ptr(),
                                                           Yref[b].data_ptr(), aux.data_ptr(), ctypes.c_int32(N),
                                                           ctypes.c_int32(K)])

            def run_batched():
                batched.launch(grid_b, blk, [Wb.data_ptr(), Sb.data_ptr(), Xin.data_ptr(), Y.data_ptr(),
                                             aux.data_ptr(), ctypes.c_int32(N), ctypes.c_int32(K), *extra], gy)
            ts, tb = timeit(run_single), timeit(run_batched)
            row.append(f"B{B}: {ts:6.1f} -> {tb:6.1f} us ({ts / tb:4.2f}x, {batched.regs}r, err {err:.1e})")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()
