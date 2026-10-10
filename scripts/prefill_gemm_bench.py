"""Prefill GEMM: int8 tensor-core Q4_0 (lab/qwen_batched.gemm_q4i8_source) vs cuBLAS on fp16-dequantized
weights, Qwen2.5-1.5B shapes, P tokens.

  pcslurm submit -- python scripts/prefill_gemm_bench.py [P=512] [WM=4 WN=2 MT=2 NT=4 KB=4 MINB=1] [MS=1 NSTG=3]

Error: max |Y - ref| / max |ref|, ref = fp32 X @ fp32 dequantized W. Ours includes the per-32 int8
quantization of X (as llama.cpp's MMQ); its time is reported separately. Timing as batched_gemv_bench.
"""
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batched_gemv_bench as BB  # noqa: E402
from qwen_e2e import check, cu  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402

SHAPES = [("qkv", 2048, 1536), ("o", 1536, 1536), ("gu", 17920, 1536), ("down", 1536, 8960)]


def main():
    cfg = {"P": 512, "WM": 4, "WN": 2, "MT": 2, "NT": 4, "KB": 4, "MINB": 1, "MS": 0, "NSTG": 3, "ABL": ""}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        cfg[k] = int(v) if v.lstrip("-").isdigit() else v
    print("config", cfg, flush=True)
    P = cfg["P"]
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    torch.manual_seed(0)
    quant = BB.Kern(QB.QUANT_Q8)
    tot_o = tot_c = 0.0
    for name, N, K in SHAPES:
        W = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int64, device="cuda").to(torch.int32)
        Sc = (torch.rand(N, K // 32, device="cuda") * 0.02).half()
        w = W.to(torch.int64) & 0xFFFFFFFF
        q = torch.stack([(w >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K).float() - 8
        w32 = q * Sc.float().repeat_interleave(32, 1)
        w16 = w32.half()
        Wp, Sp = QB.pack_q4_i8(W, Sc)
        X = torch.randn(P, K, device="cuda").to(torch.bfloat16)
        ref = X.float() @ w32.t()
        Xq = torch.zeros(P, K, dtype=torch.int8, device="cuda")
        Xd = torch.zeros(P, K // 32, device="cuda")
        Xs = torch.zeros(2 * P, K // 32, dtype=torch.int32, device="cuda")    # (c0, c1, c0, c1) quads per token pair
        nb = P * K // 32
        qargs = [X.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(), ctypes.c_int32(P), ctypes.c_int32(K)]
        gen = (lambda *a: QB.gemm_q4i8_ms_source(*a[:6], cfg["NSTG"], a[6], cfg["ABL"])) if cfg["MS"] else QB.gemm_q4i8_source
        kern = BB.Kern(gen(cfg["WM"], cfg["WN"], cfg["MT"], cfg["NT"], "resid", cfg["KB"], cfg["MINB"]))
        BM, BN = 16 * cfg["MT"] * cfg["WM"], 8 * cfg["NT"] * cfg["WN"]
        Y = torch.zeros(P, N, device="cuda")
        gargs = [Wp.data_ptr(), Sp.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(), Y.data_ptr(), 0,
                 ctypes.c_int32(N), ctypes.c_int32(K), ctypes.c_int32(P)]

        def run_q():
            quant.launch((nb + 7) // 8, 256, qargs)

        def run_g():
            kern.launch((P + BN - 1) // BN, 32 * cfg["WM"] * cfg["WN"], gargs, N // BM)
        run_q(); run_g()
        torch.cuda.synchronize()
        err = float((Y - ref).abs().max() / ref.abs().max())
        Yc = torch.matmul(X.half(), w16.t()).float()
        errc = float((Yc - ref).abs().max() / ref.abs().max())
        tq, tg = BB.timeit(run_q, 10), BB.timeit(run_g, 10)
        tc = BB.timeit(lambda: torch.matmul(X.half(), w16.t()), 10)
        tf = 2 * P * N * K / tg / 1e6
        tot_o += tq + tg
        tot_c += tc
        print(f"{name:5s} N={N:5d} K={K:5d}: ours {tg:7.1f} us ({tf:5.1f} TOPS, {kern.regs}r, err {err:.1e}) + quant {tq:5.1f} us"
              f" | cuBLAS fp16 {tc:7.1f} us (err {errc:.1e}) | {tc / (tg + tq):4.2f}x", flush=True)
    print(f"per layer: ours {tot_o:.0f} us, cuBLAS {tot_c:.0f} us", flush=True)


if __name__ == "__main__":
    main()
