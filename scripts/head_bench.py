"""Batched Q6_K LM head (tensor cores, lab/qwen_batched.head_q6_mma_source) vs cuBLAS on the head
dequantized to fp16 (what scripts/qwen_batched_decode.py used), on Qwen2.5-1.5B's real head.

  pcslurm submit -- python scripts/head_bench.py [S=.. U=.. MT=..]

Error: max |logit difference| relative to max |logit| against fp32 x @ fp32 dequantized W, for
both kernels, and argmax agreement. Timing as scripts/batched_gemv_bench.py.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batched_gemv_bench as BB  # noqa: E402
import qwen_gguf as GG  # noqa: E402
from qwen_e2e import check, cu  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402
from lab.gguf import GGUF, q6_k_blocks  # noqa: E402


def main():
    cfg = {"S": 4, "U": 2, "MT": 2}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        cfg[k] = int(v)
    print("config", cfg, flush=True)
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    g = GGUF(GG.GGUF_PATH)
    m = GG.GGUFQwen(None)
    Lw, Hw, SCw, Dw = m.head
    b, t = g.raw("output.weight")
    K, N = t.dims
    q, s = q6_k_blocks(b)
    w32 = (torch.from_numpy(q.astype(np.float32)).cuda() * torch.from_numpy(s.astype(np.float32)).cuda().repeat_interleave(16, 1)).reshape(N, K)
    w16 = w32.half()
    torch.manual_seed(0)
    for B in (1, 4, 8, 16, 32):
        X = (torch.randn(B, K, device="cuda") * 2).to(torch.bfloat16).contiguous()
        ref = X.float() @ w32.t()
        kern = BB.Kern(QB.head_q6_mma_source(cfg["S"], B, cfg["U"], cfg["MT"]))
        Y = torch.zeros(B, N, device="cuda")
        grid = (N + 16 * cfg["MT"] - 1) // (16 * cfg["MT"])
        args = [Lw.data_ptr(), Hw.data_ptr(), SCw.data_ptr(), Dw.data_ptr(), X.data_ptr(), Y.data_ptr(),
                ctypes.c_int32(N), ctypes.c_int32(K)]

        def ours():
            kern.launch(grid, 32 * cfg["S"], args)

        def cublas():
            torch.matmul(X.half(), w16.t())
        ours()
        torch.cuda.synchronize()
        y16 = cublas_out = torch.matmul(X.half(), w16.t()).float()
        scale = ref.abs().max()
        e_ours = float((Y - ref).abs().max() / scale)
        e_cub = float((cublas_out - ref).abs().max() / scale)
        am = int((Y.argmax(-1) == ref.argmax(-1)).sum()), int((y16.argmax(-1) == ref.argmax(-1)).sum())
        to, tc = BB.timeit(ours, 10), BB.timeit(cublas, 10)
        gbs = (Lw.numel() * 4 + Hw.numel() * 4 + SCw.numel() + Dw.numel() * 2) / to / 1e3
        print(f"B={B:2d}: ours {to:6.1f} us ({gbs:4.0f} GB/s, {kern.regs}r, err {e_ours:.1e}, argmax {am[0]}/{B})"
              f" | cuBLAS fp16 {tc:6.1f} us (err {e_cub:.1e}, argmax {am[1]}/{B}) | {tc / to:4.2f}x", flush=True)


if __name__ == "__main__":
    main()
