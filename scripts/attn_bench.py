"""Time the fused decode-attention kernel vs context position (graph-replayed).
  pcslurm submit -- python scripts/attn_bench.py
"""
import ctypes
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_e2e import check, cu  # noqa: E402

from lab import qwen_kernels as KS  # noqa: E402
from lab import toolchain  # noqa: E402

NH, NKV, HD, MAXLEN = 12, 2, 128, 512


def main(srcs):
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    dev = "cuda"
    qkv = torch.randn((NH + 2 * NKV) * HD, device=dev).to(torch.bfloat16)
    kc = torch.randn(NKV, MAXLEN, HD, device=dev).to(torch.bfloat16)
    vc = torch.randn_like(kc)
    cos = torch.rand(MAXLEN, HD // 2, device=dev)
    sin = torch.rand(MAXLEN, HD // 2, device=dev)
    out = torch.zeros(NH * HD, dtype=torch.bfloat16, device=dev)
    part = torch.zeros(NH * (MAXLEN // 64) * (HD + 2), device=dev)
    counter = torch.zeros(NH, dtype=torch.int32, device=dev)
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    results = {}
    for name, (src, grid, block) in srcs.items():
        fn = check(cu.cuModuleGetFunction(check(cu.cuModuleLoadData(toolchain.build(src).cubin)), b"k"))
        a = [ctypes.c_uint64(t.data_ptr()) for t in (qkv, cos, sin, pos, kc, vc, out)] + \
            [ctypes.c_int32(NH), ctypes.c_int32(NKV), ctypes.c_float(1 / math.sqrt(HD)),
             ctypes.c_uint64(part.data_ptr()), ctypes.c_uint64(counter.data_ptr())]
        p = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
        for P in (40, 160, 300, 500):
            pos.fill_(P)

            def f():
                st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
                check(cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, st, ctypes.addressof(p), 0))
            s0 = torch.cuda.Stream()
            with torch.cuda.stream(s0):
                f()
            torch.cuda.synchronize()
            out.zero_()
            with torch.cuda.stream(s0):
                f()
            torch.cuda.synchronize()
            ref = out.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(50):
                    f()
            import time as _t
            t_end = _t.perf_counter() + 0.3
            while _t.perf_counter() < t_end:
                g.replay()
                torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            results[(name, P)] = (s.elapsed_time(e) / 500 * 1e3, ref)
            print(f"{name:<10} pos {P:3d}: {results[(name, P)][0]:6.2f} us", flush=True)
    names = list(srcs)
    for P in (40, 160, 300, 500):
        if len(names) > 1:
            a, b = results[(names[0], P)][1].float(), results[(names[1], P)][1].float()
            print(f"pos {P}: max |{names[0]} - {names[1]}| = {float((a - b).abs().max()):.3e}")


if __name__ == "__main__":
    main({"attn_v1": (KS.ATTN % {"maxlen": MAXLEN}, NH, 128),
          "attn_v4": (KS.ATTN4 % {"maxlen": MAXLEN}, NH, 512)})
