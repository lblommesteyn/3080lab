"""Kernel-launch cost inside a CUDA graph vs grid size, block size, parameter count, and static smem.
Each config: 200 identical empty launches captured in one graph, clocks warmed, time per launch."""
import ctypes
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_e2e import check, cu  # noqa: E402

from lab import toolchain  # noqa: E402


def src(nparams: int, smem: int) -> str:
    ps = ", ".join(f"const unsigned* p{i}" for i in range(nparams))
    sm = f"__shared__ unsigned s[{smem // 4}]; if (threadIdx.x == 999) s[0] = 1; " if smem else ""
    return f'extern "C" __global__ void k({ps}) {{ {sm}if (threadIdx.x == 0 && blockIdx.x == 100000) ((unsigned*)p0)[0] = 1; }}'


def main():
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    buf = torch.zeros(16, dtype=torch.int32, device="cuda")
    fns = {}
    for blocks, threads, nparams, smem in [(1, 512, 1, 0), (12, 512, 1, 0), (12, 512, 10, 0), (12, 512, 10, 5696),
                                           (12, 128, 1, 0), (68, 128, 1, 0), (512, 64, 7, 0), (4480, 64, 7, 0),
                                           (12, 1024, 1, 0), (68, 512, 1, 0)]:
        key = (nparams, smem)
        if key not in fns:
            fns[key] = check(cu.cuModuleGetFunction(check(cu.cuModuleLoadData(toolchain.build(src(nparams, smem)).cubin)), b"k"))
        fn = fns[key]
        args = [ctypes.c_uint64(buf.data_ptr()) for _ in range(nparams)]
        p = (ctypes.c_void_p * nparams)(*[ctypes.addressof(a) for a in args])

        def f():
            st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
            check(cu.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, st, ctypes.addressof(p), 0))
        s0 = torch.cuda.Stream()
        with torch.cuda.stream(s0):
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(200):
                f()
        t_end = time.perf_counter() + 0.3
        while time.perf_counter() < t_end:
            g.replay()
            torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        print(f"blocks {blocks:5d} x {threads:4d} thr, {nparams:2d} params, smem {smem:5d} B: "
              f"{s.elapsed_time(e) / 2000 * 1e3:5.2f} us/launch", flush=True)


if __name__ == "__main__":
    main()
