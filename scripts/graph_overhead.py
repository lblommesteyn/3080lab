"""Cost of a kernel inside a CUDA graph on this card, vs grid size.

Captures 200 back-to-back launches of an empty kernel (each block writes one
word so it cannot be skipped), replays, and reports time per launch.
  pcslurm submit -- python scripts/graph_overhead.py
"""
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_e2e import check, cu  # noqa: E402

from lab import toolchain  # noqa: E402

SRC = r"""
extern "C" __global__ void k(int* p) { if (threadIdx.x == 0) p[blockIdx.x & 1023] = blockIdx.x; }
"""


def main():
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    mod = check(cu.cuModuleLoadData(toolchain.build(SRC).cubin))
    fn = check(cu.cuModuleGetFunction(mod, b"k"))
    buf = torch.zeros(1024, dtype=torch.int32, device="cuda")
    a = [ctypes.c_uint64(buf.data_ptr())]
    p = (ctypes.c_void_p * 1)(ctypes.addressof(a[0]))
    for blocks, threads in [(1, 32), (1, 512), (68, 128), (448, 128), (1024, 64), (4480, 32), (37984, 32)]:
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
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        print(f"grid {blocks:6d} x {threads:4d}: {s.elapsed_time(e) / 2000 * 1e3:6.2f} us per kernel in graph", flush=True)


if __name__ == "__main__":
    main()
