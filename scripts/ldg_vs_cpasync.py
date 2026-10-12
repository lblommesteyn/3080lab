"""Per-SM L2 -> SM copy rate: cp.async.cg 16 B (to shared) against ld.global.cg 16 B (to registers),
same addresses, same L2-resident buffer, 1-3 blocks of 256 threads per SM, one wave (68 b blocks).
Answers whether the multistage GEMM's copy limit (~0.7 cp.async lanes / cycle / SM in place,
1.0 alone) is the L2 -> SM path or the cp.async port.

  pcslurm submit -- python scripts/ldg_vs_cpasync.py
"""
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batched_gemv_bench as BB  # noqa: E402
from qwen_e2e import check, cu  # noqa: E402

SRC = r"""
extern "C" __global__ void __launch_bounds__(256) k(const uint4* __restrict__ src, unsigned* out, int iters, int nbuf)
{
  // each block streams through an L2-resident buffer of nbuf uint4, 8 x 16 B per thread per trip
  __shared__ __align__(16) uint4 sm[8 * 256];
  const int t = threadIdx.x;
  unsigned acc = 0;
  for (int it = 0; it < iters; ++it) {
    const uint4* q = src + ((size_t)blockIdx.x * 4096 + (size_t)it * 2048) % (nbuf - 2048);
#if CP
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      unsigned d = (unsigned)__cvta_generic_to_shared(&sm[u * 256 + t]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(q + u * 256 + t));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 1;");
#else
    uint4 v[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(q + u * 256 + t));
    #pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
#if STS
    #pragma unroll
    for (int u = 0; u < 8; ++u) sm[u * 256 + t] = v[u];
#endif
#endif
  }
#if CP
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  acc = sm[t].x;
#endif
  if (acc == 0x12345678u) out[0] = acc;
}
"""


def main():
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    nbuf = 2 * 2**20 // 16                             # 2 MB: L2-resident
    buf = torch.randint(0, 2**30, (nbuf * 4,), dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    iters = 2000
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        pass
    for name, defs in (("cp.async", "#define CP 1\n#define STS 0\n"), ("ld.cg", "#define CP 0\n#define STS 0\n"),
                       ("ld.cg+st.shared", "#define CP 0\n#define STS 1\n")):
        k = BB.Kern(defs + SRC)
        for b in (1, 2, 3):                        # 32 KB shared each: 3 fit per SM
            grid = 68 * b
            fn = lambda: k.launch(grid, 256, [buf.data_ptr(), out.data_ptr(), ctypes.c_int32(iters), ctypes.c_int32(nbuf)])  # noqa: E731
            us = min(BB.timeit(fn, 5) for _ in range(3))
            mhz = pynvml.nvmlDeviceGetClockInfo(nvml, pynvml.NVML_CLOCK_SM) if nvml else 1900
            lanes = grid * 256 * 8 * iters / 68                # 16 B pieces per SM
            print(f"{name:16s} {b} blocks/SM: {us:8.1f} us  {lanes / (us * mhz):5.2f} lanes/cycle/SM  "
                  f"{grid * 256 * 8 * iters * 16 / us / 1e6:6.0f} GB/s  (regs {k.regs}, {mhz} MHz)", flush=True)


if __name__ == "__main__":
    main()
