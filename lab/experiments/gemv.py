"""4-bit-weight GEMV (LLM decode, batch 1) at Qwen2.5-7B shapes.

y[n] = sum_k s[n, k/128] * (q[n,k] - 8) * x[k], q packed 8 per uint32.
A straightforward CUDA kernel (one warp per output row, uint4 weight loads,
float x from L1/L2, warp-shuffle reduction) compiled by ptxas: this is the
"what a reasonable engineer writes" baseline that SASS-level work must beat.

Timing: every block records globaltimer start/end; kernel time is
max(end) - min(start). Output is checked against a float64 numpy reference.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

# (name, N rows, K cols) for one Qwen2.5-7B decoder layer
SHAPES = [("qkv_kv", 512, 3584), ("q_o", 3584, 3584), ("gate_up", 18944, 3584), ("down", 3584, 18944)]
GROUP = 128
ROWS_PER_BLOCK = 4  # one warp per row


@dataclass
class GemvInt4(Experiment):
    def __post_init__(self):
        self.name = "gemv_int4"
        self.target_opcode = "FFMA"
        self.description = "int4 (group-128 scale) GEMV, one warp per row, Qwen2.5-7B layer shapes"

    def source(self, v: Variant) -> str:
        return """
extern "C" __global__ void __launch_bounds__(128) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const float4* __restrict__ X,
    float* __restrict__ Y, unsigned long long* T, int N, int K)
{
  unsigned long long g0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  float acc = 0.f;
  if (row < N) {
    const uint4* wr = W + (size_t)row * (K / 32);      // 32 weights per uint4
    const float* sr = S + (size_t)row * (K / 128);
    for (int j = lane; j < K / 32; j += 32) {
      uint4 w = wr[j];
      float s = sr[j >> 2];                             // 4 uint4 (128 weights) per group
      const float4* xp = X + j * 8;                     // 32 floats
      unsigned ws[4] = {w.x, w.y, w.z, w.w};
      float part = 0.f;
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        float4 xa = xp[2 * u], xb = xp[2 * u + 1];
        unsigned q = ws[u];
        part = fmaf((float)((int)((q >>  0) & 15) - 8), xa.x, part);
        part = fmaf((float)((int)((q >>  4) & 15) - 8), xa.y, part);
        part = fmaf((float)((int)((q >>  8) & 15) - 8), xa.z, part);
        part = fmaf((float)((int)((q >> 12) & 15) - 8), xa.w, part);
        part = fmaf((float)((int)((q >> 16) & 15) - 8), xb.x, part);
        part = fmaf((float)((int)((q >> 20) & 15) - 8), xb.y, part);
        part = fmaf((float)((int)((q >> 24) & 15) - 8), xb.z, part);
        part = fmaf((float)((int)((q >> 28) & 15) - 8), xb.w, part);
      }
      acc = fmaf(s, part, acc);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[row] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1;
  }
}
"""

    def expected(self, v):
        return {}

    def variants(self, opts: dict) -> list[Variant]:
        return [Variant(name, {"N": n, "K": k, "warps": 4, "iters": 1}) for name, n, k in SHAPES]

    def prepare(self, dev, v: Variant) -> dict:
        n, k = v.params["N"], v.params["K"]
        rng = np.random.default_rng(n * 7 + k)
        q = rng.integers(0, 16, size=(n, k), dtype=np.uint32)
        packed = np.zeros((n, k // 8), np.uint32)
        for i in range(8):
            packed |= q[:, i::8] << (4 * i)
        s = rng.uniform(0.002, 0.02, size=(n, k // GROUP)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        ref = ((q.astype(np.float64) - 8) * np.repeat(s, GROUP, axis=1).astype(np.float64)) @ x.astype(np.float64)
        blocks = (n + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
        st = {"W": dev.alloc(packed.nbytes), "S": dev.alloc(s.nbytes), "X": dev.alloc(x.nbytes),
              "Y": dev.alloc(n * 4), "T": dev.alloc(blocks * 16), "ref": ref, "blocks": blocks,
              "bytes": packed.nbytes + s.nbytes + x.nbytes + n * 4}
        dev.htod(st["W"], packed)
        dev.htod(st["S"], s)
        dev.htod(st["X"], x)
        st["out"] = st["Y"]
        st["launch"] = dict(grid=blocks, block=128, args=[
            ctypes.c_uint64(st["W"]), ctypes.c_uint64(st["S"]), ctypes.c_uint64(st["X"]),
            ctypes.c_uint64(st["Y"]), ctypes.c_uint64(st["T"]), ctypes.c_int32(n), ctypes.c_int32(k)])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        t = dev.dtoh(np.zeros(2 * st["blocks"], np.uint64), st["T"]).reshape(-1, 2)
        ns = int(t[:, 1].max() - t[:, 0].min())
        y = dev.dtoh(np.zeros(v.params["N"], np.float32), st["Y"])
        err = float(np.max(np.abs(y - st["ref"]) / (np.abs(st["ref"]) + 1e-3)))
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3,
                "warp_ops_per_cycle": 0.0, "sm_mhz_inkernel": 0.0,
                "us": ns / 1e3, "gbps": st["bytes"] / ns, "max_rel_err": err,
                "correct": err < 1e-3}

    def release(self, dev, st: dict):
        for key in ("W", "S", "X", "Y", "T"):
            dev.free(st[key])


def registry():
    e = GemvInt4()
    return {e.name: e}
