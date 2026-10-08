"""Parametrized int4 GEMV: rows per warp x dequantization scheme, Qwen2.5-7B shapes.

Knobs:
  R     rows per warp (1, 2, 4): each x chunk is loaded once and reused for R
        rows; R independent weight loads are in flight per iteration.
  deq   naive : ((q >> s) & 15) - 8 -> I2F -> FFMA        (what nvcc emits for plain C)
        magic : float bits 0x4B000000|q equal 2^23 + q exactly; one FADD of
                -(2^23+8) gives q-8 with no integer subtract and no convert
        half2 : Marlin-style: (w >> s) & 0x000F000F | 0x64006400 is a half2
                holding (1024+q_lo, 1024+q_hi); HSUB2 1032 gives (q-8) for two
                weights at once, HFMA2 against fp16 x, fp32 accumulate per
                uint32. Approximate (fp16 x and products).
Output checked against a float64 reference (tolerance by scheme).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant
from .gemv import GROUP, SHAPES

TOL = {"naive": 1e-3, "magic": 1e-3, "half2": 3e-2}


def _deq_block(deq: str, R: int) -> str:
    """Code for one uint4 (32 weights) per row r: accumulates into part[r]."""
    if deq in ("naive", "magic"):
        lines = []
        for u in range(4):
            for i in range(8):
                xv = f"x{u // 1}"  # unused
                comp = "xyzw"[i % 4]
                xs = f"xa{u}.{comp}" if i < 4 else f"xb{u}.{comp}"
                for r in range(R):
                    q = f"w{r}.{'xyzw'[u]}"
                    if deq == "naive":
                        val = f"(float)((int)(({q} >> {4 * i}) & 15) - 8)"
                    else:
                        val = f"(__int_as_float((({q} >> {4 * i}) & 15) | 0x4B000000) - 8388616.0f)"
                    lines.append(f"part[{r}] = fmaf({val}, {xs}, part[{r}]);")
        return "\n        ".join(lines)
    # half2: per uint32 q, four half2 pairs: nibbles (i, i+4) for i=0..3 via shift by 4*i
    lines = []
    for u in range(4):
        for r in range(R):
            q = f"w{r}.{'xyzw'[u]}"
            lines.append(f"{{ __half2 h = __float2half2_rn(0.f);")
            for i in range(4):
                lines.append(f"  {{ unsigned b = (({q} >> {4 * i}) & 0x000F000Fu) | 0x64006400u;"
                             f" __half2 v = __hsub2(*reinterpret_cast<__half2*>(&b), bias);"
                             f" h = __hfma2(v, xh{u}[{i}], h); }}")
            lines.append(f"  float2 f = __half22float2(h); part[{r}] += f.x + f.y; }}")
    return "\n        ".join(lines)


@dataclass
class Gemv2(Experiment):
    def __post_init__(self):
        self.name = "gemv_int4_v2"
        self.target_opcode = "FFMA"
        self.description = "int4 GEMV grid: rows/warp x dequant scheme (naive, magic fp32, half2)"

    def source(self, v: Variant) -> str:
        R, deq = v.params["R"], v.params["deq"]
        xload = []
        for u in range(4):
            xload.append(f"float4 xa{u} = xp[{2 * u}], xb{u} = xp[{2 * u + 1}];")
        if deq == "half2":
            # pair nibble i (low half) with nibble i+4 (high half): weights 8u+i and 8u+i+4
            for u in range(4):
                xs = [f"xa{u}.x", f"xa{u}.y", f"xa{u}.z", f"xa{u}.w", f"xb{u}.x", f"xb{u}.y", f"xb{u}.z", f"xb{u}.w"]
                pairs = ", ".join(f"__floats2half2_rn({xs[i]}, {xs[i + 4]})" for i in range(4))
                xload.append(f"__half2 xh{u}[4] = {{{pairs}}};")
        wl = "\n      ".join(f"uint4 w{r} = W[(size_t)(row0 + {r}) * (K / 32) + j];" for r in range(R))
        sl = "\n      ".join(f"acc[{r}] = fmaf(S[(size_t)(row0 + {r}) * (K / 128) + (j >> 2)], part[{r}], acc[{r}]);"
                             for r in range(R))
        red = "\n    ".join(
            f"{{ float a = acc[{r}]; for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);"
            f" if (lane == 0) Y[row0 + {r}] = a; }}" for r in range(R))
        return f"""
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const float4* __restrict__ X,
    float* __restrict__ Y, unsigned long long* T, int N, int K)
{{
  unsigned long long g0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  int row0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * {R};
  int lane = threadIdx.x & 31;
  const __half2 bias = __float2half2_rn(1032.f);
  (void)bias;
  if (row0 < N) {{
    float acc[{R}] = {{0}};
    for (int j = lane; j < K / 32; j += 32) {{
      {wl}
      const float4* xp = X + j * 8;
      {" ".join(xload)}
      float part[{R}] = {{0}};
      {{
        {_deq_block(deq, R)}
      }}
      {sl}
    }}
    {red}
  }}
  __syncthreads();
  if (threadIdx.x == 0) {{
    unsigned long long g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1;
  }}
}}
"""

    def expected(self, v):
        return {}

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for name, n, k in SHAPES:
            for R in (1, 2, 4):
                for deq in ("naive", "magic", "half2"):
                    out.append(Variant(f"{name}/R{R}/{deq}", {"N": n, "K": k, "R": R, "deq": deq,
                                                              "warps": 4, "iters": 1}))
        return out

    def prepare(self, dev, v: Variant) -> dict:
        from .gemv import GemvInt4
        st = GemvInt4().prepare(dev, v)
        R = v.params["R"]
        blocks = (v.params["N"] + 4 * R - 1) // (4 * R)
        dev.free(st["T"])
        st["T"] = dev.alloc(blocks * 16)
        st["blocks"] = blocks
        st["launch"]["grid"] = blocks
        st["launch"]["args"][4] = ctypes.c_uint64(st["T"])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        from .gemv import GemvInt4
        r = GemvInt4().collect(dev, v, st)
        r["correct"] = r["max_rel_err"] < TOL[v.params["deq"]]
        return r

    def release(self, dev, st: dict):
        for key in ("W", "S", "X", "Y", "T"):
            dev.free(st[key])


def registry():
    e = Gemv2()
    return {e.name: e}
