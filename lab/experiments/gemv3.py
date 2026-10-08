"""int4 GEMV, round 3: deeper knobs + a bandwidth roofline.

  R    rows per warp (4, 8)
  U    uint4 weight loads per row per iteration (1, 2): more bytes in flight
  T    threads per block (128, 256)
  deq  magic (exact fp32) | half2 (Marlin-style, fp16 x)

Also `read_roofline`: a pure streaming read of the same bytes (uint4 loads,
XOR-reduced so they cannot be elided) = the practical DRAM ceiling.

Error metric: max|y - ref| / max|ref| (normalized). The old per-row relative
metric exploded on rows where ref ~ 0 and wrongly flagged half2 as broken.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant
from .gemv import GROUP, SHAPES

TOL = {"naive": 1e-4, "magic": 1e-4, "half2": 5e-3}


def _deq(deq: str, wname: str, u: int) -> list[str]:
    """Accumulate one uint4 (32 weights) of `wname` against x chunk u into `part`."""
    out = []
    xs = [f"xa{u}_{k}.{c}" for k in range(4) for c in "xyzw"] + [f"xb{u}_{k}.{c}" for k in range(4) for c in "xyzw"]
    # weight index within the uint4: word k (0..3), nibble i (0..7) -> 8k+i; x for word k: xa{u}_k (0..3), xb{u}_k (4..7)
    for k in range(4):
        q = f"{wname}.{'xyzw'[k]}"
        xk = [f"xa{u}_{k}.{c}" for c in "xyzw"] + [f"xb{u}_{k}.{c}" for c in "xyzw"]
        if deq == "naive":
            for i in range(8):
                out.append(f"part = fmaf((float)((int)(({q} >> {4 * i}) & 15) - 8), {xk[i]}, part);")
        elif deq == "magic":
            for i in range(8):
                out.append(f"part = fmaf(__int_as_float((({q} >> {4 * i}) & 15) | 0x4B000000) - 8388616.0f, {xk[i]}, part);")
        else:
            out.append("{ __half2 h = __float2half2_rn(0.f);")
            for i in range(4):
                out.append(f"  {{ unsigned b = (({q} >> {4 * i}) & 0x000F000Fu) | 0x64006400u;"
                           f" h = __hfma2(__hsub2(*reinterpret_cast<__half2*>(&b), bias), xh{u}_{k}[{i}], h); }}")
            out.append("  float2 f = __half22float2(h); part += f.x + f.y; }")
    del xs
    return out


@dataclass
class Gemv3(Experiment):
    def __post_init__(self):
        self.name = "gemv_int4_v3"
        self.target_opcode = "FFMA"
        self.description = "int4 GEMV: rows/warp x loads in flight x block size x dequant, plus read roofline"

    def source(self, v: Variant) -> str:
        if v.params.get("roofline"):
            return """
extern "C" __global__ void k(const uint4* __restrict__ W, const float* S, const float4* X,
                             float* __restrict__ Y, unsigned long long* T, int N, int K)
{
  unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  size_t n16 = (size_t)N * K / 32;
  unsigned acc = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 w = W[i]; acc ^= w.x ^ w.y ^ w.z ^ w.w;
  }
  if (acc == 0x12345678u) Y[0] = 1.f;
  __syncthreads();
  if (threadIdx.x == 0) { unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1; }
}
"""
        R, U, Tn, deq = (v.params[x] for x in ("R", "U", "T", "deq"))
        wpb = Tn // 32
        L = []
        L.append("for (int j = lane * %d; j < K / 32; j += 32 * %d) {" % (U, U))
        for u in range(U):
            for r in range(R):
                L.append(f"  uint4 w{r}_{u} = (j + {u} < K / 32) ? W[(size_t)(row0 + {r}) * (K / 32) + j + {u}] : make_uint4(0x88888888u,0x88888888u,0x88888888u,0x88888888u);")
        for u in range(U):
            L.append(f"  const float4* xp{u} = X + (size_t)(j + {u}) * 8;")
            for k in range(4):
                L.append(f"  float4 xa{u}_{k} = (j + {u} < K / 32) ? xp{u}[{2 * k}] : make_float4(0,0,0,0);"
                         f" float4 xb{u}_{k} = (j + {u} < K / 32) ? xp{u}[{2 * k + 1}] : make_float4(0,0,0,0);")
                if deq == "half2":
                    L.append(f"  __half2 xh{u}_{k}[4] = {{__floats2half2_rn(xa{u}_{k}.x, xb{u}_{k}.x), __floats2half2_rn(xa{u}_{k}.y, xb{u}_{k}.y),"
                             f" __floats2half2_rn(xa{u}_{k}.z, xb{u}_{k}.z), __floats2half2_rn(xa{u}_{k}.w, xb{u}_{k}.w)}};")
        for u in range(U):
            for r in range(R):
                L.append("  { float part = 0.f;")
                L += ["    " + s for s in _deq(deq, f"w{r}_{u}", u)]
                L.append(f"    if (j + {u} < K / 32) acc[{r}] = fmaf(S[(size_t)(row0 + {r}) * (K / 128) + ((j + {u}) >> 2)], part, acc[{r}]); }}")
        L.append("}")
        body = "\n    ".join(L)
        red = "\n    ".join(
            f"{{ float a = acc[{r}]; for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);"
            f" if (lane == 0) Y[row0 + {r}] = a; }}" for r in range(R))
        return f"""
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__({Tn}) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const float4* __restrict__ X,
    float* __restrict__ Y, unsigned long long* T, int N, int K)
{{
  unsigned long long g0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  int row0 = (blockIdx.x * {wpb} + (threadIdx.x >> 5)) * {R};
  int lane = threadIdx.x & 31;
  const __half2 bias = __float2half2_rn(1032.f);
  (void)bias;
  if (row0 < N) {{
    float acc[{R}] = {{0}};
    {body}
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
            out.append(Variant(f"{name}/roofline", {"N": n, "K": k, "roofline": True, "warps": 8, "iters": 1}))
            out.append(Variant(f"{name}/R1U1T128/naive", {"N": n, "K": k, "R": 1, "U": 1, "T": 128, "deq": "naive",
                                                         "warps": 4, "iters": 1}))
            for R in (4, 8):
                for U in (1, 2):
                    for Tn in (128, 256):
                        for deq in ("magic", "half2"):
                            if (R, U, Tn, deq) == (8, 2, 128, "half2"):
                                continue  # spills 8 bytes; excluded
                            out.append(Variant(f"{name}/R{R}U{U}T{Tn}/{deq}",
                                               {"N": n, "K": k, "R": R, "U": U, "T": Tn, "deq": deq,
                                                "warps": Tn // 32, "iters": 1}))
        return out

    def prepare(self, dev, v: Variant) -> dict:
        from .gemv import GemvInt4
        st = GemvInt4().prepare(dev, v)
        if v.params.get("roofline"):
            blocks, tpb = 68 * 12, 256
        else:
            tpb = v.params["T"]
            rows_per_block = (tpb // 32) * v.params["R"]
            blocks = (v.params["N"] + rows_per_block - 1) // rows_per_block
        dev.free(st["T"])
        st["T"] = dev.alloc(blocks * 16)
        st["blocks"] = blocks
        st["launch"]["grid"] = blocks
        st["launch"]["block"] = tpb
        st["launch"]["args"][4] = ctypes.c_uint64(st["T"])
        if v.params.get("roofline"):
            st["bytes"] = v.params["N"] * v.params["K"] // 2
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        t = dev.dtoh(np.zeros(2 * st["blocks"], np.uint64), st["T"]).reshape(-1, 2)
        ns = int(t[:, 1].max() - t[:, 0].min())
        r = {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
             "sm_mhz_inkernel": 0.0, "us": ns / 1e3, "gbps": st["bytes"] / ns}
        if not v.params.get("roofline"):
            y = dev.dtoh(np.zeros(v.params["N"], np.float32), st["Y"])
            err = float(np.max(np.abs(y - st["ref"])) / np.max(np.abs(st["ref"])))
            r["norm_err"] = err
            r["correct"] = err < TOL[v.params["deq"]]
        return r

    def release(self, dev, st: dict):
        for key in ("W", "S", "X", "Y", "T"):
            dev.free(st[key])


def registry():
    e = Gemv3()
    return {e.name: e}
