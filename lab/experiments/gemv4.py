"""int4 GEMV round 4: single-wave scheduling + split-K for small N.

Round 3 left ~13-20% on the table to tail effects: at ~72 registers only ~7
blocks fit per SM, so large shapes ran ~2.5 waves and the last half-wave idled.

  rows mode   grid sized to one wave of resident warps; each warp owns a
              contiguous strip of ceil(N / warps) rows, processed 4 at a time
              (R4 + magic dequant inner loop). All warps finish together.
  splitk mode for small N: a block of S warps shares one 4-row group, each warp
              takes 1/S of K, partials reduced deterministically through
              shared memory (no atomics, no output memset).

The mode and its parameter are chosen from the kernel's real register count
and the occupancy formula, per shape, in prepare().
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant
from .gemv import SHAPES
from .gemv3 import _deq

SMS = 68
MAX_WARPS_PER_SM = 48


def _inner(R: int, kexpr_start: str, kexpr_end: str, step: str) -> str:
    """Accumulate rows row0..row0+R-1 over uint4 index range [start, end) with stride step."""
    L = [f"for (int j = {kexpr_start}; j < {kexpr_end}; j += {step}) {{"]
    for r in range(R):
        L.append(f"  uint4 w{r}_0 = (row0 + {r} < rowEnd) ? W[(size_t)(row0 + {r}) * KW + j]"
                 f" : make_uint4(0x88888888u,0x88888888u,0x88888888u,0x88888888u);")
    L.append("  const float4* xp0 = X + (size_t)j * 8;")
    for k in range(4):
        L.append(f"  float4 xa0_{k} = xp0[{2 * k}]; float4 xb0_{k} = xp0[{2 * k + 1}];")
    for r in range(R):
        L.append("  { float part = 0.f;")
        L += ["    " + s for s in _deq("magic", f"w{r}_0", 0)]
        L.append(f"    if (row0 + {r} < rowEnd) acc[{r}] = fmaf(S[(size_t)(row0 + {r}) * (K / 128) + (j >> 2)], part, acc[{r}]); }}")
    L.append("}")
    return "\n      ".join(L)


@dataclass
class Gemv4(Experiment):
    def __post_init__(self):
        self.name = "gemv_int4_v4"
        self.target_opcode = "FFMA"
        self.description = "int4 GEMV: single-wave row strips + deterministic split-K for small N"

    def source(self, v: Variant) -> str:
        mode = v.params["mode"]
        if mode == "rows":
            body = _inner(4, "lane", "KW", "32")
            return f"""
extern "C" __global__ void __launch_bounds__(128) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const float4* __restrict__ X,
    float* __restrict__ Y, unsigned long long* T, int N, int K, int RPW)
{{
  unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  int warp = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  int KW = K / 32;
  int rowBeg = warp * RPW, rowEnd = min(N, rowBeg + RPW);
  for (int row0 = rowBeg; row0 < rowEnd; row0 += 4) {{
    float acc[4] = {{0, 0, 0, 0}};
      {body}
    #pragma unroll
    for (int r = 0; r < 4; ++r) {{
      float a = acc[r];
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && row0 + r < rowEnd) Y[row0 + r] = a;
    }}
  }}
  __syncthreads();
  if (threadIdx.x == 0) {{ unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1; }}
}}
"""
        S_ = v.params["S"]
        body = _inner(4, "kb + lane", "ke", "32")
        return f"""
extern "C" __global__ void __launch_bounds__({32 * S_}) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const float4* __restrict__ X,
    float* __restrict__ Y, unsigned long long* T, int N, int K, int RPW)
{{
  unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  __shared__ float red[{S_}][4];
  int ws = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int KW = K / 32;
  int row0 = blockIdx.x * 4, rowEnd = min(N, row0 + 4);
  int chunk = (KW + {S_} - 1) / {S_};
  int kb = ws * chunk, ke = min(KW, kb + chunk);
  float acc[4] = {{0, 0, 0, 0}};
      {body}
  #pragma unroll
  for (int r = 0; r < 4; ++r) {{
    float a = acc[r];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[ws][r] = a;
  }}
  __syncthreads();
  if (threadIdx.x < 4 && row0 + threadIdx.x < rowEnd) {{
    float a = 0.f;
    #pragma unroll
    for (int s = 0; s < {S_}; ++s) a += red[s][threadIdx.x];
    Y[row0 + threadIdx.x] = a;
  }}
  __syncthreads();
  if (threadIdx.x == 0) {{ unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1; }}
}}
"""

    def expected(self, v):
        return {}

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for name, n, k in SHAPES:
            out.append(Variant(f"{name}/rows", {"N": n, "K": k, "mode": "rows", "warps": 4, "iters": 1}))
            for S_ in (2, 4, 8):
                out.append(Variant(f"{name}/splitk{S_}", {"N": n, "K": k, "mode": "splitk", "S": S_,
                                                          "warps": S_, "iters": 1}))
        return out

    @staticmethod
    def resident_warps(regs: int, warps_per_block: int, smem: int = 0) -> int:
        per_warp = ((regs * 32 + 255) // 256) * 256
        by_regs = 65536 // (per_warp * warps_per_block)
        by_warps = MAX_WARPS_PER_SM // warps_per_block
        blocks = max(1, min(16, by_regs, by_warps))
        return blocks * warps_per_block * SMS

    def prepare(self, dev, v: Variant) -> dict:
        from .. import toolchain
        from .gemv import GemvInt4
        st = GemvInt4().prepare(dev, v)
        n = v.params["N"]
        regs = toolchain.ptxas_resources(toolchain.build(self.source(v)).ptxas_log)["registers"]
        if v.params["mode"] == "rows":
            warps = self.resident_warps(regs, 4)
            rpw = max(1, -(-n // warps))
            blocks = -(-(-(-n // rpw)) // 4)
            tpb = 128
        else:
            rpw, tpb = 0, 32 * v.params["S"]
            blocks = -(-n // 4)
        dev.free(st["T"])
        st["T"] = dev.alloc(blocks * 16)
        st["blocks"] = blocks
        st["launch"] = dict(grid=blocks, block=tpb, args=st["launch"]["args"][:4] + [
            ctypes.c_uint64(st["T"]), ctypes.c_int32(n), ctypes.c_int32(v.params["K"]), ctypes.c_int32(rpw)])
        st["rpw"], st["regs"] = rpw, regs
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        t = dev.dtoh(np.zeros(2 * st["blocks"], np.uint64), st["T"]).reshape(-1, 2)
        ns = int(t[:, 1].max() - t[:, 0].min())
        y = dev.dtoh(np.zeros(v.params["N"], np.float32), st["Y"])
        err = float(np.max(np.abs(y - st["ref"])) / np.max(np.abs(st["ref"])))
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "us": ns / 1e3, "gbps": st["bytes"] / ns, "norm_err": err,
                "correct": err < 1e-4, "rpw": st["rpw"], "blocks": st["blocks"], "regs": st["regs"]}

    def release(self, dev, st: dict):
        for key in ("W", "S", "X", "Y", "T"):
            dev.free(st[key])


def registry():
    e = Gemv4()
    return {e.name: e}
