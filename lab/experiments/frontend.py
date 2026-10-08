"""Front end and L1 geometry.

icache: loop of N independent FFMAs (8 interleaved chains, no data stalls), code
size N*16 B swept 2 KB..256 KB. With instruction-cache hits a warp issues ~1/cycle
(4 warps = one per partition) or the SM sustains ~4/cycle (32 warps); cliffs in
cycles/instruction mark the instruction-cache levels.

l1_capacity: random pointer chase with slot stride 32/64/128/256 B over working
sets of 64-176 KB (L1 path, ld.ca). If capacity is constant in lines across
strides, L1 is tag/line-limited; if constant in bytes, data-limited.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant
from .mem import KB, Chase


@dataclass
class ICache(Experiment):
    def __post_init__(self):
        self.name = "icache"
        self.target_opcode = "FFMA"
        self.description = "issue rate vs loop code size (independent FFMAs): instruction-cache levels"

    def source(self, v):
        n = v.params["n"]
        lines = [f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x{i % 8}) : "f"(a), "f"(b));' for i in range(n)]
        decl = " ".join(f"float x{c} = in[0] + c;".replace("+ c", f"+ {c}.f") for c in range(8))
        return f"""
extern "C" __global__ void k(float* out, long long* cyc, const float* in, int iters)
{{
  {decl}
  float a = in[1], b = in[2];
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    {chr(10).join(lines)}
  }}
  long long t1 = clock64();
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
}}
"""

    def expected(self, v):
        return {"FFMA": v.params["n"]}

    def variants(self, opts):
        out = []
        for kb in (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            n = kb * 1024 // 16
            for w in (4, 32):
                out.append(Variant(f"{kb}KB/w{w}", {"n": n, "kb": kb, "warps": w,
                                                    "iters": max(4, 400_000 // n)}))
        return out

    def prepare(self, dev, v):
        w = v.params["warps"]
        st = {"out": dev.alloc(32 * w * 4), "cyc": dev.alloc(8), "in": dev.alloc(12)}
        dev.htod(st["in"], np.array([1.0, 1.0, 0.0], np.float32))
        st["launch"] = dict(grid=1, block=32 * w, args=[ctypes.c_uint64(st[x]) for x in ("out", "cyc", "in")]
                            + [ctypes.c_int32(v.params["iters"])])
        return st

    def collect(self, dev, v, st):
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        per_warp = v.params["iters"] * v.params["n"]
        return {"cycles": cyc, "ops_per_thread": per_warp, "cycles_per_op": cyc / per_warp,
                "warp_ops_per_cycle": v.params["warps"] * per_warp / cyc, "sm_mhz_inkernel": 0.0}

    def release(self, dev, st):
        for x in ("out", "cyc", "in"):
            dev.free(st[x])


@dataclass
class L1Capacity(Chase):
    def __post_init__(self):
        super().__post_init__()
        self.name = "l1_capacity"
        self.target_opcode = "LDG.E.64.STRONG.SM"
        self.description = "L1 capacity vs slot stride (lines vs bytes)"

    def variants(self, opts):
        out = []
        for stride in (32, 64, 128, 256):
            for kb in range(64, 177, 8):
                out.append(Variant(f"s{stride}/{kb}KB", {"bytes": kb * KB, "stride": stride, "iters": 1600,
                                                         "warps": 1, "cold": False}))
        return out

    def prepare(self, dev, v):
        self.stride = v.params["stride"]
        return super().prepare(dev, v)

    def collect(self, dev, v, st):
        self.stride = v.params["stride"]
        r = super().collect(dev, v, st)
        r["lines"] = v.params["bytes"] // max(128, v.params["stride"])
        return r


def registry():
    exps = [ICache(), L1Capacity(space="global"), GridBarrier()]
    return {e.name: e for e in exps}


@dataclass
class GridBarrier(Experiment):
    """Cost of a software grid-wide barrier (atomic arrive + spin on a generation counter) with
    every block co-resident, vs. a kernel boundary in a CUDA graph (~1 us + ramp, graph_overhead).
    Decides whether a persistent per-token megakernel can beat ~200 launches."""

    def __post_init__(self):
        self.name = "grid_barrier"
        self.description = "software grid barrier cost vs blocks (all co-resident)"

    def source(self, v):
        return r"""
extern "C" __global__ void k(unsigned* bar, int n, long long* cyc, unsigned long long* ns, unsigned* timedout)
{
  // bar[0] = arrive count, bar[1] = generation
  long long t0 = clock64(); unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  for (int it = 0; it < n; ++it) {
    __syncthreads();
    if (threadIdx.x == 0) {
      volatile unsigned* gen = bar + 1;
      unsigned g = *gen;
      __threadfence();
      if (atomicAdd(bar, 1u) == gridDim.x - 1) { bar[0] = 0; __threadfence(); atomicAdd(bar + 1, 1u); }
      else {
        // hard spin cap: a non-co-resident grid would otherwise hang the (shared) GPU forever;
        // this machine does not reset hung kernels (no TDR)
        unsigned long long spins = 0;
        while (*gen == g) { if (++spins > 20000000ull) { *timedout = 1u; break; } }
      }
      __threadfence();
    }
    __syncthreads();
  }
  long long t1 = clock64(); unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (blockIdx.x == 0 && threadIdx.x == 0) { cyc[0] = t1 - t0; ns[0] = g1 - g0; }
}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        out = []
        for b in (68, 136, 272, 544):
            for t in (128, 256):
                per_sm = b // 68
                if per_sm * t > 1536 or per_sm > 16:   # must be co-resident or a software barrier deadlocks
                    continue
                out.append(Variant(f"blocks{b}/t{t}", {"blocks": b, "threads": t, "n": 2000, "warps": t // 32, "iters": 1}))
        return out

    def prepare(self, dev, v):
        st = {"bar": dev.alloc(8), "cyc": dev.alloc(8), "ns": dev.alloc(8), "to": dev.alloc(4)}
        dev.memset(st["bar"], 8)
        dev.memset(st["to"], 4)
        st["launch"] = dict(grid=v.params["blocks"], block=v.params["threads"],
                            args=[ctypes.c_uint64(st["bar"]), ctypes.c_int32(v.params["n"]),
                                  ctypes.c_uint64(st["cyc"]), ctypes.c_uint64(st["ns"]), ctypes.c_uint64(st["to"])])
        return st

    def collect(self, dev, v, st):
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        ns = int(dev.dtoh(np.zeros(1, np.uint64), st["ns"])[0])
        n = v.params["n"]
        to = int(dev.dtoh(np.zeros(1, np.uint32), st["to"])[0])
        return {"cycles": cyc, "ns": ns, "ops_per_thread": n, "cycles_per_op": cyc / n, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": cyc / max(ns, 1) * 1e3, "us_per_barrier": ns / n / 1e3, "correct": to == 0}

    def release(self, dev, st):
        for x in ("bar", "cyc", "ns", "to"):
            dev.free(st[x])
