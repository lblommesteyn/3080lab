"""Memory system under load (whole GPU), Phase 3 proper.

  mem_coalesce   warps read 32 x 4 B with a lane stride of s bytes, from an
                 L2-resident (2 MB) or DRAM (256 MB) buffer, L1 bypassed (ld.cg).
                 Useful GB/s vs s exposes the transfer granularity (sectors/lines).
  mem_width      perfectly coalesced 4/8/16 B per lane loads, L2 and DRAM.
  mem_mlp        DRAM bandwidth vs active SMs and warps per SM (one 16 B load in
                 flight per thread per iteration): Little's-law curve.
  smem_banks     LDS throughput vs lane stride in 4 B words (bank conflicts).

Timing: per-block globaltimer start/end; kernel time = max(end) - min(start).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

MB = 1 << 20


def _span(dev, st) -> int:
    t = dev.dtoh(np.zeros(2 * st["blocks"], np.uint64), st["T"]).reshape(-1, 2)
    return int(t[:, 1].max() - t[:, 0].min())


TIMED_HEAD = 'unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));'
TIMED_TAIL = """
  __syncthreads();
  if (threadIdx.x == 0) { unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1; }"""


@dataclass
class Coalesce(Experiment):
    def __post_init__(self):
        self.name = "mem_coalesce"
        self.description = "useful bandwidth vs lane stride (L1 bypassed), L2-resident and DRAM"

    def source(self, v):
        return f"""
extern "C" __global__ void __launch_bounds__(256) k(const unsigned char* __restrict__ B, unsigned long long mask,
    int stride, int iters, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long warp = (blockIdx.x * 256ull + threadIdx.x) >> 5;
  unsigned lane = threadIdx.x & 31, acc = 0;
  unsigned long long nw = gridDim.x * 8ull;
  for (int i = 0; i < iters; ++i) {{
    unsigned long long e = (warp + i * nw) * 32 + lane;            // element index
    unsigned long long a = (e * (unsigned long long)stride) & mask;
    unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(B + a));
    acc ^= v;
  }}
  if (acc == 0x9e3779b9u) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        out = []
        for level, size in (("L2", 2 * MB), ("DRAM", 256 * MB)):
            for s in (4, 8, 16, 32, 64, 128, 256):
                out.append(Variant(f"{level}/stride{s}", {"size": size, "stride": s, "iters": 64, "warps": 8}))
        return out

    def prepare(self, dev, v):
        blocks = 68 * 6
        size = v.params["size"]
        st = {"B": dev.alloc(size), "sink": dev.alloc(4), "T": dev.alloc(blocks * 16), "blocks": blocks}
        dev.memset(st["B"], size)
        st["useful"] = blocks * 256 * v.params["iters"] * 4
        st["launch"] = dict(grid=blocks, block=256, args=[
            ctypes.c_uint64(st["B"]), ctypes.c_uint64(size - 1), ctypes.c_int32(v.params["stride"]),
            ctypes.c_int32(v.params["iters"]), ctypes.c_uint64(st["sink"]), ctypes.c_uint64(st["T"])])
        return st

    def collect(self, dev, v, st):
        ns = _span(dev, st)
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "useful_gbps": st["useful"] / ns}

    def release(self, dev, st):
        for x in ("B", "sink", "T"):
            dev.free(st[x])


@dataclass
class Width(Coalesce):
    def __post_init__(self):
        self.name = "mem_width"
        self.description = "coalesced 4/8/16 B per lane loads, L2-resident vs DRAM, L1 bypassed"

    def source(self, v):
        w = v.params["width"]
        ty, n = {4: ("u32", 1), 8: ("v2.u32", 2), 16: ("v4.u32", 4)}[w]
        regs = ", ".join(f"%{i}" for i in range(n))
        outs = ", ".join(f'"=r"(r{i})' for i in range(n))
        decl = " ".join(f"unsigned r{i};" for i in range(n))
        dst = "{" + regs + "}" if n > 1 else "%0"
        fold = " ^ ".join(f"r{i}" for i in range(n))
        return f"""
extern "C" __global__ void __launch_bounds__(256) k(const unsigned char* __restrict__ B, unsigned long long mask,
    int stride, int iters, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long tid = blockIdx.x * 256ull + threadIdx.x, nt = gridDim.x * 256ull;
  unsigned acc = 0;
  for (int i = 0; i < iters; ++i) {{
    unsigned long long a = ((tid + i * nt) * {w}ull) & mask;
    {decl}
    asm volatile("ld.global.cg.{ty} {dst}, [%{n}];" : {outs} : "l"(B + a));
    acc ^= {fold};
  }}
  if (acc == 0x9e3779b9u) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        out = []
        for level, size, iters in (("L2", 2 * MB, 64), ("DRAM", 256 * MB, 256)):
            for w in (4, 8, 16):
                out.append(Variant(f"{level}/{w}B", {"size": size, "width": w, "stride": w, "iters": iters, "warps": 8}))
        return out

    def prepare(self, dev, v):
        st = super().prepare(dev, v)
        st["useful"] = st["blocks"] * 256 * v.params["iters"] * v.params["width"]
        return st


@dataclass
class MLP(Experiment):
    """DRAM bandwidth vs SMs used and warps per SM: bytes in flight vs achieved bandwidth."""

    def __post_init__(self):
        self.name = "mem_mlp"
        self.description = "DRAM bandwidth vs active SMs x warps per SM (16 B/thread/iteration)"

    def source(self, v):
        return f"""
extern "C" __global__ void k(const uint4* __restrict__ B, unsigned long long n16, int iters, unsigned* sink,
                             unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  unsigned long long nt = (unsigned long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (int i = 0; i < iters; ++i) {{
    uint4 w; unsigned long long idx = (tid + i * nt) & (n16 - 1);  // n16 is a power of two
    asm volatile("ld.global.cg.v4.u32 {{%0,%1,%2,%3}}, [%4];" : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w) : "l"(B + idx));
    acc ^= w.x ^ w.y ^ w.z ^ w.w;
  }}
  if (acc == 0x9e3779b9u) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        out = []
        for sms in (1, 4, 17, 34, 68):
            for wps in (1, 2, 4, 8, 16, 32):   # <= 1024 threads per block
                out.append(Variant(f"sms{sms}/w{wps}", {"sms": sms, "wps": wps, "warps": wps, "iters": 1}))
        return out

    def prepare(self, dev, v):
        size = 512 * MB
        if not hasattr(self, "_buf"):
            self._buf = dev.alloc(size)
            dev.memset(self._buf, size)
            self._refs = 0
        self._refs += 1
        sms, wps = v.params["sms"], v.params["wps"]
        # one block per SM (blocks <= SMs and <=1 block/SM keeps the SM count exact),
        # block = wps warps; iterations sized for ~32 MB of traffic
        threads = 32 * wps
        iters = max(4, (32 * MB) // (16 * threads * sms))
        st = {"sink": dev.alloc(4), "T": dev.alloc(sms * 16), "blocks": sms,
              "bytes": 16 * threads * sms * iters}
        st["launch"] = dict(grid=sms, block=threads, args=[
            ctypes.c_uint64(self._buf), ctypes.c_uint64(size // 16), ctypes.c_int32(iters),
            ctypes.c_uint64(st["sink"]), ctypes.c_uint64(st["T"])])
        return st

    def collect(self, dev, v, st):
        ns = _span(dev, st)
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "gbps": st["bytes"] / ns,
                "bytes_in_flight_per_sm": 16 * 32 * v.params["wps"]}

    def release(self, dev, st):
        dev.free(st["sink"]), dev.free(st["T"])
        self._refs -= 1
        if self._refs == 0:
            dev.free(self._buf)
            del self._buf


@dataclass
class SmemBanks(Experiment):
    def expected(self, v):
        return {"LDS": 4}

    def __post_init__(self):
        self.name = "smem_banks"
        self.description = "LDS throughput vs lane stride in 4 B words (bank conflicts), 32 warps on one SM"

    def source(self, v):
        return f"""
extern "C" __global__ void __launch_bounds__(1024) k(int stride, int iters, unsigned* sink, long long* cyc)
{{
  __shared__ unsigned sm[32 * 33 * 4 + 512];
  for (int i = threadIdx.x; i < 32 * 33 * 4 + 512; i += blockDim.x) sm[i] = i;
  __syncthreads();
  unsigned lane = threadIdx.x & 31, acc = 0;
  unsigned a = (lane * stride) % (32 * 33 * 4);
  // shared-window address of sm[0] (sm_80+ reserves 1 KB of system smem, so it is not 0)
  unsigned smbase = (unsigned)__cvta_generic_to_shared(sm);
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    // ptxas hoists loop-invariant asm-volatile LDS; use ld.volatile and an iteration-dependent
    // offset that is a multiple of 32 words (same banks every iteration)
    unsigned b = smbase + ((a + (i & 31) * 128) % (32 * 33 * 4)) * 4;
    unsigned v0, v1, v2, v3;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v0) : "r"(b));
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v1) : "r"(b + 128 * 4));
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v2) : "r"(b + 256 * 4));
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v3) : "r"(b + 384 * 4));
    acc ^= v0 ^ v1 ^ v2 ^ v3;
  }}
  long long t1 = clock64();
  if (acc == 0x9e3779b9u) sink[0] = acc;
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
}}
"""

    def variants(self, opts):
        return [Variant(f"stride{s}", {"stride": s, "iters": 4096, "warps": 32}) for s in (0, 1, 2, 3, 4, 8, 16, 32, 33)]

    def prepare(self, dev, v):
        st = {"sink": dev.alloc(4), "cyc": dev.alloc(8)}
        st["launch"] = dict(grid=1, block=1024, args=[ctypes.c_int32(v.params["stride"]), ctypes.c_int32(v.params["iters"]),
                                                      ctypes.c_uint64(st["sink"]), ctypes.c_uint64(st["cyc"])])
        return st

    def collect(self, dev, v, st):
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        lds = 32 * v.params["iters"] * 4  # warp-level LDS instructions across the SM
        return {"cycles": cyc, "ops_per_thread": v.params["iters"] * 4, "cycles_per_op": cyc / (v.params["iters"] * 4),
                "warp_ops_per_cycle": lds / cyc, "sm_mhz_inkernel": 0.0}

    def release(self, dev, st):
        dev.free(st["sink"]), dev.free(st["cyc"])


def registry():
    exps = [Coalesce(), Width(), MLP(), SmemBanks(), L1Wavefronts()]
    return {e.name: e for e in exps}


@dataclass
class L1Wavefronts(Experiment):
    """L1-hit load throughput vs lane stride on one SM: how many cycles an LDG costs when the
    warp's 32 addresses touch k distinct 128 B lines (k = 1..32). 16 KB buffer, ld.ca (L1)."""

    def __post_init__(self):
        self.name = "l1_wavefronts"
        self.description = "L1-hit LDG throughput vs lane stride (lines touched per instruction), one SM"

    def source(self, v):
        w = v.params["width"]
        ty, n = {4: ("u32", 1), 16: ("v4.u32", 4)}[w]
        regs = ", ".join(f"%{i}" for i in range(n))
        dst = "{" + regs + "}" if n > 1 else "%0"
        outs = ", ".join(f'"=r"(r{i})' for i in range(n))
        decl = " ".join(f"unsigned r{i};" for i in range(n))
        fold = " ^ ".join(f"r{i}" for i in range(n))
        return f"""
extern "C" __global__ void __launch_bounds__(1024) k(const unsigned char* __restrict__ B, int stride, int iters,
                                                    unsigned* sink, long long* cyc)
{{
  unsigned lane = threadIdx.x & 31, acc = 0;
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    unsigned off = ((lane * stride) + (i & 7) * 4096 + (threadIdx.x >> 5) * 16) & 16383;
    {decl}
    asm volatile("ld.global.{v.params.get('cop', 'ca')}.{ty} {dst}, [%{n}];" : {outs} : "l"(B + off));
    acc ^= {fold};
  }}
  long long t1 = clock64();
  if (acc == 0x9e3779b9u) sink[0] = acc;
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
}}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        out = []
        for cop in ("ca", "nc"):
            for w in (4, 16):
                for s in (w, 32, 64, 128, 256):
                    out.append(Variant(f"{cop}/{w}B/stride{s}", {"width": w, "stride": s, "iters": 4096, "warps": 32,
                                                                 "cop": cop}))
        return out

    def prepare(self, dev, v):
        st = {"B": dev.alloc(16384), "sink": dev.alloc(4), "cyc": dev.alloc(8)}
        dev.memset(st["B"], 16384)
        st["launch"] = dict(grid=1, block=1024, args=[ctypes.c_uint64(st["B"]), ctypes.c_int32(v.params["stride"]),
                                                      ctypes.c_int32(v.params["iters"]), ctypes.c_uint64(st["sink"]),
                                                      ctypes.c_uint64(st["cyc"])])
        return st

    def collect(self, dev, v, st):
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        instr = 32 * v.params["iters"]
        lines = min(32, max(1, 32 * v.params["stride"] // 128)) if v.params["stride"] >= 4 else 1
        return {"cycles": cyc, "ops_per_thread": v.params["iters"], "cycles_per_op": cyc / v.params["iters"],
                "warp_ops_per_cycle": instr / cyc, "cycles_per_ldg_sm": cyc / instr,
                "lines_per_instr": lines, "sm_mhz_inkernel": 0.0}

    def release(self, dev, st):
        for x in ("B", "sink", "cyc"):
            dev.free(st[x])
