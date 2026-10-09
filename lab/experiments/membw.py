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
    exps = [Coalesce(), Width(), MLP(), SmemBanks(), L1Wavefronts(), SplitSector(), SplitGap(), SplitMech(), SplitPhase()]
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


@dataclass
class SplitSector(Coalesce):
    """DRAM streaming where each 32 B sector is read by TWO instructions (lane stride 32 B,
    16 B each: the GEMV U=2 pattern) vs contiguous 16 B per lane (each sector read once)."""

    def __post_init__(self):
        self.name = "mem_split_sector"
        self.description = "DRAM bandwidth: split-sector (U=2 GEMV) pattern vs contiguous, via L1 (.nc) or not (.cg)"

    def source(self, v):
        cop = v.params["cop"]
        split = v.params["pattern"] == "split"
        a0 = "(base + lane * 32)" if split else "(base + lane * 16)"
        a1 = "(base + lane * 32 + 16)" if split else "(base + 512 + lane * 16)"
        return f"""
extern "C" __global__ void __launch_bounds__(256) k(const unsigned char* __restrict__ B, unsigned long long mask,
    int stride, int iters, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long warp = (blockIdx.x * 256ull + threadIdx.x) >> 5, nw = gridDim.x * 8ull;
  unsigned lane = threadIdx.x & 31, acc = 0;
  for (int i = 0; i < iters; ++i) {{
    unsigned long long base = ((warp + i * nw) * 1024ull) & mask;
    unsigned x0, x1, x2, x3, y0, y1, y2, y3;
    asm volatile("ld.global.{cop}.v4.u32 {{%0,%1,%2,%3}}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "l"(B + {a0}));
    asm volatile("ld.global.{cop}.v4.u32 {{%0,%1,%2,%3}}, [%4];" : "=r"(y0), "=r"(y1), "=r"(y2), "=r"(y3) : "l"(B + {a1}));
    acc ^= x0 ^ x1 ^ x2 ^ x3 ^ y0 ^ y1 ^ y2 ^ y3;
  }}
  if (acc == 0x9e3779b9u) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        out = []
        for bps in (1, 6):                       # blocks of 8 warps per SM: low vs full occupancy
            for pat in ("contig", "split"):
                for cop in ("nc", "cg"):
                    out.append(Variant(f"bps{bps}/{pat}/{cop}", {"size": 256 * MB, "stride": 0,
                                                                 "iters": 128 * 6 // bps, "warps": 8,
                                                                 "pattern": pat, "cop": cop, "bps": bps}))
        return out

    def prepare(self, dev, v):
        st = super().prepare(dev, v)
        blocks = 68 * v.params["bps"]
        dev.free(st["T"])
        st["T"] = dev.alloc(blocks * 16)
        st["blocks"] = blocks
        st["launch"]["grid"] = blocks
        st["launch"]["args"][5] = ctypes.c_uint64(st["T"])
        st["useful"] = blocks * 256 * v.params["iters"] * 32
        return st


@dataclass
class SplitGap(Coalesce):
    """Split-sector loads where the second half of each sector is requested G instructions after
    the first (GEMV U=2 order: first halves of G rows, then second halves). G=1 is adjacent."""

    def __post_init__(self):
        self.name = "mem_split_gap"
        self.description = "DRAM bandwidth when the two halves of each sector are requested G loads apart"

    def source(self, v):
        G, pat = v.params["G"], v.params["pattern"]
        first = [f"(base + {g} * 1024 + lane * 32)" if pat == "split" else f"(base + {g} * 1024 + lane * 16)" for g in range(G)]
        second = [f"(base + {g} * 1024 + lane * 32 + 16)" if pat == "split" else f"(base + {g} * 1024 + 512 + lane * 16)" for g in range(G)]
        loads = []
        for idx, a in enumerate(first + second):
            loads.append(f'asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];" : "=r"(r{idx}.x), "=r"(r{idx}.y), "=r"(r{idx}.z), "=r"(r{idx}.w) : "l"(B + {a}));')
        decl = " ".join(f"uint4 r{i};" for i in range(2 * G))
        fold = " ^ ".join(f"r{i}.x ^ r{i}.y ^ r{i}.z ^ r{i}.w" for i in range(2 * G))
        return f"""
extern "C" __global__ void __launch_bounds__(256) k(const unsigned char* __restrict__ B, unsigned long long mask,
    int stride, int iters, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long warp = (blockIdx.x * 256ull + threadIdx.x) >> 5, nw = gridDim.x * 8ull;
  unsigned lane = threadIdx.x & 31, acc = 0;
  for (int i = 0; i < iters; ++i) {{
    unsigned long long base = ((warp + i * nw) * {1024 * G}ull) & mask;
    {decl}
    {chr(10).join("    " + l for l in loads)}
    acc ^= {fold};
  }}
  if (acc == 0x9e3779b9u) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        return [Variant(f"G{G}/{pat}", {"size": 256 * MB, "stride": 0, "iters": max(8, 256 // G), "warps": 8,
                                        "pattern": pat, "G": G})
                for G in (1, 2, 4, 8) for pat in ("contig", "split")]

    def prepare(self, dev, v):
        blocks = 68 * 2
        st = super().prepare(dev, v)
        dev.free(st["T"])
        st["T"] = dev.alloc(blocks * 16)
        st["blocks"] = blocks
        st["launch"]["grid"] = blocks
        st["launch"]["args"][5] = ctypes.c_uint64(st["T"])
        st["useful"] = blocks * 256 * v.params["iters"] * 32 * v.params["G"]
        return st


@dataclass
class SplitMech(SplitGap):
    """Why do far-apart sector halves cost DRAM bandwidth? Varies, independently:
      G     rows whose first halves are requested before any second half (intervening requests)
      fill  dependent FFMAs between the two batches (a pure time gap, no extra requests)
      wait  second halves issued only after the first halves' DATA arrived (address depends on it)
      cache .nc (L1-allocating, LDG.CONSTANT) or .cg (L1 bypass)
    If only `fill` hurts: a pending miss is not merged. If `wait` restores full speed: the second
    half hits once the fill completed. If only G hurts: requests in between evict/serialize."""

    def __post_init__(self):
        self.name = "mem_split_mech"
        self.description = "split-sector penalty mechanism: request gap vs time gap vs completed fill"

    def source(self, v):
        G, pat, fill, wait, cache = (v.params[k] for k in ("G", "pattern", "fill", "wait", "cache"))
        q = "nc" if cache == "nc" else "cg"
        first = [f"(base + {g} * 1024 + lane * 32)" if pat == "split" else f"(base + {g} * 1024 + lane * 16)"
                 for g in range(G)]
        second = [f"(base2 + {g} * 1024 + lane * 32 + 16)" if pat == "split" else f"(base2 + {g} * 1024 + 512 + lane * 16)"
                  for g in range(G)]
        ld = lambda idx, a: (f'asm volatile("ld.global.{q}.v4.u32 {{%0,%1,%2,%3}}, [%4];" : "=r"(r{idx}.x), '  # noqa: E731
                             f'"=r"(r{idx}.y), "=r"(r{idx}.z), "=r"(r{idx}.w) : "l"(B + {a}));')
        firsts = "\n    ".join(ld(i, a) for i, a in enumerate(first))
        seconds = "\n    ".join(ld(G + i, a) for i, a in enumerate(second))
        dep = " ^ ".join(f"r{i}.x" for i in range(G))
        base2 = f"base + ((unsigned long long)({dep}) & zmask)" if wait else "base"
        # time gap: nanosleep is a scheduling barrier for ptxas (an FFMA filler got spread around)
        filler = f'asm volatile("nanosleep.u32 {fill};" ::: "memory");' if fill else ""
        decl = " ".join(f"uint4 r{i};" for i in range(2 * G))
        fold = " ^ ".join(f"r{i}.x ^ r{i}.y ^ r{i}.z ^ r{i}.w" for i in range(2 * G))
        return f"""
extern "C" __global__ void __launch_bounds__(256) k(const unsigned char* __restrict__ B, unsigned long long mask,
    int stride, int iters, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long warp = (blockIdx.x * 256ull + threadIdx.x) >> 5, nw = gridDim.x * 8ull;
  unsigned long long zmask = (unsigned long long)stride;          // runtime 0
  unsigned lane = threadIdx.x & 31, acc = 0;
  float f = (float)lane, fz = (float)stride;
  for (int i = 0; i < iters; ++i) {{
    unsigned long long base = ((warp + i * nw) * {1024 * G}ull) & mask;
    {decl}
    {firsts}
    {filler}
    unsigned long long base2 = {base2};
    {seconds}
    acc ^= {fold};
  }}
  if (acc == 0x9e3779b9u || f == 12345.f) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        out = []
        for cache in ("nc", "cg"):
            for G in (1, 2, 4, 8, 16, 32):
                for pat in ("contig", "split"):
                    out.append(Variant(f"{cache}/G{G}/{pat}", {"size": 256 * MB, "stride": 0, "iters": max(8, 256 // G),
                                                              "warps": 8, "pattern": pat, "G": G, "fill": 0,
                                                              "wait": False, "cache": cache}))
            for fill in ((100, 400, 1600) if cache == "cg" else ()):   # ptxas hoists .nc loads across it
                out.append(Variant(f"{cache}/G1/split/fill{fill}", {"size": 256 * MB, "stride": 0, "iters": 256,
                                                                     "warps": 8, "pattern": "split", "G": 1,
                                                                     "fill": fill, "wait": False, "cache": cache}))
            if cache == "nc":
                # cold-start hypothesis (optimize benchmark): is the re-fetch a steady-state effect?
                # few trips per warp, one wave, G=16 (reuse distance far beyond L2)
                for it in (1, 2, 4, 32):
                    for pat in ("contig", "split"):
                        out.append(Variant(f"nc/G16/{pat}/it{it}", {"size": 256 * MB, "stride": 0, "iters": it,
                                                                   "warps": 8, "pattern": pat, "G": 16, "fill": 0,
                                                                   "wait": False, "cache": "nc"}))
            for G in (8, 32):
                out.append(Variant(f"{cache}/G{G}/split/wait", {"size": 256 * MB, "stride": 0,
                                                                 "iters": max(8, 256 // G), "warps": 8,
                                                                 "pattern": "split", "G": G, "fill": 0,
                                                                 "wait": True, "cache": cache}))
                out.append(Variant(f"{cache}/G{G}/contig/wait", {"size": 256 * MB, "stride": 0,
                                                                  "iters": max(8, 256 // G), "warps": 8,
                                                                  "pattern": "contig", "G": G, "fill": 0,
                                                                  "wait": True, "cache": cache}))
        return out


@dataclass
class SplitPhase(SplitGap):
    """Does hoisting the later sector halves pay off in every launch regime? (optimize benchmark:
    multi-wave GEMVs gain 1.3-1.6x, short sub-wave ones lose up to 8%.)
    Each warp runs T trips: first halves of G rows -> C dependent FFMAs on that data -> second halves.
      late:    second-half addresses depend on the compute result (forced late, like ptxas's GEMV order)
      hoisted: second halves issued with the first halves (what lab/optimize.py produces)
    Grid size sets the number of waves; occupancy is fixed by __launch_bounds__(128, 12)."""

    def __post_init__(self):
        self.name = "mem_split_phase"
        self.description = "late vs hoisted sector halves across waves and trips (regime test)"

    def source(self, v):
        G, C, order = v.params["G"], v.params["C"], v.params["order"]
        ld = lambda idx, a: (f'asm volatile("ld.global.nc.v4.u32 {{%0,%1,%2,%3}}, [%4];" : "=r"(r{idx}.x), '  # noqa: E731
                             f'"=r"(r{idx}.y), "=r"(r{idx}.z), "=r"(r{idx}.w) : "l"(B + {a}));')
        first = "\n      ".join(ld(g, f"(base + {g} * 1024 + lane * 32)") for g in range(G))
        second = "\n      ".join(ld(G + g, f"(base2 + {g} * 1024 + lane * 32 + 16)") for g in range(G))
        dep = " ^ ".join(f"r{g}.x" for g in range(G))
        chain = "\n      ".join('asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(f) : "f"(fz));' for _ in range(C))
        decl = " ".join(f"uint4 r{i};" for i in range(2 * G))
        fold = " ^ ".join(f"r{i}.x ^ r{i}.y ^ r{i}.z ^ r{i}.w" for i in range(2 * G))
        if order == "late":
            body = f"""{first}
      f += (float)({dep});
      {chain}
      unsigned long long base2 = base + ((unsigned long long)__float_as_uint(f) & zmask);
      {second}"""
        else:
            # "early": first halves of all rows, then second halves, then compute (ptxas schedules it);
            # "paired" = "early" rewritten by lab/optimize.py (each later half right after its partner)
            body = f"""unsigned long long base2 = base;
      {first}
      {second}
      f += (float)({dep});
      {chain}"""
        return f"""
extern "C" __global__ void __launch_bounds__(128, 12) k(const unsigned char* __restrict__ B, unsigned long long mask,
    int stride, int iters, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  unsigned long long warp = (blockIdx.x * 128ull + threadIdx.x) >> 5, nw = gridDim.x * 4ull;
  unsigned long long zmask = (unsigned long long)stride;          // runtime 0
  unsigned lane = threadIdx.x & 31, acc = 0;
  float f = (float)lane, fz = (float)stride;
  for (int i = 0; i < iters; ++i) {{
    unsigned long long base = ((warp + i * nw) * {1024 * G}ull) & mask;
    {decl}
    {body}
    acc ^= {fold};
  }}
  if (acc == 0x9e3779b9u || f == 12345.f) sink[0] = acc;
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        out = []
        for waves in (0.25, 0.5, 1, 2, 4):
            for trips in (2, 8):
                for order in ("late", "early", "paired"):
                    out.append(Variant(f"w{waves}/t{trips}/{order}", {"size": 256 * MB, "stride": 0, "iters": trips,
                                                                       "warps": 4, "G": 8, "C": 256, "order": order,
                                                                       "waves": waves, "pattern": "split"}))
        return out

    def build_key(self, v):
        return v.params["order"]

    def transform(self, cubin, v):
        if v.params["order"] != "paired":
            return cubin
        from lab import optimize
        c = optimize.candidates(cubin, "k")["hoist+rename"]
        if "cubin" not in c or c["moves"] == 0 or optimize.validate(c["cubin"], cubin):
            raise SystemExit("optimizer could not pair the halves")
        return c["cubin"]

    def prepare(self, dev, v):
        blocks = max(1, int(68 * 12 * v.params["waves"]))          # 12 blocks of 128 per SM = one wave
        st = Coalesce.prepare(self, dev, v)
        dev.free(st["T"])
        st["T"] = dev.alloc(blocks * 16)
        st["blocks"] = blocks
        st["launch"]["grid"] = blocks
        st["launch"]["block"] = 128
        st["launch"]["args"][5] = ctypes.c_uint64(st["T"])
        st["useful"] = blocks * 128 * v.params["iters"] * 32 * v.params["G"]
        return st
