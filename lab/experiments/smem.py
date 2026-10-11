"""Shared-memory width and the cp.async (global -> shared) path, for the multistage GEMM model.

  smem_width    conflict-free LDS.32 / .64 / .128 and LDSM.x4 throughput on one SM vs warps:
                bytes per SM cycle (smem_banks measured only 32-bit loads).
  l2_cpasync    GPU-wide cp.async.cg ring with no compute (wait_group NSTG - 2 + one barrier per
                stage, as gemm_q4i8_ms_source): achieved L2 -> shared bandwidth vs blocks per SM,
                stage bytes, ring depth, access pattern, and F = blocks reading the same data
                (the GEMM's X tile is read by every row-block of a wave, its weights by 8 blocks).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant
from .membw import TIMED_HEAD, TIMED_TAIL, _span

SMS = 68


@dataclass
class SmemWidth(Experiment):
    def __post_init__(self):
        self.name = "smem_width"
        self.description = "conflict-free LDS.32/.64/.128 and LDSM.x4 bytes per SM cycle vs warps (one block, one SM)"

    def expected(self, v):
        return {}

    def source(self, v):
        w = v.params["width"]
        if w == "ldsm":
            ld = ('asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" '
                  ': "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(b + OFF));')
            addr = "(lane * 16)"               # 32 row addresses, 16 B each, contiguous: conflict-free
            fold = "r0 ^ r1 ^ r2 ^ r3"
        else:
            ty, n = {4: ("u32", 1), 8: ("v2.u32", 2), 16: ("v4.u32", 4)}[w]
            regs = ", ".join(f"%{i}" for i in range(n))
            dst = "{" + regs + "}" if n > 1 else "%0"
            outs = ", ".join(f'"=r"(r{i})' for i in range(n))
            ld = f'asm volatile("ld.volatile.shared.{ty} {dst}, [%{n}];" : {outs} : "r"(b + OFF));'
            addr = f"(lane * {w})"
            fold = " ^ ".join(f"r{i}" for i in range(n))
        body = "\n    ".join(ld.replace("OFF", str(k * 1024)) + f" acc ^= {fold};" for k in range(4))
        return f"""
extern "C" __global__ void __launch_bounds__(1024) k(int iters, unsigned* sink, long long* cyc)
{{
  __shared__ __align__(16) unsigned sm[8192 + 1024];
  for (int i = threadIdx.x; i < 8192 + 1024; i += blockDim.x) sm[i] = i;
  __syncthreads();
  unsigned lane = threadIdx.x & 31, acc = 0, r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  unsigned smbase = (unsigned)__cvta_generic_to_shared(sm);
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    unsigned b = smbase + {addr} + (i & 7) * 4096;     // iteration-dependent, same banks
    {body}
  }}
  long long t1 = clock64();
  if (acc == 0x9e3779b9u) sink[0] = acc;
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
}}
"""

    def variants(self, opts):
        out = []
        for w in (4, 8, 16, "ldsm"):
            for warps in (4, 8, 16, 32):
                out.append(Variant(f"{'LDSM.x4' if w == 'ldsm' else f'LDS{8 * w}'}/w{warps}",
                                   {"width": w, "warps": warps, "iters": 2048}))
        return out

    def prepare(self, dev, v):
        st = {"sink": dev.alloc(4), "cyc": dev.alloc(8)}
        st["launch"] = dict(grid=1, block=32 * v.params["warps"], args=[
            ctypes.c_int32(v.params["iters"]), ctypes.c_uint64(st["sink"]), ctypes.c_uint64(st["cyc"])])
        return st

    def collect(self, dev, v, st):
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        w = v.params["width"]
        per = 512 if w == "ldsm" else 32 * w
        n = v.params["warps"] * v.params["iters"] * 4
        return {"cycles": cyc, "ops_per_thread": v.params["iters"] * 4, "cycles_per_op": cyc / (v.params["iters"] * 4),
                "warp_ops_per_cycle": n / cyc, "bytes_per_cycle": n * per / cyc, "sm_mhz_inkernel": 0.0}

    def release(self, dev, st):
        dev.free(st["sink"]), dev.free(st["cyc"])


@dataclass
class L2CpAsync(Experiment):
    def __post_init__(self):
        self.name = "l2_cpasync"
        self.description = ("cp.async.cg ring, no compute: L2 -> smem GB/s vs blocks/SM, stage bytes, NSTG, "
                            "pattern, blocks sharing data")

    def expected(self, v):
        return {}

    def source(self, v):
        p = v.params
        S, NSTG, nthr = p["stage"], p["nstg"], 32 * p["warps"]
        if p["pattern"] == "rows":      # GEMM X: 64 B per row per stage, rows 1536 B apart
            src = "base + (size_t)(i >> 2) * 1536 + (s % 24) * 64 + (i & 3) * 16"
        elif p["pattern"].startswith("scatter"):   # GEMM activation scales: 16 or 32 B per line, lines 2240 B apart
            per = int(p["pattern"][7:]) // 16
            src = f"base + (size_t)(i / {per}) * 2240 + (s % 140) * 16 * {per} + (i % {per}) * 16"
        elif p["pattern"] == "stream":  # GEMM weights: each region streamed once (DRAM), by F blocks at once
            src = f"base + (size_t)s * {S} + i * 16"
        else:
            src = f"base + ((size_t)s * {S} + i * 16) % {p['region']}"
        return f"""
__device__ __forceinline__ void cp16(unsigned d, const void* src) {{
  asm volatile("cp.async.{'cg' if p['cp'] == 16 else 'ca'}.shared.global [%0], [%1], {p['cp']};" :: "r"(d), "l"(src));
}}
extern "C" __global__ void __launch_bounds__({nthr}) k(const unsigned char* __restrict__ B, int share, int stages,
    unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  __shared__ __align__(16) unsigned char ring[{NSTG * S}];
  const unsigned rb = (unsigned)__cvta_generic_to_shared(ring);
  const unsigned char* base = B + (size_t)(blockIdx.x / share) * {p['region']};
  auto issue = [&](int slot, int s) {{
    #pragma unroll
    for (int i = threadIdx.x; i < {S // 16}; i += {nthr})
      if ((threadIdx.x & 31) < {p['lanes']}) cp16(rb + slot * {S} + i * 16, {src});
  }};
  #pragma unroll
  for (int i = 0; i < {NSTG - 1}; ++i) {{ issue(i, i); asm volatile("cp.async.commit_group;"); }}
  int lslot = {NSTG - 1};
  #pragma unroll 1
  for (int s = 0; s < stages; ++s) {{
    asm volatile("cp.async.wait_group %0;" :: "n"({NSTG - 2}));
    __syncthreads();
    if (s + {NSTG - 1} < stages) issue(lslot, s + {NSTG - 1});
    asm volatile("cp.async.commit_group;");
    lslot = lslot == {NSTG - 1} ? 0 : lslot + 1;
  }}
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  if (ring[threadIdx.x] == 0x5a && threadIdx.x == 1023) sink[0] = 1;
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        out = []
        add = lambda lab, **kw: out.append(Variant(lab, {"warps": 4, "bps": 3, "stage": 8192, "nstg": 3, "share": 1,  # noqa: E731
                                                          "pattern": "contig", "region": 16384, "cp": 16, "lanes": 32,
                                                          **kw}))
        # per-instruction or per-byte cost? same instruction count, fewer bytes: half the lanes, or 8 / 4 B copies
        add("contig/b3/lanes16", lanes=16)
        add("contig/b3/lanes8", lanes=8)
        add("contig/b3/cp8", cp=8)
        add("contig/b3/cp4", cp=4)
        for bps in (1, 2, 3):                                   # occupancy
            for share in (1, 8, 34):
                add(f"contig/b{bps}/F{share}", bps=bps, share=share)
        for nstg in (2, 4):                                     # ring depth
            add(f"contig/b3/S{nstg}", nstg=nstg)
        add("contig/b2/w8", bps=2, warps=8)
        add("contig/b2/w8/16K", bps=2, warps=8, stage=16384, region=32768)
        add("contig/b3/4K", stage=4096)
        for share in (8, 34):                                   # the GEMM's X: 64 B per row, rows 1536 B apart
            add(f"rows/b3/F{share}", pattern="rows", share=share, region=128 * 1536)
        for pat in ("scatter16", "scatter32"):
            for share in (8, 34):
                add(f"{pat}/b3/F{share}", pattern=pat, share=share, stage=4096,
                    region=(4096 // (16 * int(pat[7:]) // 16)) * 2240)
        for share in (1, 8, 34):                                # DRAM stream read by F blocks concurrently:
            for stage in (2048, 8192):                          # do the F misses merge in L2?
                add(f"stream/b3/F{share}/{stage // 1024}K", pattern="stream", share=share, stage=stage,
                    region=256 * stage)
        return out

    def prepare(self, dev, v):
        p = v.params
        blocks = SMS * p["bps"]
        stages = 256
        size = (blocks // p["share"] + 1) * p["region"]
        st = {"B": dev.alloc(size), "sink": dev.alloc(4), "T": dev.alloc(blocks * 16), "blocks": blocks}
        dev.memset(st["B"], size)
        st["bytes"] = blocks * stages * p["stage"] * p["lanes"] // 32 * p["cp"] // 16
        st["footprint_mb"] = size / 2**20
        st["launch"] = dict(grid=blocks, block=32 * p["warps"], args=[
            ctypes.c_uint64(st["B"]), ctypes.c_int32(p["share"]), ctypes.c_int32(stages),
            ctypes.c_uint64(st["sink"]), ctypes.c_uint64(st["T"])])
        return st

    def collect(self, dev, v, st):
        ns = _span(dev, st)
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "gbps": st["bytes"] / ns, "footprint_mb": st["footprint_mb"]}

    def release(self, dev, st):
        for x in ("B", "sink", "T"):
            dev.free(st[x])


@dataclass
class MemEpilogue(Experiment):
    """DRAM bandwidth of GEMM-epilogue-shaped traffic: one 4 B (or 2 B) element per thread per step,
    coalesced, over a 64 MB output (as gemm_q4i8_ms_source's epilogue loop)."""

    def __post_init__(self):
        self.name = "mem_epilogue"
        self.description = "DRAM GB/s for fp32 y += a (read-modify-write), fp32 store, bf16 store"

    def expected(self, v):
        return {}

    def source(self, v):
        op = {"rmw": "Y[i] += 1.0f;", "st32": "Y[i] = (float)i;",
              "st16": "((unsigned short*)Y)[i] = (unsigned short)i;"}[v.params["op"]]
        return f"""
extern "C" __global__ void __launch_bounds__(256) k(float* Y, unsigned long long n, unsigned* sink, unsigned long long* T)
{{
  {TIMED_HEAD}
  for (unsigned long long i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull) {{ {op} }}
  {TIMED_TAIL}
}}
"""

    def variants(self, opts):
        return [Variant(f"{op}/b{bps}", {"op": op, "bps": bps, "warps": 8}) for op in ("rmw", "st32", "st16")
                for bps in (2, 6)]

    def prepare(self, dev, v):
        size = 64 << 20
        blocks = SMS * v.params["bps"]
        st = {"B": dev.alloc(size), "sink": dev.alloc(4), "T": dev.alloc(blocks * 16), "blocks": blocks}
        dev.memset(st["B"], size)
        esz = 2 if v.params["op"] == "st16" else 4
        n = size // esz
        st["bytes"] = size * (2 if v.params["op"] == "rmw" else 1)
        st["launch"] = dict(grid=blocks, block=256, args=[
            ctypes.c_uint64(st["B"]), ctypes.c_uint64(n), ctypes.c_uint64(st["sink"]), ctypes.c_uint64(st["T"])])
        return st

    def collect(self, dev, v, st):
        ns = _span(dev, st)
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "gbps": st["bytes"] / ns}

    def release(self, dev, st):
        for x in ("B", "sink", "T"):
            dev.free(st[x])


def registry():
    exps = [SmemWidth(), L2CpAsync(), MemEpilogue()]
    return {e.name: e for e in exps}
