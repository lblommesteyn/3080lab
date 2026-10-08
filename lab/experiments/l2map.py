"""L2 structure and TLB reach.

l2_latency_map: one block per SM (first block to claim its %smid) times every
128 B line of an L2-resident 2 MB buffer (ld.cg, L1 bypassed) with the
clock -> load -> wait -> clock pattern from cache.py. Full matrix [SM, line]
is saved to results/l2map_<n>.npy; the record keeps per-SM summaries.

tlb_reach: pointer chase over one line in each of N pages (page stride P),
2 GB buffer. The touched lines total N*128 B, so they stay cache-resident and
latency above the L2-hit baseline is address translation. The chain is
(re)written by the kernel itself each launch, so all variants share a buffer.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

SMS = 68
LINE = 128


@dataclass
class L2Map(Experiment):
    size: int = 2 << 20

    def __post_init__(self):
        self.name = "l2_latency_map"
        self.description = "per-SM latency of every L2 line (2 MB buffer, L1 bypassed)"

    def source(self, v):
        return r"""
extern "C" __global__ void k(const unsigned char* __restrict__ B, int nlines, unsigned* claim,
                             unsigned short* lat, int* smids)
{
  unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  if (threadIdx.x != 0) return;
  if (atomicCAS(&claim[sm], 0u, 1u) != 0u) return;     // one measuring block per SM
  smids[blockIdx.x] = (int)sm;
  unsigned acc = 0;
  for (int i = 0; i < nlines; ++i) {                    // warm: every line into L2
    unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(B + (size_t)i * 128)); acc += v;
  }
  for (int i = 0; i < nlines; ++i) {
    long long t0, t1 = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    const unsigned char* a = B + (size_t)i * 128 + ((unsigned long long)t0 >> 63);
    unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a));
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0x7fffffff; @p mov.u64 %0, %%clock64; }" : "+l"(t1) : "r"(v));
    acc += v;
    lat[(size_t)sm * nlines + i] = (unsigned short)min((long long)65535, t1 - t0);
  }
  if (acc == 0x12345678u) smids[0] = -2;
}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        return [Variant("all_sms", {"warps": 1, "iters": 1})]

    def prepare(self, dev, v):
        n = self.size // LINE
        blocks = SMS * 16
        st = {"B": dev.alloc(self.size), "claim": dev.alloc(SMS * 4), "lat": dev.alloc(SMS * n * 2),
              "smids": dev.alloc(blocks * 4), "n": n, "blocks": blocks}
        dev.memset(st["B"], self.size)
        st["launch"] = dict(grid=blocks, block=32, args=[
            ctypes.c_uint64(st["B"]), ctypes.c_int32(n), ctypes.c_uint64(st["claim"]),
            ctypes.c_uint64(st["lat"]), ctypes.c_uint64(st["smids"])])
        st["base"] = st["B"]
        return st

    def collect(self, dev, v, st):
        from ..runner import RESULTS
        lat = dev.dtoh(np.zeros(SMS * st["n"], np.uint16), st["lat"]).reshape(SMS, st["n"])
        dev.memset(st["claim"], SMS * 4)          # reset for the next trial
        dev.memset(st["lat"], SMS * st["n"] * 2)
        RESULTS.mkdir(exist_ok=True)
        k = len(list(RESULTS.glob("l2map_*.npy")))
        np.save(RESULTS / f"l2map_{k}.npy", lat)
        np.save(RESULTS / f"l2map_{k}_base.npy", np.array([st["base"]], dtype=np.uint64))
        got = lat.max(1) > 0
        med = np.median(lat[got], axis=1)
        return {"cycles": float(np.median(lat[got])), "ops_per_thread": 1, "cycles_per_op": float(np.median(lat[got])),
                "warp_ops_per_cycle": 0.0, "sm_mhz_inkernel": 0.0, "sms_measured": int(got.sum()),
                "median_lat_min_sm": float(med.min()), "median_lat_max_sm": float(med.max()),
                "map_file": f"l2map_{k}.npy"}

    def release(self, dev, st):
        for x in ("B", "claim", "lat", "smids"):
            dev.free(st[x])


@dataclass
class TLB(Experiment):
    def __post_init__(self):
        self.name = "tlb_reach"
        self.description = "chase one line per page over N pages: latency above L2-hit = translation cost"

    def source(self, v):
        return r"""
extern "C" __global__ void k(unsigned char* B, const unsigned long long* offs, const unsigned* nxt, int N,
                             int steps, long long* cyc, unsigned long long* out)
{
  if (threadIdx.x != 0) return;
  for (int i = 0; i < N; ++i)
    *(unsigned long long*)(B + offs[i]) = (unsigned long long)(B + offs[nxt[i]]);
  __threadfence();
  unsigned long long p = (unsigned long long)(B + offs[0]);
  for (int i = 0; i < N; ++i) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));   // warm caches + TLBs
  long long t0 = clock64();
  for (int i = 0; i < steps; ++i) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
  long long t1 = clock64();
  out[0] = p; cyc[0] = t1 - t0;
}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        out = []
        for page, label in ((64 << 10, "64KB"), (2 << 20, "2MB")):
            for n in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192, 16384):
                if n * page > (2 << 30):
                    continue
                out.append(Variant(f"{label}/N{n}", {"page": page, "N": n, "steps": 4096, "warps": 1, "iters": 1}))
        return out

    def prepare(self, dev, v):
        if not hasattr(self, "_buf"):
            self._buf, self._refs = dev.alloc(2 << 30), 0
        self._refs += 1
        n, page = v.params["N"], v.params["page"]
        rng = np.random.default_rng(n * 31 + page)
        order = rng.permutation(n)
        nxt = np.empty(n, np.uint32)
        nxt[order] = np.roll(order, -1)
        # one line per page, at a random line offset within the page (spreads L2 sets)
        offs = (np.arange(n, dtype=np.uint64) * np.uint64(page)
                + rng.integers(0, min(page, 1 << 20) // LINE, n).astype(np.uint64) * np.uint64(LINE))
        st = {"offs": dev.alloc(n * 8), "nxt": dev.alloc(n * 4), "cyc": dev.alloc(8), "out": dev.alloc(8)}
        dev.htod(st["offs"], offs)
        dev.htod(st["nxt"], nxt)
        st["launch"] = dict(grid=1, block=32, args=[
            ctypes.c_uint64(self._buf), ctypes.c_uint64(st["offs"]), ctypes.c_uint64(st["nxt"]), ctypes.c_int32(n),
            ctypes.c_int32(v.params["steps"]), ctypes.c_uint64(st["cyc"]), ctypes.c_uint64(st["out"])])
        return st

    def collect(self, dev, v, st):
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        return {"cycles": cyc, "ops_per_thread": v.params["steps"], "cycles_per_op": cyc / v.params["steps"],
                "warp_ops_per_cycle": 0.0, "sm_mhz_inkernel": 0.0,
                "coverage_mb": v.params["N"] * v.params["page"] / 2**20}

    def release(self, dev, st):
        for x in ("offs", "nxt", "cyc", "out"):
            dev.free(st[x])
        self._refs -= 1
        if self._refs == 0:
            dev.free(self._buf)
            del self._buf


def registry():
    exps = [L2Map(), TLB()]
    return {e.name: e for e in exps}
