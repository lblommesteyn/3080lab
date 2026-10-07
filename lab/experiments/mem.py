"""Pointer chasing: latency vs working-set size (Phase 1 loads, Phase 3 hierarchy).

Each element holds the absolute address of the next, so the timed chain is a
pure sequence of dependent loads with no address arithmetic:  p = *p.
Elements sit `stride` bytes apart; the visit order is either a single random
cycle (Sattolo) or sequential. An untimed warm walk precedes the timed walk
so we measure steady-state residency, not cold misses (unless --cold).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

KB, MB = 1 << 10, 1 << 20
DEFAULT_SIZES = [2 * KB << i for i in range(18)]  # 2 KB .. 256 MB


def sattolo(n: int, rng: np.random.Generator) -> np.ndarray:
    """Successor array of one random cycle covering all n slots."""
    order = rng.permutation(n)
    nxt = np.empty(n, dtype=np.int64)
    nxt[order] = np.roll(order, -1)
    return nxt


def _fmt(n: int) -> str:
    return f"{n // MB}MB" if n >= MB and n % MB == 0 else f"{n // KB}KB"


@dataclass
class Chase(Experiment):
    space: str = "global"   # global | shared
    cache: str = "ca"       # ca (L1+L2) | cg (L2 only) for global
    pattern: str = "random"
    stride: int = 128
    body: int = 64

    def __post_init__(self):
        suffix = "" if self.space == "shared" or self.cache == "ca" else f"_{self.cache}"
        pat = "" if self.pattern == "random" else f"_{self.pattern}"
        self.name = f"chase_{self.space}{suffix}{pat}"
        # Observed sm_86 encodings: inline-asm ld.ca -> STRONG.SM, ld.cg -> STRONG.GPU, plain C (int->ptr) -> generic LD.E.64
        self.target_opcode = {"ca": "LDG.E.64.STRONG.SM", "cg": "LDG.E.64.STRONG.GPU",
                              "plain": "LD.E.64"}[self.cache] if self.space == "global" else "LDS"
        self.description = (f"dependent {self.space} loads ({self.pattern}, stride {self.stride}B"
                            f"{', ld.' + self.cache if self.space == 'global' else ''}) vs working set")

    def source(self, v: Variant) -> str:
        if self.space == "shared":
            load = 'asm volatile("ld.shared.u32 %0, [%0];" : "+r"(p));'
            return f"""
extern "C" __global__ void k(const unsigned* init, long long* cyc, unsigned long long* ns,
                             unsigned* out, int n_elems, int warm, int iters)
{{
  extern __shared__ unsigned sm[];
  unsigned base = (unsigned)__cvta_generic_to_shared(sm);
  for (int i = threadIdx.x; i < n_elems; i += blockDim.x) sm[i * {self.stride // 4}] = base + init[i] * {self.stride};
  __syncthreads();
  if (threadIdx.x) return;
  unsigned p = base;
  #pragma unroll 1
  for (int i = 0; i < warm; ++i) {{ {load} }}
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    {" ".join([load] * self.body)}
  }}
  long long t1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = p; cyc[0] = t1 - t0; ns[0] = g1 - g0;
}}
"""
        if self.cache == "plain":  # ordinary C dereference: whatever ptxas picks by default
            load = "p = *(const unsigned long long*)p;"
        else:
            load = f'asm volatile("ld.global.{self.cache}.u64 %0, [%0];" : "+l"(p));'
        return f"""
extern "C" __global__ void k(const unsigned long long* arr, long long* cyc, unsigned long long* ns,
                             unsigned long long* out, int n_elems, int warm, int iters)
{{
  unsigned long long p = (unsigned long long)arr;
  #pragma unroll 1
  for (int i = 0; i < warm; ++i) {{ {load} }}
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    {" ".join([load] * self.body)}
  }}
  long long t1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = p; cyc[0] = t1 - t0; ns[0] = g1 - g0;
}}
"""

    def expected_body_ops(self) -> int:
        return self.body

    def variants(self, opts: dict) -> list[Variant]:
        sizes = opts.get("sizes") or DEFAULT_SIZES
        if self.space == "shared":
            sizes = [s for s in sizes if s <= 48 * KB] or [48 * KB]  # >48 KB needs a func attribute opt-in
        loads = int(opts.get("iters") or 200_000)
        return [Variant(_fmt(s), {"bytes": s, "iters": max(1, loads // self.body), "warps": 1,
                                  "cold": bool(opts.get("cold"))}) for s in sizes]

    def prepare(self, dev, v: Variant) -> dict:
        size = v.params["bytes"]
        n = max(2, size // self.stride)
        rng = np.random.default_rng(size)
        nxt = sattolo(n, rng) if self.pattern == "random" else (np.arange(n) + 1) % n
        # Warm walk covers the set once (capped: beyond ~16 MB at 128 B it cannot stay in L2 anyway,
        # and long kernels risk the 2 s Windows TDR).
        warm = 0 if v.params["cold"] else min(n, 1 << 17)
        st = {"cyc": dev.alloc(8), "ns": dev.alloc(8), "out": dev.alloc(8), "n": n}
        if self.space == "shared":
            st["buf"] = dev.alloc(n * 4)
            dev.htod(st["buf"], nxt.astype(np.uint32))
            launch = dict(grid=1, block=256, smem=n * self.stride)
        else:
            # Byte layout: slot i lives at i*stride; it holds the address of slot nxt[i].
            words = self.stride // 8
            host = np.zeros(n * words, dtype=np.uint64)
            st["buf"] = dev.alloc(size)
            host[np.arange(n) * words] = np.uint64(st["buf"]) + nxt.astype(np.uint64) * np.uint64(self.stride)
            dev.htod(st["buf"], host)
            launch = dict(grid=1, block=32, smem=0)
        launch["args"] = [ctypes.c_uint64(st["buf"]), ctypes.c_uint64(st["cyc"]), ctypes.c_uint64(st["ns"]),
                          ctypes.c_uint64(st["out"]), ctypes.c_int32(n), ctypes.c_int32(warm),
                          ctypes.c_int32(v.params["iters"])]
        st["launch"] = launch
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        ns = int(dev.dtoh(np.zeros(1, np.uint64), st["ns"])[0])
        loads = v.params["iters"] * self.body
        return {
            "cycles": cyc, "ns": ns, "ops_per_thread": loads,
            "cycles_per_op": cyc / loads,
            "ns_per_op": ns / loads,
            "warp_ops_per_cycle": loads / cyc,
            "sm_mhz_inkernel": cyc / max(ns, 1) * 1e3,
        }

    def release(self, dev, st: dict):
        for key in ("buf", "cyc", "ns", "out"):
            dev.free(st[key])


def registry() -> dict[str, Experiment]:
    exps = [Chase(space="global"), Chase(space="global", cache="cg"), Chase(space="global", cache="plain"),
            Chase(space="shared", stride=4), Chase(space="global", pattern="seq"), Carveout(space="global")]
    return {e.name: e for e in exps}


@dataclass
class Carveout(Chase):
    """L1 capacity vs the preferred shared-memory carveout (percent of max smem).
    -1 = driver default. Unified L1/smem on sm_86 is 128 KB per SM."""
    carveouts: tuple = (-1, 0, 50, 100)

    def __post_init__(self):
        super().__post_init__()
        self.name = "chase_l1_carveout"
        self.description = "L1 hit latency vs working set, per preferred smem carveout"

    def variants(self, opts: dict) -> list[Variant]:
        sizes = opts.get("sizes") or [k * KB for k in range(16, 137, 8)]
        loads = int(opts.get("iters") or 100_000)
        return [Variant(f"c{c}/{_fmt(s)}", {"bytes": s, "iters": max(1, loads // self.body), "warps": 1,
                                             "cold": False, "carveout": c})
                for c in self.carveouts for s in sizes]

    def kernel_attrs(self, v: Variant) -> dict[str, int]:
        c = v.params["carveout"]
        return {} if c < 0 else {"PREFERRED_SHARED_MEMORY_CARVEOUT": c}
