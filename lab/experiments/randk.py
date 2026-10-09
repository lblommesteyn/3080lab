"""Random kernels for out-of-sample predictor evaluation.

Each variant is a seeded random program: k independent chains per thread
(float or int), each a random sequence of ops from a menu, interleaved in a
random order, run with a random warp count. Nothing here was used to fit the
model's parameters. Timing only (values may overflow; timing is
value-independent for these ops).
"""
from __future__ import annotations

import ctypes
import random
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

FLOAT_OPS = {
    "ffma": "fma.rn.f32 {d}, {d}, %1, %2;",
    "fadd": "add.f32 {d}, {d}, %1;",
    "fmul": "mul.f32 {d}, {d}, %1;",
    "rsqrt": "rsqrt.approx.ftz.f32 {d}, {d};",
    "ex2": "ex2.approx.ftz.f32 {d}, {d};",
}
INT_OPS = {
    "imad": "mad.lo.s32 {d}, {d}, %3, %4;",
    "shl": "shl.b32 {d}, {d}, %3;",
    "shfl": "shfl.sync.idx.b32 {d}, {d}, %5, 0x1f, 0xffffffff;",
}


@dataclass
class RandomKernels(Experiment):
    n_kernels: int = 60
    seed0: int = 1000

    def __post_init__(self):
        self.name = "rand_kernels"
        self.description = "seeded random ALU/MUFU/SHFL programs for out-of-sample prediction"

    def _program(self, seed: int) -> dict:
        rng = random.Random(seed)
        k = rng.choice([1, 2, 3, 4, 6, 8])
        warps = rng.choice([1, 2, 4, 8, 16, 32])
        length = rng.choice([16, 32, 48, 64])
        chains = []
        for _ in range(k):
            is_float = rng.random() < 0.6
            menu = FLOAT_OPS if is_float else INT_OPS
            weights = [5, 2, 2, 1, 1] if is_float else [3, 3, 1]
            ops = rng.choices(list(menu), weights=weights, k=length)
            chains.append({"float": is_float, "ops": ops})
        # random interleave: repeatedly pick a chain with ops left
        cursor = [0] * k
        order = []
        while any(cursor[c] < length for c in range(k)):
            c = rng.choice([c for c in range(k) if cursor[c] < length])
            order.append((c, chains[c]["ops"][cursor[c]]))
            cursor[c] += 1
        return {"k": k, "warps": warps, "length": length, "chains": chains, "order": order}

    def source(self, v: Variant) -> str:
        p = self._program(v.params["seed"])
        lines = []
        for c, op in p["order"]:
            is_float = p["chains"][c]["float"]
            tmpl = (FLOAT_OPS if is_float else INT_OPS)[op].format(d="%0")
            cons = "f" if is_float else "r"
            lines.append(f'asm volatile("{tmpl}" : "+{cons}"(x{c}) : "f"(fa), "f"(fb), "r"(ia), "r"(ib), "r"(lane1));')
        decl = "\n  ".join(
            (f"float x{c} = fin[0] + (float)threadIdx.x;" if ch["float"] else f"int x{c} = iin[0] + (int)threadIdx.x;")
            for c, ch in enumerate(p["chains"]))
        fold = " + ".join(f"(float)x{c}" for c in range(p["k"]))
        body = "\n      ".join(lines)
        return f"""
extern "C" __global__ void k(float* out, long long* cyc, unsigned long long* ns,
                             const float* fin, const int* iin, int iters)
{{
  {decl}
  float fa = fin[1], fb = fin[2];
  int ia = iin[1], ib = iin[2];
  int lane1 = (threadIdx.x + 1) & 31;
  __syncthreads();
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
      {body}
  }}
  long long t1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = {fold};
  if ((threadIdx.x & 31) == 0) {{ cyc[tid >> 5] = t1 - t0; ns[tid >> 5] = g1 - g0; }}
}}
"""

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for i in range(int(opts.get("n") or self.n_kernels)):
            seed = self.seed0 + i
            p = self._program(seed)
            ops = p["k"] * p["length"]
            slow = sum(op in ("rsqrt", "ex2", "shfl") for _, op in p["order"]) / ops
            # keep each launch well under the ~2 s TDR even when MUFU/SHFL-bound at 32 warps
            per_thread = int(opts.get("iters") or (60_000 if slow > 0.2 else 300_000))
            out.append(Variant(f"seed{seed}", {"seed": seed, "warps": p["warps"], "k": p["k"],
                                               "iters": max(1, per_thread // ops)}))
        return out

    def prepare(self, dev, v: Variant) -> dict:
        w = v.params["warps"]
        st = {"out": dev.alloc(32 * w * 4), "cyc": dev.alloc(w * 8), "ns": dev.alloc(w * 8),
              "fin": dev.alloc(12), "iin": dev.alloc(12)}
        dev.htod(st["fin"], np.array([1.0, 1.0, 0.0], np.float32))
        dev.htod(st["iin"], np.array([1, 1, 0], np.int32))
        st["launch"] = dict(grid=1, block=32 * w,
                            args=[ctypes.c_uint64(st[x]) for x in ("out", "cyc", "ns", "fin", "iin")]
                            + [ctypes.c_int32(v.params["iters"])])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        w = v.params["warps"]
        cyc = dev.dtoh(np.zeros(w, np.int64), st["cyc"])
        ns = dev.dtoh(np.zeros(w, np.uint64), st["ns"])
        cmax = int(cyc.max())
        p = self._program(v.params["seed"])
        ops = v.params["iters"] * p["k"] * p["length"]
        import hashlib
        y = dev.dtoh(np.zeros(32 * w, np.float32), st["out"])
        return {"y_hash": hashlib.sha1(y.tobytes()).hexdigest()[:12],
                "cycles": cmax, "ns": int(ns.max()), "ops_per_thread": ops,
                "cycles_per_op": cmax / ops, "warp_ops_per_cycle": w * ops / cmax,
                "sm_mhz_inkernel": cmax / max(int(ns.max()), 1) * 1e3}

    def release(self, dev, st: dict):
        for x in ("out", "cyc", "ns", "fin", "iin"):
            dev.free(st[x])


@dataclass
class ReschedRandom(RandomKernels):
    """Phase 8 scheduler on random kernels: ptxas vs our critical-path schedule vs a random legal
    order (a correctness stress test). Outputs must match bitwise (y_hash)."""
    n_kernels: int = 40

    def __post_init__(self):
        self.name = "resched_rand"
        self.description = "lab/resched.py block scheduler vs ptxas on seeded random kernels"

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for v in super().variants(opts):
            for pol in ("ptxas", "crit", "random", "identity"):
                out.append(Variant(f"{v.label}/{pol}", dict(v.params, policy=pol)))
        return out

    def build_key(self, v: Variant):
        return v.params["policy"]

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        if v.params["policy"] == "ptxas":
            return cubin
        from lab import resched
        out, _ = resched.reschedule(cubin, "k", policy=v.params["policy"], seed=v.params["seed"])
        return out


def registry():
    a = RandomKernels()
    b = RandomKernels(seed0=2000)  # held-out set for model versions tuned on set A
    b.name = "rand_kernels_b"
    c = ReschedRandom()
    return {a.name: a, b.name: b, c.name: c}
