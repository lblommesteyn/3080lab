"""Tensor cores (mma.sync) on sm_86: latency, throughput, SASS, and correctness.

Each op is issued with inline PTX. Latency: one accumulator chain (D feeds C).
Throughput: k independent accumulators per warp, swept over warps per SM.
A/B fragments hold zeros in the timing kernels (results stay finite; tensor-core
timing is not value dependent), so correctness is checked separately by
tc_correct, which feeds known matrices through the documented PTX fragment
layouts and compares with numpy.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

# name -> (ptx shape/types, #A regs, #B regs, #C regs, C constraint, SASS opcode prefix, m, n, k)
TC = {
    "f16_f32":  ("m16n8k16.row.col.f32.f16.f16.f32", 4, 2, 4, "f", "HMMA.16816.F32", 16, 8, 16),
    "f16_f16":  ("m16n8k16.row.col.f16.f16.f16.f16", 4, 2, 2, "r", "HMMA.16816.F16", 16, 8, 16),
    "bf16_f32": ("m16n8k16.row.col.f32.bf16.bf16.f32", 4, 2, 4, "f", "HMMA.16816.F32.BF16", 16, 8, 16),
    "tf32_f32": ("m16n8k8.row.col.f32.tf32.tf32.f32", 4, 2, 4, "f", "HMMA.1688.F32.TF32", 16, 8, 8),
    "s8_s32":   ("m16n8k32.row.col.s32.s8.s8.s32", 4, 2, 4, "r", "IMMA.16832.S8.S8", 16, 8, 32),
    "u8s8_s32": ("m16n8k32.row.col.s32.u8.s8.s32", 4, 2, 4, "r", "IMMA.16832.U8.S8", 16, 8, 32),   # Q4 GEMM operand types
    "s4_s32":   ("m16n8k64.row.col.s32.s4.s4.s32", 4, 2, 4, "r", "IMMA.16864.S4.S4", 16, 8, 64),
    "b1_s32":   ("m16n8k256.row.col.s32.b1.b1.s32.and.popc", 4, 2, 4, "r", "BMMA.168256.AND.POPC", 16, 8, 256),
}


def _mma_asm(op: str, dname: str) -> str:
    shape, na, nb, nc, cc, *_ = TC[op]
    d = ", ".join(f"%{i}" for i in range(nc))
    a = ", ".join(f"%{nc + i}" for i in range(na))
    b = ", ".join(f"%{nc + na + i}" for i in range(nb))
    outs = ", ".join(f'"+{cc}"({dname}[{i}])' for i in range(nc))
    ins = ", ".join([f'"r"(a[{i}])' for i in range(na)] + [f'"r"(b[{i}])' for i in range(nb)])
    return (f'asm volatile("mma.sync.aligned.{shape} {{{d}}}, {{{a}}}, {{{b}}}, {{{d}}};"'
            f" : {outs} : {ins});")


@dataclass
class TensorCore(Experiment):
    op: str = "f16_f32"
    chains: int = 1
    body: int = 64

    def __post_init__(self):
        kind = "lat" if self.chains == 1 else "tput"
        self.name = f"tc_{kind}_{self.op}"
        self.target_opcode = TC[self.op][5]
        self.description = f"mma.sync {TC[self.op][0]} ({self.target_opcode}), {self.chains} chain(s)"

    def source(self, v):
        nc, cc = TC[self.op][3], TC[self.op][4]
        ct = "float" if cc == "f" else "unsigned"
        decl = "\n  ".join(f"{ct} d{c}[{nc}] = {{{', '.join(['(' + ct + ')in[0]'] * nc)}}};" for c in range(self.chains))
        lines = []
        for _ in range(self.body):
            for c in range(self.chains):
                lines.append(_mma_asm(self.op, f"d{c}"))
        fold = " + ".join(f"(float)d{c}[0]" for c in range(self.chains))
        return f"""
extern "C" __global__ void k(float* out, long long* cyc, unsigned long long* ns, const float* in, int iters)
{{
  unsigned a[4] = {{0, 0, 0, 0}}, b[2] = {{0, 0}};
  if (in[1] == 12345.f) {{ a[0] = 1; b[0] = 1; }}   /* runtime-opaque zeros */
  {decl}
  __syncthreads();
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
    {chr(10).join("    " + l for l in lines)}
  }}
  long long t1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = {fold};
  if ((threadIdx.x & 31) == 0) {{ cyc[tid >> 5] = t1 - t0; ns[tid >> 5] = g1 - g0; }}
}}
"""

    def expected(self, v):
        return {self.target_opcode: self.body * self.chains}

    def variants(self, opts):
        warps = opts.get("warps") or ([1] if self.chains == 1 else [1, 2, 4, 8, 16, 32])
        it = max(1, int(opts.get("iters") or (200_000 if self.chains == 1 else 40_000)) // self.body)
        return [Variant(f"warps={w}", {"warps": w, "iters": it}) for w in warps]

    def prepare(self, dev, v):
        w = v.params["warps"]
        st = {"out": dev.alloc(32 * w * 4), "cyc": dev.alloc(w * 8), "ns": dev.alloc(w * 8), "in": dev.alloc(8)}
        dev.htod(st["in"], np.array([1.0, 0.0], np.float32))
        st["launch"] = dict(grid=1, block=32 * w, args=[ctypes.c_uint64(st[x]) for x in ("out", "cyc", "ns", "in")]
                            + [ctypes.c_int32(v.params["iters"])])
        return st

    def collect(self, dev, v, st):
        w = v.params["warps"]
        cyc = dev.dtoh(np.zeros(w, np.int64), st["cyc"])
        ns = dev.dtoh(np.zeros(w, np.uint64), st["ns"])
        cmax = int(cyc.max())
        n_mma = v.params["iters"] * self.body * self.chains          # per warp
        m, n, k = TC[self.op][6:9]
        flop = 2 * m * n * k * n_mma * w
        return {"cycles": cmax, "ns": int(ns.max()), "ops_per_thread": n_mma,
                "cycles_per_op": cmax / (v.params["iters"] * self.body),   # latency view (per chain step)
                "warp_ops_per_cycle": w * n_mma / cmax,                    # mma/cycle/SM
                "flop_per_cycle_sm": flop / cmax,
                "sm_mhz_inkernel": cmax / max(int(ns.max()), 1) * 1e3}

    def release(self, dev, st):
        for x in ("out", "cyc", "ns", "in"):
            dev.free(st[x])


@dataclass
class TCCorrect(Experiment):
    """One m16n8k16 f16->f32 and one m16n8k32 s8->s32 mma per warp with random A/B/C laid out per
    the PTX ISA fragment tables; compare with numpy. Validates our understanding of the layouts."""

    def __post_init__(self):
        self.name = "tc_correct"
        self.description = "fragment-layout correctness of mma.sync vs numpy (f16->f32 m16n8k16, s8->s32 m16n8k32)"

    def source(self, v):
        return r"""
#include <cuda_fp16.h>
extern "C" __global__ void k(const __half* A, const __half* B, const float* C, float* D,
                             const signed char* A8, const signed char* B8, const int* C8, int* D8)
{
  int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  // f16 m16n8k16: A row-major 16x16, B col-major 16x8 (B[k][n] stored at n*16 + k), C/D 16x8 row-major
  unsigned a[4], b[2]; float c[4];
  auto pk = [](__half lo, __half hi) { return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16); };
  a[0] = pk(A[g * 16 + 2 * t], A[g * 16 + 2 * t + 1]);
  a[1] = pk(A[(g + 8) * 16 + 2 * t], A[(g + 8) * 16 + 2 * t + 1]);
  a[2] = pk(A[g * 16 + 2 * t + 8], A[g * 16 + 2 * t + 9]);
  a[3] = pk(A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9]);
  b[0] = pk(B[g * 16 + 2 * t], B[g * 16 + 2 * t + 1]);
  b[1] = pk(B[g * 16 + 2 * t + 8], B[g * 16 + 2 * t + 9]);
  c[0] = C[g * 8 + 2 * t]; c[1] = C[g * 8 + 2 * t + 1]; c[2] = C[(g + 8) * 8 + 2 * t]; c[3] = C[(g + 8) * 8 + 2 * t + 1];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  D[g * 8 + 2 * t] = c[0]; D[g * 8 + 2 * t + 1] = c[1]; D[(g + 8) * 8 + 2 * t] = c[2]; D[(g + 8) * 8 + 2 * t + 1] = c[3];
  // s8 m16n8k32: A row-major 16x32, B col-major 32x8 (B[k][n] at n*32 + k)
  unsigned a8[4], b8[2]; int c8[4];
  auto p4 = [](const signed char* p) { return (unsigned)(unsigned char)p[0] | ((unsigned)(unsigned char)p[1] << 8) |
                                               ((unsigned)(unsigned char)p[2] << 16) | ((unsigned)(unsigned char)p[3] << 24); };
  a8[0] = p4(A8 + g * 32 + 4 * t);       a8[1] = p4(A8 + (g + 8) * 32 + 4 * t);
  a8[2] = p4(A8 + g * 32 + 4 * t + 16);  a8[3] = p4(A8 + (g + 8) * 32 + 4 * t + 16);
  b8[0] = p4(B8 + g * 32 + 4 * t);       b8[1] = p4(B8 + g * 32 + 4 * t + 16);
  c8[0] = C8[g * 8 + 2 * t]; c8[1] = C8[g * 8 + 2 * t + 1]; c8[2] = C8[(g + 8) * 8 + 2 * t]; c8[3] = C8[(g + 8) * 8 + 2 * t + 1];
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c8[0]), "+r"(c8[1]), "+r"(c8[2]), "+r"(c8[3]) : "r"(a8[0]), "r"(a8[1]), "r"(a8[2]), "r"(a8[3]), "r"(b8[0]), "r"(b8[1]));
  D8[g * 8 + 2 * t] = c8[0]; D8[g * 8 + 2 * t + 1] = c8[1]; D8[(g + 8) * 8 + 2 * t] = c8[2]; D8[(g + 8) * 8 + 2 * t + 1] = c8[3];
}
"""

    def expected(self, v):
        return {}

    def variants(self, opts):
        return [Variant("layouts", {"warps": 1, "iters": 1})]

    def prepare(self, dev, v):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((16, 16)).astype(np.float16)
        Bm = rng.standard_normal((16, 8)).astype(np.float16)          # logical K x N
        C = rng.standard_normal((16, 8)).astype(np.float32)
        A8 = rng.integers(-128, 128, (16, 32)).astype(np.int8)
        B8 = rng.integers(-128, 128, (32, 8)).astype(np.int8)
        C8 = rng.integers(-1000, 1000, (16, 8)).astype(np.int32)
        st = {"ref": A.astype(np.float32) @ Bm.astype(np.float32) + C,
              "ref8": A8.astype(np.int32) @ B8.astype(np.int32) + C8}
        bufs = {"A": A, "B": np.ascontiguousarray(Bm.T), "C": C, "A8": A8, "B8": np.ascontiguousarray(B8.T), "C8": C8}
        for kk, arr in bufs.items():
            st[kk] = dev.alloc(arr.nbytes)
            dev.htod(st[kk], arr)
        st["D"], st["D8"] = dev.alloc(16 * 8 * 4), dev.alloc(16 * 8 * 4)
        order = ("A", "B", "C", "D", "A8", "B8", "C8", "D8")
        st["launch"] = dict(grid=1, block=32, args=[ctypes.c_uint64(st[x]) for x in order])
        return st

    def collect(self, dev, v, st):
        D = dev.dtoh(np.zeros((16, 8), np.float32), st["D"])
        D8 = dev.dtoh(np.zeros((16, 8), np.int32), st["D8"])
        e16 = float(np.max(np.abs(D - st["ref"])))
        e8 = int(np.max(np.abs(D8 - st["ref8"])))
        return {"cycles": 1, "ops_per_thread": 1, "cycles_per_op": 1, "warp_ops_per_cycle": 0.0, "sm_mhz_inkernel": 0.0,
                "f16_max_abs_err": e16, "s8_max_abs_err": e8, "correct": e16 < 1e-2 and e8 == 0}

    def release(self, dev, st):
        for x in ("A", "B", "C", "D", "A8", "B8", "C8", "D8"):
            dev.free(st[x])


def registry():
    exps = [TCCorrect()]
    for op in TC:
        exps += [TensorCore(op=op, chains=1), TensorCore(op=op, chains=4, body=16)]
    return {e.name: e for e in exps}
