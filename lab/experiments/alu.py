"""Phase 1 instruction-table experiments: dependent chains (latency) and
interleaved independent chains (throughput) for one instruction at a time.

Each op is pinned with inline PTX so the frontend cannot fold it, and the timed
loop carries `#pragma unroll 1` around an explicitly unrolled body so the SASS
loop body is exactly what we generated. The runner verifies that claim against
the disassembly before reporting a number.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

# name -> (C type, asm constraint, PTX template, SASS opcode, init x, a, b)
# {d} is the accumulator, {a}/{b} are loop-invariant registers.
OPS: dict[str, tuple] = {
    "ffma":  ("float", "f", "fma.rn.f32 {d}, {d}, {a}, {b};", "FFMA", 1.0, 1.0, 0.0),
    "fadd":  ("float", "f", "add.f32 {d}, {d}, {a};", "FADD", 1.0, 0.0, 0.0),
    "fmul":  ("float", "f", "mul.f32 {d}, {d}, {a};", "FMUL", 1.0, 1.0, 0.0),
    "iadd":  ("int", "r", "add.s32 {d}, {d}, {a};", "IADD3", 1, 1, 0),
    "imad":  ("int", "r", "mad.lo.s32 {d}, {d}, {a}, {b};", "IMAD", 1, 1, 0),
    "lop3":  ("int", "r", "xor.b32 {d}, {d}, {a};", "LOP3.LUT", 1, 3, 0),
    "shl":   ("int", "r", "shl.b32 {d}, {d}, {a};", "SHF.L.U32", 1, 0, 0),
    "hfma2": ("unsigned", "r", "fma.rn.f16x2 {d}, {d}, {a}, {b};", "HFMA2", 0x3C003C00, 0x3C003C00, 0),
    "dfma":  ("double", "d", "fma.rn.f64 {d}, {d}, {a}, {b};", "DFMA", 1.0, 1.0, 0.0),
    "rsqrt": ("float", "f", "rsqrt.approx.ftz.f32 {d}, {d};", "MUFU.RSQ", 1.0, 0.0, 0.0),
    "ex2":   ("float", "f", "ex2.approx.ftz.f32 {d}, {d};", "MUFU.EX2", 0.0, 0.0, 0.0),
    "sin":   ("float", "f", "sin.approx.ftz.f32 {d}, {d};", "MUFU.SIN", 0.5, 0.0, 0.0),
    "shfl":  ("unsigned", "r", "shfl.sync.idx.b32 {d}, {d}, {a}, 0x1f, 0xffffffff;", "SHFL.IDX", 1, 0, 0),
}

# ptxas folds these chains at -O1+ (x+a+a -> one IADD3, x^a^a -> x), so they
# default to -O0, which keeps one SASS op per PTX op. -O0 also changes control
# codes, which is itself something to measure (compare with --ptxas=-O3).
DEFAULT_PTXAS = {"iadd": ["-O0"], "lop3": ["-O0"]}
# Loop-invariant operand overrides. A uniform shfl source lane makes the chain
# idempotent and ptxas deletes it; a per-lane rotation cannot be folded.
SLOW_PIPE_ITERS = {"dfma": 50_000, "rsqrt": 250_000, "ex2": 250_000, "sin": 250_000, "shfl": 250_000}
A_EXPR = {"shfl": "(unsigned)((threadIdx.x + 1) & 31)"}


def reference(op: str, x0, a, b, n: int):
    """Host-side expected value for integer chains (exact), else None."""
    m = 0xFFFFFFFF
    if op == "iadd":
        return (x0 + a * n) & m
    if op == "lop3":
        return x0 ^ (a if n % 2 else 0)
    if op == "imad":
        return (x0 * pow(a, n, 1 << 32) + b * sum(pow(a, i, 1 << 32) for i in range(min(n, 4)))) & m if a == 1 and b == 0 else None
    return None


def _ptx_operands(tmpl: str) -> list[str]:
    return [k for k in ("a", "b") if "{" + k + "}" in tmpl]


@dataclass
class Chain(Experiment):
    op: str = "ffma"
    chains: int = 1          # independent accumulators per thread (ILP)
    body: int = 256          # ops per chain per loop iteration
    lane_seed: int = 0       # 1: every chain starts at in[0] + lane (needed to see shfl rotation)

    def __post_init__(self):
        kind = "dependent" if self.chains == 1 else "independent"
        self.name = f"{kind}_{self.op}"
        self.target_opcode = OPS[self.op][3]
        if not self.ptxas_flags:
            self.ptxas_flags = list(DEFAULT_PTXAS.get(self.op, []))
        self.description = (f"{self.chains} chain(s) of {self.op} ({self.target_opcode}), "
                            f"{'latency' if self.chains == 1 else 'throughput'} probe")

    # ---- kernel generation -------------------------------------------------
    def source(self, v: Variant) -> str:
        ctype, cons, tmpl, *_ = OPS[self.op]
        ops = _ptx_operands(tmpl)
        k = self.chains
        lines = []
        # Interleave chains so independent ops sit adjacent: c0 c1 .. c(k-1) c0 ..
        for _ in range(self.body):
            for c in range(k):
                pt = tmpl.format(d="%0", a="%1", b="%2")
                ins = ", ".join(f'"{cons}"({o})' for o in ops)
                lines.append(f'asm volatile("{pt}" : "+{cons}"(x{c}){" : " + ins if ins else ""});')
        body = "\n      ".join(lines)
        decl = "\n  ".join(f"{ctype} x{c} = in[0] + ({ctype})(threadIdx.x * {int(c > 0 or self.lane_seed)});" for c in range(k))
        fold = " + ".join(f"x{c}" for c in range(k))
        return f"""
extern "C" __global__ void k({ctype}* out, long long* cyc, unsigned long long* ns,
                             const {ctype}* in, int iters)
{{
  {decl}
  {ctype} a = {A_EXPR.get(self.op, "in[1]")}, b = in[2];
  (void)b;
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
  if ((threadIdx.x & 31) == 0) {{
    cyc[tid >> 5] = t1 - t0;
    ns[tid >> 5] = g1 - g0;
  }}
}}
"""

    def expected_body_ops(self) -> int:
        return self.body * self.chains

    # ---- execution ---------------------------------------------------------
    def variants(self, opts: dict) -> list[Variant]:
        warps = opts.get("warps") or ([1] if self.chains == 1 else [1, 4, 8, 16, 32])
        # Independent runs at 32 warps retire 32x the ops; keep slow pipes under the ~2 s Windows TDR.
        default = 10_000_000 if self.chains == 1 else SLOW_PIPE_ITERS.get(self.op, 1_000_000)
        total = int(opts.get("iters") or default)
        iters = max(1, total // self.body)
        return [Variant(f"warps={w}", {"warps": w, "iters": iters}) for w in warps]

    def prepare(self, dev, v: Variant) -> dict:
        ctype, _, _, _, x0, a, b = OPS[self.op]
        if getattr(self, "inputs", None):
            x0, a, b = self.inputs
        np_t = {"float": np.float32, "int": np.int32, "unsigned": np.uint32, "double": np.float64}[ctype]
        w = v.params["warps"]
        st = {
            "out": dev.alloc(32 * w * np.dtype(np_t).itemsize),
            "cyc": dev.alloc(w * 8),
            "ns": dev.alloc(w * 8),
            "in": dev.alloc(3 * np.dtype(np_t).itemsize),
        }
        dev.htod(st["in"], np.array([x0, a, b], dtype=np_t))
        st["launch"] = dict(grid=1, block=32 * w,
                            args=[ctypes.c_uint64(st["out"]), ctypes.c_uint64(st["cyc"]),
                                  ctypes.c_uint64(st["ns"]), ctypes.c_uint64(st["in"]),
                                  ctypes.c_int32(v.params["iters"])])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        w = v.params["warps"]
        cyc = dev.dtoh(np.zeros(w, np.int64), st["cyc"])
        ns = dev.dtoh(np.zeros(w, np.uint64), st["ns"])
        ops_per_thread = v.params["iters"] * self.body * self.chains
        cmax = int(cyc.max())
        ctype, _, _, _, x0, a, b = OPS[self.op]
        ref = reference(self.op, x0, a, b, ops_per_thread) if self.chains == 1 else None
        correct = None
        if ref is not None:
            out = dev.dtoh(np.zeros(32 * w, np.uint32), st["out"])
            correct = bool(out[0] == ref)
        return {
            "correct": correct,
            "cycles": cmax,
            "cycles_per_warp": cyc.tolist(),
            "ns": int(ns.max()),
            "ops_per_thread": ops_per_thread,
            # latency view: cycles per op along one chain
            "cycles_per_op": cmax / (v.params["iters"] * self.body),
            # throughput view: warp-instructions retired per cycle on this SM
            "warp_ops_per_cycle": w * ops_per_thread / cmax,
            "sm_mhz_inkernel": cmax / max(int(ns.max()), 1) * 1e3,
        }

    def release(self, dev, st: dict):
        for key in ("out", "cyc", "ns", "in"):
            dev.free(st[key])


def registry() -> dict[str, Experiment]:
    from .probes import probes
    out: dict[str, Experiment] = dict(probes())
    for op in OPS:
        dep = Chain(op=op, chains=1)
        ind = Chain(op=op, chains=8, body=64)
        out[dep.name], out[ind.name] = dep, ind
    return out
