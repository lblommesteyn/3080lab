"""Probes that go below what ptxas emits.

stall_<op>: patch the stall count of every fixed-latency op in a dependent
chain and check the arithmetic. x = x*1 + 1 (or x + 1) counts its own
executions, so a stall shorter than the true latency shows up as lost
increments: the next op read the register before the previous write landed.

mix_ffma_<partner>: 32 warps, 8 independent chains per thread, f of them FFMA
and 8-f a partner op. Tests GA102's split FP32 datapath (16 dedicated FP32
lanes + 16 shared FP32/INT32 lanes per SM partition): which partner ops share
a pipe with FFMA shows up in how throughput bends with the mix.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .alu import OPS, Chain
from .base import Experiment, Variant


@dataclass
class StallProbe(Chain):
    stalls: tuple = (0, 1, 2, 3, 4, 5, 6)

    def __post_init__(self):
        super().__post_init__()
        self.name = f"stall_{self.op}"
        self.description = f"patch stall field of dependent {self.target_opcode}; lost increments = stall too short"
        one = 1.0 if OPS[self.op][0] == "float" else 1
        self.inputs = (0 * one, one, one)  # x0=0, a=1, b=1: x*1+1 or x+1 counts executions

    def variants(self, opts: dict) -> list[Variant]:
        total = int(opts.get("iters") or 2_000_000)  # < 2^24 so float counts stay exact
        iters = max(1, total // self.body)
        vs = [Variant("original", {"warps": 1, "iters": iters, "stall": None})]
        return vs + [Variant(f"stall={s}", {"warps": 1, "iters": iters, "stall": s}) for s in self.stalls]

    def build_key(self, v):
        return v.params["stall"]

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        from .. import patch, sass, toolchain
        s = v.params["stall"]
        if s is None:
            return cubin
        body = sass.loop_body(sass.parse(toolchain.disassemble(cubin)))
        tgt = [i for i in body if i.opcode == self.target_opcode]
        if any(i.wbar != 7 for i in tgt):
            raise SystemExit(f"{self.target_opcode} is scoreboarded here; stall patching does not apply")
        stalls = [i.stall for i in tgt]
        common = max(set(stalls), key=stalls.count)
        return patch.set_control(cubin, self.kernel_name,
                                 {i.offset: {"stall": s} for i in tgt if i.stall == common})

    def collect(self, dev, v: Variant, st: dict) -> dict:
        r = super().collect(dev, v, st)
        is_float = OPS[self.op][0] == "float"
        x = dev.dtoh(np.zeros(1, np.float32 if is_float else np.int32), st["out"])[0]
        n = r["ops_per_thread"]
        r["final_value"] = float(x)
        r["lost_fraction"] = 1.0 - float(x) / n
        r["correct"] = bool(float(x) == n)
        return r


@dataclass
class Mix(Experiment):
    partner: str = "shl"
    chains: int = 8
    body: int = 32
    warps: int = 32

    def __post_init__(self):
        self.name = f"mix_ffma_{self.partner}"
        self.target_opcode = "FFMA"
        self.partner_opcode = OPS[self.partner][3]
        self.description = f"throughput of FFMA:{self.partner_opcode} mixes, {self.warps} warps x {self.chains} chains"

    def _kinds(self, f: int) -> list[str]:
        # Spread the f FFMA chains evenly among the slots.
        return ["ffma" if (i * f) // self.chains != ((i + 1) * f) // self.chains else self.partner
                for i in range(self.chains)]

    def source(self, v: Variant) -> str:
        kinds = self._kinds(v.params["ffma_chains"])
        decl, lines = [], []
        for c, op in enumerate(kinds):
            ctype = OPS[op][0]
            src = "fin" if ctype == "float" else "iin"
            decl.append(f"{ctype} x{c} = ({ctype}){src}[0] + ({ctype})(threadIdx.x * {c});")
        for _ in range(self.body):
            for c, op in enumerate(kinds):
                ctype, cons, tmpl = OPS[op][:3]
                av, bv = ("fa", "fb") if ctype == "float" else ("ia", "ib")
                ins = [f'"{cons}"({av})'] + ([f'"{cons}"({bv})'] if "{b}" in tmpl else [])
                pt = tmpl.format(d="%0", a="%1", b="%2")
                lines.append(f'asm volatile("{pt}" : "+{cons}"(x{c}) : {", ".join(ins)});')
        fold = " + ".join(f"(float)x{c}" for c in range(self.chains))
        nl = "\n      "
        return f"""
extern "C" __global__ void k(float* out, long long* cyc, unsigned long long* ns,
                             const float* fin, const int* iin, int iters)
{{
  {nl.join(decl)}
  float fa = fin[1], fb = fin[2];
  int ia = iin[1], ib = iin[2];
  unsigned ua = (unsigned)iin[3], ub = (unsigned)iin[3];
  (void)fb; (void)ib; (void)ua; (void)ub;
  __syncthreads();
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
      {nl.join(lines)}
  }}
  long long t1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = {fold};
  if ((threadIdx.x & 31) == 0) {{ cyc[tid >> 5] = t1 - t0; ns[tid >> 5] = g1 - g0; }}
}}
"""

    def expected(self, v: Variant) -> dict[str, int]:
        f = v.params["ffma_chains"]
        return {"FFMA": f * self.body, self.partner_opcode: (self.chains - f) * self.body}

    def variants(self, opts: dict) -> list[Variant]:
        total = int(opts.get("iters") or 500_000)
        iters = max(1, total // self.body)
        return [Variant(f"ffma={f}/{self.chains}", {"ffma_chains": f, "warps": self.warps, "iters": iters})
                for f in range(self.chains + 1)]

    def prepare(self, dev, v: Variant) -> dict:
        w = v.params["warps"]
        st = {"out": dev.alloc(32 * w * 4), "cyc": dev.alloc(w * 8), "ns": dev.alloc(w * 8),
              "fin": dev.alloc(12), "iin": dev.alloc(16)}
        dev.htod(st["fin"], np.array([1.0, 1.0, 0.0], np.float32))
        # hfma2 partner reads its operands through the int path as packed half2 1.0
        dev.htod(st["iin"], np.array([1, 0, 0, 0x3C003C00], np.int32))
        st["launch"] = dict(grid=1, block=32 * w,
                            args=[ctypes.c_uint64(st[k]) for k in ("out", "cyc", "ns", "fin", "iin")]
                            + [ctypes.c_int32(v.params["iters"])])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        w = v.params["warps"]
        cyc = dev.dtoh(np.zeros(w, np.int64), st["cyc"])
        ns = dev.dtoh(np.zeros(w, np.uint64), st["ns"])
        ops = v.params["iters"] * self.body * self.chains
        cmax = int(cyc.max())
        return {"cycles": cmax, "ns": int(ns.max()), "ops_per_thread": ops,
                "cycles_per_op": cmax / (v.params["iters"] * self.body),
                "warp_ops_per_cycle": w * ops / cmax,
                "sm_mhz_inkernel": cmax / max(int(ns.max()), 1) * 1e3}

    def release(self, dev, st: dict):
        for k in ("out", "cyc", "ns", "fin", "iin"):
            dev.free(st[k])


def probes() -> dict[str, Experiment]:
    exps = [StallProbe(op=o) for o in ("ffma", "fadd", "imad")]
    exps += [Mix(partner=p) for p in ("shl", "imad", "hfma2")]
    return {e.name: e for e in exps}
