"""Producer -> consumer forwarding latency per opcode PAIR (the scheduler's dependence table).

One thread chain alternates producer P and consumer C (x = P(x); x = C(x); ...). Every P has its
stall patched to s and every C gets a safe 8, so only the P->C distance varies. The hardware has no
data-hazard interlock: if s is below the P->C forwarding latency, C reads the stale register and
P's effect is lost, so the final value differs from the unpatched ptxas build (the reference). P
must not be an identity op, or its own stale reads would be invisible. Same-op pairs patch all.

Found because ptxas writes 4 between same-pipe dependents (IMAD->IMAD, SHF->SHF) but 5 across
pipes (IMAD->SHF), and a scheduler using 4 everywhere produced wrong answers.

Operands: %1 = 1, %2 = 3, %3 = 0x5A5A5A5A, %4 = lane, %5 = 1.0f, %6 = -1.0f, %7 = -1e30f, %8 = 1.
"""
from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

OPS = {  # name -> (PTX, SASS opcode)
    "imad3": ("mad.lo.s32 %0, %0, %2, %1;", "IMAD"),         # x*3+1
    "imad1": ("mad.lo.s32 %0, %0, %1, %1;", "IMAD"),         # x+1, gentle on float bits
    "iadd": ("add.s32 %0, %0, %1;", "IADD3"),
    "rot": ("shf.l.wrap.b32 %0, %0, %0, %8;", "SHF.L.W.U32.HI"),
    "xork": ("xor.b32 %0, %0, %3;", "LOP3.LUT"),
    "xor1": ("xor.b32 %0, %0, %1;", "LOP3.LUT"),
    "shfl": ("shfl.sync.idx.b32 %0, %0, %4, 0x1f, 0xffffffff;", "SHFL.IDX"),
    "ffma": ("fma.rn.f32 %0, %0, %5, %5;", "FFMA"),
    "fadd": ("add.f32 %0, %0, %5;", "FADD"),
    "fneg": ("mul.f32 %0, %0, %6;", "FMUL"),
    "fmax": ("max.f32 %0, %0, %7;", "FMNMX"),
}
INT_P, INT_C = ("imad3", "iadd", "rot", "xork"), ("imad3", "iadd", "rot", "xork", "shfl")
FLT_P, FLT_C = ("ffma", "fadd", "fneg"), ("ffma", "fadd", "fneg", "fmax", "shfl")
GENTLE = ("imad1", "iadd", "xor1")
BODY = 32
ITERS = 2000


def pairs():
    out = [(p, c) for p in INT_P for c in INT_C]
    out += [(p, c) for p in FLT_P for c in FLT_C]
    out += [(p, c) for p in GENTLE for c in ("ffma", "fadd", "fneg", "fmax")]
    out += [(p, c) for p in FLT_P for c in GENTLE]
    return out


@dataclass
class PairLatency(Experiment):
    stalls: tuple = (1, 2, 3, 4, 5, 6, 7)

    def __post_init__(self):
        self.name = "pair_latency"
        self.description = "forwarding latency per producer->consumer opcode pair (patched chains)"

    def source(self, v: Variant) -> str:
        cons = ('"+r"(x) : "r"(one), "r"(three), "r"(key), "r"(lane), "r"(fone), "r"(fm1), "r"(fninf), '
                '"r"(rot)')
        body = "\n      ".join(f'asm volatile("{OPS[n][0]}" : {cons});'
                               for _ in range(BODY) for n in (v.params["p"], v.params["c"]))
        return f"""
extern "C" __global__ void k(unsigned* out, const unsigned* in, int iters)
{{
  unsigned x = in[0], one = in[1], three = in[2], key = in[3], fone = in[4], fm1 = in[5], fninf = in[6],
           rot = in[7], lane = threadIdx.x;
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
      {body}
  }}
  out[threadIdx.x] = x;
}}
"""

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for p, c in pairs():
            for s in (None,) + self.stalls:
                out.append(Variant(f"{p}>{c}/" + ("ptxas" if s is None else f"s{s}"),
                                   {"p": p, "c": c, "stall": s, "warps": 1, "iters": ITERS}))
        return out

    def build_key(self, v):
        return v.params["stall"]

    def expected(self, v: Variant) -> dict[str, int]:
        return {}

    @staticmethod
    def chain(ins, v):
        """Chain instructions of the timed loop (those writing the chain register x), in order."""
        import re
        from collections import Counter
        from .. import sass
        body = sass.loop_body(ins)

        def dest(i):
            regs = re.findall(r"R\d+", i.text)
            return regs[0] if regs else None
        cnt = Counter(dest(i) for i in body if i.opcode not in ("BRA", "ISETP.GE.AND"))
        x = cnt.most_common(1)[0][0]
        return [i for i in body if dest(i) == x and i.opcode not in ("BRA",)]

    @staticmethod
    def producers(ch, v):
        """The most common P-instance -> C-instance opcode edge (chain alternates P, C, P, C ...).
        Returns (edge, producer instructions) where edge = (P opcode, C opcode)."""
        from collections import Counter
        edges = Counter((a.opcode, b.opcode) for a, b in zip(ch[0::2], ch[1::2]))
        if not edges:
            return None, []
        e = edges.most_common(1)[0][0]
        return e, [a for a, b in zip(ch[0::2], ch[1::2]) if (a.opcode, b.opcode) == e]

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        from .. import patch, sass, toolchain
        s = v.params["stall"]
        if s is None:
            return cubin
        ch = self.chain(sass.parse(toolchain.disassemble(cubin)), v)
        prod = {i.offset for i in self.producers(ch, v)[1]}
        edits = {i.offset: {"stall": s if i.offset in prod else 8, "yield": 1} for i in ch}
        return patch.set_control(cubin, "k", edits)

    def prepare(self, dev, v: Variant) -> dict:
        st = {"out": dev.alloc(32 * 4), "in": dev.alloc(32)}
        f = lambda z: struct.unpack("<I", struct.pack("<f", z))[0]  # noqa: E731
        x0 = f(1.0) if v.params["p"] in FLT_P or v.params["c"] in FLT_C[:4] else 7
        dev.htod(st["in"], np.array([x0, 1, 3, 0x5A5A5A5A, f(1.0), f(-1.0), f(-1e30), 1], np.uint32))
        st["launch"] = dict(grid=1, block=32, args=[ctypes.c_uint64(st["out"]), ctypes.c_uint64(st["in"]),
                                                    ctypes.c_int32(v.params["iters"])])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        y = dev.dtoh(np.zeros(32, np.uint32), st["out"])
        return {"cycles": 1, "ns": 1, "ops_per_thread": 1, "cycles_per_op": 0.0, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "value": int(y[0]), "lanes_equal": bool((y == y[0]).all())}

    def release(self, dev, st: dict):
        dev.free(st["out"])
        dev.free(st["in"])


MUFU = {"rsq": ("rsqrt.approx.ftz.f32", "MUFU.RSQ"), "ex2": ("ex2.approx.ftz.f32", "MUFU.EX2"),
        "lg2": ("lg2.approx.ftz.f32", "MUFU.LG2"), "sin": ("sin.approx.ftz.f32", "MUFU.SIN"),
        "rcp": ("rcp.approx.ftz.f32", "MUFU.RCP")}
COUNTERS = {"ffma": ("fma.rn.f32 %0, %0, %2, %2;", "FFMA"), "fadd": ("add.f32 %0, %0, %2;", "FADD"),
            "iadd": ("add.s32 %0, %0, %3;", "IADD3"), "imad1": ("mad.lo.s32 %0, %0, %3, %3;", "IMAD")}


@dataclass
class MufuLatency(PairLatency):
    """Fixed-latency producer -> MUFU consumer. Iterating a MUFU contracts to a fixed point, so a
    chain hides stale reads; instead x counts up (the producer), y = MUFU(x) is a side value, and
    acc += y. A stale read sums g(x - 1) instead of g(x)."""

    def __post_init__(self):
        self.name = "mufu_latency"
        self.description = "fixed-latency producer -> MUFU consumer forwarding latency"

    def source(self, v: Variant) -> str:
        cp = COUNTERS[v.params["p"]][0]
        mp = MUFU[v.params["c"]][0]
        lines = []
        for _ in range(BODY):
            lines.append(f'asm volatile("{cp}" : "+r"(x) : "r"(one), "r"(fone), "r"(ione));')
            lines.append(f'asm volatile("{{ .reg .f32 t; {mp} t, %1; add.f32 %0, %0, t; }}" : "+f"(acc) : "r"(x));')
        body = "\n      ".join(lines)
        return f"""
extern "C" __global__ void k(unsigned* out, const unsigned* in, int iters)
{{
  unsigned x = in[0], one = in[1], fone = in[4], ione = in[1];
  float acc = 0.f;
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {{
      {body}
  }}
  out[threadIdx.x] = __float_as_uint(acc) ^ x;
}}
"""

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for p in COUNTERS:
            for c in MUFU:
                for s in (None,) + self.stalls + (8, 9):
                    out.append(Variant(f"{p}>{c}/" + ("ptxas" if s is None else f"s{s}"),
                                       {"p": p, "c": c, "stall": s, "warps": 1, "iters": 2}))
        return out

    @staticmethod
    def producers(ch, v):
        return None, ch

    def chain(self, ins, v):
        from .. import sass
        body = sass.loop_body(ins)
        op = COUNTERS[v.params["p"]][1]
        out = []
        for i, n in zip(body, body[1:]):
            if i.opcode == op and n.opcode.startswith("MUFU"):
                out.append(i)
        return out

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        from .. import patch, sass, toolchain
        s = v.params["stall"]
        if s is None:
            return cubin
        ch = self.chain(sass.parse(toolchain.disassemble(cubin)), v)
        return patch.set_control(cubin, "k", {i.offset: {"stall": s, "yield": 1} for i in ch})

    def prepare(self, dev, v: Variant) -> dict:
        st = super().prepare(dev, v)
        f = lambda z: struct.unpack("<I", struct.pack("<f", z))[0]  # noqa: E731
        x0 = f(1.5) if v.params["p"] in ("ffma", "fadd") else f(0.75)
        dev.htod(st["in"], np.array([x0, 1, 3, 0, f(1.0), 0, 0, 1], np.uint32))
        return st


def registry():
    e, m = PairLatency(), MufuLatency()
    return {e.name: e, m.name: m}
