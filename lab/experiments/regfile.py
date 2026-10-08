"""Register-file probes: bank parity and the operand reuse cache.

Huerta et al. (MICRO'25) describe 2 register banks per sub-core with one read
port each, plus a small reuse cache. A 3-source FFMA whose operands all sit in
one bank should then need extra read cycles unless the reuse cache supplies them.

We take the independent-FFMA kernel (8 chains x 64, 512 FFMA per loop) and
rewrite every FFMA in the timed loop to
    FFMA S_i, A, S_i, C        (S_i cycles through 8 scratch registers)
with the parity of S, A, C chosen per variant, then set or clear the reuse
flags. Scratch registers R32-R63 lie above everything ptxas allocated (the
kernel uses ~25), and the kernel's register count is raised to 64 so they are
legally allocated. Values are garbage but FFMA timing is value-independent,
and nothing live is overwritten. Timing only; results are not checked.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .alu import Chain
from .base import Variant

SCRATCH_EVEN = [48, 50, 52, 54, 56, 58, 60, 62]
SCRATCH_ODD = [49, 51, 53, 55, 57, 59, 61, 63]


@dataclass
class BankProbe(Chain):
    def __post_init__(self):
        self.op, self.chains, self.body = "ffma", 8, 64
        super().__post_init__()
        self.name = "regbank_ffma"
        self.description = "FFMA S,A,S,C with chosen register-bank parity and reuse flags"

    def variants(self, opts: dict) -> list[Variant]:
        warps = opts.get("warps") or [4, 32]
        iters = max(1, int(opts.get("iters") or 1_000_000) // self.body)
        out = [Variant(f"w{w}/orig", {"warps": w, "iters": iters, "pattern": None}) for w in warps]
        for w in warps:
            for sp, ap, cp in itertools.product("EO", repeat=3):
                for reuse in ("noreuse", "reuseAC"):
                    out.append(Variant(f"w{w}/S{sp}A{ap}C{cp}/{reuse}",
                                       {"warps": w, "iters": iters, "pattern": (sp, ap, cp, reuse)}))
        return out

    def build_key(self, v):
        return v.params["pattern"]

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        from .. import patch, sass, toolchain
        pat = v.params["pattern"]
        if pat is None:
            return cubin
        sp, ap, cp, reuse = pat
        S = SCRATCH_EVEN if sp == "E" else SCRATCH_ODD
        A = 32 if ap == "E" else 33
        C = 34 if cp == "E" else 35
        body = sass.loop_body(sass.parse(toolchain.disassemble(cubin)))
        c = patch.set_regcount(cubin, self.kernel_name, 64)
        for n, ins in enumerate(i for i in body if i.opcode == "FFMA"):
            s = S[n % 8]
            c = patch.set_regs(c, self.kernel_name, ins.offset, rd=s, ra=A, rb=s, rc=C)
            c = patch.set_control(c, self.kernel_name, {ins.offset: {"reuse": 0b101 if reuse == "reuseAC" else 0}})
        # sanity: the rewrite must disassemble to exactly what we intended
        got = [i for i in sass.loop_body(sass.parse(toolchain.disassemble(c))) if i.opcode == "FFMA"]
        want = f"R{A}" + (".reuse" if reuse == "reuseAC" else "")
        if not all(f", {want}, " in g.text for g in got):
            raise SystemExit(f"register rewrite failed: {got[0].text}")
        return c

    def collect(self, dev, v: Variant, st: dict) -> dict:
        r = super().collect(dev, v, st)
        r["correct"] = None  # garbage values by design
        # per partition: warp-instr issued per cycle on each SM sub-core
        r["issue_per_cycle_per_partition"] = r["warp_ops_per_cycle"] / min(4, v.params["warps"])
        return r


def registry():
    e = BankProbe()
    return {e.name: e}
