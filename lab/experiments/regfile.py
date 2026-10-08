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

# With regcount N only R0..R(N-3) are usable: R62/R63 fault at N=64 (diag_r62, diag_r63).
SCRATCH_EVEN = [36, 38, 40, 42, 44, 46, 48, 50]
SCRATCH_ODD = [37, 39, 41, 43, 45, 47, 49, 51]


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
        out += [Variant(f"w{w}/diag_{d}", {"warps": w, "iters": iters, "pattern": d})
                for w in warps for d in ("regcount_only", "reuse_only", "one_ffma", "rd_only", "ra_only",
                                         "rc_only", "low_regs", "rd_ra", "rd_rc", "ra_rc", "all3",
                                         "all3_reuse0", "all3_reuse5", "r56", "r60", "r61", "r62", "r63")]
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
        if pat == "regcount_only":
            return patch.set_regcount(cubin, self.kernel_name, 64)
        if pat == "reuse_only":
            body = sass.loop_body(sass.parse(toolchain.disassemble(cubin)))
            return patch.set_control(cubin, self.kernel_name,
                                     {i.offset: {"reuse": 0} for i in body if i.opcode == "FFMA"})
        if isinstance(pat, str):  # bisection diagnostics
            body = sass.loop_body(sass.parse(toolchain.disassemble(cubin)))
            ff = [i for i in body if i.opcode == "FFMA"]
            c = patch.set_regcount(cubin, self.kernel_name, 64)
            if pat == "one_ffma":
                return patch.set_regs(c, self.kernel_name, ff[3].offset, rd=48, ra=32, rb=48, rc=34)
            for n, i in enumerate(ff):
                kw = {} if pat.startswith("r") and pat[1:].isdigit() else {"rd_only": dict(rd=48 + n % 8, rb=48 + n % 8), "ra_only": dict(ra=32),
                      "rc_only": dict(rc=34), "low_regs": dict(ra=ff[0].lo >> 24 & 0xFF),
                      "rd_ra": dict(rd=48 + n % 8, rb=48 + n % 8, ra=32),
                      "rd_rc": dict(rd=48 + n % 8, rb=48 + n % 8, rc=34),
                      "ra_rc": dict(ra=32, rc=34),
                      "all3": dict(rd=48 + n % 8, rb=48 + n % 8, ra=32, rc=34),
                      "all3_reuse0": dict(rd=48 + n % 8, rb=48 + n % 8, ra=32, rc=34),
                      "all3_reuse5": dict(rd=48 + n % 8, rb=48 + n % 8, ra=32, rc=34)}[pat]
                if pat.startswith("r") and pat[1:].isdigit():  # write one high register
                    kw = dict(rd=int(pat[1:]), rb=int(pat[1:])) if n == 0 else {}
                c = patch.set_regs(c, self.kernel_name, i.offset, **kw)
                if pat in ("all3_reuse0", "all3_reuse5"):
                    c = patch.set_control(c, self.kernel_name, {i.offset: {"reuse": 0 if pat == "all3_reuse0" else 5}})
            return c
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
