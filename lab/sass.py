"""Parse `nvdisasm -hex` output into instructions with raw bytes and Ampere control bits.

Volta/Turing/Ampere instructions are 128 bits. The top bits [105:125] carry the
scheduling control word (Jia et al., "Dissecting the NVIDIA Volta GPU
Architecture via Microbenchmarking", 2018):

    [105:108] stall cycles   [109] yield   [110:112] write barrier
    [113:115] read barrier   [116:121] wait-barrier mask   [122:125] reuse cache

Barrier index 7 means "none".
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_LINE1 = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*/\*\s*(0x[0-9a-f]{16})\s*\*/")
_LINE2 = re.compile(r"^\s*/\*\s*(0x[0-9a-f]{16})\s*\*/\s*$")
_LABEL = re.compile(r"^(\.L_x_\d+):")


@dataclass
class Instr:
    offset: int
    text: str
    lo: int
    hi: int
    label: str | None = None

    @property
    def raw(self) -> str:
        return f"{self.hi:016x}{self.lo:016x}"

    @property
    def ctrl(self) -> int:
        return (self.hi >> 41) & 0x1FFFFF

    @property
    def stall(self) -> int:
        return self.ctrl & 0xF

    @property
    def yield_(self) -> int:
        return (self.ctrl >> 4) & 1

    @property
    def wbar(self) -> int:
        return (self.ctrl >> 5) & 7

    @property
    def rbar(self) -> int:
        return (self.ctrl >> 8) & 7

    @property
    def wait(self) -> int:
        return (self.ctrl >> 11) & 0x3F

    @property
    def reuse(self) -> int:
        return (self.ctrl >> 17) & 0xF

    @property
    def opcode(self) -> str:
        t = re.sub(r"^@!?U?P[T0-9]+\s+", "", self.text.strip())
        return t.split()[0] if t else ""

    def ctrl_str(self) -> str:
        b = lambda v: "-" if v == 7 else str(v)  # noqa: E731
        w = "".join(str(i) for i in range(6) if self.wait >> i & 1) or "-"
        return f"S{self.stall:02d} {'Y' if self.yield_ else '-'} W{b(self.wbar)} R{b(self.rbar)} wait:{w:<6} reuse:{self.reuse:x}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(raw=self.raw, opcode=self.opcode, stall=self.stall, yield_=self.yield_,
                 wbar=self.wbar, rbar=self.rbar, wait=self.wait, reuse=self.reuse)
        d.pop("lo"), d.pop("hi")
        return d


def parse(text: str) -> list[Instr]:
    out: list[Instr] = []
    pending: tuple[int, str, int] | None = None
    label = None
    for line in text.splitlines():
        if m := _LABEL.match(line.strip()):
            label = m.group(1)
            continue
        if m := _LINE1.search(line):
            pending = (int(m.group(1), 16), m.group(2).strip().rstrip(";").strip(), int(m.group(3), 16))
            continue
        if pending and (m := _LINE2.match(line)):
            off, txt, lo = pending
            out.append(Instr(off, txt, lo, int(m.group(1), 16), label))
            pending, label = None, None
    return out


def loops(instrs: list[Instr]) -> list[list[Instr]]:
    """Every backward-branch region [label .. BRA label]."""
    out = []
    for i, ins in enumerate(instrs):
        if ins.opcode == "BRA" and (m := re.search(r"\(\.L_x_\d+\)", ins.text)):
            tgt = m.group(0)[1:-1]
            starts = [j for j, x in enumerate(instrs[:i]) if x.label == tgt]
            if starts:
                out.append(instrs[starts[0]: i + 1])
    return out


def loop_body(instrs: list[Instr]) -> list[Instr]:
    """The timed loop: by construction our kernels' largest loop (explicitly
    unrolled body under `#pragma unroll 1`); warm-up loops are short."""
    ls = loops(instrs)
    return max(ls, key=len) if ls else []


def opcode_histogram(instrs: list[Instr]) -> dict[str, int]:
    h: dict[str, int] = {}
    for ins in instrs:
        h[ins.opcode] = h.get(ins.opcode, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def format_listing(instrs: list[Instr], raw: bool = True) -> str:
    lines = []
    for ins in instrs:
        lab = f"{ins.label}:\n" if ins.label else ""
        r = f"  {ins.raw}" if raw else ""
        lines.append(f"{lab}  /*{ins.offset:04x}*/  {ins.text:<44} {ins.ctrl_str()}{r}")
    return "\n".join(lines)
