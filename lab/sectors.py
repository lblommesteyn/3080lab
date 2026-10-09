"""Static split-sector analysis for global loads in SASS (no source, no R8U2 assumptions).

Address provenance is an affine abstract interpretation over the warp:

    value = root + const + coef * lane

  root   a symbolic, lane-uniform part (a kernel parameter, blockIdx, a uniform product ...);
         None means zero. Structurally built, so two registers with equal roots differ only
         by their constants and lane coefficients.
  const  a known byte offset
  coef   bytes per lane (None = not affine in the lane, e.g. lane * unknown stride)

Each LDG [Rb.64 + imm] gets the value of Rb plus imm. Two loads with the same root touch the same
32 B sectors in a fraction f of their sectors, computed exactly from (const, coef, width) by
enumerating the base alignment mod 32 over multiples of the access width. A pair splits sectors
when f > 0 and the loads are separate requests; whether that costs DRAM traffic depends on the
reuse distance between them (lab/costmodel.py, measured by mem_split_mech).

The analysis only feeds the cost model; correctness of any rewrite comes from lab/schedule.py's
legality checks and lab/verify.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import sass

SECTOR = 32


@dataclass(frozen=True)
class Val:
    root: object
    const: int | None
    coef: int | None

    def uniform(self) -> bool:
        return self.coef == 0


_fresh_count = [0]


def fresh(tag, uniform: bool) -> Val:
    return Val(("u", tag) if uniform else ("t", tag), 0, 0 if uniform else None)


ZERO = Val(None, 0, 0)


def add(a: Val, b: Val, neg_b: bool = False) -> Val:
    if b.root is not None and neg_b:
        return fresh(("neg", a, b), a.uniform() and b.uniform())
    if a.root is not None and b.root is not None:
        root = ("add",) + tuple(sorted([repr(a.root), repr(b.root)]))
    else:
        root = a.root if a.root is not None else b.root
    sign = -1 if neg_b else 1
    const = None if a.const is None or b.const is None else a.const + sign * b.const
    coef = None if a.coef is None or b.coef is None else a.coef + sign * b.coef
    if const is None:
        root, const = ("addc", repr(root), repr(a), repr(b)), 0
    return Val(root, const, coef)


def mul_const(a: Val, k: int) -> Val:
    root = None if a.root is None else ("mul", repr(a.root), k)
    return Val(root, None if a.const is None else a.const * k, None if a.coef is None else a.coef * k)


def mul(a: Val, b: Val, tag) -> Val:
    if b.root is None and b.coef == 0 and b.const is not None:
        return mul_const(a, b.const)
    if a.root is None and a.coef == 0 and a.const is not None:
        return mul_const(b, a.const)
    if a.uniform() and b.uniform():
        return Val(("mul", repr(a), repr(b)), 0, 0)
    return fresh(tag, False)


def _warp_scale(x: Val):
    """x = 32*s*warp + lane*coef (from SR_TID.X scaled by s)? returns s, else None."""
    if x.root == ("warp32",):
        return 1
    if isinstance(x.root, tuple) and len(x.root) == 3 and x.root[0] == "mul" and x.root[1] == repr(("warp32",)):
        return x.root[2]
    return None


def _imm(tok: str) -> int | None:
    tok = tok.strip()
    m = re.fullmatch(r"-?0x[0-9a-fA-F]+|-?\d+", tok)
    return int(tok, 0) if m else None


@dataclass
class Load:
    offset: int
    text: str
    width: int                     # bytes per lane
    addr: Val
    in_loop: bool
    index: int                     # position in issue order within its region
    cache: str                     # "nc" (L1-allocating) or "cg"
    warp_dep: bool = True          # address depends on the warp/block id (each warp streams its own data)


@dataclass
class Pair:
    first: Load
    second: Load
    overlap: float                 # fraction of the second load's sectors the first already touched
    loads_between: int
    bytes_between: int             # per warp, requested strictly between the two
    gap_instr: int


@dataclass
class Analysis:
    loads: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    loop: tuple | None = None      # (first offset, last offset)


def _width(op: str) -> int:
    for w, n in ((".128", 16), (".64", 8), (".U8", 1), (".S8", 1), (".U16", 2), (".S16", 2)):
        if w in op:
            return n
    return 4


def sectors_touched(v: Val, imm: int, width: int, base_mod: int) -> set:
    c = (v.const or 0) + imm + base_mod
    out = set()
    for lane in range(32):
        a = c + lane * v.coef
        for b in range(a, a + width, SECTOR if width > SECTOR else width):
            out.add(b // SECTOR)
        out.add((a + width - 1) // SECTOR)
    return out


def overlap(a: Load, b: Load) -> float:
    """Fraction of b's sectors also touched by a, averaged over base alignments mod 32 that keep
    both accesses naturally aligned."""
    if a.addr.root != b.addr.root or a.addr.coef is None or b.addr.coef is None:
        return 0.0
    if a.addr.const is None or b.addr.const is None:
        return 0.0
    step = max(a.width, b.width, 4)
    fr = []
    for m in range(0, SECTOR, step):
        sa = sectors_touched(a.addr, 0, a.width, m)
        sb = sectors_touched(b.addr, 0, b.width, m)
        fr.append(len(sa & sb) / len(sb))
    return sum(fr) / len(fr)


def streams(ld: Load) -> bool:
    """Each warp reads its own addresses (depends on the warp or block id), i.e. DRAM streaming.
    Lane-only addresses (a vector every warp reads) hit in L1/L2 after the first warp."""
    return ld.warp_dep


def warp_bytes(ld: Load) -> int:
    """DRAM/L2 bytes one warp-instruction requests (sectors touched x 32), alignment-averaged."""
    if ld.addr.coef is None:
        return 32 * ld.width          # unknown stride: assume scattered, at least width per lane
    n = [len(sectors_touched(ld.addr, 0, ld.width, m)) for m in range(0, SECTOR, max(ld.width, 4))]
    return round(sum(n) / len(n) * SECTOR)


class _Interp:
    def __init__(self):
        self.r: dict[str, Val] = {}
        self.u: dict[str, Val] = {}
        self.local: dict = {}                 # spill slots: (stack base, byte offset) -> Val
        self.dep: dict = {}                   # register / spill slot -> depends on the warp or block id

    def _slot(self, addr: str):
        m = re.match(r"\[(R\d+)(?:\+(-?0x[0-9a-f]+))?\]", addr.strip())
        if not m:
            return None
        return (repr(self.get(m.group(1))), int(m.group(2), 16) if m.group(2) else 0)

    def get(self, tok: str) -> Val:
        tok = tok.strip().lstrip("-~|").replace(".reuse", "")
        tok = re.sub(r"\.(64|X|H1|B1|B2|B3)$", "", tok)
        if tok in ("RZ", "URZ"):
            return ZERO
        k = _imm(tok)
        if k is not None:
            return Val(None, k, 0)
        m = re.fullmatch(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", tok)
        if m:
            return Val(("c", int(m.group(1), 16), int(m.group(2), 16)), 0, 0)
        if tok.startswith("UR"):
            return self.u.get(tok, fresh(("ur", tok), True))
        if tok.startswith("R"):
            return self.r.get(tok, fresh(("live-in", tok), False))
        return fresh(("tok", tok), False)

    def step(self, ins: sass.Instr):
        t = re.sub(r"\(\*.*?\*\)", "", ins.text).strip().replace(".reuse", "")   # drop nvdisasm comments
        pred = re.match(r"^@(!?)(U?P\w+)\s+", t)
        t = re.sub(r"^@!?U?P\w+\s+", "", t)
        parts = t.split(None, 1)
        op = parts[0]
        args = [a.strip() for a in parts[1].split(",")] if len(parts) > 1 else []
        if not args:
            return
        dst = args[0]
        if op.startswith(("STL", "LDL")):            # spills: keep register values through the stack
            w = 4 if ".128" in op else 2 if ".64" in op else 1
            if op.startswith("STL") and len(args) >= 2:
                slot, src_reg = self._slot(args[0]), args[1]
                if slot is not None and re.fullmatch(r"R\d+", src_reg):
                    for k in range(w):
                        self.local[(slot[0], slot[1] + 4 * k)] = self.get(f"R{int(src_reg[1:]) + k}")
                        self.dep[("slot", slot[0], slot[1] + 4 * k)] = self.dep.get(f"R{int(src_reg[1:]) + k}", False)
                return
            if op.startswith("LDL") and len(args) >= 2 and re.fullmatch(r"R\d+", dst):
                slot = self._slot(args[1])
                for k in range(w):
                    v = self.local.get((slot[0], slot[1] + 4 * k)) if slot else None
                    self.set(f"R{int(dst[1:]) + k}", v if v is not None else fresh((ins.offset, k), False), pred)
                    self.dep[f"R{int(dst[1:]) + k}"] = bool(slot) and self.dep.get(("slot", slot[0], slot[1] + 4 * k), True)
                return
        if not re.fullmatch(r"U?R\d+", dst):
            # predicate-first forms (ISETP etc.) or stores: no register result we track
            if op.startswith(("ST", "BRA", "EXIT", "BAR", "CALL", "RET")):
                return
            dregs = [a for a in args if re.fullmatch(r"U?R\d+", a)]
            if op.startswith(("SHFL",)) and len(args) > 1:
                self.set(args[1], fresh((ins.offset, "shfl"), False), pred)
            return
        tag = (ins.offset, dst)
        src = [a for a in args[1:] if not re.fullmatch(r"!?U?P\w+", a)]
        d = any(self.dep.get(re.sub(r"\.\w+$", "", a.lstrip("-~|")), False) for a in src)             or (op.startswith("S2R") and len(args) > 1 and args[1] in ("SR_TID.X", "SR_CTAID.X", "SR_CTAID.Y", "SR_LANEID"))
        self.dep[dst] = d or (bool(pred) and self.dep.get(dst, False))
        if op.startswith(("IMAD.WIDE", "UIMAD.WIDE")) or ".64" in op:
            n_ = int(dst.lstrip("UR"))
            self.dep[("UR" if dst.startswith("UR") else "R") + str(n_ + 1)] = self.dep[dst]
        v = None
        try:
            if op.startswith("CS2R") and args[1] == "SRZ":
                v = ZERO
            elif op.startswith("S2R") or op.startswith("CS2R") or op.startswith("S2UR"):
                sr = args[1]
                if sr == "SR_TID.X":
                    v = Val(("warp32",), 0, 1)
                elif sr == "SR_LANEID":
                    v = Val(None, 0, 1)
                else:
                    v = fresh(("sr", sr), sr.startswith("SR_CTAID") or sr.startswith("SR_NTID"))
            elif op in ("MOV", "UMOV", "IMAD.MOV.U32", "IMAD.MOV") or op.startswith("ULDC") or op.startswith("LDC"):
                v = self.get(src[-1])
            elif op.startswith("IADD3") and not op.startswith("IADD3.X"):
                vals = []
                for a in src[:3]:
                    vals.append((self.get(a), a.startswith("-")))
                v = ZERO
                for x, neg in vals:
                    v = add(v, x, neg)
            elif op == "IMAD.IADD":
                v = add(mul(self.get(src[0]), self.get(src[1]), tag), self.get(src[2]))
            elif op.startswith("IMAD.SHL"):
                v = mul(self.get(src[0]), self.get(src[1]), tag)
            elif op.startswith("IMAD.WIDE") or op == "IMAD" or op == "UIMAD" or op.startswith("UIMAD.WIDE"):
                v = add(mul(self.get(src[0]), self.get(src[1]), tag), self.get(src[2]))
            elif op.startswith("LEA") and not op.startswith("LEA.HI"):
                s = _imm(src[-1])
                v = add(mul_const(self.get(src[0]), 1 << s), self.get(src[1])) if s is not None else None
            elif op.startswith("SHF.R") and _imm(src[1]) is not None and src[0] == "RZ":
                x, k = self.get(src[2]), _imm(src[1])
                sc = _warp_scale(x)
                if x.uniform():
                    v = Val(("shr", repr(x), k), 0, 0)
                elif sc is not None and x.const == 0 and x.coef is not None and (1 << k) <= 32 * sc                         and 31 * x.coef < (1 << k):
                    v = Val(("warpid", 32 * sc >> k), 0, 0)            # (32*s*warp + lane*c) >> k
                elif x.coef == 1 and "warp32" in repr(x.root) and k >= 5:
                    # (blockIdx * blockDim + tid) >> k, k >= 5: the global warp id. Exact when the
                    # non-tid part is a multiple of 32 (block sizes are), assumed: cost model only.
                    v = Val(("warpid", repr(x.root), k), 0, 0)
            elif op.startswith("SHF.L") or op.startswith("USHF.L"):
                s = _imm(src[1])
                v = mul_const(self.get(src[0]), 1 << s) if s is not None and src[2] in ("RZ", "URZ") else None
            elif op.startswith("LOP3.LUT") and len(src) >= 4:
                a, b = self.get(src[0]), _imm(src[1])
                lut = src[3]
                if lut == "0xc0" and b is not None and src[2] == "RZ":          # a & imm
                    sc = _warp_scale(a)
                    if a.uniform():
                        v = Val(("and", repr(a), b), 0, 0)
                    elif sc is not None and a.const == 0 and a.coef is not None and (32 * sc) & (32 * sc - 1) == 0                             and b < 32 * sc and all((l * a.coef) & b == l * a.coef for l in range(32)):
                        v = Val(None, 0, a.coef)                               # warp part masked away
                    elif a.coef not in (None, 0) and b > 0 and 31 * abs(a.coef) < (b & -b):
                        # mask clears every bit the lane can reach ((gid << s) & ~(2^z - 1)): the result
                        # is warp-uniform, assuming the other terms are 2^z-aligned (cost model only)
                        v = Val(("andmask", repr(a.root), b), 0, 0)
        except (IndexError, TypeError):
            v = None
        if v is None:
            unif = all(self.get(a).uniform() for a in src if re.fullmatch(r"U?R\d+(\.\w+)?", a.lstrip("-~|"))) \
                and not op.startswith(("LD", "SHFL", "S2R"))
            v = fresh(tag, unif or op.startswith("U"))
        self.set(dst, v, pred)
        if v.root is None and not pred:              # a pure function of the lane (e.g. tid & 31): same in every warp
            self.dep[dst] = False
        if op.startswith(("IMAD.WIDE", "UIMAD.WIDE")) or ".64" in op:
            n = int(dst.lstrip("UR"))
            hi = ("UR" if dst.startswith("UR") else "R") + str(n + 1)
            self.set(hi, fresh((ins.offset, "hi"), v.uniform()), pred)

    def set(self, dst: str, v: Val, pred):
        table = self.u if dst.startswith("UR") else self.r
        if pred:
            old = table.get(dst)
            if old != v:
                v = Val(("phi", repr(old), repr(v)), 0,
                        old.coef if old is not None and old.coef == v.coef else None)
        table[dst] = v


def analyze(cubin: bytes | None = None, ins: list | None = None) -> Analysis:
    from . import toolchain
    if ins is None:
        ins = sass.parse(toolchain.disassemble(cubin))
    body = sass.loop_body(ins)
    lo, hi = (body[0].offset, body[-1].offset) if body else (None, None)
    it = _Interp()
    # straight code before the loop, then the loop body to a fixed point on its registers
    pre = [x for x in ins if lo is None or x.offset < lo]
    for x in pre:
        it.step(x)
    if body:
        dep_entry = dict(it.dep)
        entry_r, entry_u, entry_l = dict(it.r), dict(it.u), dict(it.local)
        for _ in range(4):
            snap_r, snap_u, snap_l = dict(it.r), dict(it.u), dict(it.local)
            for x in body:
                it.step(x)
            # merge back edge into loop-header values
            changed = False
            for tbl, ent, snap in ((it.r, entry_r, snap_r), (it.u, entry_u, snap_u), (it.local, entry_l, snap_l)):
                for k in set(tbl) | set(snap):
                    a, b = snap.get(k), tbl.get(k)
                    if a != b:
                        coef = a.coef if a is not None and b is not None and a.coef == b.coef else None
                        e0 = ent.get(k)                        # keep provenance (warp/cta dependence)
                        merged = Val(("loopvar", k, repr(e0.root) if e0 is not None else None), 0, coef)
                        if snap.get(k) != merged:
                            changed = True
                        tbl[k] = merged
                    else:
                        tbl[k] = a
            for k_, v_ in dep_entry.items():                 # loop-header dependence = entry OR back edge
                it.dep[k_] = it.dep.get(k_, False) or v_
            if not changed:
                break
    res = Analysis(loop=(lo, hi))
    # final pass to read addresses (loop header values now stable)
    for region, xs in (("loop", body), ("pre", pre)):
        state_r, state_u, state_l = dict(it.r), dict(it.u), dict(it.local)
        if region == "pre":
            it2 = _Interp()
        else:
            it2 = _Interp()
            it2.r, it2.u, it2.local = state_r, state_u, state_l
            it2.dep = dict(it.dep)
        idx = 0
        for x in xs:
            if x.opcode.startswith("LDG"):
                m = re.search(r"\[(R\d+)(?:\.64)?(?:\+(-?0x[0-9a-f]+))?\]", x.text)
                if m:
                    base = it2.get(m.group(1))
                    imm = int(m.group(2), 16) if m.group(2) else 0
                    addr = Val(base.root, None if base.const is None else base.const + imm, base.coef)
                    cache = "cg" if ".STRONG" in x.opcode else "nc"
                    res.loads.append(Load(x.offset, x.text, _width(x.opcode), addr, region == "loop", idx, cache,
                                          it2.dep.get(m.group(1), True)))
                    idx += 1
            it2.step(x)
    # pairs: for each load, the closest EARLIER load (same region) sharing sectors with it
    pos = {x.offset: k for k, x in enumerate(ins)}
    for region in (True, False):
        lds = [l for l in res.loads if l.in_loop == region]
        for j, b in enumerate(lds):
            # the partner is the earlier load sharing the MOST of b's sectors (closest on ties); the
            # closest one with any overlap can be a neighbouring row touching a single sector
            best = None
            for i in range(j - 1, -1, -1):
                f = overlap(lds[i], b)
                if f > 0 and (best is None or f > best[1] + 1e-9):
                    best = (i, f)
            if best is not None:
                i, f = best
                between = lds[i + 1:j]
                res.pairs.append(Pair(lds[i], b, f, len(between), sum(warp_bytes(x) for x in between),
                                      pos[b.offset] - pos[lds[i].offset]))
    return res
