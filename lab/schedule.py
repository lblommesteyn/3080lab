"""Phase 7: SASS instruction motion inside a compiled kernel.

hoist(): move one instruction earlier within a straight-line region, only when every
conservative legality rule holds, then repair control bits:
  * the moved instruction's register sources must not be (re)defined in the skipped range;
  * its destination(s) must not be read or written in the skipped range (WAR/WAW);
  * no store / barrier / atomic / control flow / unmodeled instruction in the range;
  * predicate and uniform registers it mentions must not be mentioned in the range;
  * a fixed-latency producer of a source must sit >= MIN_FIXED_GAP stall-cycles before the new
    position; a scoreboarded producer's barrier is added to the moved instruction's wait mask;
  * the instruction before the old position absorbs the moved instruction's stall (timing of
    everything else is preserved or lengthened, never shortened);
  * reuse flags on the instructions adjacent to the old and new positions are cleared.

fix_split_sectors(): the pass motivated by mem_split_gap / gemv_load_order. ptxas defers the
second half of 32 B sectors split across two LDG.128 (same base register, offsets o and o+16)
by up to ~1000 instructions; L1 only merges the second request within ~2 loads. Hoist the
second load next to the first.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from . import operands as opnd
from . import patch, rename, sass, toolchain

MIN_FIXED_GAP = 6          # cycles between a fixed-latency producer and a consumer we move up
BARRIERS = ("ST", "ATOM", "RED", "MEMBAR", "BAR", "CCTL", "LDGSTS", "BRA", "CALL", "RET", "EXIT", "WARPSYNC",
            "BSSY", "BSYNC", "DEPBAR", "NANOSLEEP", "YIELD", "S2R", "CS2R")
_PRED = re.compile(r"\bU?P[0-6]\b")
_UREG = re.compile(r"\bUR\d+\b")
_REG = re.compile(r"(?<![U\w])R(\d+)\b")


@dataclass
class RegUse:
    defs: set
    uses: set
    ok: bool          # all register operands understood


def reg_use(i: sass.Instr, cache: dict) -> RegUse:
    m = re.match(r"^CS2R\s+R(\d+),\s*SRZ\s*$", i.text.strip())
    if m:                                          # zeroing idiom: writes a register pair, reads nothing
        r = int(m.group(1))
        return RegUse({r, r + 1}, set(), True)
    toks = opnd.reg_tokens(i.text)
    widths = rename.token_widths(i)
    info = cache.get(opnd.form_key(i), {"fields": {}})
    if widths is None or not info["fields"] or len(widths) != len(toks):
        return RegUse(set(), set(), False)
    dtok = info["fields"].get("d")
    defs, uses = set(), set()
    for k, t in enumerate(toks):
        if t == "RZ":
            continue
        r = int(t[1:])
        regs = set(range(r, r + widths[k]))
        (defs if k == dtok else uses).update(regs)
    if re.match(r"^@!?P", i.text.strip()) and defs:
        uses |= defs                               # predicated write keeps the old value live
    return RegUse(defs, uses, True)


def _is_fixed_latency(i: sass.Instr) -> bool:
    return i.wbar == 7 and not i.opcode.startswith(("LD", "MUFU", "SHFL", "DFMA", "S2R"))


def hoist(cubin: bytes, kernel: str, src_off: int, dst_after_off: int) -> tuple[bytes, str]:
    """Move instruction at src_off to directly after dst_after_off. Returns (cubin, reason);
    reason is 'ok' or why the move was refused (cubin unchanged)."""
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    idx = {x.offset: k for k, x in enumerate(ins)}
    s, d = idx[src_off], idx[dst_after_off]
    if not d < s - 1:
        return cubin, "not an upward move"
    m = ins[s]
    mu = reg_use(m, cache)
    if not mu.ok:
        return cubin, "moved instruction has unmodeled operands"
    m_preds, m_urs = set(_PRED.findall(m.text)), set(_UREG.findall(m.text))
    skipped = ins[d + 1:s]
    for x in skipped:
        if x.opcode.startswith(BARRIERS) and not (x.opcode == "CS2R" and x.text.rstrip().endswith("SRZ")):
            return cubin, f"barrier/memory/control instruction in range: {x.text}"
        xu = reg_use(x, cache)
        if not xu.ok:
            return cubin, f"unmodeled instruction in range: {x.text}"
        if xu.defs & mu.uses:
            return cubin, f"source redefined in range: {x.text}"
        if (xu.uses | xu.defs) & mu.defs:
            return cubin, f"destination touched in range: {x.text}"
        x_preds = set(_PRED.findall(x.text))
        shared = m_preds & x_preds
        # predicates: only a def on either side orders the pair (read-read sharing is free)
        if shared and ((_pred_defs(m.text) & shared) or (_pred_defs(x.text) & shared))                 or m_urs & set(_UREG.findall(x.text)):
            return cubin, f"predicate/uniform register shared with: {x.text}"
    # producers of the moved instruction's sources that sit before the new position
    extra_wait = 0
    cyc = 0
    for k in range(d, -1, -1):
        x = ins[k]
        if x.opcode.startswith(("BRA", "EXIT", "CALL", "RET")) or x.label:
            pass
        xu = reg_use(x, cache)
        if not xu.ok:
            if x.opcode.startswith(("S2R", "CS2R")) and xu.defs & mu.uses:
                return cubin, "source from special-register read"
            if set(_REG.findall(x.text)) & {str(r) for r in mu.uses}:
                return cubin, f"source may come from unmodeled instruction: {x.text}"
        if xu.defs & mu.uses:
            if x.wbar != 7:
                extra_wait |= 1 << x.wbar
            elif _is_fixed_latency(x) and cyc < MIN_FIXED_GAP:
                return cubin, f"fixed-latency producer too close ({cyc} cycles): {x.text}"
            if cyc >= 64 and x.wbar == 7:
                break
        cyc += x.stall if x.stall else 32
        if x.label and k < d:
            break   # do not reason across a block entry; earlier producers are older than the label
    # build the new instruction order inside [d+1, s]
    elf = bytearray(cubin)
    sec = patch.text_section(elf, kernel)
    raw = [bytes(elf[sec.offset + x.offset: sec.offset + x.offset + 16]) for x in ins[d + 1:s + 1]]
    moved, rest = raw[-1], raw[:-1]
    new = [moved] + rest
    base = sec.offset + ins[d + 1].offset
    for k, w in enumerate(new):
        elf[base + 16 * k: base + 16 * k + 16] = w
    c = bytes(elf)
    # control-bit repair
    edits = {}
    m_new = ins[d + 1].offset                       # moved instruction now lives here
    edits[m_new] = {"wait": (m.wait | extra_wait) & 0x3F}
    prev_old = ins[s].offset                        # instruction that now precedes the old slot
    pv = ins[s - 1]
    st = (pv.stall if pv.stall else 1) + (m.stall if m.stall else 1)
    if st > 15:
        return cubin, "stall overflow when absorbing the moved instruction's stall"
    edits[prev_old] = {"stall": st, "reuse": 0}
    edits[ins[d].offset] = {**edits.get(ins[d].offset, {}), "reuse": 0}
    edits[m_new].update(reuse=0)
    c = patch.set_control(c, kernel, edits)
    return c, "ok"


def rename_dest(cubin: bytes, kernel: str, off: int, new_base: int) -> tuple[bytes, str]:
    """Give the (wide) destination of the load at `off` fresh registers new_base.. and rewrite every
    use of exactly that value (its def-use webs, flow-sensitive), so the load can later be hoisted
    over code that reuses its old registers. Refuses if any affected occurrence is unmodeled."""
    from . import regalloc
    g = regalloc.build(cubin, kernel)
    W = regalloc.webs(g)
    ins = g["ins"]
    k = next(i for i, x in enumerate(ins) if x.offset == off)
    defs = [o for o in g["occs"][k] if o.is_def]
    if not defs:
        return cubin, "no destination"
    r0 = min(o.reg for o in defs)
    webs = {W["occ_web"][id(o)]: o.reg - r0 for o in defs}
    edits = {}                                    # ins index -> {field: new reg}
    for w, delta in webs.items():
        for o in W["members"][w]:
            if not o.mapped and o.ins != k:
                return cubin, f"web touches unmodeled instruction: {ins[o.ins].text}"
            info = g["cache"].get(opnd.form_key(ins[o.ins]))
            t2f = {t: f for f, t in info["fields"].items()}
            if o.tok not in t2f:
                return cubin, f"unmapped token in {ins[o.ins].text}"
            fld = {"d": "rd", "a": "ra", "b": "rb", "c": "rc"}[t2f[o.tok]]
            if o.ins == k:
                edits.setdefault(o.ins, {})[fld] = new_base          # wide dest token names the base
            else:
                edits.setdefault(o.ins, {})[fld] = new_base + delta
    c = cubin
    for i, kw in edits.items():
        c = patch.set_regs(c, kernel, ins[i].offset, **kw)
    need = new_base + 4 + 3
    if need > rename._regcount(c, kernel):
        c = patch.set_regcount(c, kernel, need)
    return c, "ok"


def split_sector_pairs(cubin: bytes, kernel: str = "k") -> list[tuple[int, int, int]]:
    """(first_offset, second_offset, gap_in_instructions) for LDG pairs sharing a base register
    with offsets o (32-aligned) and o+16, where the +16 load comes later."""
    ins = sass.parse(toolchain.disassemble(cubin))
    body = sass.loop_body(ins) or ins
    pos = {x.offset: k for k, x in enumerate(body)}
    by = {}
    for x in body:
        if x.opcode.startswith("LDG.E.128"):
            m = re.search(r"\[R(\d+)\.64(?:\+(-?0x[0-9a-f]+))?\]", x.text)
            if m:
                by.setdefault(m.group(1), []).append((int(m.group(2) or "0", 16), x))
    out = []
    for lst in by.values():
        offs = {o: x for o, x in lst}
        for o, x in lst:
            if o % 32 == 0 and (o + 16) in offs:
                y = offs[o + 16]
                a, b = sorted((x, y), key=lambda z: pos[z.offset])
                out.append((a.offset, b.offset, pos[b.offset] - pos[a.offset]))
    return out


def _find(cubin: bytes, lo: int, hi: int, text: str):
    lst = [y for y in sass.parse(toolchain.disassemble(cubin)) if lo < y.offset < hi and y.text == text]
    return lst[-1] if lst else None


def _try_move(cubin: bytes, kernel: str, a: int, b: int, rename_regs: bool, fresh_base: int):
    """Move the load at b up next to a, repairing what blocks it. Returns (cubin, reason, regs_used)."""
    notes, anchor, renamed = [], a, False
    for _ in range(16):
        c2, why = hoist(cubin, kernel, b, anchor)
        if why == "ok":
            return c2, "ok" + (f" ({', '.join(notes)})" if notes else ""), renamed
        blocker = why.split(": ", 1)[1] if ": " in why else ""
        bdefs = reg_use(next(y for y in sass.parse(toolchain.disassemble(cubin)) if y.offset == b),
                        opnd.discover(cubin, kernel)).defs
        x = _find(cubin, anchor, b, blocker) if blocker else None
        xdefs = set(int(r) for r in _REG.findall(blocker.split(",")[0])) if blocker else set()
        if why.startswith("fixed-latency producer too close"):
            # leave room after the address producer: slide the landing point down
            nxt = [y.offset for y in sass.parse(toolchain.disassemble(cubin)) if anchor < y.offset < b]
            if len(nxt) < 2 or anchor - a >= 8 * 16:
                break
            anchor = nxt[0]
            notes.append("slid")
            continue
        if why.startswith("predicate/uniform") and x is not None:
            gp = re.match(r"^@!?(P\d)\s", next(y for y in sass.parse(toolchain.disassemble(cubin))
                                              if y.offset == b).text.strip())
            if gp and gp.group(1) in _pred_defs(x.text):
                anchor = x.offset
                notes.append("after guard def")
                continue
            break
        if why.startswith(("source redefined", "destination touched")) and x is not None:
            if _is_const_def(x.text) and xdefs & bdefs:
                # the not-taken-path default of a guarded load: carry it up first
                c3, w3 = hoist(cubin, kernel, x.offset, anchor)
                if w3 == "ok":
                    cubin, anchor = c3, anchor + 16
                    notes.append("carried default")
                    continue
            if rename_regs and not renamed and (why.startswith("destination touched") or xdefs & bdefs):
                c3, w3 = rename_dest(cubin, kernel, b, fresh_base)
                if w3 == "ok":
                    cubin, renamed = c3, True
                    notes.append(f"renamed to R{fresh_base}")
                    continue
                return cubin, f"rename refused: {w3}", False
        break
    return cubin, why, False


def fix_split_sectors(cubin: bytes, kernel: str = "k", min_gap: int = 3, max_moves: int = 128,
                      rename_regs: bool = True) -> tuple[bytes, list]:
    """Repeatedly hoist the second half of the widest-gap split-sector pair next to its first half.
    Pairs are re-found after every move (moves shift offsets); refused pairs are not retried."""
    log, refused = [], set()
    for _ in range(max_moves):
        ins = {x.offset: x for x in sass.parse(toolchain.disassemble(cubin))}
        cand = [(a, b, g) for a, b, g in split_sector_pairs(cubin, kernel)
                if g >= min_gap and (ins[a].text, ins[b].text) not in refused]
        if not cand:
            break
        a, b, g = max(cand, key=lambda t: t[2])
        top = max(int(r) for x in ins.values() for r in _REG.findall(x.text))
        c2, why, _ = _try_move(cubin, kernel, a, b, rename_regs, (top + 4) & ~3)
        log.append((g, ins[b].text, why))
        if why.startswith("ok"):
            cubin = c2
        else:
            refused.add((ins[a].text, ins[b].text))
    return cubin, log


def _is_const_def(text: str) -> bool:
    """Unpredicated register def with no register sources (MOV imm, IMAD.MOV.U32 Rd, RZ, RZ, imm, CS2R Rd, SRZ)."""
    t = text.strip()
    if t.startswith("@"):
        return False
    m = re.match(r"^(MOV|IMAD\.MOV\.U32|CS2R)\s+R\d+,\s*(.*)$", t)
    return bool(m) and not _REG.findall(m.group(2))


# ---- predicate fields (found by perturbation: scripts in README round 5) --------------------
def set_guard(cubin: bytes, kernel: str, off: int, pred: int) -> bytes:
    """Guard predicate index lives in lo[12:15] (bit 15 = negate, left unchanged)."""
    elf = bytearray(cubin)
    sec = patch.text_section(elf, kernel)
    base = sec.offset + off
    lo, = struct.unpack_from("<Q", elf, base)
    struct.pack_into("<Q", elf, base, (lo & ~(7 << 12)) | (pred << 12))
    return bytes(elf)


def set_isetp_dest(cubin: bytes, kernel: str, off: int, pred: int) -> bytes:
    """ISETP's destination predicate lives in hi[17:20]."""
    elf = bytearray(cubin)
    sec = patch.text_section(elf, kernel)
    base = sec.offset + off + 8
    hi, = struct.unpack_from("<Q", elf, base)
    struct.pack_into("<Q", elf, base, (hi & ~(7 << 17)) | (pred << 17))
    return bytes(elf)


def _pred_defs(text: str) -> set:
    """Predicates an instruction writes (setp outputs and carry-outs)."""
    t = re.sub(r"^@!?U?P\w+\s+", "", text.strip())
    out = set()
    m = re.match(r"^(?:[IFHD]SETP|PLOP3|LOP3|P2R|R2P)\S*\s+(P\d)(?:,\s*(P\d))?", t)
    if m:
        out.update(g for g in m.groups() if g)
    m = re.match(r"^\S+\s+R\d+,\s*(P\d)\b", t)        # IADD3 Rd, P0, ... / LEA Rd, P1, ...: carry out
    if m:
        out.add(m.group(1))
    return out


def fix_split_sectors_guarded(cubin: bytes, kernel: str = "k", min_gap: int = 3) -> tuple[bytes, list]:
    """Like fix_split_sectors, but first hoists the ISETP that guards the late loads, retargeted to a
    free predicate register, so the guarded loads can move up next to their sector partners."""
    log = []
    ins = sass.parse(toolchain.disassemble(cubin))
    used = set(re.findall(r"\bP([0-6])\b", " ".join(x.text for x in ins)))
    free = [p for p in (6, 5, 4) if str(p) not in used]
    pairs = [p for p in split_sector_pairs(cubin, kernel) if p[2] >= min_gap]
    pos = {x.offset: k for k, x in enumerate(ins)}
    by_guard = {}
    for a, b, g in pairs:
        m = re.match(r"^@(!?)P(\d)\s", ins[pos[b]].text.strip())
        if m:
            by_guard.setdefault(int(m.group(2)), []).append((a, b, g))
    for gp, lst in by_guard.items():
        if not free:
            log.append((gp, "no free predicate register"))
            break
        last_b = max(pos[b] for _, b, _ in lst)
        # latest definition of P<gp> before the first guarded late load
        first_b = min(pos[b] for _, b, _ in lst)
        xk = next((k for k in range(first_b - 1, -1, -1) if f"P{gp}" in _pred_defs(ins[k].text)), None)
        if xk is None or not ins[xk].opcode.startswith("ISETP"):
            log.append((gp, f"guard not defined by a hoistable ISETP ({ins[xk].text if xk is not None else 'none'})"))
            continue
        # consumers of X's value: until the next def of P<gp>
        nxt = next((k for k in range(xk + 1, len(ins)) if f"P{gp}" in _pred_defs(ins[k].text)), len(ins))
        users, bad = [], None
        for k in range(xk + 1, nxt):
            t = ins[k].text.strip()
            guard_use = re.match(rf"^@!?P{gp}\s", t) is not None
            body_use = re.search(rf"\bP{gp}\b", re.sub(r"^@!?P\d\s+", "", t)) is not None
            if body_use:
                bad = t
                break
            if guard_use:
                users.append(ins[k].offset)
        if bad:
            log.append((gp, f"predicate value also consumed as an operand: {bad}"))
            continue
        pk = free.pop()
        c = set_isetp_dest(cubin, kernel, ins[xk].offset, pk)
        for off in users:
            c = set_guard(c, kernel, off, pk)
        # hoist the ISETP to just after the earliest first-half load of these pairs
        a0 = min((a for a, _, _ in lst), key=lambda o: pos[o])
        c2, why = hoist(c, kernel, ins[xk].offset, a0)
        log.append((gp, f"ISETP -> P{pk}, {len(users)} guards retargeted, hoist: {why}"))
        if why != "ok":
            continue
        cubin = c2
    c3, log2 = fix_split_sectors(cubin, kernel, min_gap)
    return c3, log + log2
