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


_WIDTH = {".64": 2, ".128": 4}


def _local_mem_use(i: sass.Instr):
    """LDL/STL (spill refill/store): parsed directly, operand discovery does not cover them."""
    t = re.sub(r"^@!?P\d\s+", "", i.text.strip()).replace(".reuse", "")
    m = re.match(r"^(LDL|STL)(\.\w+)*\s+(.*)$", t)
    if not m:
        return None
    w = next((n for k, n in _WIDTH.items() if k in i.opcode), 1)
    ops = m.group(3)
    addr = re.search(r"\[(R\d+)?", ops)
    a = {int(addr.group(1)[1:])} if addr and addr.group(1) else set()
    regs = [int(x) for x in re.findall(r"(?<![\[\w])R(\d+)(?![\.\w])", ops.split("[")[0] if m.group(1) == "LDL" else ops.split("]")[-1])]
    if m.group(1) == "LDL":
        d = set(range(regs[0], regs[0] + w)) if regs else set()
        u = set(a)
        if re.match(r"^@!?P", i.text.strip()):
            u |= d
        return RegUse(d, u, True)
    return RegUse(set(), a | (set(range(regs[0], regs[0] + w)) if regs else set()), True)


def reg_use(i: sass.Instr, cache: dict) -> RegUse:
    lm = _local_mem_use(i) if i.opcode.startswith(("LDL", "STL")) else None
    if lm is not None:
        return lm
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
        # Memory ordering: local memory (spill LDL/STL) never aliases global memory, and a
        # .CONSTANT (ld.global.nc) load reads data that is read-only for the whole kernel, so it
        # commutes with every store. Register dependencies are still checked below.
        local = x.opcode.startswith(("LDL", "STL"))
        nc_over_store = m.opcode.startswith("LDG") and ".CONSTANT" in m.opcode and x.opcode.startswith(("ST", "LD"))             and not x.opcode.startswith(("STS", "LDS")) or False
        memory_ok = m.opcode.startswith("LDG") and (local or nc_over_store)
        if x.opcode.startswith(BARRIERS) and not (x.opcode == "CS2R" and x.text.rstrip().endswith("SRZ"))                 and not memory_ok:
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
    # producers of the moved instruction's sources that sit before the new position. Distances
    # use the measured pair latencies (lab/resched.latency: 4 same pipe, 5 across, ptxas's own
    # distance for unmeasured ops), and include the guard/operand PREDICATES, not just registers.
    from .resched import fixed_latency_table, latency
    table = fixed_latency_table(ins, cache)
    m_pred_uses = {p for p in _PRED.findall(m.text) if p not in _pred_defs(m.text)}
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
        if (xu.defs & mu.uses) or (_pred_defs(x.text) & m_pred_uses):
            if x.wbar != 7:
                extra_wait |= 1 << x.wbar
            elif _is_fixed_latency(x) and cyc < max(MIN_FIXED_GAP, latency(x, table, m)):
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
    if st > 11:                                     # encoding: yield set allows stalls 1..11 only
        edits[prev_old]["yield"] = 0
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
    for _ in range(48):                               # slides of up to 32 slots need room
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
            slides = sum(1 for n_ in notes if n_ == "slid")
            if len(nxt) < 2 or slides >= 32:                 # up to 32 slots: covers a 13-cycle predicate
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
                      rename_regs: bool = True, finder=None) -> tuple[bytes, list]:
    """Repeatedly hoist the second half of the widest-gap split-sector pair next to its first half.
    Pairs are re-found after every move (moves shift offsets); refused pairs are not retried."""
    log, refused = [], set()
    for _ in range(max_moves):
        ins = {x.offset: x for x in sass.parse(toolchain.disassemble(cubin))}
        cand = [(a, b, g) for a, b, g in (finder or split_sector_pairs)(cubin, kernel)
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


def fix_split_sectors_guarded(cubin: bytes, kernel: str = "k", min_gap: int = 3, finder=None,
                              rename_regs: bool = True) -> tuple[bytes, list]:
    """Like fix_split_sectors, but first hoists the ISETP that guards the late loads, retargeted to a
    free predicate register, so the guarded loads can move up next to their sector partners."""
    log = []
    ins = sass.parse(toolchain.disassemble(cubin))
    used = set(re.findall(r"\bP([0-6])\b", " ".join(x.text for x in ins)))
    free = [p for p in (6, 5, 4) if str(p) not in used]
    pairs = [p for p in (finder or split_sector_pairs)(cubin, kernel) if p[2] >= min_gap]
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
    c3, log2 = fix_split_sectors(cubin, kernel, min_gap, finder=finder, rename_regs=rename_regs)
    return c3, log + log2


def untangle_barriers(cubin: bytes, kernel: str = "k") -> tuple[bytes, list]:
    """Remove false scoreboard dependences created by hoisting (optimize benchmark, qo7 vs gu7: the
    same rewritten binary wins 1.5x on a large grid and loses on a small one).

    A hoisted load keeps its original write barrier. If an instruction between the load and its
    first consumer waits on that barrier for some OTHER load, it now also waits for the hoisted
    load: harmless when bandwidth-bound, a full memory latency per trip when latency-bound.
    Each such load is moved to a barrier no instruction in the loop uses, and that barrier is added
    to the wait mask of every instruction that consumes (or overwrites) its destination."""
    from . import patch
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    body = sass.loop_body(ins)
    if not body:
        return cubin, []
    used = set()
    for x in body:
        used |= {x.wbar, x.rbar} - {7}
        used |= {b for b in range(6) if x.wait >> b & 1}
    free = [b for b in range(6) if b not in used]
    log, edits = [], {}
    n = len(body)
    for k, ld in enumerate(body):
        if not ld.opcode.startswith("LDG") or ld.wbar == 7:
            continue
        u = reg_use(ld, cache)
        if not u.ok or not u.defs:
            continue
        # first instruction (in the iteration) that reads or rewrites the loaded registers
        first = next((j for j in range(k + 1, n) if (lambda r: r.ok and (r.uses | r.defs) & u.defs)(reg_use(body[j], cache))), None)
        if first is None:
            continue
        def waited_between(bar):
            return any(body[j].wait >> bar & 1 for j in range(k + 1, first)) or                 any((edits.get(body[j].offset, {}).get("wait", 0) >> bar) & 1 for j in range(k + 1, first))
        if not waited_between(ld.wbar):
            continue
        # any barrier nobody waits on before this load's first consumer will do: later waits on it run
        # after that consumer, i.e. after the load completed, so they cannot be delayed by it
        cands = [bb for bb in range(6) if bb != ld.rbar and not waited_between(bb)]
        cands.sort(key=lambda bb: bb not in free)       # prefer barriers the loop does not use at all
        if not cands:
            log.append((ld.offset, "every barrier is waited on before the first consumer"))
            continue
        b = cands[0]
        edits.setdefault(ld.offset, {})["wbar"] = b
        # every later reader/writer of the destination, until it is redefined, waits on b too
        live = set(u.defs)
        for j in range(k + 1, n):
            r = reg_use(body[j], cache)
            if not r.ok:
                edits.setdefault(body[j].offset, {})["wait"] = (body[j].wait | edits.get(body[j].offset, {}).get("wait", 0) | 1 << b)
                continue
            if (r.uses | r.defs) & live:
                w = edits.get(body[j].offset, {}).get("wait", body[j].wait)
                edits.setdefault(body[j].offset, {})["wait"] = w | (1 << b)
            live -= r.defs - r.uses
            if not live:
                break
        # the loop back edge: if still live at the end, the branch waits too
        if live:
            br = body[-1]
            w = edits.get(br.offset, {}).get("wait", br.wait)
            edits.setdefault(br.offset, {})["wait"] = w | (1 << b)
        log.append((ld.offset, f"W{ld.wbar} -> W{b}"))
    if not edits:
        return cubin, log
    out = patch.set_control(cubin, kernel, edits)
    # a barrier arms a cycle late: the instruction right after a retargeted load must not wait on it
    ins2 = sass.parse(toolchain.disassemble(out))
    fix = {}
    for a_, b_ in zip(ins2, ins2[1:]):
        if a_.wbar != 7 and b_.wait >> a_.wbar & 1 and a_.stall < 2:
            fix[a_.offset] = {"stall": 2}
    if fix:
        out = patch.set_control(out, kernel, fix)
    return out, log


def dedicate_barrier(cubin: bytes, kernel: str = "k", original: bytes | None = None) -> tuple[bytes, str]:
    """Give the long-lived (hoisted) loads one scoreboard barrier of their own.

    `untangle_barriers` cannot help when every barrier is waited on inside a hoisted load's window
    (issue -> first consumer), as in the rewritten GEMVs. Here: long loads = loads whose barrier is
    waited on before their first consumer. All of them move to one barrier B*; every other producer
    on B* moves to another barrier; inside the long loads' windows, waits on B* are replaced by waits
    on the barriers those other producers now use; elsewhere B* is kept and those barriers are added
    (a superset, never fewer waits). Long loads' consumers also wait on B*. The result must pass the
    hazard verifier against `cubin`, or the caller keeps `cubin`."""
    from . import patch
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    body = sass.loop_body(ins)
    if not body:
        return cubin, "no loop"
    n = len(body)
    uses = [reg_use(x, cache) for x in body]

    def first_consumer(k):
        d = uses[k].defs
        return next((j for j in range(k + 1, n) if uses[j].ok and (uses[j].uses | uses[j].defs) & d), None)

    prods = []
    for k, x in enumerate(body):
        if x.wbar != 7 and uses[k].ok and uses[k].defs:
            prods.append((k, first_consumer(k)))
    moved = None
    if original is not None:
        # the loads the rewrite moved: their position relative to the other loads changed
        from .verify import match
        oi = sass.parse(toolchain.disassemble(original))
        mp = match(oi, ins)
        lds = [x for x in body if x.opcode.startswith("LDG") and x.offset in mp]
        # loads that kept their relative order = longest increasing subsequence of original positions;
        # the rest were moved by the rewrite
        seq = [mp[x.offset] for x in lds]
        import bisect
        tails, tail_idx, prev = [], [], [-1] * len(seq)
        for i, v in enumerate(seq):
            p_ = bisect.bisect_left(tails, v)
            if p_ == len(tails):
                tails.append(v)
                tail_idx.append(i)
            else:
                tails[p_] = v
                tail_idx[p_] = i
            prev[i] = tail_idx[p_ - 1] if p_ > 0 else -1
        keep, i = set(), tail_idx[-1] if tail_idx else -1
        while i >= 0:
            keep.add(i)
            i = prev[i]
        moved = {x.offset for i, x in enumerate(lds) if i not in keep}
    long_ = [(k, fc) for k, fc in prods if body[k].opcode.startswith("LDG") and fc is not None
             and (moved is None or body[k].offset in moved)
             and any(body[j].wait >> body[k].wbar & 1 for j in range(k + 1, fc))]
    if not long_:
        return cubin, "no false dependence"
    long_idx = {k for k, _ in long_}
    windows = [(k, fc) for k, fc in long_]
    best = None
    rbars = {x.rbar for x in body} - {7}
    body_offs = {x.offset for x in body}
    outside = set()
    for x in ins:                                      # barriers armed outside the loop: external producers
        if x.offset not in body_offs:
            outside |= {x.wbar, x.rbar} - {7}
    for B in range(6):
        if B in rbars or B in outside:                 # waits on B must be fully accounted for by the loop
            continue
        others = [k for k, _ in prods if k not in long_idx and body[k].wbar == B]
        # cost: other producers that must move off B (each may add some over-waiting)
        if best is None or len(others) < best[1]:
            best = (B, len(others), others)
    if best is None:
        return cubin, "every barrier is a read barrier somewhere in the loop"
    B, _, others = best
    # move the others to the barrier least used by the remaining producers (never B)
    load = {b: 0 for b in range(6)}
    for k, _ in prods:
        if k not in long_idx and k not in others:
            load[body[k].wbar] += 1
    newbar = {}
    for k in others:
        b2 = min((b for b in range(6) if b != B and b != body[k].rbar), key=lambda b: load[b])
        newbar[k] = b2
        load[b2] += 1
    moved_bits = 0
    for b2 in newbar.values():
        moved_bits |= 1 << b2
    edits = {}
    for k in long_idx:
        edits[body[k].offset] = {"wbar": B}
    for k, b2 in newbar.items():
        edits[body[k].offset] = {"wbar": b2}
    in_window = [any(a < j < fc for a, fc in windows) for j in range(n)]
    # write-after-read: overwrites of a long load's address registers may have been protected by a
    # wait on its WRITE barrier (completion implies the read), even when it has a read barrier;
    # those writers must now wait on B as well
    war = set()
    for k in long_idx:
        srcs = set(uses[k].uses) - set(uses[k].defs)
        for j in range(k + 1, n):
            if uses[j].ok and uses[j].defs & srcs:
                war.add(j)
                srcs -= uses[j].defs
            if not srcs:
                break
    # A wait on B covered exactly the producers armed on B since the previous wait on B (scoreboard
    # waits are sticky). Replace it by those producers' NEW barriers; scan twice for loop-carried ones.
    repl = {}
    armed = []
    for pass_ in range(2):
        for j, x in enumerate(body):
            if x.wait >> B & 1:
                bits = 0
                for k in armed:
                    bits |= 1 << newbar[k]
                if pass_ == 1 or j not in repl:
                    repl[j] = repl.get(j, 0) | bits
                armed = []
            if j in newbar:
                armed.append(j)
    for j, x in enumerate(body):
        w = x.wait
        if w >> B & 1:
            w = (w & ~(1 << B)) | repl.get(j, 0)
            if not in_window[j]:
                w |= 1 << B                            # outside the windows B is harmless: keep it
        if any(fc == j for k, fc in long_ if fc is not None) or j in war:
            w |= 1 << B
        if w != x.wait:
            edits.setdefault(x.offset, {})["wait"] = w
    out = patch.set_control(cubin, kernel, edits)
    ins2 = sass.parse(toolchain.disassemble(out))
    fix = {}
    for a_, b_ in zip(ins2, ins2[1:]):                     # a barrier arms one cycle after issue
        if a_.wbar != 7 and b_.wait >> a_.wbar & 1 and a_.stall < 2:
            fix[a_.offset] = {"stall": 2}
    if fix:
        out = patch.set_control(out, kernel, fix)
    return out, f"{len(long_idx)} long loads -> W{B}, {len(others)} other producers moved"
