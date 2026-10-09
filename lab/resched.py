"""Phase 8: our own instruction scheduler on top of ptxas's register allocation.

Every basic block of a kernel is list-scheduled from a dependence DAG, and its control words are
regenerated: stall counts from measured latencies, wait masks from the new order, reuse flags from
the new neighbours. Register allocation is ptxas's (no renaming), so the scheduler can only reorder.

Safety model (see README Phase 8):
  * Barrier indices stay ptxas's. A scoreboard is a counter, so waiting on a shared index is only
    slower, never wrong; keeping the indices means consumers outside the block (and values pending
    from outside it) still wait on the right barrier. New wait mask = original mask | new needs.
  * Fixed-latency results never straddle a block exit: the terminator (or the block's last stall)
    is delayed until every fixed-latency result of the block is written. Every block gets this rule,
    rescheduled or not, so every block entry is clean.
  * Fixed latencies: 4 cycles for the ops measured in Phase 1/2; for any other op, the smallest
    producer->consumer stall sum ptxas itself used in this cubin (ptxas never goes below the true
    latency), else 15.
  * Blocks with stores, atomics, fences, clock/special-register reads, uniform predicates or
    operands the field discovery cannot model keep their order (they still get the exit rule).
"""
from __future__ import annotations

import random
import re
import struct

from . import operands as opnd
from . import patch, sass, toolchain
from .model import port_of_v1, sb_latency
from .schedule import _pred_defs, reg_use

CONTROL = ("BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "WARPSYNC", "BAR", "BPT",
           "KILL", "NANOSLEEP", "YIELD", "BMOV", "RPCMOV")
UNSAFE = ("ST", "ATOM", "RED", "MEMBAR", "CCTL", "LDGSTS", "DEPBAR", "S2R", "CS2R", "ERRBAR", "FENCE",
          "VOTE", "MATCH", "B2R", "R2B", "LDGDEPBAR", "ARRIVES", "SUATOM", "SURED", "SUST", "LEPC")
REUSE_SLOTS = {"a": 1, "b": 2, "c": 4}
VARUNIT = ("MUFU", "SHFL", "LD", "S2R", "I2F", "F2I", "F2F", "FRND", "POPC", "FLO", "BREV", "DFMA", "DADD",
           "DMUL", "HMMA", "IMMA", "DSETP")
_PRED = re.compile(r"(?<!U)\bP([0-6])\b")
_UREG = re.compile(r"\bUR(\d+)\b")


def _base(op: str) -> str:
    return op.split(".")[0]


def is_control(i: sass.Instr) -> bool:
    return _base(i.opcode) in CONTROL


def is_var(i: sass.Instr) -> bool:
    """Variable latency = ptxas gave it a write or read scoreboard."""
    return i.wbar != 7 or i.rbar != 7


def blocks(ins: list[sass.Instr]) -> list[tuple[int, int]]:
    """[start, end) index ranges of basic blocks: a label starts one, a control instruction ends one."""
    out, s = [], 0
    for k, x in enumerate(ins):
        if x.label and k > s:
            out.append((s, k))
            s = k
        if is_control(x):
            out.append((s, k + 1))
            s = k + 1
    if s < len(ins):
        out.append((s, len(ins)))
    return out


def _resources(i: sass.Instr, cache: dict):
    """(defs, uses) over registers 'R<n>', predicates 'P<n>', uniform registers 'U<n>'; None if unmodelled."""
    t = i.text.strip()
    if re.search(r"\bUP[0-6]\b", t):
        return None
    if is_control(i):                                   # terminators: only their guard predicate
        return set(), {f"P{p}" for p in _PRED.findall(t)}
    if re.match(r"^CS2R\s+R\d+,\s*SRZ\s*$", t):
        r = int(re.findall(r"R(\d+)", t)[0])
        return {f"R{r}", f"R{r + 1}"}, set()
    ru = reg_use(i, cache)
    if not ru.ok:
        return None
    defs = {f"R{r}" for r in ru.defs}
    uses = {f"R{r}" for r in ru.uses}
    pdefs = _pred_defs(t)
    for p in _PRED.findall(t):
        if f"P{p}" not in pdefs:
            uses.add(f"P{p}")
    defs |= pdefs
    if pdefs and t.startswith("@"):
        uses |= pdefs                                   # predicated write keeps the old value live
    urs = _UREG.findall(t)
    if urs:
        op = i.opcode
        if op.startswith("U") or op.startswith(("S2UR", "R2UR")):
            first = int(urs[0])
            width = 2 if ".64" in op else 1
            defs |= {f"U{first + w}" for w in range(width)}
            uses |= {f"U{u}" for u in urs[1:]}
        else:
            uses |= {f"U{u}" for u in urs}
            for u in urs:                               # 64-bit uniform operands: both halves
                if ".64" in i.text or "WIDE" in i.opcode:
                    uses.add(f"U{int(u) + 1}")
    return defs, uses


def fixed_latency_table(ins: list[sass.Instr], cache: dict) -> dict[str, int]:
    """Smallest stall-sum distance ptxas left between a fixed-latency producer (by opcode) and a RAW
    consumer, within straight-line code. A floor ptxas itself trusted."""
    best: dict[str, int] = {}
    res = [_resources(x, cache) for x in ins]
    for (s, e) in blocks(ins):
        for p in range(s, e):
            if ins[p].wbar != 7 or res[p] is None or not res[p][0]:
                continue
            dist, live = 0, set(res[p][0])
            for c in range(p + 1, e):
                dist += ins[c - 1].stall or 32
                if res[c] is None:
                    break
                if res[c][1] & live:
                    if ins[c].wait:                      # ptxas may lean on the wait: not evidence
                        break
                    best[ins[p].opcode] = min(best.get(ins[p].opcode, 99), dist)
                    break
                live -= res[c][0]
                if not live or dist > 15:
                    break
    return best


def pipe(op: str) -> str | None:
    """Pipe class of the opcodes measured by pair_latency (README Phase 8)."""
    base = _base(op)
    if base in ("FFMA", "FADD", "FMUL") or op in ("IMAD", "IMAD.IADD"):
        return "fma"
    if (base == "IADD3" and op != "IADD3.X") or op == "LOP3.LUT" or base in ("SHF", "FMNMX"):
        return "alu"
    return None


def latency(i: sass.Instr, table: dict[str, int], consumer: sass.Instr | None = None) -> int:
    """Cycles from fixed-latency producer i until a dependent consumer may issue (no interlock).
    Measured (pair_latency): 4 within a pipe, 5 across the FMA/ALU pipes, 4 into SHFL. Unmeasured
    producers: the smallest distance ptxas itself used, but never below 5 (consumer pipe unknown)."""
    p = pipe(i.opcode)
    if p is not None:
        if consumer is None:
            return 5
        if consumer.opcode.startswith("SHFL") or pipe(consumer.opcode) == p:
            return 4
        return 5
    if i.opcode in table:
        return min(15, max(5, table[i.opcode]))
    return 15


def var_latency(i: sass.Instr) -> int:
    """Estimated (not guaranteed) latency of a scoreboarded op, for priorities only."""
    if i.opcode.startswith(("LDG", "LD.")) or i.opcode == "LD":
        return 498
    return sb_latency(i, "L1")


def _port(i: sass.Instr) -> tuple[str, int]:
    """Issue port and cycles it is held per warp instruction (model.py v1 tables, measured)."""
    port, cost, _, _ = port_of_v1(i.opcode)
    return port, cost


def _set_stall(hi: int, st: int) -> int:
    """Encoding rule (nvdisasm, all opcodes tried): with the yield bit set the stall must be 1..11;
    ptxas writes 12..15 with yield clear. Reuse needs yield set, so it is cleared too."""
    hi = patch.set_field(hi, "stall", st)
    if st > 11:
        hi = patch.set_field(patch.set_field(hi, "yield", 0), "reuse", 0)
    return hi


def _memory(i: sass.Instr) -> bool:
    return _base(i.opcode).startswith(("LD", "ST", "ATOM", "RED", "SU", "TEX", "TLD"))


def _fixed_producer(i: sass.Instr, r) -> bool:
    """Writes a register or predicate through a fixed-latency pipe."""
    if i.wbar != 7 or is_control(i) or _memory(i):
        return False
    if r is not None:
        return bool(r[0])
    return re.match(r"^(@!?P\w+\s+)?\S+\s+(U?R\d+|P\d)", i.text.strip()) is not None


def schedule_block(ins: list[sass.Instr], res: list, table: dict, policy: str = "crit",
                   rng: random.Random | None = None, reg_bars: dict | None = None):
    """Returns (order, stalls, waits): the new instruction order (indices into ins), each one's stall,
    and each one's extra wait mask. ins/res are the block's instructions and resources."""
    n = len(ins)
    term = n - 1 if is_control(ins[-1]) else None
    body = [k for k in range(n) if k != term]
    # dependence edges: succ[u] = [(v, min_distance, wait_bit_or_None)]
    preds = {k: [] for k in range(n)}
    last_def: dict[str, int] = {}
    readers: dict[str, list[int]] = {}
    for v in range(n):
        dv, uv = res[v]
        for r in uv:                                         # RAW
            u = last_def.get(r)
            if u is not None:
                if ins[u].wbar != 7:                         # scoreboarded result
                    preds[v].append((u, 2, "w"))
                else:                                        # stall-timed result (even if read-barriered)
                    preds[v].append((u, latency(ins[u], table, ins[v]), None))
        for r in dv:
            u = last_def.get(r)
            if u is not None:                                # WAW
                if ins[u].wbar != 7:
                    preds[v].append((u, 2, "w"))
                else:
                    lv = 2 if ins[v].wbar != 7 else 4
                    preds[v].append((u, max(1, latency(ins[u], table) - lv + 1), None))
            for u in readers.get(r, []):                     # WAR
                if u == v:
                    continue
                if is_var(ins[u]):
                    preds[v].append((u, 2, "r" if ins[u].rbar != 7 else "w"))
                else:
                    preds[v].append((u, 1, None))
        for r in uv:
            readers.setdefault(r, []).append(v)
        for r in dv:
            last_def[r] = v
            readers[r] = []
    # A variable-latency unit op with no write barrier (ptxas times it by distance, e.g. a dead MUFU
    # whose register is later overwritten) keeps at least ptxas's original cycle distance to every
    # later reader/writer of its destination and to the block end.
    orig_t, tt = [], 0
    for k in range(n):
        orig_t.append(tt)
        tt += ins[k].stall or 32
    for u in range(n):
        if ins[u].wbar == 7 and _base(ins[u].opcode).startswith(VARUNIT):
            for v in range(u + 1, n):
                if res[u][0] & (res[v][0] | res[v][1]):
                    preds[v].append((u, orig_t[v] - orig_t[u], None))
            if term is not None:
                preds[term].append((u, orig_t[term] - orig_t[u], None))
    # the terminator waits for everything, and for every fixed-latency result (exit rule)
    if term is not None:
        for u in body:
            preds[term].append((u, latency(ins[u], table) if _fixed_producer(ins[u], res[u]) else 1, None))
    # priority: longest latency path to the end
    succ = {k: [] for k in range(n)}
    for v in range(n):
        for u, d, _ in preds[v]:
            succ[u].append(v)
    prio = [0] * n
    for u in reversed(range(n)):
        own = var_latency(ins[u]) if ins[u].wbar != 7 else latency(ins[u], table)
        prio[u] = own + max((prio[v] for v in succ[u]), default=0)
    issued: dict[int, int] = {}
    port_free: dict[str, int] = {}
    order: list[int] = []
    t = 0
    left = set(range(n))
    while left:
        ready = []
        for v in left:
            if v == term and len(left) > 1:
                continue
            if all(u in issued for u, _, _ in preds[v]):
                # crit picks the order by EXPECTED completion: a consumer of a scoreboarded result
                # is not "ready" until the producer is likely done, or the warp just sits at the wait
                earliest = max([issued[u] + (max(d, var_latency(ins[u])) if kind == "w" and policy == "crit" else d)
                                for u, d, kind in preds[v]] + [t])
                ready.append((earliest, v))
        now = [v for e, v in ready if e <= t]
        if policy == "identity":                         # strict program order: an audit, not a schedule
            nxt = min(left)
            now = [v for v in now if v == nxt]
        if not now:
            if policy == "identity":
                t = max(t + 1, next(e for e, v in ready if v == min(left)))
            else:
                t = max(t + 1, min(e for e, _ in ready))
            continue
        free = [k for k in now if port_free.get(_port(ins[k])[0], 0) <= t]
        if free and policy == "crit":
            now = free                                    # don't stall a pipe that is still busy
        if policy == "random":
            v = rng.choice(now)
        elif policy == "identity":
            v = min(now)
        else:
            v = max(now, key=lambda k: (prio[k], -k))
        port, cost = _port(ins[v])
        issued[v] = max(t, port_free.get(port, 0))
        port_free[port] = issued[v] + cost
        order.append(v)
        left.discard(v)
        t = issued[v] + 1
    # encode: replay the chosen order with only the hard requirements (waits do the blocking)
    issued, pf, prev = {}, {}, None
    for v in order:
        e = max([issued[u] + d for u, d, _ in preds[v]] + [issued[prev] + 1 if prev is not None else 0])
        port, cost = _port(ins[v])
        issued[v] = max(e, pf.get(port, 0))
        pf[port] = issued[v] + cost
        prev = v
    # stalls = issue gaps; the block's last instruction also covers the exit rule
    stalls = []
    for a, b in zip(order, order[1:]):
        stalls.append(issued[b] - issued[a])
    last = order[-1]
    if term is None:
        done = max([issued[u] + latency(ins[u], table) for u in body if _fixed_producer(ins[u], res[u])]
                   + [issued[u] + orig_t[-1] + (ins[-1].stall or 32) - orig_t[u] for u in body
                      if ins[u].wbar == 7 and _base(ins[u].opcode).startswith(VARUNIT)]
                   + [issued[last] + 1])
        st_last = done - issued[last]
        if ins[last].wbar != 7 or ins[last].rbar != 7:
            st_last = max(st_last, 2)                 # the next block may wait on it; a barrier arms late
        stalls.append(st_last)
    else:
        stalls.append(ins[term].stall)                      # keep ptxas's branch stall
    if any(s > 15 for s in stalls):
        raise ValueError(f"stall gap {max(stalls)} > 15 in block")
    # External = an original wait on a barrier that no earlier instruction of the block armed (it
    # protects a value from a predecessor, or drains for a successor). Those waits all go on the
    # block's FIRST issued instruction, so external values are resolved before anything here can
    # re-arm the same barrier (which would make the wait a false dependence on an in-block op).
    armed, entry = 0, 0
    for v in range(n):
        entry |= ins[v].wait & ~armed
        for bar in (ins[v].wbar, ins[v].rbar):
            if bar != 7:
                armed |= 1 << bar
    # Write barriers are reallocated in the new order (ptxas's numbering assumed its order: shared
    # barriers become false waits once ops move). Producers still in flight at the ORIGINAL block
    # exit keep their barrier: a later block waits on that number.
    orig_w = {v: ins[v].wbar for v in range(n)}
    pinned = {u for u, kind in _pending(range(n), ins, {v: ins[v].wait for v in range(n)}, orig_w)
              if kind == "w"}
    cands = [b_ for b_ in range(6) if (armed | entry) >> b_ & 1] or list(range(6))
    outstanding = {b_: set() for b_ in range(6)}
    new_w, waits = {}, {}
    for v in order:
        m = entry if v == order[0] else 0
        for u, _, kind in preds[v]:
            if kind == "w":
                m |= 1 << new_w[u]
            elif kind == "r":
                m |= 1 << ins[u].rbar
        waits[v] = m
        for b_ in range(6):
            if m >> b_ & 1:
                outstanding[b_].clear()
        if ins[v].rbar != 7:
            outstanding[ins[v].rbar].add(v)
        if ins[v].wbar == 7:
            new_w[v] = 7
            continue
        if v in pinned or policy == "identity":
            b_ = ins[v].wbar
        else:
            free = [c for c in cands if not outstanding[c] and c != ins[v].rbar]
            free = free or [c for c in range(6) if not outstanding[c] and c != ins[v].rbar]
            b_ = free[0] if free else min(cands, key=lambda c: len(outstanding[c]))
        new_w[v] = b_
        outstanding[b_].add(v)
    # Ops in flight at exit in the new order but resolved by exit in the original: successors may
    # rely on that, so the terminator waits on their barriers.
    pend_new = _pending(order, ins, waits, new_w)
    pend_old = _pending(range(n), ins, {v: ins[v].wait for v in range(n)}, orig_w)
    extra = pend_new - pend_old
    if extra:
        if term is None:
            raise ValueError("fallthrough block would leave a scoreboarded op in flight")
        for v, kind in extra:
            waits[term] |= 1 << (new_w[v] if kind == "w" else ins[v].rbar)
    return order, stalls, waits, new_w


def _pending(order, ins, waits, wbar) -> set:
    """Unresolved scoreboard events after `order`: (index, "w") for a result not yet covered by a
    later wait on its write barrier, (index, "r") for operands not yet covered by a wait on its read
    barrier (or write barrier: completion implies the reads happened)."""
    pend = set()
    for v in order:
        w = waits[v]
        keep = set()
        for u, kind in pend:
            bars = {wbar[u]} if kind == "w" else {ins[u].rbar, wbar[u]}
            if not any(b != 7 and w >> b & 1 for b in bars):
                keep.add((u, kind))
        pend = keep
        if wbar[v] != 7:
            pend.add((v, "w"))
        if ins[v].rbar != 7:
            pend.add((v, "r"))
    return pend


def _write(elf: bytearray, sec, idx: int, lo: int, hi: int):
    struct.pack_into("<QQ", elf, sec.offset + 16 * idx, lo, hi)


def _reuse_flags(prev: sass.Instr, nxt: sass.Instr, cache: dict, reuse_ops: set) -> int:
    if prev.opcode not in reuse_ops or nxt.opcode not in reuse_ops or prev.yield_ == 0:
        return 0
    a, b = opnd.operands(prev, cache), opnd.operands(nxt, cache)
    if a.wide or b.wide or not a.complete or not b.complete:
        return 0
    defs = reg_use(prev, cache).defs
    flags = 0
    for slot, reg in a.srcs:
        if reg != 255 and reg not in defs and (slot, reg) in b.srcs and slot in REUSE_SLOTS:
            flags |= REUSE_SLOTS[slot]
    return flags


def reschedule(cubin: bytes, kernel: str = "k", policy: str = "crit", seed: int = 0,
               only_loops: bool = False, validate: bool = True, reuse: bool = True) -> tuple[bytes, list]:
    """Reschedule every safe basic block. Returns (cubin, log)."""
    reuse_on = reuse
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    table = fixed_latency_table(ins, cache)
    reuse_ops = {x.opcode for x in ins if x.reuse}
    rng = random.Random(seed)
    reg_bars: dict[str, int] = {}                 # register -> mask of barriers any scoreboarded op uses for it
    for x in ins:
        rr = _resources(x, cache)
        if rr is None:
            continue
        if x.wbar != 7:
            for d in rr[0] | rr[1]:
                reg_bars[d] = reg_bars.get(d, 0) | 1 << x.wbar
        if x.rbar != 7:
            for u in rr[1]:
                reg_bars[u] = reg_bars.get(u, 0) | 1 << x.rbar
    elf = bytearray(cubin)
    sec = patch.text_section(elf, kernel)
    loop_idx = set()
    if only_loops:
        for body in sass.loops(ins):
            loop_idx |= {ins.index(x) for x in body}
    log = []
    for (s, e) in blocks(ins):
        blk = ins[s:e]
        res = [_resources(x, cache) for x in blk]
        unsafe = [x.text for x, r in zip(blk, res) if r is None or (_base(x.opcode).startswith(UNSAFE) and
                                                                    not x.text.strip().endswith("SRZ"))]
        movable = not unsafe and len(blk) >= 3 and (not only_loops or s in loop_idx)
        sched = None
        if movable:
            try:
                sched = schedule_block(blk, res, table, policy, rng, reg_bars)
            except ValueError as err:
                unsafe = [str(err)]
        if sched is None:
            # keep the order, but enforce the exit rule on the last stall before leaving the block
            t, ready = 0, 0
            for k, x in enumerate(blk):
                if k == len(blk) - 1:
                    break
                if _fixed_producer(x, res[k]):
                    ready = max(ready, t + latency(x, table))
                t += x.stall or 32
            last = blk[-1]
            if is_control(last) and len(blk) >= 2:
                need = ready - t
                if need > 0:
                    k = len(blk) - 2
                    while need > 0 and k >= 0:
                        add = min(need, 15 - blk[k].stall)
                        if add > 0 and not _memory(blk[k]):
                            hi = _set_stall(struct.unpack_from("<Q", elf, sec.offset + blk[k].offset + 8)[0],
                                            blk[k].stall + add)
                            struct.pack_into("<Q", elf, sec.offset + blk[k].offset + 8, hi)
                            need -= add
                        k -= 1
                    log.append((s, "kept order, padded exit"))
            elif not is_control(last):
                if _fixed_producer(last, res[-1]):
                    ready = max(ready, t + latency(last, table))
                need = ready - t
                if need > last.stall and not _memory(last):
                    hi = _set_stall(last.hi, min(15, need))
                    struct.pack_into("<Q", elf, sec.offset + last.offset + 8, hi)
                    log.append((s, "kept order, padded fallthrough"))
            if unsafe and len(blk) >= 3:
                log.append((s, f"kept order: {unsafe[0]}"))
            continue
        order, stalls, waits, new_w = sched
        for pos in range(len(order) - 1):            # a barrier arms one cycle after issue
            a_, b_ = order[pos], order[pos + 1]
            mask = waits[b_]
            if any(bar != 7 and mask >> bar & 1 for bar in (new_w[a_], blk[a_].rbar)) and stalls[pos] < 2:
                stalls[pos] = 2
        new = [blk[k] for k in order]
        for pos, (x, st) in enumerate(zip(new, stalls)):
            hi = x.hi
            hi = patch.set_field(hi, "wait", waits[order[pos]])
            hi = patch.set_field(hi, "wbar", new_w[order[pos]])
            reuse = 0
            if reuse_on and pos + 1 < len(new):
                reuse = _reuse_flags(x, new[pos + 1], cache, reuse_ops)
            hi = _set_stall(patch.set_field(hi, "reuse", reuse), st)
            _write(elf, sec, (blk[0].offset // 16) + pos, x.lo, hi)
        moved = sum(1 for a, b in zip(order, range(len(order))) if a != b)
        log.append((s, f"scheduled {len(blk)} instrs, {moved} moved"))
    out = bytes(elf)
    if validate:
        toolchain.disassemble(out)                          # nvdisasm doubles as an encoding validator
    return out, log
