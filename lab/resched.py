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

import dataclasses
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
# with wide=True these may move: memory ops keep their order whenever either side writes memory
WIDE_OK = ("ST", "STS", "STG", "STL", "ATOM", "ATOMS", "ATOMG", "RED", "S2R")
_TIMER = re.compile(r"SR_(CLOCK|GLOBALTIMER)")


def _writes_mem(i: sass.Instr) -> bool:
    return _base(i.opcode).startswith(("ST", "ATOM", "RED", "SU"))
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


PRED_BRANCH = 13
# BAR.SYNC.DEFER_BLOCKING -> first memory op: ptxas never leaves less than 6 cycles (66 barriers over
# the suite + PolyBench); 2 gave wrong results on the suite's layernorm (Oct 18 2026)
BAR_MEM = 6


def latency(i: sass.Instr, table: dict[str, int], consumer: sass.Instr | None = None) -> int:
    """Cycles from fixed-latency producer i until a dependent consumer may issue (no interlock).
    Measured (pair_latency): 4 within a pipe, 5 across the FMA/ALU pipes. Into SHFL pair_latency
    said 4, but FADD -> SHFL at 4 gave wrong reductions on the suite's spmv_csr and layernorm (Oct 18
    2026; ptxas always leaves 5), so SHFL counts as another pipe. Unmeasured producers: the smallest
    distance ptxas itself used, but never below 5 (consumer pipe unknown)."""
    if consumer is not None and is_control(consumer):
        # a predicate read by a branch/EXIT: ptxas never leaves less than 13 (183 of 240 pairs over
        # the suite + PolyBench at exactly 13, none below); 12 broke spmv_csr (Oct 18 2026)
        return PRED_BRANCH
    p = pipe(i.opcode)
    if p is not None:
        if consumer is None:
            return 5
        if pipe(consumer.opcode) == p:
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
    if _memory(i):                     # ptxas spaces memory ops >= 4 cycles; 1 gave wrong results
        cost = max(cost, 4)            # (suite transpose, Oct 18 2026; lab/verify.py MEM_GAP)
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
                   rng: random.Random | None = None, reg_bars: dict | None = None,
                   bar_alloc: str = "first", keep_bars: bool = False, self_loop: bool = False,
                   soft_war: bool = False, sticky_waits: bool = False, wide: bool = False,
                   mem_floor: int = 0):
    """Returns (order, stalls, waits): the new instruction order (indices into ins), each one's stall,
    and each one's extra wait mask. ins/res are the block's instructions and resources. mem_floor:
    no memory op issues before this cycle (a block entered from a barrier, see BAR_MEM)."""
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
                    if ins[u].rbar == 7 and soft_war:          # checked after scheduling (rb_fixpoint)
                        preds[v].append((u, 1, None))
                    else:
                        preds[v].append((u, 2, "r" if ins[u].rbar != 7 else "w"))
                else:
                    preds[v].append((u, 1, None))
        for r in uv:
            readers.setdefault(r, []).append(v)
        for r in dv:
            last_def[r] = v
            readers[r] = []
    if wide:                                                 # memory order: any pair with a writer
        for v in range(n):
            if not _memory(ins[v]):
                continue
            for u in range(v):
                if _memory(ins[u]) and (_writes_mem(ins[u]) or _writes_mem(ins[v])):
                    preds[v].append((u, 1, None))
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
    # An op that keeps ptxas's barrier number (a read barrier always; a write barrier when the op is
    # still in flight at the block exit, e.g. a load prefetched for the next trip of a loop) must not
    # move above an instruction that waited on that number before it in ptxas's order: that wait
    # would then also wait for this op (a counter), a false dependence costing the op's full latency.
    if keep_bars:
        pin0 = {u for u, kind in _pending(range(n), ins, {v: ins[v].wait for v in range(n)},
                                          {v: ins[v].wbar for v in range(n)}) if kind == "w"}
        for u in range(n):
            bars = ([ins[u].rbar] if ins[u].rbar != 7 else []) + ([ins[u].wbar] if u in pin0 else [])
            for b_ in bars:
                for v in range(u):
                    if ins[v].wait >> b_ & 1 and v != term:
                        preds[u].append((v, 1, None))
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
    if self_loop:
        # a value defined here and read earlier in the block feeds the NEXT trip: the producer inherits
        # its reader's priority (else a prefetch for the next trip has no in-block consumer, ranks
        # lowest and sinks to the loop end, shrinking the prefetch distance ptxas built)
        first_read: dict[str, int] = {}
        defined: set = set()
        for v in range(n):
            for r in res[v][1]:
                if r not in defined and r not in first_read:
                    first_read[r] = v
            defined |= res[v][0]
        last_def = {}
        for v in range(n):
            for r in res[v][0]:
                last_def[r] = v
        floor = [0] * n
        for r, u in last_def.items():
            c = first_read.get(r)
            if c is not None and c < u:
                own = var_latency(ins[u]) if ins[u].wbar != 7 else latency(ins[u], table)
                floor[u] = max(floor[u], own + prio[c])
        for u in reversed(range(n)):
            own = var_latency(ins[u]) if ins[u].wbar != 7 else latency(ins[u], table)
            prio[u] = max(floor[u], own + max((prio[v] for v in succ[u]), default=0))
    issued: dict[int, int] = {}
    port_free: dict[str, int] = {}
    order: list[int] = []
    t = 0
    left = set(range(n))
    sb = policy == "sb"
    crit = policy in ("crit", "sb")
    if sb:
        # scoreboards as a scheduled resource: bpend[b] = expected completions of ops armed on b and
        # not yet cleared by a wait on b (a wait blocks until ALL of them are done)
        pin_sb = {u for u, kind in _pending(range(n), ins, {v: ins[v].wait for v in range(n)},
                                            {v: ins[v].wbar for v in range(n)}) if kind == "w"}
        rset = {x.rbar for x in ins if x.rbar != 7}
        sb_cands = [c for c in range(6) if c not in rset] or list(range(6))
        bpend = {b_: [] for b_ in range(6)}
        sb_assign: dict[int, int] = {}

        covered: set = set()

        def wait_bars(v):
            return {sb_assign[u] for u, _, kind in preds[v] if kind == "w" and (u, "w") not in covered} |                    {ins[u].rbar for u, _, kind in preds[v] if kind == "r" and (u, "r") not in covered}

        def pick_bar(v, at):
            if v in pin_sb:
                return ins[v].wbar, 0
            done = at + var_latency(ins[v])
            mine = wait_bars(v)
            best = None
            for b_ in sb_cands:
                if b_ == ins[v].rbar:
                    continue
                pend = [] if b_ in mine else bpend[b_]           # v's own wait clears b_ first
                c = sum(max(0, done - d) for d in pend)
                if best is None or (c, len(pend), b_) < best[0]:
                    best = ((c, len(pend), b_), b_)
            return best[1], best[0][0]
    while left:
        ready = []
        for v in left:
            if v == term and len(left) > 1:
                continue
            if all(u in issued for u, _, _ in preds[v]):
                # crit picks the order by EXPECTED completion: a consumer of a scoreboarded result
                # is not "ready" until the producer is likely done, or the warp just sits at the wait
                earliest = max([issued[u] + (max(d, var_latency(ins[u])) if kind == "w" and crit else d)
                                for u, d, kind in preds[v]] + [t])
                if mem_floor and _memory(ins[v]):
                    earliest = max(earliest, mem_floor)
                if sb:                                       # a wait lasts until its barrier drains
                    for b_ in wait_bars(v):
                        earliest = max([earliest] + bpend[b_])
                ready.append((earliest, v))
        now = [v for e, v in ready if e <= t]
        if sb and now:
            # a scoreboarded op that would delay a pending wait (no barrier free of earlier,
            # sooner-done ops) waits until something else cannot go
            ok = [v for v in now if ins[v].wbar == 7 or pick_bar(v, t)[1] <= 64]
            now = ok or now
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
        if free and crit:
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
        if sb:
            wb = wait_bars(v)
            for b_ in wb:
                bpend[b_] = []
            for u in order:
                if sb_assign.get(u, 7) in wb:
                    covered.update(((u, "w"), (u, "r")))
                if ins[u].rbar in wb:
                    covered.add((u, "r"))
            if ins[v].wbar != 7:
                b_, _ = pick_bar(v, issued[v])
                sb_assign[v] = b_
                bpend[b_].append(issued[v] + var_latency(ins[v]))
            if ins[v].rbar != 7:
                bpend[ins[v].rbar].append(issued[v] + 30)
        order.append(v)
        left.discard(v)
        t = issued[v] + 1
    # encode: replay the chosen order with only the hard requirements (waits do the blocking)
    issued, pf, prev = {}, {}, None
    for v in order:
        e = max([issued[u] + d for u, d, _ in preds[v]] + [issued[prev] + 1 if prev is not None else 0])
        if mem_floor and _memory(ins[v]):
            e = max(e, mem_floor)
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
    if soft_war:                                  # keep read-barrier indices for reads only
        rset = {x.rbar for x in ins if x.rbar != 7}
        cands = [c for c in cands if c not in rset] or [c for c in range(6) if c not in rset] or cands
    outstanding = {b_: set() for b_ in range(6)}
    new_w, waits = {}, {}
    sticky = policy == "sb" or sticky_waits
    resolved: set = set()
    seen: list = []
    if policy == "sb":
        new_w = dict(sb_assign)
    elif bar_alloc == "cost" and policy != "identity":
        new_w = _alloc_barriers(order, ins, preds, issued, pinned, cands, entry)
    for v in order:
        m = entry if v == order[0] else 0
        for u, _, kind in preds[v]:
            if sticky and (u, kind) in resolved:
                continue                                    # an earlier wait already covered it
            if kind == "w":
                m |= 1 << new_w[u]
            elif kind == "r":
                m |= 1 << ins[u].rbar
        waits[v] = m
        if sticky:
            for u in seen:
                for kind, b_ in (("w", new_w.get(u, 7)), ("r", ins[u].rbar)):
                    if b_ != 7 and m >> b_ & 1:
                        resolved.add((u, kind))
                        if kind == "w":
                            resolved.add((u, "r"))         # completion implies its reads happened
            seen.append(v)
        for b_ in range(6):
            if m >> b_ & 1:
                outstanding[b_].clear()
        if ins[v].rbar != 7:
            outstanding[ins[v].rbar].add(v)
        if ins[v].wbar == 7:
            new_w[v] = 7
            continue
        if v in new_w:                                      # cost allocator chose it already
            outstanding[new_w[v]].add(v)
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


def _alloc_barriers(order, ins, preds, issued, pinned, cands, entry) -> dict:
    """Write barriers for the scoreboarded ops of a scheduled block, chosen to minimise false waits.

    A scoreboard is a counter: a wait on barrier b blocks until EVERY op armed on b so far is done.
    With more loads in flight than barriers, sharing is unavoidable, and the first-free rule can put
    a late load on the barrier an early load's consumer waits on (the consumer then waits for the
    late load: 5x more scoreboard stall on the R8U2 down GEMV). Here each op goes, in issue order,
    to the barrier where it adds the least expected delay to the waits still pending on that barrier
    (expected completion = issue + var_latency), and nothing when its completion is earlier."""
    INF = 1 << 30
    pos = {v: i for i, v in enumerate(order)}
    wpos, rpos = {}, {}
    for v in order:
        for u, _, kind in preds[v]:
            if kind == "w":
                wpos[u] = min(wpos.get(u, INF), pos[v])
            elif kind == "r":
                rpos[u] = min(rpos.get(u, INF), pos[v])
    occ = {b: [] for b in range(6)}                 # barrier -> [(arm_pos, done, wait_pos)]

    def pending(b, p):
        waits_b = [w for _, _, w in occ[b] if w < p]
        return [o for o in occ[b] if o[0] < p and o[2] >= p and not any(o[0] < w < p for w in waits_b)]

    new_w = {}
    for v in order:
        p = pos[v]
        if ins[v].wbar != 7:
            done = issued[v] + var_latency(ins[v])
            if v in pinned:
                b = ins[v].wbar
            else:
                best = None
                for b_ in [c for c in cands if c != ins[v].rbar] or [c for c in range(6) if c != ins[v].rbar]:
                    pend = pending(b_, p)
                    cost = sum(max(0, done - d) for _, d, _ in pend)
                    wv = wpos.get(v, INF)
                    cost += sum(max(0, d - done) for a, d, w in pend if w > wv)
                    key = (cost, len(pend), b_)
                    if best is None or key < best[0]:
                        best = (key, b_)
                b = best[1]
            new_w[v] = b
            occ[b].append((p, done, wpos.get(v, INF)))
        if ins[v].rbar != 7:
            occ[ins[v].rbar].append((p, issued[v] + 30, rpos.get(v, INF)))
    return new_w


def _war_short(blk, res, order, stalls) -> set:
    """Variable-latency ops without a read barrier whose source register is now overwritten sooner
    (in issue cycles) than in ptxas's order. ptxas never overwrites such a source within 60 cycles
    without a barrier (GEMV and random kernels), so its own distance is the only evidence of safety."""
    t_new, t = {}, 0
    for k, v in enumerate(order):
        t_new[v] = t
        t += stalls[k]
    t_old, t = [], 0
    for x in blk:
        t_old.append(t)
        t += x.stall or 32
    short = set()
    for u, x in enumerate(blk):
        if not is_var(x) or x.rbar != 7 or res[u] is None:
            continue
        src = set(res[u][1])
        for v in range(u + 1, len(blk)):
            hit = res[v][0] & src
            if hit:
                if t_new[v] - t_new[u] < t_old[v] - t_old[u]:
                    short.add(u)
                    break
                src -= hit
            if not src:
                break
    return short


def _rb_fixpoint(blk, res, table, policy, rng, reg_bars, bar_alloc, keep_bars, loop_blk, sticky=False,
                 mem_floor=0):
    """Schedule with WAR edges on barrier-less variable-latency ops relaxed; an op whose source is
    then overwritten sooner than ptxas allowed gets a read barrier (ptxas's own read-barrier index
    if the block has one, else a barrier the block never uses), and the block is rescheduled with
    that op's WAR edges waiting on it. Up to 4 rounds; still short -> ValueError (keep the order)."""
    used = {x.rbar for x in blk if x.rbar != 7}
    unused = [b for b in range(6) if all(x.wbar != b and x.rbar != b and not x.wait >> b & 1 for x in blk)]
    rb = (min(used) if used else unused[0] if unused else None)
    cur = list(blk)
    for _ in range(4):
        sched = schedule_block(cur, res, table, policy, rng, reg_bars, bar_alloc, keep_bars, self_loop=loop_blk,
                               soft_war=True, sticky_waits=sticky, mem_floor=mem_floor)
        short = _war_short(cur, res, sched[0], sched[1])
        if not short:
            return cur, sched
        if rb is None:
            raise ValueError("WAR too short and no free read barrier")
        cur = [dataclasses.replace(x, hi=patch.set_field(x.hi, "rbar", rb)) if k in short else x
               for k, x in enumerate(cur)]
    raise ValueError("WAR distances did not converge")


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


def _form(i: sass.Instr) -> tuple:
    """Opcode plus operand kinds (register or not): reuse encodings are valid per form, e.g. an FFMA
    with an immediate third operand rejects reuse bits a register-register FFMA accepts."""
    ops = re.sub(r"^@!?U?P[T0-9]+\s+", "", i.text.strip()).rstrip(" ;").split(None, 1)
    toks = [t.strip() for t in ops[1].split(",")] if len(ops) > 1 else []
    return (i.opcode, tuple("R" if re.match(r"^-?\|?R\d", t) else "X" for t in toks))


def _reuse_flags(prev: sass.Instr, nxt: sass.Instr, cache: dict, reuse_ops: dict) -> int:
    if _form(prev) not in reuse_ops or nxt.opcode not in {k[0] for k in reuse_ops} or prev.yield_ == 0:
        return 0
    a, b = opnd.operands(prev, cache), opnd.operands(nxt, cache)
    if a.wide or b.wide or not a.complete or not b.complete:
        return 0
    defs = reg_use(prev, cache).defs
    flags = 0
    for slot, reg in a.srcs:
        if reg != 255 and reg not in defs and (slot, reg) in b.srcs and slot in REUSE_SLOTS:
            flags |= REUSE_SLOTS[slot]
    return flags & reuse_ops[_form(prev)]                  # only bits ptxas used on this form


def reschedule(cubin: bytes, kernel: str = "k", policy: str = "crit", seed: int = 0,
               only_loops: bool = False, validate: bool = True, reuse: bool = True,
               bar_alloc: str = "first", keep_bars: bool = False, loop_aware: bool = False,
               read_bars: bool = False, sticky_waits: bool = False, wide: bool = False,
               only_blocks: set | None = None) -> tuple[bytes, list]:
    """Reschedule every safe basic block (only those starting at an index in only_blocks, if given;
    scripts/suite_bisect.py). Returns (cubin, log)."""
    if policy == "sb_wide":                     # alias: scoreboard-aware order, stores/atomics schedulable
        policy, wide = "sb", True
    reuse_on = reuse
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    table = fixed_latency_table(ins, cache)
    reuse_ops: dict = {}                               # form -> reuse bits ptxas set on it in this cubin
    for x in ins:
        if x.reuse:
            reuse_ops[_form(x)] = reuse_ops.get(_form(x), 0) | x.reuse
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
    self_loops = set()
    for body in sass.loops(ins):
        k0 = ins.index(body[0])
        self_loops.add((k0, k0 + len(body)))
    for (s, e) in blocks(ins):
        blk = ins[s:e]
        res = [_resources(x, cache) for x in blk]
        unsafe = [x.text for x, r in zip(blk, res) if r is None or (_base(x.opcode).startswith(UNSAFE) and
                                                                    not x.text.strip().endswith("SRZ") and
                                                                    not (wide and _base(x.opcode) in WIDE_OK and
                                                                         not _TIMER.search(x.text)))]
        movable = (not unsafe and len(blk) >= 3 and (not only_loops or s in loop_idx)
                   and (only_blocks is None or s in only_blocks))
        sched = None
        if movable:
            loop_blk = loop_aware and any(a <= s and e <= z and (z - a) - (e - s) <= 2 for a, z in self_loops)
            prev = ins[s - 1] if s else None
            floor = max(0, BAR_MEM - (prev.stall or 32)) if prev is not None and _base(prev.opcode) == "BAR" else 0
            try:
                if read_bars:
                    blk, sched = _rb_fixpoint(blk, res, table, policy, rng, reg_bars, bar_alloc, keep_bars, loop_blk,
                                              sticky_waits, mem_floor=floor)
                else:
                    sched = schedule_block(blk, res, table, policy, rng, reg_bars, bar_alloc, keep_bars,
                                           self_loop=loop_blk, sticky_waits=sticky_waits, wide=wide,
                                           mem_floor=floor)
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
            hi = patch.set_field(hi, "rbar", x.rbar)              # _rb_fixpoint may have added one
            reuse = 0
            if reuse_on == "ptxas" and order == list(range(len(order))):
                reuse = x.reuse                            # unchanged order: ptxas's flags stay valid
            elif reuse_on and pos + 1 < len(new):
                reuse = _reuse_flags(x, new[pos + 1], cache, reuse_ops)
            hi = _set_stall(patch.set_field(hi, "reuse", reuse), st)
            _write(elf, sec, (blk[0].offset // 16) + pos, x.lo, hi)
        moved = sum(1 for a, b in zip(order, range(len(order))) if a != b)
        log.append((s, f"scheduled {len(blk)} instrs, {moved} moved"))
    out = bytes(elf)
    if validate:
        toolchain.disassemble(out)                          # nvdisasm doubles as an encoding validator
    return out, log
