"""Static hazard checker for a (rescheduled) cubin, independent of the scheduler's DAG.

Walks the code in execution order: prologue, the main loop body unrolled three times (so loop-carried
hazards are seen), then the epilogue; not-taken forward branches are ignored. Tracks each register's
last writer and checks every read and write against the hardware rules measured in 3080lab:

  * fixed-latency writer: reader must issue >= latency(writer, reader) cycles later (stall sums);
  * scoreboarded writer (wbar b): some instruction at or after the writer, up to the reader, must
    wait on b, at least 2 cycles after the writer issued (waits are sticky once passed);
  * WAW against an in-flight scoreboarded write needs the same wait;
  * WAR against a scoreboarded reader that holds a read barrier needs a wait on that barrier.
"""
from __future__ import annotations

import re

from . import operands as opnd
from . import sass, toolchain
from .resched import _fixed_producer, _resources, fixed_latency_table, is_control, latency


def _path(ins: list[sass.Instr]) -> list[int]:
    body = sass.loop_body(ins)
    if not body:
        return list(range(len(ins)))
    s, e = ins.index(body[0]), ins.index(body[-1])
    tail = []
    for k in range(e + 1, len(ins)):
        tail.append(k)
        if ins[k].opcode == "EXIT" and not ins[k].text.startswith("@"):
            break
    return list(range(0, s)) + list(range(s, e + 1)) * 3 + tail


def check(cubin: bytes, kernel: str = "k", reference: bytes | None = None, structured: bool = False) -> list:
    """reference: the original ptxas cubin, whose stall distances calibrate unmeasured opcodes (the
    same table the scheduler used)."""
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    ref = sass.parse(toolchain.disassemble(reference)) if reference else ins
    table = fixed_latency_table(ref, opnd.discover(reference, kernel) if reference else cache)
    res = [_resources(x, cache) for x in ins]
    last = {}                 # resource -> (step, t, instr index, barrier or None, resolved?)
    readers = {}              # resource -> [(step, t, idx, rbar)] pending scoreboarded reads
    armed = {b: [] for b in range(6)}   # barrier -> [(t, resource keys)]
    out, recs = [], []
    t = 0
    for step, k in enumerate(_path(ins)):
        x = ins[k]
        # waits resolve everything armed on those barriers at least 2 cycles ago
        for b in range(6):
            if x.wait >> b & 1:
                keep = []
                for (tp, keys, kind) in armed[b]:
                    if t - tp >= 2:
                        for key in keys:
                            if kind == "w" and key in last and last[key][3] == b and last[key][1] == tp:
                                last[key] = last[key][:4] + (True,)
                            if kind == "w":                 # completion implies its operands were read
                                for rk in list(readers):
                                    readers[rk] = [q for q in readers[rk] if q[1] != tp]
                            if kind == "r":
                                readers[key] = [r for r in readers.get(key, []) if not (r[3] == b and r[1] == tp)]
                    else:
                        keep.append((tp, keys, kind))
                armed[b] = keep
        r = res[k]
        if is_control(x):
            r = (set(), {f"P{p}" for p in re.findall(r"(?<!U)P([0-6])", x.text)})
        if r is None:
            t += x.stall or 32
            continue
        defs, uses = r
        for u in uses:
            if u in last:
                ps, pt, pk, pb, ok = last[u]
                if pb is None:
                    need = latency(ins[pk], table, x)
                    if t - pt < need:
                        recs.append(("RAW", ins[pk].offset, x.offset, t - pt, need))
                        out.append(f"RAW {u}: {ins[pk].text!r} @{ins[pk].offset:#x} -> {x.text!r} @{x.offset:#x} "
                                   f"after {t - pt} < {need} cycles")
                elif not ok:
                    recs.append(("RAW-SB", ins[pk].offset, x.offset, None, None))
                    out.append(f"RAW {u}: scoreboarded {ins[pk].text!r} @{ins[pk].offset:#x} not waited on "
                               f"(bar {pb}) before {x.text!r} @{x.offset:#x}")
        for d in defs:
            if d in last and last[d][3] is not None and not last[d][4] and d not in uses:
                ps, pt, pk, pb, ok = last[d]
                recs.append(("WAW", ins[pk].offset, x.offset, None, None))
                out.append(f"WAW {d}: in-flight {ins[pk].text!r} @{ins[pk].offset:#x} overwritten by "
                           f"{x.text!r} @{x.offset:#x}")
            for rr in readers.get(d, []):
                recs.append(("WAR", ins[rr[2]].offset, x.offset, None, None))
                out.append(f"WAR {d}: pending read by {ins[rr[2]].text!r} @{ins[rr[2]].offset:#x} clobbered by "
                           f"{x.text!r} @{x.offset:#x}")
        var = x.wbar != 7
        for d in defs:
            last[d] = (step, t, k, x.wbar if var else None, False)
        if var:
            armed[x.wbar].append((t, list(defs), "w"))
        if x.rbar != 7:
            for u in uses:
                readers.setdefault(u, []).append((step, t, k, x.rbar))
            armed[x.rbar].append((t, list(uses), "r"))
        if not var and not _fixed_producer(x, r):
            pass
        t += x.stall or 32
    if structured:
        return sorted(set(recs), key=str)
    # de-duplicate (the loop is unrolled three times)
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


_LO_MASK = ~((0xF << 12) | (0xFFFFFF << 16)) & (2**64 - 1)     # guard predicate, Rd, Ra, Rb
_HI_MASK = ~(0xFF | (0x7 << 17) | (0x1FFFFF << 41)) & (2**64 - 1)  # Rc, setp dest, control word


def match(orig: list, new: list) -> dict:
    """new offset -> original offset, matching encodings with register/predicate fields and control
    bits masked (a rewrite only moves words and patches those fields). Duplicates match in order."""
    from collections import defaultdict, deque
    q = defaultdict(deque)
    for x in orig:
        q[(x.lo & _LO_MASK, x.hi & _HI_MASK, x.opcode)].append(x.offset)
    out = {}
    for x in new:
        k = (x.lo & _LO_MASK, x.hi & _HI_MASK, x.opcode)
        if q[k]:
            out[x.offset] = q[k].popleft()
    return out


def new_hazards(new_cubin: bytes, orig_cubin: bytes, kernel: str = "k") -> list:
    """Hazards in the rewrite that the original does not already have for the same instruction
    pair (instructions matched by encoding, so register renames and guard retargets do not break the
    comparison). A pre-existing fixed-latency hazard counts as new only if its distance shrank."""
    oi = sass.parse(toolchain.disassemble(orig_cubin))
    ni = sass.parse(toolchain.disassemble(new_cubin))
    mp = match(oi, ni)
    old = {}
    for kind, p, c, dist, need in check(orig_cubin, kernel, reference=orig_cubin, structured=True):
        old[(kind, p, c)] = min(dist if dist is not None else -1, old.get((kind, p, c), 10**9))
    bad = []
    for kind, p, c, dist, need in check(new_cubin, kernel, reference=orig_cubin, structured=True):
        key = (kind, mp.get(p), mp.get(c))
        if None not in key[1:] and key in old and (dist is None or dist >= old[key]):
            continue
        by = {x.offset: x for x in ni}
        bad.append(f"{kind}: {by[p].text!r} @{p:#x} -> {by[c].text!r} @{c:#x}"
                   + (f" after {dist} < {need} cycles" if dist is not None else ""))
    return bad
