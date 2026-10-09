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


def check(cubin: bytes, kernel: str = "k", reference: bytes | None = None) -> list[str]:
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
    out = []
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
                        out.append(f"RAW {u}: {ins[pk].text!r} @{ins[pk].offset:#x} -> {x.text!r} @{x.offset:#x} "
                                   f"after {t - pt} < {need} cycles")
                elif not ok:
                    out.append(f"RAW {u}: scoreboarded {ins[pk].text!r} @{ins[pk].offset:#x} not waited on "
                               f"(bar {pb}) before {x.text!r} @{x.offset:#x}")
        for d in defs:
            if d in last and last[d][3] is not None and not last[d][4] and d not in uses:
                ps, pt, pk, pb, ok = last[d]
                out.append(f"WAW {d}: in-flight {ins[pk].text!r} @{ins[pk].offset:#x} overwritten by "
                           f"{x.text!r} @{x.offset:#x}")
            for rr in readers.get(d, []):
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
    # de-duplicate (the loop is unrolled three times)
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq
