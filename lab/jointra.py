"""Joint register allocation + scheduling, minimal prototype: split hot-loop live ranges into spare
registers to remove false (WAR/WAW) dependences, then reschedule with the scoreboard-aware policy.

ptxas reuses register numbers aggressively inside unrolled loops (e.g. every LDG of a 4x unrolled
loop writes the same R10). Each reuse adds a write-after-read edge: the next load may not issue
until the previous value's last reader has, so loads that could overlap are serialized. A web
(lab/regalloc.webs: the definitions that reach common uses) can take any register that no
interfering web holds; giving the hot webs that share a register their own fresh registers removes
those edges, and lab/resched rebuilds the dependence DAG from the renamed code.

  split(cubin, extra)   recolor: each hot-loop web that shares its register with another hot web
                        gets a fresh register, from registers no web uses, up to (highest used + extra).
                        Webs written by variable-latency ops (loads, MUFU, SHFL) go first.
  optimize(cubin, ...)  split, then lab/resched.reschedule(policy="sb", late_entry, sticky_waits).

Safety is inherited: regalloc.apply proves every use keeps the same reaching definitions and that no
non-register bit changed; reschedule validates hazards (lab/verify) against the original.
"""
from __future__ import annotations

from collections import Counter

from . import regalloc, resched, sass

VARLAT = ("LDG", "LD", "LDS", "MUFU", "SHFL", "S2R", "LDC")


def split(cubin: bytes, kernel: str = "k", extra: int = 8) -> tuple[bytes, dict]:
    g = regalloc.build(cubin, kernel)
    W = regalloc.webs(g)
    E = regalloc.interference(g, W)
    body = sass.loop_body(g["ins"])
    body_off = {i.offset for i in body}
    color = {w: w[1] for w in W["members"]}
    free = [w for w in W["members"] if w not in W["pinned"]]
    hot = [w for w in free if any(g["ins"][o.ins].offset in body_off for o in W["members"][w])]
    used = set(color.values()) | {o.reg + j for lst in g["occs"] for o in lst for j in range(o.width)}
    top = max(r for r in used if r != 255)
    pool = [r for r in range(2, top + extra + 1) if r not in used]
    share = Counter(color[w] for w in hot)

    def first_def(w):
        ks = [o.ins for o in W["members"][w] if o.is_def]
        return min(ks) if ks else 10 ** 9

    def var(w):
        return any(o.is_def and g["ins"][o.ins].opcode.startswith(VARLAT) for o in W["members"][w])

    todo = sorted((w for w in hot if share[color[w]] > 1), key=lambda w: (not var(w), first_def(w)))
    moved = 0
    for w in todo:
        if not pool or share[color[w]] <= 1:
            continue
        c = next((r for r in pool if all(color[x] != r for x in E[w])), None)
        if c is None:
            continue
        pool.remove(c)
        share[color[w]] -= 1
        color[w] = c
        share[c] += 1
        moved += 1
    info = {"hot_webs": len(hot), "shared_hot": len(todo), "moved": moved, "top_before": top,
            "top_after": max(c for c in color.values() if c != 255)}
    if not moved:
        return cubin, info
    out = regalloc.apply(cubin, {"g": g, "W": W, "color": color}, kernel)
    return out, info


def _crit(g, W, body_k, color, cache, table):
    """Longest path through the loop body (program-order DAG: RAW at resched latencies, WAR/WAW 1,
    memory order kept) under a register coloring. Returns (length, false edges on the path), each
    false edge as (earlier k, later k, web of the later definition)."""
    ins = g["ins"]
    regs = []
    for k in body_k:
        d, u = {}, {}
        for o in g["occs"][k]:
            w = W["occ_web"][id(o)]
            for j in range(o.width):
                (d if o.is_def else u)[color[w] + j] = w
        res = resched._resources(ins[k], cache)
        other = (set(), set()) if res is None else ({x for x in res[0] if x[0] != "R"},
                                                    {x for x in res[1] if x[0] != "R"})
        regs.append(({f"R{r}": w for r, w in d.items()} | {x: None for x in other[0]},
                     {f"R{r}": w for r, w in u.items()} | {x: None for x in other[1]}))
    n = len(body_k)
    t, back = [0] * n, [None] * n
    last_def, readers = {}, {}
    for v in range(n):
        iv = ins[body_k[v]]
        cand = [(0, None)]
        if resched._memory(iv):
            for p in range(v):
                ip = ins[body_k[p]]
                if resched._memory(ip) and (resched._writes_mem(ip) or resched._writes_mem(iv)):
                    cand.append((t[p] + 1, (p, None)))
        d, u = regs[v]
        for x in u:
            if x in last_def:
                p = last_def[x]
                ip = ins[body_k[p]]
                lat = resched.var_latency(ip) if ip.wbar != 7 else resched.latency(ip, table, iv)
                cand.append((t[p] + lat, (p, None)))
        for x, w in d.items():
            if x in last_def:
                cand.append((t[last_def[x]] + 1, (last_def[x], w)))
            for p in readers.get(x, ()):
                cand.append((t[p] + 1, (p, w)))
        t[v], back[v] = max(cand, key=lambda c: c[0])
        for x in u:
            readers.setdefault(x, []).append(v)
        for x in d:
            last_def[x], readers[x] = v, []
    v = max(range(n), key=lambda i: t[i]) if n else 0
    false = []
    while n and back[v] is not None:
        p, w = back[v]
        if w is not None:
            false.append((body_k[p], body_k[v], w))
        v = p
    return (max(t) + 1 if n else 0), false


def split_crit(cubin: bytes, kernel: str = "k", extra: int = 8, max_moves: int = 64) -> tuple[bytes, dict]:
    """Rename only webs whose definition ends a false edge on the loop's critical path; repeat until
    the path has no false edge, the pool is empty, or the path stops shrinking."""
    from . import operands as opnd
    g = regalloc.build(cubin, kernel)
    W = regalloc.webs(g)
    E = regalloc.interference(g, W)
    cache = opnd._load_cache()
    body = sass.loop_body(g["ins"])
    off = {x.offset: k for k, x in enumerate(g["ins"])}
    body_k = [off[x.offset] for x in body]
    table = resched.fixed_latency_table(body, cache)
    color = {w: w[1] for w in W["members"]}
    used = set(color.values()) | {o.reg + j for lst in g["occs"] for o in lst for j in range(o.width)}
    top = max(r for r in used if r != 255)
    pool = [r for r in range(2, top + extra + 1) if r not in used]
    cp0, false = _crit(g, W, body_k, color, cache, table)
    cp, moved, tried = cp0, 0, set()
    while false and pool and moved < max_moves:
        cands = [w for _, _, w in false if w not in W["pinned"] and w not in tried]
        if not cands:
            break
        w = cands[0]
        tried.add(w)
        c = next((r for r in pool if all(color[x] != r for x in E[w])), None)
        if c is None:
            continue
        old = color[w]
        color[w] = c
        cp1, false1 = _crit(g, W, body_k, color, cache, table)
        if cp1 < cp:
            pool.remove(c)
            cp, false, moved = cp1, false1, moved + 1
        else:
            color[w] = old
    info = {"cp_before": cp0, "cp_after": cp, "moved": moved, "top_before": top,
            "top_after": max(c for c in color.values() if c != 255)}
    if not moved:
        return cubin, info
    return regalloc.apply(cubin, {"g": g, "W": W, "color": color}, kernel), info


def optimize(cubin: bytes, kernel: str = "k", extra: int = 8, policy: str = "sb") -> tuple[bytes, dict]:
    c, info = split(cubin, kernel, extra)
    out, log = resched.reschedule(c, kernel, policy=policy, late_entry=True, sticky_waits=True)
    info["resched_log"] = len(log)
    return out, info
