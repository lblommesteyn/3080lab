"""Live-range (web) register renaming for compiled sm_86 kernels.

ptxas reuses register numbers for unrelated values, so renaming by register
*number* is blocked whenever any one of those values touches a wide operand.
Here each register is split into webs (maximal sets of definitions that reach
common uses); webs are recolored independently subject to interference, to
minimize register-bank conflicts in the hot loop.

Pipeline: CFG -> reaching definitions -> webs -> liveness -> interference ->
annealed coloring -> rewrite -> proof. The proof recomputes reaching
definitions on the rewritten kernel: every use must see exactly the same set
of defining instructions as before, which is semantic equivalence for a pure
renaming (opcodes and non-register bits are untouched, checked by text).
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass

from . import operands as opnd
from . import patch, rename, sass, toolchain

ENTRY = -1  # pseudo-definition for registers live into the kernel


@dataclass
class Occ:
    ins: int         # instruction index
    tok: int         # register token index in the instruction text
    reg: int
    is_def: bool
    width: int       # 1, 2, 4 (wide occurrences pin their web)
    mapped: bool     # token has a discovered encoding field (renamable)


def _branch_target(text: str):
    m = re.search(r"`\((\.L_x_\d+)\)", text)
    return m.group(1) if m else None


def build(cubin: bytes, kernel: str = "k") -> dict:
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    n = len(ins)
    label_at = {i.label: k for k, i in enumerate(ins) if i.label}
    # ---- occurrences ----
    occs: list[list[Occ]] = []
    for k, i in enumerate(ins):
        toks = opnd.reg_tokens(i.text)
        info = cache.get(opnd.form_key(i), {"fields": {}})
        f2t = info["fields"]
        widths = rename.token_widths(i)
        dest_tok = f2t.get("d")
        predicated = bool(re.match(r"^@!?P", i.text.strip()))
        lst = []
        for t, tok in enumerate(toks):
            if tok == "RZ":
                continue
            r = int(tok[1:])
            w = None if widths is None else widths[t]
            mapped = t in f2t.values() and w is not None
            is_def = (t == dest_tok)
            if w is None or not info["fields"]:
                # unknown instruction: treat every register as both used and defined, pinned
                lst.append(Occ(k, t, r, False, 2, False))
                lst.append(Occ(k, t, r, True, 2, False))
                continue
            regs = range(r, r + w) if w > 1 else [r]
            for rr in regs:
                if is_def:
                    if predicated:
                        lst.append(Occ(k, t, rr, False, w, mapped))  # predicated def keeps old value live
                    lst.append(Occ(k, t, rr, True, w, mapped))
                else:
                    lst.append(Occ(k, t, rr, False, w, mapped))
        occs.append(lst)
    # ---- CFG ----
    succ = [[] for _ in range(n)]
    for k, i in enumerate(ins):
        op = i.opcode
        pred = bool(re.match(r"^@!?P", i.text.strip()))
        tgt = _branch_target(i.text)
        if op in ("BRA", "CALL.REL.NOINC", "CALL.REL"):
            if tgt in label_at:
                succ[k].append(label_at[tgt])
            if pred and k + 1 < n:
                succ[k].append(k + 1)
        elif op in ("EXIT", "RET", "RET.REL", "RET.REL.NODEC"):
            if pred and k + 1 < n:
                succ[k].append(k + 1)
        elif k + 1 < n:
            succ[k].append(k + 1)
    pred_of = [[] for _ in range(n)]
    for k, ss in enumerate(succ):
        for s in ss:
            pred_of[s].append(k)
    return {"ins": ins, "occs": occs, "succ": succ, "pred": pred_of, "cache": cache}


def reaching(g: dict, reg_of=None) -> dict:
    """For each (ins, reg) use: frozenset of defining instruction indices (ENTRY if live-in)."""
    n = len(g["ins"])
    occs = g["occs"]
    reg_of = reg_of or (lambda o: o.reg)
    gen = [dict() for _ in range(n)]   # reg -> def index produced at k
    for k in range(n):
        for o in occs[k]:
            if o.is_def:
                gen[k][reg_of(o)] = k
    IN = [defaultdict(frozenset) for _ in range(n)]
    OUT = [defaultdict(frozenset) for _ in range(n)]
    work = list(range(n))
    entry = defaultdict(lambda: frozenset({ENTRY}))
    while work:
        k = work.pop()
        new_in = defaultdict(frozenset)
        sources = [OUT[p] for p in g["pred"][k]] or [entry]
        regs = set()
        for s in sources:
            regs.update(s.keys())
        if not g["pred"][k]:
            regs.update(r for o in occs[k] for r in [reg_of(o)])
        for r in regs:
            acc = frozenset()
            for s in sources:
                acc |= s[r]
            new_in[r] = acc
        new_out = defaultdict(frozenset, new_in)
        for r, d in gen[k].items():
            new_out[r] = frozenset({d})
        IN[k] = new_in
        if new_out != OUT[k]:
            OUT[k] = new_out
            work.extend(g["succ"][k])
    uses = {}
    for k in range(n):
        for o in occs[k]:
            if not o.is_def:
                r = reg_of(o)
                d = IN[k][r] if r in IN[k] else frozenset({ENTRY})
                uses[(k, o.tok, o.reg)] = d
    return uses


def webs(g: dict) -> dict:
    """Union-find over definitions: defs reaching a common use share a web."""
    uses = reaching(g)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for k, lst in enumerate(g["occs"]):
        for o in lst:
            if o.is_def:
                find((k, o.reg))
    for (k, tok, reg), ds in uses.items():
        ds = sorted(ds)
        keys = [(d, reg) for d in ds]
        for key in keys:
            find(key)
        for a, b in zip(keys, keys[1:]):
            union(a, b)
    web_of_def = {key: find(key) for key in parent}
    # occurrence -> web id
    occ_web = {}
    for k, lst in enumerate(g["occs"]):
        for o in lst:
            if o.is_def:
                occ_web[id(o)] = web_of_def[(k, o.reg)]
            else:
                ds = uses[(k, o.tok, o.reg)]
                occ_web[id(o)] = web_of_def[(min(ds), o.reg)]
    members = defaultdict(list)
    for k, lst in enumerate(g["occs"]):
        for o in lst:
            members[occ_web[id(o)]].append(o)
    pinned = {w for w, os in members.items()
              if any((not o.mapped) or o.width > 1 or o.reg == 1 for o in os) or w[0] == ENTRY}
    return {"occ_web": occ_web, "members": dict(members), "pinned": pinned, "uses": uses}


def interference(g: dict, W: dict) -> dict:
    """Webs live at the same point interfere. Liveness per web via backward dataflow."""
    n = len(g["ins"])
    occs, occ_web = g["occs"], W["occ_web"]
    use_w = [set(occ_web[id(o)] for o in occs[k] if not o.is_def) for k in range(n)]
    def_w = [set(occ_web[id(o)] for o in occs[k] if o.is_def) for k in range(n)]
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for k in range(n - 1, -1, -1):
            out = set().union(*(live_in[s] for s in g["succ"][k])) if g["succ"][k] else set()
            inn = use_w[k] | (out - def_w[k])
            if out != live_out[k] or inn != live_in[k]:
                live_out[k], live_in[k] = out, inn
                changed = True
    edges = defaultdict(set)
    for k in range(n):
        group = live_out[k] | def_w[k]
        for a in group:
            for b in group:
                if a != b:
                    edges[a].add(b)
    return edges


def optimize(cubin: bytes, kernel: str = "k", max_reg: int | None = None, steps: int = 6000,
             seed: int = 0) -> dict:
    g = build(cubin, kernel)
    W = webs(g)
    E = interference(g, W)
    body = sass.loop_body(g["ins"])
    body_idx = {i.offset for i in body}
    used = {o.reg for lst in g["occs"] for o in lst}
    max_reg = max_reg if max_reg is not None else max(used)
    color = {w: w[1] for w in W["members"]}  # web id = (def index, original reg)
    free = [w for w in W["members"] if w not in W["pinned"]]
    # only webs that appear in the hot loop matter for the objective
    hot = [w for w in free if any(g["ins"][o.ins].offset in body_idx for o in W["members"][w])]
    occ_web = W["occ_web"]

    def cost(col):
        total, prev = 0, {}
        for i in body:
            k = next(j for j, x in enumerate(g["ins"]) if x.offset == i.offset) if False else None  # noqa
        return _loop_cost(g, body, occ_web, col)

    def ok(w, c, col):
        if c > max_reg or c == 1:
            return False
        return all(col[x] != c for x in E[w])

    best = cur = cost(color)
    base = cur
    best_col = dict(color)
    rng = random.Random(seed)
    temp = 2.0
    for _ in range(steps if hot else 0):
        w = rng.choice(hot)
        c = rng.randrange(0, max_reg + 1)
        if c == color[w] or not ok(w, c, color):
            continue
        old = color[w]
        color[w] = c
        new = cost(color)
        if new <= cur or rng.random() < math.exp((cur - new) / temp):
            cur = new
            if new < best:
                best, best_col = new, dict(color)
        else:
            color[w] = old
        temp = max(0.05, temp * 0.999)
    return {"g": g, "W": W, "E": E, "base_cost": base, "best_cost": best, "color": best_col,
            "hot_webs": len(hot), "free_webs": len(free), "webs": len(W["members"])}


def _loop_cost(g, body, occ_web, color) -> int:
    """Static register-read cycles for one warp running the hot loop twice (the second
    pass sees the loop-carried reuse state), using the (bank, slot) reuse cache."""
    idx = {x.offset: j for j, x in enumerate(g["ins"])}
    cache = g["cache"]
    rc: dict = {}
    total = 0
    for rep in range(2):
        total = 0
        for i in body:
            k = idx[i.offset]
            info = cache.get(opnd.form_key(i))
            o = opnd.operands(i, cache)
            if o.wide or not o.complete or not info:
                total += 1
                rc.clear()
                continue
            tok_slot = {t: f for f, t in info["fields"].items() if f != "d"}
            srcs = [(tok_slot[oc.tok], color[occ_web[id(oc)]]) for oc in g["occs"][k]
                    if not oc.is_def and oc.tok in tok_slot]
            banks = [0, 0]
            for slot, reg in srcs:
                if reg != 255 and rc.get((reg & 1, slot)) != reg:
                    banks[reg & 1] += 1
            total += max(1, max(banks))
            for slot, reg in srcs:
                if reg == 255:
                    continue
                if i.yield_ and i.reuse >> "abc".index(slot) & 1:
                    rc[(reg & 1, slot)] = reg
                else:
                    rc.pop((reg & 1, slot), None)
    return total


def apply(cubin: bytes, res: dict, kernel: str = "k") -> bytes:
    """Rewrite every occurrence with its web's color; prove equivalence."""
    g, W, color = res["g"], res["W"], res["color"]
    c = cubin
    for k, i in enumerate(g["ins"]):
        info = g["cache"].get(opnd.form_key(i))
        if not info:
            continue
        t2f = {t: f for f, t in info["fields"].items()}
        kw = {}
        for o in g["occs"][k]:
            w = W["occ_web"][id(o)]
            if w in W["pinned"] or color[w] == o.reg:
                continue
            fld = t2f[o.tok]
            kw[{"d": "rd", "a": "ra", "b": "rb", "c": "rc"}[fld]] = color[w]
        if kw:
            c = patch.set_regs(c, kernel, i.offset, **kw)
    need = max(color.values()) + 3
    if need > rename._regcount(c, kernel):
        c = patch.set_regcount(c, kernel, need)
    prove(cubin, c, kernel)
    return c


def prove(orig: bytes, new: bytes, kernel: str = "k"):
    """Every use must see the same defining instructions; opcodes/non-register text unchanged."""
    g0, g1 = build(orig, kernel), build(new, kernel)
    strip = lambda t: re.sub(r"(?<![U\w])R(\d+|Z)\b", "R", t)  # noqa: E731
    for a, b in zip(g0["ins"], g1["ins"]):
        if strip(a.text) != strip(b.text) or a.ctrl != b.ctrl:
            raise RuntimeError(f"non-register change at {a.offset:#x}: {a.text} -> {b.text}")
    u0, u1 = reaching(g0), reaching(g1)
    by_pos0 = {(k, t): d for (k, t, r), d in u0.items()}
    by_pos1 = {(k, t): d for (k, t, r), d in u1.items()}
    for pos, d0 in by_pos0.items():
        if by_pos1.get(pos) != d0:
            raise RuntimeError(f"reaching definitions changed at ins {pos[0]} tok {pos[1]}: {sorted(d0)} -> {sorted(by_pos1.get(pos, []))}")
