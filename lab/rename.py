"""Global register renaming to remove register-bank conflicts (post-ptxas).

A rename is a bijection on register numbers applied to every occurrence in the
kernel, so the program is unchanged except for which physical register (and
therefore which bank) each value lives in. Only "free" registers are renamed:
those that appear exclusively as 32-bit operands of fully mapped instruction
forms (see operands.py). Anything touching register pairs/quads, unknown forms,
or R1 (stack pointer) stays fixed.

Safety: after rewriting, the kernel is disassembled and every instruction
must equal the original text with the substitution applied, token by token.
Correctness on hardware is then checked by comparing outputs bitwise.
"""
from __future__ import annotations

import math
import random
import re

from . import operands as opnd
from . import patch, sass, toolchain


def _num(tok: str) -> int:
    return 255 if tok == "RZ" else int(tok[1:])


_RENAMABLE_WIDE_FREE = ("SHFL",)  # in operands.WIDE_HINTS only to skip RF modeling; operands are 32-bit


def token_widths(i: sass.Instr) -> list[int] | None:
    """Register width (1/2/4) of each register token, or None if unknown (pin all)."""
    toks = opnd.reg_tokens(i.text)
    op = i.opcode
    if op.startswith(("DFMA", "DADD", "DMUL", "DSETP", "HMMA", "IMMA", "LDSM", "LDGSTS", "ATOM", "RED",
                      "TEX", "SU", "CALL", "RET", "BSSY", "BSYNC")):
        return None
    addr = re.findall(r"\[([^\]]*)\]", i.text)
    addr_toks = []
    for a in addr:
        addr_toks += [(f"R{m}", ".64" in a) for m in re.findall(r"(?<![U\w])R(\d+|Z)\b", a)]
    data_w = 4 if ".128" in op else 2 if ".64" in op and not op.startswith("IMAD") else 1
    widths, ai = [], 0
    in_addr = set(t for t, _ in addr_toks)
    for k, t in enumerate(toks):
        if t in in_addr and ai < len(addr_toks) and addr_toks[ai][0] == t:
            widths.append(2 if addr_toks[ai][1] else 1)
            ai += 1
        elif op.startswith("IMAD.WIDE") and k in (0, len(toks) - 1):
            widths.append(2)  # 64-bit destination and 64-bit addend
        else:
            widths.append(data_w)
    return widths


def analyze(cubin: bytes, kernel: str = "k") -> dict:
    cache = opnd.discover(cubin, kernel)
    ins = sass.parse(toolchain.disassemble(cubin))
    used, fixed = set(), {1}
    for i in ins:
        toks = opnd.reg_tokens(i.text)
        nums = [_num(t) for t in toks]
        used.update(n for n in nums if n != 255)
        o = opnd.operands(i, cache)
        widths = token_widths(i)
        info = cache.get(opnd.form_key(i), {"fields": {}})
        mapped_tokens = set(info["fields"].values())
        for k, n in enumerate(nums):
            if n == 255:
                continue
            w = None if widths is None else widths[k]
            if w is None or k not in mapped_tokens or not o.complete:
                fixed.update((n & ~1, n | 1))  # unknown: pin with its possible pair partner
            elif w > 1:
                base = n & ~(w - 1)
                fixed.update(range(base, base + w))
    free = sorted(r for r in used if r not in fixed)
    return {"instrs": ins, "cache": cache, "used": used, "fixed": fixed, "free": free}


def rf_cost(body: list, cache: dict, mapping: dict) -> int:
    """Static register-read cycles for one warp running the loop body back to back,
    with the reuse cache (same slot, same register) supplying operands for free."""
    total, prev = 0, {}
    for i in body:
        o = opnd.operands(i, cache)
        if o.wide or not o.complete:
            total += 1
            prev = {}
            continue
        banks = [0, 0]
        for slot, reg in o.srcs:
            if reg == 255 or prev.get(slot) == reg:
                continue
            banks[mapping.get(reg, reg) & 1] += 1
        total += max(1, max(banks))
        prev = {slot: reg for slot, reg in o.srcs if i.reuse >> "abc".index(slot) & 1}
    return total


def search(cubin: bytes, kernel: str = "k", max_reg: int | None = None, steps: int = 4000,  # max_reg: highest target register (occupancy budget)
           seed: int = 0) -> dict:
    """Simulated annealing over renames of free registers to minimize rf_cost."""
    a = analyze(cubin, kernel)
    body = sass.loop_body(a["instrs"])
    regcount = max(a["used"] - {255}) + 3
    max_reg = max_reg or regcount - 3
    # candidate target registers: any number <= max_reg not pinned
    pool = [r for r in range(max_reg + 1) if r not in a["fixed"]]
    mapping = {r: r for r in a["free"]}
    inverse = {v: k for k, v in mapping.items()}
    rng = random.Random(seed)
    cost = best = rf_cost(body, a["cache"], mapping)
    best_map = dict(mapping)
    base = cost
    temp = 2.0
    for step in range(steps):
        r = rng.choice(a["free"])
        tgt = rng.choice(pool)
        cur = mapping[r]
        if tgt == cur:
            continue
        other = inverse.get(tgt)  # swap with whoever holds tgt (if a free reg does)
        mapping[r] = tgt
        if other is not None:
            mapping[other] = cur
        new = rf_cost(body, a["cache"], mapping)
        if new <= cost or rng.random() < math.exp((cost - new) / temp):
            cost = new
            inverse = {v: k for k, v in mapping.items()}
            if new < best:
                best, best_map = new, dict(mapping)
        else:
            mapping[r] = cur
            if other is not None:
                mapping[other] = tgt
        temp = max(0.05, temp * 0.999)
    return {"base_cost": base, "best_cost": best, "mapping": {k: v for k, v in best_map.items() if k != v},
            "max_reg": max(list(best_map.values()) + [max(a["used"] - {255})])}


def apply(cubin: bytes, mapping: dict, kernel: str = "k") -> bytes:
    """Rewrite every register operand per mapping, raise regcount if needed, verify text."""
    if not mapping:
        return cubin
    a = analyze(cubin, kernel)
    c = cubin
    for i in a["instrs"]:
        info = a["cache"].get(opnd.form_key(i))
        toks = opnd.reg_tokens(i.text)
        if not info or not toks:
            continue
        kw = {}
        for fld, t in info["fields"].items():
            n = _num(toks[t])
            if n in mapping:
                kw[{"d": "rd", "a": "ra", "b": "rb", "c": "rc"}[fld]] = mapping[n]
        if kw:
            c = patch.set_regs(c, kernel, i.offset, **kw)
    need = max(max(mapping.values()), max(a["used"] - {255})) + 3
    cur = patch.text_section(c, kernel)  # noqa: F841 (ensures kernel exists)
    c = patch.set_regcount(c, kernel, max(need, _regcount(c, kernel)))
    _verify(a["instrs"], sass.parse(toolchain.disassemble(c)), mapping)
    return c


def _regcount(cubin: bytes, kernel: str) -> int:
    m = re.search(r"SHI_REGISTERS=(\d+)", toolchain.disassemble(cubin))
    return int(m.group(1)) if m else 0


def _verify(orig: list, new: list, mapping: dict):
    def sub(text):
        return re.sub(r"(?<![U\w])R(\d+)\b", lambda m: f"R{mapping.get(int(m.group(1)), int(m.group(1)))}", text)
    if len(orig) != len(new):
        raise RuntimeError("instruction count changed")
    for o, n in zip(orig, new):
        if sub(o.text) != n.text or (o.lo >> 0 & 0xFFF) != (n.lo & 0xFFF):
            raise RuntimeError(f"rename verify failed at {o.offset:#x}:\n  want {sub(o.text)}\n  got  {n.text}")
