"""Empirical register-operand map for sm_86 SASS.

For every instruction *form* (low 12 bits of the encoding: opcode + operand
form, e.g. 0x223 = FFMA R,R,R,R), find which encoding byte holds which
register token by perturbation: set a candidate field to a marker register,
re-disassemble, and see which token changed. Results are cached in
data/operand_fields.json so each form is probed once.

Candidate fields (Volta+ layout): Rd lo[16:24), Ra lo[24:32), Rb lo[32:40),
Rc hi[0:8). Slot names a/b/c match the reuse-flag bits (1/2/4).
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import patch, sass, toolchain

FIELDS = {"d": ("lo", 16), "a": ("lo", 24), "b": ("lo", 32), "c": ("hi", 0)}
CACHE = Path(__file__).resolve().parents[1] / "data" / "operand_fields.json"
_REG = re.compile(r"(?<![U\w])R(\d+|Z)\b")
MARKERS = (77, 78)  # two markers so a token that already equals one is still detected

# Opcodes whose register operands span register pairs/quads or have implicit
# extra registers: their registers are never renamed and their reads are not
# modeled as single-bank reads.
WIDE_HINTS = ("64", "128", "WIDE", "DFMA", "DADD", "DMUL", "DSETP", "HMMA", "IMMA", "LDSM",
              "LDGSTS", "SHFL", "ATOM", "RED", "TEX", "SUST", "SULD", "CALL", "RET", "BSSY", "BSYNC")


def form_key(ins: sass.Instr) -> str:
    return f"{ins.lo & 0xFFF:03x}"


def reg_tokens(text: str) -> list[str]:
    """Register tokens in operand order ('R12', 'RZ'), excluding uniform regs and predicates."""
    body = re.sub(r"^@!?P\w+\s+", "", text.strip())
    parts = body.split(None, 1)
    return [f"R{m}" for m in _REG.findall(parts[1])] if len(parts) > 1 else []


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save_cache(c: dict):
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=1, sort_keys=True))


def _set_byte(elf: bytearray, sec, off: int, fld: str, val: int):
    half, shift = FIELDS[fld]
    base = sec.offset + off + (0 if half == "lo" else 8)
    w, = struct.unpack_from("<Q", elf, base)
    struct.pack_into("<Q", elf, base, (w & ~(0xFF << shift)) | (val << shift))


def discover(cubin: bytes, kernel: str = "k") -> dict:
    """Probe every unseen form in this kernel; returns the full cache."""
    cache = _load_cache()
    ins = sass.parse(toolchain.disassemble(cubin))
    todo = {}
    for i in ins:
        k = form_key(i)
        if k not in cache and reg_tokens(i.text):
            todo.setdefault(k, i)
    if not todo:
        return cache
    sec = patch.text_section(cubin, kernel)
    for k, ex in todo.items():
        toks = reg_tokens(ex.text)
        found: dict[str, int] = {}
        for fld in FIELDS:
            hits = set()
            for marker in MARKERS:
                elf = bytearray(cubin)
                _set_byte(elf, sec, ex.offset, fld, marker)
                try:
                    new = [x for x in sass.parse(toolchain.disassemble(bytes(elf))) if x.offset == ex.offset]
                except RuntimeError:
                    continue  # invalid encoding: not a register field for this form
                if not new:
                    continue
                nt = reg_tokens(new[0].text)
                if len(nt) != len(toks):
                    continue
                diff = [j for j, (a, b) in enumerate(zip(toks, nt)) if a != b]
                if len(diff) == 1 and nt[diff[0]] == f"R{marker}":
                    hits.add(diff[0])
            if len(hits) == 1:
                found[fld] = hits.pop()
        cache[k] = {"opcode": ex.opcode, "example": ex.text, "tokens": len(toks), "fields": found}
    _save_cache(cache)
    return cache


@dataclass
class Operands:
    dest: int | None                     # register number written (None if not a plain register)
    srcs: list = field(default_factory=list)   # [(slot, reg)] for slots a/b/c read as registers
    wide: bool = False
    complete: bool = True                # every register token mapped to a field


def operands(ins: sass.Instr, cache: dict) -> Operands:
    toks = reg_tokens(ins.text)
    info = cache.get(form_key(ins))
    wide = any(h in ins.opcode for h in WIDE_HINTS) or ".64" in ins.text or ".128" in ins.text
    if not toks:
        return Operands(None, [], wide, True)
    if not info:
        return Operands(None, [], wide, False)
    f = info["fields"]
    by_tok = {t: fld for fld, t in f.items()}
    num = lambda s: 255 if s == "RZ" else int(s[1:])  # noqa: E731
    dest = num(toks[f["d"]]) if "d" in f else None
    srcs = [(fld, num(toks[t])) for fld, t in f.items() if fld != "d"]
    complete = len(by_tok) == len(toks)
    return Operands(dest, srcs, wide, complete)
