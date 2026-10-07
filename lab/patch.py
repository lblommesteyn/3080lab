"""Edit SASS control bits (and swap instructions) directly in a cubin.

A cubin is ELF64; each kernel's code lives in section `.text.<name>` as a
sequence of 16-byte instructions. The control word sits in bits [41:62) of
the high 64-bit half (bits [105:126) of the instruction), so changing stall
counts, barriers or wait masks is a bit edit at a fixed offset; no assembler
or re-encoding is needed.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

CTRL_SHIFT = 41
FIELDS = {  # name -> (bit offset within control word, width)
    "stall": (0, 4), "yield": (4, 1), "wbar": (5, 3), "rbar": (8, 3), "wait": (11, 6), "reuse": (17, 4),
}


@dataclass
class Section:
    name: str
    offset: int
    size: int


def sections(elf: bytes) -> dict[str, Section]:
    if elf[:4] != b"\x7fELF" or elf[4] != 2:
        raise ValueError("not an ELF64 cubin")
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    hdrs = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]
    strtab_off = hdrs[shstrndx][4]

    def name(off):
        end = elf.index(b"\0", strtab_off + off)
        return elf[strtab_off + off:end].decode()

    return {name(h[0]): Section(name(h[0]), h[4], h[5]) for h in hdrs}


def text_section(elf: bytes, kernel: str = "k") -> Section:
    return sections(elf)[f".text.{kernel}"]


def read_hi(elf: bytes | bytearray, sec: Section, index: int) -> int:
    return struct.unpack_from("<Q", elf, sec.offset + 16 * index + 8)[0]


def write_hi(elf: bytearray, sec: Section, index: int, hi: int):
    struct.pack_into("<Q", elf, sec.offset + 16 * index + 8, hi)


def get_field(hi: int, field: str) -> int:
    off, width = FIELDS[field]
    return (hi >> (CTRL_SHIFT + off)) & ((1 << width) - 1)


def set_field(hi: int, field: str, value: int) -> int:
    off, width = FIELDS[field]
    mask = ((1 << width) - 1) << (CTRL_SHIFT + off)
    if not 0 <= value < (1 << width):
        raise ValueError(f"{field}={value} does not fit {width} bits")
    return (hi & ~mask) | (value << (CTRL_SHIFT + off))


def set_control(cubin: bytes, kernel: str, edits: dict[int, dict[str, int]]) -> bytes:
    """edits: {instruction byte offset (as nvdisasm prints it): {field: value}}."""
    elf = bytearray(cubin)
    sec = text_section(elf, kernel)
    for off, fields in edits.items():
        if off % 16 or off >= sec.size:
            raise ValueError(f"bad instruction offset {off:#x}")
        hi = read_hi(elf, sec, off // 16)
        for f, v in fields.items():
            hi = set_field(hi, f, v)
        write_hi(elf, sec, off // 16, hi)
    return bytes(elf)


def swap(cubin: bytes, kernel: str, off_a: int, off_b: int) -> bytes:
    """Swap two whole instructions (control bits travel with them). Only safe for
    non-branch instructions within one basic block; the caller is responsible
    for re-checking dependencies and barriers."""
    elf = bytearray(cubin)
    sec = text_section(elf, kernel)
    a, b = sec.offset + off_a, sec.offset + off_b
    elf[a:a + 16], elf[b:b + 16] = elf[b:b + 16], elf[a:a + 16]
    return bytes(elf)
