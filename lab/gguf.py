"""Minimal GGUF reader with Q4_0 / Q6_K decoding (ggml block layouts), numpy only.

Q4_0 block (32 weights, 18 B): fp16 d, 16 B qs; weight i = (qs[i] & 15) - 8,
weight i+16 = (qs[i] >> 4) - 8 (i < 16); value = d * weight.
Q6_K block (256 weights, 210 B): ql[128], qh[64], int8 scales[16], fp16 d.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 8: ("Q8_0", 32, 34), 14: ("Q6_K", 256, 210)}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass
class Tensor:
    name: str
    dims: tuple      # ggml order: dims[0] is the contiguous (K) axis
    type: int
    offset: int


class GGUF:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta, self.tensors = {}, {}
        with open(self.path, "rb") as f:
            magic, ver, ntens, nkv = struct.unpack("<4sIQQ", f.read(24))
            assert magic == b"GGUF", magic

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:
                    at, n = struct.unpack("<IQ", f.read(12))
                    return [rval(at) for _ in range(n)]
                fmt = _SCALAR[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

            for _ in range(nkv):
                k = rstr()
                t, = struct.unpack("<I", f.read(4))
                self.meta[k] = rval(t)
            for _ in range(ntens):
                name = rstr()
                nd, = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                t, off = struct.unpack("<IQ", f.read(12))
                self.tensors[name] = Tensor(name, dims, t, off)
            align = self.meta.get("general.alignment", 32)
            self.data_start = (f.tell() + align - 1) // align * align
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def raw(self, name: str) -> tuple[np.ndarray, Tensor]:
        t = self.tensors[name]
        _, blk, bsz = GGML[t.type]
        n = int(np.prod(t.dims))
        nbytes = n // blk * bsz
        start = self.data_start + t.offset
        return np.asarray(self.mm[start:start + nbytes]), t

    def f32(self, name: str) -> np.ndarray:
        b, t = self.raw(name)
        kind = GGML[t.type][0]
        if kind == "F32":
            a = b.view(np.float32)
        elif kind == "F16":
            a = b.view(np.float16).astype(np.float32)
        elif kind == "Q4_0":
            d, q = q4_0_blocks(b)
            a = (d[:, None] * (q.astype(np.float32) - 8)).reshape(-1)
        elif kind == "Q6_K":
            q, s = q6_k_blocks(b)
            a = (q.astype(np.float32) * np.repeat(s, 16, axis=1)).reshape(-1)
        else:
            raise NotImplementedError(kind)
        return a.reshape(tuple(reversed(t.dims)))


def q4_0_blocks(b: np.ndarray):
    """-> d float32 [nblk], q uint8 [nblk, 32] in weight order (0..15 low nibbles, 16..31 high)."""
    blk = b.reshape(-1, 18)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = blk[:, 2:]
    q = np.concatenate([qs & 15, qs >> 4], axis=1)
    return d, q


def q6_k_blocks(b: np.ndarray):
    """-> q int8 [nblk, 256] in [-32, 31], s float32 [nblk, 16] (= d * scale per 16 weights)."""
    blk = b.reshape(-1, 210)
    ql, qh = blk[:, :128].astype(np.int16), blk[:, 128:192].astype(np.int16)
    sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = blk[:, 208:210].copy().view(np.float16).astype(np.float32)
    q = np.zeros((blk.shape[0], 256), np.int16)
    for half in range(2):
        L = ql[:, 64 * half: 64 * half + 64]
        H = qh[:, 32 * half: 32 * half + 32]
        base = 128 * half
        q[:, base + 0: base + 32] = ((L[:, :32] & 15) | (((H >> 0) & 3) << 4)) - 32
        q[:, base + 32: base + 64] = ((L[:, 32:] & 15) | (((H >> 2) & 3) << 4)) - 32
        q[:, base + 64: base + 96] = ((L[:, :32] >> 4) | (((H >> 4) & 3) << 4)) - 32
        q[:, base + 96: base + 128] = ((L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)) - 32
    # sub-block scale index: weights [16k, 16k+16) use sc[k]
    s = d * sc
    return q.astype(np.int8), s
