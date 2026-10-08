"""Fused-path GEMV (Q4_0 layout: 32-weight groups, fp16 scales, bf16 x) vs row-groups
per block G and split-K S, at Qwen2.5-1.5B shapes. Tests whether cutting block count
(dispatch costs ~49 ns/block/SM, scripts/graph_overhead.py) speeds up the GEMVs."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant
from .gemv4 import QWEN15_SHAPES


@dataclass
class Gemv5(Experiment):
    def __post_init__(self):
        self.name = "gemv_q4_0_blocks"
        self.target_opcode = "FFMA"
        self.description = "Q4_0-layout GEMV: row-groups per block G x split-K S"

    def source(self, v):
        from .. import qwen_kernels as K
        return K.gemv_source(v.params["S"], "logits", "g32f16", v.params["G"], timed=True)

    def expected(self, v):
        return {}

    def variants(self, opts):
        out = []
        for name, n, k in QWEN15_SHAPES:
            for S_ in (1, 2):
                for G in (1, 2, 4, 8):
                    out.append(Variant(f"{name}/S{S_}G{G}", {"N": n, "K": k, "S": S_, "G": G,
                                                            "warps": S_ * G, "iters": 1}))
        return out

    def prepare(self, dev, v):
        n, k, S_, G = (v.params[x] for x in ("N", "K", "S", "G"))
        rng = np.random.default_rng(n + k)
        q = rng.integers(0, 16, size=(n, k), dtype=np.uint32)
        packed = np.zeros((n, k // 8), np.uint32)
        for i in range(8):
            packed |= q[:, i::8] << (4 * i)
        d = rng.uniform(0.002, 0.02, size=(n, k // 32)).astype(np.float16)
        x = rng.standard_normal(k).astype(np.float32)
        xb = (x.view(np.uint32) >> 16).astype(np.uint16)          # bf16 by truncation
        xr = (xb.astype(np.uint32) << 16).view(np.float32)
        ref = ((q.astype(np.float64) - 8) * np.repeat(d.astype(np.float64), 32, axis=1)) @ xr.astype(np.float64)
        blocks = -(-n // (4 * G))
        st = {"W": dev.alloc(packed.nbytes), "S": dev.alloc(d.nbytes), "X": dev.alloc(xb.nbytes),
              "Y": dev.alloc(n * 4), "T": dev.alloc(blocks * 16), "ref": ref, "blocks": blocks,
              "bytes": packed.nbytes + d.nbytes}
        dev.htod(st["W"], packed)
        dev.htod(st["S"], d)
        dev.htod(st["X"], xb)
        st["launch"] = dict(grid=blocks, block=32 * S_ * G, args=[
            ctypes.c_uint64(st["W"]), ctypes.c_uint64(st["S"]), ctypes.c_uint64(st["X"]), ctypes.c_uint64(st["Y"]),
            ctypes.c_uint64(0), ctypes.c_int32(n), ctypes.c_int32(k), ctypes.c_uint64(st["T"])])
        return st

    def collect(self, dev, v, st):
        t = dev.dtoh(np.zeros(2 * st["blocks"], np.uint64), st["T"]).reshape(-1, 2)
        ns = int(t[:, 1].max() - t[:, 0].min())
        y = dev.dtoh(np.zeros(v.params["N"], np.float32), st["Y"])
        err = float(np.max(np.abs(y - st["ref"])) / np.max(np.abs(st["ref"])))
        return {"ns": ns, "cycles": ns, "ops_per_thread": 1, "cycles_per_op": ns / 1e3, "warp_ops_per_cycle": 0.0,
                "sm_mhz_inkernel": 0.0, "us": ns / 1e3, "gbps": st["bytes"] / ns, "norm_err": err,
                "correct": err < 1e-4}

    def release(self, dev, st):
        for key in ("W", "S", "X", "Y", "T"):
            dev.free(st[key])


def registry():
    e = Gemv5()
    return {e.name: e}
