"""ptxas vs our register allocation on the same kernel.

For each variant of a base experiment, run the ptxas cubin ("orig") and the
cubin after live-range re-coloring for register-bank conflicts ("ra").
Outputs must match bitwise (checked in finalize); the cycle ratio is the
speedup. Allocation results are cached on disk keyed by the cubin hash, so the
search runs locally before the GPU job (scripts/validate.py warms the cache).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import Experiment, Variant

CACHE = Path(__file__).resolve().parents[2] / "data" / "regalloc_cache"


@dataclass
class Realloc(Experiment):
    base_name: str = "mix_ffma_shl"
    max_reg: int = 61
    steps: int = 3000
    only: tuple = ()

    def __post_init__(self):
        self.name = f"realloc_{self.base_name}"
        self.description = f"ptxas vs bank-aware live-range re-coloring on {self.base_name}"

    @property
    def base(self):
        from . import registry
        return registry()[self.base_name]

    def source(self, v):
        return self.base.source(v)

    def expected(self, v):
        return self.base.expected(v)

    def variants(self, opts):
        out = []
        for b in self.base.variants(opts):
            if self.only and b.label not in self.only:
                continue
            for mode in ("orig", "ra"):
                out.append(Variant(f"{b.label}/{mode}", {**b.params, "mode": mode, "base_label": b.label}))
        return out

    def build_key(self, v):
        return v.params["mode"]

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        if v.params["mode"] == "orig":
            return cubin
        CACHE.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(cubin + f"{self.max_reg}/{self.steps}".encode()).hexdigest()[:24]
        f = CACHE / f"{key}.cubin"
        if f.exists():
            return f.read_bytes()
        from .. import regalloc
        res = regalloc.optimize(cubin, max_reg=self.max_reg, steps=self.steps)
        out = regalloc.apply(cubin, res)  # raises unless equivalence is proven
        f.write_bytes(out)
        (CACHE / f"{key}.txt").write_text(f"base_cost {res['base_cost']} best_cost {res['best_cost']}\n")
        return out

    def prepare(self, dev, v):
        return self.base.prepare(dev, v)

    def collect(self, dev, v, st):
        r = self.base.collect(dev, v, st)
        n = 32 * v.params.get("warps", 1)
        r["out_hash"] = hash(dev.dtoh(np.zeros(n, np.float32), st["out"]).tobytes())
        r.pop("correct", None)
        return r

    def release(self, dev, st):
        self.base.release(dev, st)

    def finalize(self, results):
        for label, rs in results.items():
            if not label.endswith("/ra"):
                continue
            gold = {r["out_hash"] for r in results.get(label[:-3] + "/orig", [])}
            for r in rs:
                r["correct"] = len(gold) == 1 and r["out_hash"] in gold


def registry():
    exps = [Realloc(base_name="mix_ffma_shl"), Realloc(base_name="mix_ffma_imad"),
            Realloc(base_name="rand_kernels"), Realloc(base_name="rand_kernels_b"),
            Realloc(base_name="independent_ffma"), Realloc(base_name="mix_ffma_hfma2")]
    return {e.name: e for e in exps}
