"""Scripted L1 experiments: the host writes a list of line accesses, the GPU
replays them one at a time from a single thread and times the ones marked.

Each op is a 32-bit word: bit 31 = time this access, low bits = line index.
Lines are 128 B apart in one buffer. The op list is read with ld.global.cg
(L2 only) and timings are stored with st.global.cg, so the script machinery
never touches L1. Each data access is consumed before the next op issues
(in-order issue), so L1 sees exactly the scripted order.

Scenarios:
  capacity_fwd / capacity_rev   flush, fill N lines, re-read them forward / reverse
  policy_K<k>                   flush, fill N, re-touch the first quarter,
                                stream k new lines, probe ONE old line per round.
                                LRU evicts the oldest *untouched* lines;
                                FIFO evicts the oldest *inserted* lines; random scatters.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .base import Experiment, Variant

LINE = 128
FLUSH_BASE = 1 << 14          # flush lines live far from test lines
FLUSH_LINES = 4096            # 512 KB of streaming evicts any L1 content
NEW_BASE = 1 << 13            # lines used for "stream K new lines"
TIMED = 1 << 31
HIT_THRESHOLD = 120           # cycles: L1 ~35+overhead, L2 ~238+


def _flush() -> list[int]:
    return list(range(FLUSH_BASE, FLUSH_BASE + FLUSH_LINES))


@dataclass
class CacheScript(Experiment):
    n_lines: int = 768           # 96 KB: largest size measured fully L1-resident
    probe_step: int = 4          # policy: probe every 4th old line (one round each)

    def __post_init__(self):
        self.name = "l1_replacement"
        self.target_opcode = "LDG.E.STRONG.SM"
        self.description = "scripted L1 access sequences with per-access timing (capacity + LRU/FIFO/random)"

    def source(self, v: Variant) -> str:
        return """
extern "C" __global__ void k(const unsigned char* base, const unsigned* ops, int nops,
                             unsigned* times, long long* cyc, unsigned long long* ns, unsigned* sinkout)
{
  if (threadIdx.x) return;
  unsigned sink = 0;
  int ti = 0;
  unsigned long long g0, g1;
  long long s0, s1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(s0));
  #pragma unroll 1
  for (int i = 0; i < nops; ++i) {
    unsigned op;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(op) : "l"(ops + i));
    unsigned long long addr;
    asm volatile("mad.wide.u32 %0, %1, 128, %2;" : "=l"(addr) : "r"(op & 0x7fffffffu), "l"(base));
    unsigned v;
    if (op >> 31) {
      // t0 feeds the address (t0>>63 is 0 at runtime but ptxas cannot prove it), so the load
      // issues after t0. t1 is a clock read predicated on the loaded value, so it issues after
      // the load completes. transform() verifies this order in the SASS.
      long long t0, t1 = 0;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
      unsigned long long a2 = addr + ((unsigned long long)t0 >> 63);
      asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(a2));
      asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0x7fffffff; @p mov.u64 %0, %%clock64; }"
                   : "+l"(t1) : "r"(v));
      asm volatile("add.u32 %0, %0, %1;" : "+r"(sink) : "r"(v));
      asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(times + ti), "r"((unsigned)(t1 - t0)));
      ++ti;
    } else {
      asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(addr));
      asm volatile("add.u32 %0, %0, %1;" : "+r"(sink) : "r"(v));
    }
  }
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(s1));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  sinkout[0] = sink; cyc[0] = s1 - s0; ns[0] = g1 - g0;
}
"""

    def expected(self, v: Variant) -> dict[str, int]:
        return {}  # shape is checked in transform(): clock -> load -> wait -> clock

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        """Refuse to run unless the timed path is ordered clock, L1 load, consumer wait, clock."""
        from .. import sass, toolchain
        ins = sass.parse(toolchain.disassemble(cubin))
        clocks = [i for i, x in enumerate(ins) if "SR_CLOCKLO" in x.text]
        loads = [i for i, x in enumerate(ins) if x.opcode == "LDG.E.STRONG.SM"]
        ok = False
        for a, b in zip(clocks, clocks[1:]):
            mid = [i for i in loads if a < i < b]
            if len(mid) == 1:
                ld = ins[mid[0]]
                waits = [x for x in ins[mid[0] + 1:b + 1] if x.wait >> ld.wbar & 1] if ld.wbar != 7 else []
                ok = ok or bool(waits)
        if not ok:
            raise SystemExit("timed load is not bracketed by clock reads with a wait in between")
        return cubin

    # ---- scripts ----------------------------------------------------------
    def _script(self, v: Variant) -> tuple[list[int], list[tuple]]:
        n = v.params["n"]
        kind = v.params["kind"]
        ops: list[int] = []
        meta: list[tuple] = []  # one entry per timed op: (round, line)
        if kind in ("capacity_fwd", "capacity_rev"):
            ops += _flush() + list(range(n))
            order = range(n) if kind == "capacity_fwd" else range(n - 1, -1, -1)
            for ln in order:
                ops.append(TIMED | ln)
                meta.append((0, ln))
        else:
            k = v.params["k"]
            touched = range(n // 4)
            for r, p in enumerate(range(0, n, self.probe_step)):
                ops += _flush() + list(range(n)) + list(touched)
                ops += list(range(NEW_BASE, NEW_BASE + k))
                ops.append(TIMED | p)
                meta.append((r, p))
        return ops, meta

    def variants(self, opts: dict) -> list[Variant]:
        out = []
        for n in opts.get("lines") or [512, 640, 704, 768, 800, 832, 864, 896, 960, 1024]:
            for kind in ("capacity_fwd", "capacity_rev"):
                out.append(Variant(f"{kind}/N={n}", {"kind": kind, "n": n, "warps": 1, "iters": 1}))
        n = self.n_lines
        for k in (n // 8, n // 4, n // 2):
            out.append(Variant(f"policy/N={n}/K={k}", {"kind": "policy", "n": n, "k": k, "warps": 1, "iters": 1}))
        return out

    def prepare(self, dev, v: Variant) -> dict:
        ops, meta = self._script(v)
        ops_np = np.array(ops, dtype=np.uint32)
        st = {"meta": meta, "ntimed": len(meta),
              "base": dev.alloc((FLUSH_BASE + FLUSH_LINES) * LINE),
              "ops": dev.alloc(ops_np.nbytes), "times": dev.alloc(len(meta) * 4),
              "cyc": dev.alloc(8), "ns": dev.alloc(8), "sink": dev.alloc(4), "nops": len(ops)}
        dev.memset(st["base"], (FLUSH_BASE + FLUSH_LINES) * LINE)
        dev.htod(st["ops"], ops_np)
        st["launch"] = dict(grid=1, block=32, args=[
            ctypes.c_uint64(st["base"]), ctypes.c_uint64(st["ops"]), ctypes.c_int32(len(ops)),
            ctypes.c_uint64(st["times"]), ctypes.c_uint64(st["cyc"]), ctypes.c_uint64(st["ns"]),
            ctypes.c_uint64(st["sink"])])
        return st

    def collect(self, dev, v: Variant, st: dict) -> dict:
        t = dev.dtoh(np.zeros(st["ntimed"], np.uint32), st["times"])
        cyc = int(dev.dtoh(np.zeros(1, np.int64), st["cyc"])[0])
        ns = int(dev.dtoh(np.zeros(1, np.uint64), st["ns"])[0])
        hits = t < HIT_THRESHOLD
        return {
            "cycles": cyc, "ns": ns, "ops_per_thread": st["nops"],
            "cycles_per_op": cyc / st["nops"], "warp_ops_per_cycle": st["nops"] / cyc,
            "sm_mhz_inkernel": cyc / max(ns, 1) * 1e3,
            "hit_fraction": float(hits.mean()),
            "timed_cycles_median": float(np.median(t)),
            "lines": [m[1] for m in st["meta"]],
            "timings": t.tolist(),
        }

    def release(self, dev, st: dict):
        for key in ("base", "ops", "times", "cyc", "ns", "sink"):
            dev.free(st[key])


def registry() -> dict[str, Experiment]:
    e = CacheScript()
    return {e.name: e}
