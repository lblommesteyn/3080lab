"""3080lab optimize: detect split-sector loads, predict, rewrite, validate.

Pipeline for one kernel of a cubin:
  1. analyze   lab/sectors.py: affine address analysis of every LDG; split pairs = loads whose
               sectors overlap a later load's (per-lane stride, base, offsets, width).
  2. predict   lab/costmodel.py: runtime per loop trip of the original.
  3. rewrite   candidates built with lab/schedule.py (hoist the later half next to its partner,
               guard predicates retargeted, destination webs renamed when needed):
                 hoist+rename   every streaming split pair with requests in between
                 hoist-only     the same, refusing any move that would need new registers
  4. validate  nvdisasm decodes it; lab/verify.py finds no hazard the original does not have;
               register count stays within 255.
  5. decide    guided: keep the best candidate if predicted speedup >= MIN_GAIN, else the
               original. blind: keep the hoist+rename candidate whenever it validates.

MIN_GAIN = 1.02 sits above the run-to-run noise of the timing harness (CV ~1%).
"""
from __future__ import annotations

import re

from . import costmodel, sass, schedule, sectors, toolchain, verify

MIN_GAIN = 1.02


def split_finder(min_between: int = 1, min_overlap: float = 0.25):
    """Pair finder for lab/schedule.py: (first offset, second offset, instruction gap)."""
    def find(cubin: bytes, kernel: str = "k"):
        a = sectors.analyze(cubin)
        out = []
        for p in a.pairs:
            if p.overlap >= min_overlap and p.loads_between >= min_between and sectors.streams(p.second):
                out.append((p.first.offset, p.second.offset, max(1, p.gap_instr)))
        return out
    return find


def _new_hazards(out: bytes, orig: bytes) -> list:
    return verify.new_hazards(out, orig)


DEDICATE = False     # opt-in: barrier dedication removes false waits but cost 1% geomean overall (README)


def candidates(cubin: bytes, kernel: str = "k", dedicate: bool | None = None) -> dict:
    fnd = split_finder()
    out = {}
    for name, rename in (("hoist+rename", True), ("hoist-only", False)):
        try:
            c, log = schedule.fix_split_sectors_guarded(cubin, kernel, min_gap=1, finder=fnd, rename_regs=rename)
        except Exception as e:                       # an unsupported encoding: refuse, keep going
            out[name] = {"error": repr(e)[:200]}
            continue
        moved = sum(1 for item in log if len(item) == 3 and str(item[2]).startswith("ok"))
        out[name] = {"cubin": c, "moves": moved, "log": log}
        if moved and (DEDICATE if dedicate is None else dedicate):
            # hoisted loads keep ptxas's barrier: give them their own, or every wait on that barrier in
            # their window becomes a false dependence (latency-bound launches lose up to 8%, README)
            try:
                c2, why = schedule.dedicate_barrier(c, kernel, original=cubin)
            except Exception as e:                     # refuse, keep the plain rewrite
                c2, why = c, f"dedicate failed: {e!r}"[:200]
            if c2 is not c and validate(c2, cubin) is None:
                out[name + "+dedicate"] = {"cubin": c2, "moves": moved, "log": log + [("dedicate", why)]}
    return out


def validate(c: bytes, orig: bytes) -> str | None:
    """None if valid, else the reason it is refused."""
    try:
        text = toolchain.disassemble(c)
    except RuntimeError as e:
        return f"nvdisasm rejects it: {str(e)[:120]}"
    if costmodel.regcount(text) > 255:
        return "register count above 255"
    hz = _new_hazards(c, orig)
    if hz:
        return f"verifier: {hz[0]}"
    return None


def optimize(cubin: bytes, kernel: str = "k", launch: costmodel.Launch = costmodel.Launch(),
             mode: str = "guided", min_gain: float = MIN_GAIN, model_version: int | None = None) -> tuple[bytes, dict]:
    rep = {"mode": mode, "kernel": kernel, "model_version": model_version or costmodel.MODEL_VERSION,
           "trips": launch.trips}
    a = sectors.analyze(cubin)
    rep["split_pairs"] = [{"first": p.first.offset, "second": p.second.offset, "overlap": round(p.overlap, 3),
                           "loads_between": p.loads_between, "streaming": sectors.streams(p.second)}
                          for p in a.pairs if p.overlap >= 0.25]
    base = costmodel.predict(cubin, launch, kernel, model_version)
    rep["original"] = {k: base[k] for k in ("regs", "t_ns", "t_mem_ns", "t_sm_ns", "bound",
                                             "refetched_bytes_per_warp_trip", "unique_bytes_per_warp_trip")}
    rep["original"]["pairs"] = base["pairs"]
    rep["candidates"] = {}
    best = None
    cands = candidates(cubin, kernel)
    for name, c in cands.items():
        entry = {}
        if "error" in c:
            entry["refused"] = c["error"]
        elif c["moves"] == 0:
            entry["refused"] = "no legal move"
        else:
            why = validate(c["cubin"], cubin)
            if why:
                entry["refused"] = why
            else:
                p = costmodel.predict(c["cubin"], launch, kernel, model_version, sm_reference=cubin)
                entry.update(moves=c["moves"], regs=p["regs"], t_ns=p["t_ns"], bound=p["bound"],
                             predicted_speedup=base["t_ns"] / p["t_ns"])
                key = (round(p["t_ns"], 1), 0 if name.endswith("+dedicate") else 1)
                if best is None or key < best[3]:
                    best = (name, c["cubin"], p["t_ns"], key)
        rep["candidates"][name] = entry
    chosen, out = "original", cubin
    if mode == "blind":
        for name in ("hoist+rename+dedicate", "hoist+rename"):
            if "predicted_speedup" in rep["candidates"].get(name, {}):
                chosen, out = name, cands[name]["cubin"]
                break
    elif best is not None and base["t_ns"] / best[2] >= min_gain:
        chosen, out = best[0], best[1]
    rep["chosen"] = chosen
    rep["_cubins"] = {k: v["cubin"] for k, v in cands.items()
                      if "cubin" in v and "predicted_speedup" in rep["candidates"].get(k, {})}
    rep["predicted_speedup"] = base["t_ns"] / (rep["candidates"][chosen]["t_ns"] if chosen != "original"
                                               else base["t_ns"])
    return out, rep


def summary(rep: dict) -> str:
    lines = [f"kernel {rep['kernel']}: {len(rep['split_pairs'])} split-sector pairs "
             f"({sum(p['streaming'] for p in rep['split_pairs'])} streaming)"]
    o = rep["original"]
    unit = "us (whole kernel)" if rep.get("trips") else "us per loop trip"
    lines.append(f"  original: {o['regs']} regs, predicted {o['t_ns'] / 1e3:.2f} {unit} ({o['bound']}-bound), "
                 f"re-fetch {o['refetched_bytes_per_warp_trip'] / max(1, o['unique_bytes_per_warp_trip']):.0%} of unique bytes")
    for name, c in rep["candidates"].items():
        if "refused" in c:
            lines.append(f"  {name}: refused ({c['refused']})")
        else:
            lines.append(f"  {name}: {c['moves']} moves, {c['regs']} regs, predicted {c['predicted_speedup']:.3f}x")
    lines.append(f"  chosen: {rep['chosen']} (predicted {rep['predicted_speedup']:.3f}x, mode {rep['mode']})")
    return "\n".join(lines)
