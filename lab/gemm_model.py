"""Occupancy cost model for multistage (cp.async ring) tensor-core GEMMs: whole-SM simulation.

lab/model.py simulates the warps of ONE block running a straight loop. A multistage GEMM adds what
that model does not have, and those are exactly the things a tile/occupancy choice trades:

  * several blocks per SM (registers and shared memory decide how many), each with its own warps
    spread over the 4 partitions, and a block-wide BAR.SYNC every stage;
  * IMMA on a per-partition tensor port (16 cycles per m16n8k32 int8, latency 24: `tc_*`), with
    the result registers interlocked in hardware (ptxas sets no scoreboard on IMMA);
  * shared memory: LDS / LDSM on one SM-wide port, max(2, bytes / 128) cycles per warp request
    (`smem_banks`: one request per 2 cycles), latency 23;
  * LDGSTS (cp.async) through an SM-wide copy port that processes ONE ACTIVE LANE PER CYCLE, and a
    lane alone in its 32 B sector costs 2 (`l2_cpasync`: half the lanes take half the time; 8 B and
    4 B copies take as long as 16 B; 16 B pieces one per line run at half rate). At 16 B per lane
    that is the familiar ~2.1 TB/s, but it is an SM-side limit, so it does not depend on what the
    other SMs do. A stage's group completes at DRAM latency (its weights are a first touch; the 8
    blocks sharing them merge in L2: `l2_cpasync` stream/F8). LDGDEPBAR closes a group and
    DEPBAR.LE SB0, n waits until at most n groups are outstanding (counted, in order);
  * per stage, the lanes a block copies come from the tile shape (not from the predicated SASS:
    which lanes are active is a runtime property).

T_kernel = LAUNCH_US + sum over waves of [ simulated main loop of the most loaded SM + the wave's epilogue ].
Blocks of a wave are equal length and start together, so their epilogues coincide: the output tile
traffic (fp32 read-modify-write for a residual epilogue: 8 B per output) is a DRAM burst at the
`mem_epilogue` rate that does not overlap the loop. Constants are microbenchmark values
(`smem_width`, `l2_cpasync`, `mem_epilogue`, `tc_*`) except BAR_LAT.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from . import sass
from .model import BRANCH_TAKEN_PENALTY, STALL0_CYCLES, port_of_v1
from .model_gpu import LAUNCH_US              # kernel in a graph: ~1 us empty + ramp (graph_overhead)

SMS = 68
F_CLK_GHZ = 1.74                  # sustained int8 GEMM is power-capped (NVML: SW power cap, ~365 W, 1725-1785 MHz;
                                  # light kernels run at 1.95). Cycle terms scale with it, DRAM terms do not.
L2_LAT = 238                      # chase_global, L2-resident
CP_LANES_PER_CYC = 1.0            # l2_cpasync: 2.11 TB/s at 16 B / lane = 0.995 lanes / cycle / SM
SMEM_LAT = 23
SMEM_BPC = 128                    # bytes / cycle, min 2 cycles per request
IMMA_COST, IMMA_LAT = 16, 24
BAR_LAT = 20                      # last arrival -> release (assumed)
DRAM_GBPS = 710                   # mem_width / mem_mlp peak
DRAM_LAT = 570                    # loaded DRAM latency, cycles (README, calibrated through the SM simulator)
EPI_BYTES = {"resid": 8, "store": 2, "biasf": 2, "swiglu": 1, "swiglu16": 1}   # per output element
EPI_GBPS = {"resid": 562, "store": 690, "biasf": 690, "swiglu": 690, "swiglu16": 690}   # mem_epilogue, 16 warps/SM
EPI_UNROLL = 4                    # ptxas unrolls the epilogue loop 4x: 4 loads in flight per thread
SMEM_PER_SM = 100 * 1024
SMEM_RESERVED = 1024              # per block (Ampere)
REGS_PER_SM, MAX_WARPS, MAX_BLOCKS = 65536, 48, 16

_REG = re.compile(r"\bR(\d+)\b")


def blocks_per_sm(regs: int, nthr: int, smem: int) -> dict:
    wpb = (nthr + 31) // 32
    per_warp = ((regs * 32 + 255) // 256) * 256
    lim = {"regs": REGS_PER_SM // (per_warp * wpb), "warps": MAX_WARPS // wpb,
           "smem": SMEM_PER_SM // (smem + SMEM_RESERVED), "hw": MAX_BLOCKS}
    b = max(1, min(lim.values()))
    return {"blocks": b, "limit": min(lim, key=lim.get), **lim}


def _width(op: str) -> int:
    return 4 if ".128" in op else 2 if ".64" in op else 1


def _regs_touched(ins) -> set[int]:
    """Registers an instruction reads or writes (widths expanded), to interlock on IMMA results."""
    t = ins.text.split(";")[0]
    t = re.sub(r"^@!?U?P\w+\s+", "", t.strip())
    op = ins.opcode
    regs = set()
    if op.startswith("IMMA") or op.startswith("HMMA"):
        ops = [int(x) for x in _REG.findall(t)]
        for r, w in zip(ops, (4, 4, 2, 4)):
            regs |= set(range(r, r + w)) if r != 255 else set()
        return regs
    parts = t.split(None, 1)
    if len(parts) < 2:
        return regs
    args = parts[1].split(",")
    w = _width(op)
    for k, a in enumerate(args):
        for r in _REG.findall(a):
            r = int(r)
            if r == 255:
                continue
            ww = w if (k == 0 and op.startswith(("LDS", "LDSM", "LDG"))) else 1
            if op.startswith("LDSM") and k == 0:
                ww = 4 if ".4" in op else 2 if ".2" in op else 1
            if op.startswith(("STS", "STG")) and k == 1:
                ww = w
            regs |= set(range(r, r + ww))
    return regs


def _imma_dst(ins) -> set[int]:
    r = int(_REG.findall(ins.text)[0])
    return set(range(r, r + 4))


@dataclass
class W:
    blk: int
    part: int
    pc: int = 0
    stage: int = 0
    ready: float = 0.0
    sb: list = field(default_factory=lambda: [0.0] * 6)
    groups: list = field(default_factory=list)     # completion time of each committed cp.async group
    open_grp: float = 0.0
    reg_ready: dict = field(default_factory=dict)
    at_bar: bool = False
    stage_end: list = field(default_factory=list)


def simulate_sm(body: list, *, nblocks: int, wpb: int, stage_lanes: float, prologue_groups: int,
                stages: int, copy_lat: int = DRAM_LAT) -> dict:
    """Cycle simulation of `nblocks` blocks x `wpb` warps on one SM running `stages` trips of the
    multistage loop `body`. Returns per-warp stage end times."""
    nb = len(body)
    info = []
    n_ldgsts = sum(1 for x in body if x.opcode.startswith("LDGSTS")) or 1
    for x in body:
        op = x.opcode
        info.append({
            "op": op, "touch": _regs_touched(x),
            "imma": op.startswith("IMMA"),
            "imma_dst": _imma_dst(x) if op.startswith("IMMA") else set(),
            "lds": op.startswith(("LDS", "LDSM")),
            "smem_bytes": 32 * 4 * (_width(op) if op.startswith("LDS") and not op.startswith("LDSM")
                                    else (4 if ".4" in op else 2 if ".2" in op else 1)),
            "ldgsts": op.startswith("LDGSTS"),
            "commit": op == "LDGDEPBAR",
            "depbar": op.startswith("DEPBAR"),
            "depn": int(re.search(r"0x([0-9a-f]+)", x.text).group(1), 16) if op.startswith("DEPBAR") else 0,
            "bar": op.startswith("BAR.SYNC") or op == "BAR",
            "port": port_of_v1(op),
            "stall": x.stall or STALL0_CYCLES,
        })
    cyc_per_ldgsts = stage_lanes / CP_LANES_PER_CYC / (wpb * n_ldgsts)
    warps = [W(b, (b * wpb + w) % 4) for b in range(nblocks) for w in range(wpb)]
    l2_free = 0.0
    smem_free = 0.0
    tensor_free = [0.0] * 4
    part_free = [dict() for _ in range(4)]
    bar_arrived = [0] * nblocks
    # prologue: each block's warps commit prologue_groups full stages at t = 0
    for b in range(nblocks):
        bw = [w for w in warps if w.blk == b]
        for g in range(prologue_groups):
            done = 0.0
            for w in bw:
                start = max(0.0, l2_free)
                l2_free = start + stage_lanes / CP_LANES_PER_CYC / wpb
                done = max(done, l2_free + copy_lat)
            for w in bw:
                w.groups.append(done)
    alive = list(warps)
    last = [None] * 4
    t = 0.0
    while alive:
        issued = False
        for p in range(4):
            cands = [w for w in alive if w.part == p and w.ready <= t and not w.at_bar]
            if not cands:
                continue
            cands.sort(key=lambda w: (w is not last[p], w.blk, w.pc))
            for w in cands:
                x = info[w.pc]
                ins = body[w.pc]
                if any(ins.wait >> b & 1 and w.sb[b] > t for b in range(6)):
                    continue
                if x["touch"] and any(w.reg_ready.get(r, 0) > t for r in x["touch"]):
                    continue
                if x["depbar"]:
                    n = x["depn"]
                    if len(w.groups) > n and w.groups[-(n + 1)] > t:
                        continue
                port, cost, _, _ = x["port"]
                if part_free[p].get(port, 0) > t:
                    continue
                if x["imma"]:
                    if tensor_free[p] > t:
                        continue
                    tensor_free[p] = t + IMMA_COST
                    for r in x["imma_dst"]:
                        w.reg_ready[r] = t + IMMA_LAT
                elif x["lds"]:
                    start = max(t, smem_free)
                    smem_free = start + max(2, x["smem_bytes"] / SMEM_BPC)
                    done = smem_free + SMEM_LAT
                    if ins.wbar != 7:
                        w.sb[ins.wbar] = max(w.sb[ins.wbar], done)
                    else:   # LDSM without scoreboard is impossible; be safe: interlock its dest
                        pass
                elif x["ldgsts"]:
                    start = max(t, l2_free)
                    l2_free = start + cyc_per_ldgsts
                    w.open_grp = max(w.open_grp, l2_free + copy_lat)
                    smem_free = max(smem_free, t) + 2          # the smem write of the copy
                elif x["commit"]:
                    w.groups.append(max(w.open_grp, t))
                    w.open_grp = 0.0
                else:
                    part_free[p][port] = t + cost
                    if ins.wbar != 7:
                        w.sb[ins.wbar] = max(w.sb[ins.wbar], t + 20)
                if ins.rbar != 7:
                    w.sb[ins.rbar] = max(w.sb[ins.rbar], t + 2)
                w.ready = t + x["stall"]
                w.pc += 1
                if x["bar"]:
                    w.at_bar = True
                    bar_arrived[w.blk] += 1
                    if bar_arrived[w.blk] == wpb:
                        bar_arrived[w.blk] = 0
                        for v in warps:
                            if v.blk == w.blk:
                                v.at_bar = False
                                v.ready = max(v.ready, t + BAR_LAT)
                if w.pc == nb:
                    w.pc = 0
                    w.stage += 1
                    w.ready += BRANCH_TAKEN_PENALTY
                    w.stage_end.append(w.ready)
                    if w.stage >= stages:
                        # stop at the first finisher: past it fewer blocks share the SM, and a block the
                        # scheduler favoured (lowest id) would leave the others to run alone (too fast)
                        done = [min(v.stage for v in warps if v.blk == b) for b in range(nblocks)]
                        return {"warps": warps, "t": w.ready, "block_stages": done,
                                "stage_cyc": w.ready * nblocks / max(1, sum(done))}
                last[p] = w
                issued = True
                break
        if issued:
            t += 1
        elif alive:
            nxt = [w.ready for w in alive if w.ready > t and not w.at_bar]
            t = max(t + 1, min(nxt)) if nxt else t + 1
    return {"warps": warps, "t_end": max(w.stage_end[-1] for w in warps)}


@dataclass(frozen=True)
class Tile:
    WM: int
    WN: int
    MT: int
    NT: int
    KB: int
    NSTG: int
    MINB: int = 1
    ABL: str = ""

    @property
    def BM(self):
        return 16 * self.MT * self.WM

    @property
    def BN(self):
        return 8 * self.NT * self.WN

    @property
    def nthr(self):
        return 32 * self.WM * self.WN

    def stage_bytes(self) -> int:
        """Global bytes one block copies per stage: int8 X, per token pair its scales (8 KB B) and
        sums (16 KB B), 4-bit weights, fp16 weight scales."""
        if self.ABL == "noload":
            return 0
        return self.BN * 32 * self.KB + (self.BN // 2) * 24 * self.KB + self.BM * 16 * self.KB + self.BM * 2 * self.KB

    def stage_lanes(self) -> float:
        """Copy-port cycles one block costs per stage (gemm_q4i8_ms_source loader): each 16 B piece
        is a lane; per stream max(lanes, 2 x sectors) (l2_cpasync)."""
        if self.ABL == "noload":
            return 0
        c = lambda lanes, nbytes: max(lanes, 2 * -(-nbytes // 32))  # noqa: E731
        KB, BN, BM = self.KB, self.BN, self.BM
        tw = (BM // 16) * (KB // 2)
        return (BN * c(2 * KB, 32 * KB)                                  # X: 32 KB B per token
                + (BN // 2) * (c(KB // 2, 8 * KB) + c(KB, 16 * KB))      # scales, sums per token pair
                + c(tw * 32, tw * 512) + c(tw * 4, tw * 64))             # weight tiles + their scales


def build(tile: Tile, epi: str = "resid"):
    from . import qwen_batched as QB
    from . import toolchain
    b = toolchain.build(QB.gemm_q4i8_ms_source(tile.WM, tile.WN, tile.MT, tile.NT, epi, tile.KB, tile.NSTG,
                                               tile.MINB, tile.ABL))
    res = toolchain.ptxas_resources(b.ptxas_log)
    ins = sass.parse(b.sass)
    ring = [l for l in sass.loops(ins) if any(x.opcode.startswith("DEPBAR") for x in l)]   # not the epilogue loop
    return {"body": max(ring, key=len), "regs": res["registers"], "smem": res["static_smem"],
            "spills": res.get("spill_stores", 0) + res.get("spill_loads", 0)}


SIM_STAGES = 12


def predict(tile: Tile, N: int, K: int, P: int, built: dict | None = None, epi: str = "resid") -> dict:
    """Predicted kernel time (us) of one GEMM Y[P, N] = X[P, K] W[N, K]^T."""
    bld = built or build(tile, epi)
    occ = blocks_per_sm(bld["regs"], tile.nthr, bld["smem"])
    bps = occ["blocks"]
    nblk = (N // tile.BM) * math.ceil(P / tile.BN)
    stages = K // (32 * tile.KB)
    wpb = tile.nthr // 32
    cache: dict = {}
    t_cyc, left, waves = 0.0, nblk, []
    while left > 0:
        in_wave = min(left, bps * SMS)
        b_here = min(bps, math.ceil(in_wave / SMS))          # the most loaded SM
        key = b_here
        if key not in cache:
            # SM throughput while all its blocks run (in stages of one block), then a block's loop is
            # its stages at that rate plus the ring fill (prologue copies at DRAM latency)
            r = simulate_sm(bld["body"], nblocks=b_here, wpb=wpb, stage_lanes=tile.stage_lanes(),
                            prologue_groups=tile.NSTG - 1, stages=SIM_STAGES)
            cache[key] = stages * r["stage_cyc"] + DRAM_LAT, r["stage_cyc"]
        # the wave's output tiles: bandwidth (mem_epilogue) or, for few blocks, the loop's DRAM round trips
        trips = math.ceil(tile.BM * tile.BN / tile.nthr / EPI_UNROLL)
        epi_c = max(in_wave * tile.BM * tile.BN * EPI_BYTES[epi] / (EPI_GBPS[epi] / F_CLK_GHZ), trips * DRAM_LAT)
        t_cyc += cache[key][0] + epi_c
        waves.append({"blocks": in_wave, "per_sm": b_here, "copy_cyc": round(b_here * tile.stage_lanes()),
                      "stage_cyc": round(cache[key][1], 1), "loop_us": round(cache[key][0] / F_CLK_GHZ / 1e3, 2),
                      "epi_us": round(epi_c / F_CLK_GHZ / 1e3, 2)})
        left -= in_wave
    return {"us": t_cyc / F_CLK_GHZ / 1e3 + LAUNCH_US, "regs": bld["regs"], "smem": bld["smem"], "occ": occ,
            "blocks": nblk, "stages": stages, "waves": waves}
