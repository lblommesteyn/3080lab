"""Occupancy sweep of the multistage int8 GEMM: the SAME kernel at 1, 2, 3, ... blocks per SM.

Occupancy is forced down by launching with unused dynamic shared memory (the code, registers and
tile stay identical), so the only thing that changes is how many warps share an SM. One wave of
exactly b blocks per SM (grid 4 x 17b, so 68 b blocks); per-stage cycles are the slope of kernel time
over K (launch, ring fill and epilogue cancel) times the SM clock sampled by NVML during the timing.

  pcslurm submit --shared -- python scripts/occupancy_sweep.py [tiles=A,D] [abl=,noload,nocompute] [out=x.json]

Tiles are named (A, D, E, ...) or given as WM.WN.MT.NT.KB.NSTG.MINB.
"""
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batched_gemv_bench as BB  # noqa: E402
from qwen_e2e import check, cu  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402
from lab import toolchain  # noqa: E402

TILES = {"A": (2, 2, 2, 4, 2, 3, 3), "D": (4, 2, 2, 4, 4, 2, 2), "B": (4, 2, 2, 4, 2, 3, 2),
         "E": (2, 2, 4, 4, 2, 3, 2), "F": (4, 1, 2, 8, 2, 3, 2), "H": (2, 2, 2, 4, 4, 2, 2),
         "T": (4, 2, 2, 2, 4, 2, 1)}
SMEM_SM = 100 * 1024
GX = 4                                     # token blocks; 17 b weight blocks -> 68 b blocks, b per SM
STAGES = (24, 72)


class Clock:
    """Mean SM clock (MHz) and power (W) over a window, sampled by NVML every 5 ms."""
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def __enter__(self):
        self.mhz, self.w, self.run = [], [], True
        def loop():
            while self.run:
                self.mhz.append(pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM))
                self.w.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3)
                time.sleep(0.005)
        self.th = threading.Thread(target=loop)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.run = False
        self.th.join()


class PadKern:
    """The kernel, launched with `dyn` bytes of unused dynamic shared memory."""
    def __init__(self, src):
        b = toolchain.build(src)
        r = toolchain.ptxas_resources(b.ptxas_log)
        self.regs, self.smem = r["registers"], r["static_smem"]
        self.fn = check(cu.cuModuleGetFunction(check(cu.cuModuleLoadData(b.cubin)), b"k"))
        self.keep = []

    def pad_for(self, blocks):
        """Dynamic smem that leaves room for exactly `blocks` blocks per SM (1 KB reserved per block)."""
        per = SMEM_SM // blocks - 512 if blocks > 1 else 60 * 1024
        return max(0, per - 1024 - self.smem)

    def occupancy(self, nthr, dyn):
        check(cu.cuFuncSetAttribute(self.fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    max(dyn, 1)))
        return check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.fn, nthr, dyn))

    def launch(self, grid, gy, block, dyn, args):
        a = [x if isinstance(x, (ctypes.c_uint64, ctypes.c_int32)) else ctypes.c_uint64(x) for x in args]
        p = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
        self.keep.append((a, p))
        st_ = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(self.fn, grid, gy, 1, block, 1, 1, dyn, st_, ctypes.addressof(p), 0))


def run_tile(name, t, abl, clk):
    WM, WN, MT, NT, KB, NSTG, MINB = t
    BM, BN, nthr = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
    kern = PadKern(QB.gemm_q4i8_ms_source(WM, WN, MT, NT, "resid", KB, NSTG, MINB, abl))
    nat = kern.occupancy(nthr, 0)
    rows = []
    for b in range(1, nat + 1):
        dyn = 0 if b == nat else kern.pad_for(b)
        occ = kern.occupancy(nthr, dyn)
        if occ != b:
            print(f"  {name} b={b}: padding gives {occ} blocks/SM, skipped", flush=True)
            continue
        P, N = GX * BN, 17 * b * BM
        times, mhz, watts = [], [], []
        for S in STAGES:
            K = S * 32 * KB
            torch.manual_seed(0)
            W = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int64, device="cuda").to(torch.int32)
            Sc = (torch.rand(N, K // 32, device="cuda") * 0.02).half()
            Wp, Sp = QB.pack_q4_i8(W, Sc)
            Xq = torch.randint(-127, 128, (P, K), dtype=torch.int8, device="cuda")
            Xd = torch.rand(P, K // 32, device="cuda") * 0.01
            Xs = torch.randint(-2**20, 2**20, (2 * P, K // 32), dtype=torch.int32, device="cuda")
            Y = torch.zeros(P, N, device="cuda")
            args = [Wp.data_ptr(), Sp.data_ptr(), Xq.data_ptr(), Xd.data_ptr(), Xs.data_ptr(), Y.data_ptr(), 0,
                    ctypes.c_int32(N), ctypes.c_int32(K), ctypes.c_int32(P)]
            fn = lambda: kern.launch(GX, N // BM, nthr, dyn, args)  # noqa: E731
            with clk:
                us = min(BB.timeit(fn, 50) for _ in range(3))
            times.append(us)
            mhz.append(sum(clk.mhz) / max(1, len(clk.mhz)))
            watts.append(sum(clk.w) / max(1, len(clk.w)))
        slope = (times[1] - times[0]) / (STAGES[1] - STAGES[0])
        f = mhz[1]
        row = {"tile": name, "cfg": list(t), "abl": abl, "blocks_sm": b, "warps_sm": b * nthr // 32,
               "regs": kern.regs, "smem": kern.smem, "dyn": dyn, "us": [round(x, 2) for x in times],
               "mhz": round(f), "watts": round(watts[1]), "stage_us": round(slope, 4),
               "stage_cyc": round(slope * f, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    cfg = {"tiles": "A,D", "abl": ",noload,nocompute", "out": "occupancy_sweep.json"}
    for a in sys.argv[1:]:
        k, v = a.split("=", 1)
        cfg[k] = v
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    pynvml_util = Clock()
    print("GPU util at start:", pynvml.nvmlDeviceGetUtilizationRates(pynvml_util.h).gpu, "%", flush=True)
    out = []
    for name in cfg["tiles"].split(","):
        t = TILES[name] if name in TILES else tuple(int(x) for x in name.split("."))
        for abl in cfg["abl"].split(","):
            out += run_tile(name, t, abl, pynvml_util)
            Path(cfg["out"]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
