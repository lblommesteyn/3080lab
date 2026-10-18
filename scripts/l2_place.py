"""Does L2 slice-aware placement speed up a hot set? (Experiment C, Oct 2026)

The Oct 8 contention map (`results/l2_unit_labels.npy`, scripts/l2_slices.py) labels every 128 B
line of one 2 MB buffer with its L2 slice (37 clean units of 402-428 lines, one unsplit pair, one
oversized unit). This script asks two questions, with predictions written before measuring:

1. Does the map transfer? It was measured on one physical 2 MB page. The linear part of the hash
   uses address bits <= 20 only (page-invariant: cudaMalloc pages are 2 MB), but the 5-way selector
   was not a function of bits < 21, so it may use physical bits we do not control. For each page of
   a fresh allocation, an "adversarial" hot set (K lines all from one unit) is timed against a
   "balanced" set (K lines spread over all units). If the full map transfers, adversarial is ~K/37x
   slower when slice-bound; if only the 8 linear classes transfer, the adversarial lines spread over
   the 5 slices of one class and the penalty is ~5x smaller.
2. Does placement help? Every SM's warps read (ld.global.cg, full 128 B lines) or atomically add to
   a hot set of K lines laid out contiguously (the naive layout), at random, or balanced across
   units. Prediction if hot-set traffic is slice-bound: time proportional to the busiest unit's
   line count under the map.

  pcslurm submit -- python scripts/l2_place.py [pages=16] [reps=15]
"""
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import toolchain  # noqa: E402
from lab.gpu import Device, Kernel  # noqa: E402

PAGE = 2 << 20
NL = PAGE // 128

SRC = r"""
// each warp walks the hot list (K line offsets), 8 independent full-line requests in flight
extern "C" __global__ void rd(const unsigned char* __restrict__ B, const unsigned* __restrict__ off, int K, int iters,
                              unsigned* out)
{
  const int lane = threadIdx.x & 31, w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  unsigned acc = 0;
  int j = (w * 7) % K;
  for (int it = 0; it < iters; ++it) {
    unsigned v[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      int jj = j + u; jj -= jj >= K ? K : 0;
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v[u]) : "l"(B + off[jj] + lane * 4));
    }
    #pragma unroll
    for (int u = 0; u < 8; ++u) acc += v[u];
    j += 8; while (j >= K) j -= K;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
extern "C" __global__ void at(unsigned char* B, const unsigned* __restrict__ off, int K, int iters)
{
  const int lane = threadIdx.x & 31, w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int j = (w * 7) % K;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      int jj = j + u; jj -= jj >= K ? K : 0;
      asm volatile("red.global.add.u32 [%0], 1;" :: "l"(B + off[jj] + lane * 4) : "memory");
    }
    j += 8; while (j >= K) j -= K;
  }
}
"""


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    npages, reps = int(kv.get("pages", 16)), int(kv.get("reps", 15))
    old = np.load(ROOT / "results" / "l2_unit_labels.npy")
    base_old = json.load(open(ROOT / "results" / "l2_slices.json"))["base"]
    # labels by line offset within a 2 MB page
    lab = np.empty(NL, np.int64)
    lab[((base_old % PAGE) // 128 + np.arange(NL)) % NL] = old
    sizes = np.bincount(lab)
    clean = [u for u in range(len(sizes)) if 395 <= sizes[u] <= 430]
    print(f"old base {base_old:#x} (page offset {base_old % PAGE:#x}); {len(clean)} clean units", flush=True)
    rng = np.random.default_rng(0)
    by_unit = {u: rng.permutation(np.nonzero(lab == u)[0]) for u in clean}

    def busiest(lines):
        return int(np.bincount(lab[lines]).max())

    def sets(K):
        s = {}
        start = int(rng.integers(0, NL - K))
        s["contig"] = np.arange(start, start + K)
        s["random"] = rng.choice(NL, K, replace=False)
        s["balanced"] = np.array([by_unit[clean[i % len(clean)]][i // len(clean)] for i in range(K)])
        s["one_unit"] = by_unit[clean[int(rng.integers(len(clean)))]][:K]
        return s

    dev = Device()
    cub = toolchain.build(SRC).cubin
    krd, kat = Kernel.load(dev, cub, "rd"), Kernel.load(dev, cub, "at")
    total = (npages + 1) * PAGE
    raw = dev.alloc(total)
    dev.memset(raw, total)
    base = (raw + PAGE - 1) // PAGE * PAGE                           # 2 MB-aligned start
    out = dev.alloc(16)
    offd = dev.alloc(4 * 4096)
    grid, block = 68 * 4, 256

    def timeit(kern, page, lines, iters):
        offs = (lines.astype(np.uint32) * 128)
        dev.htod(offd, offs)
        pb = base + page * PAGE
        args = [ctypes.c_uint64(pb), ctypes.c_uint64(offd), ctypes.c_int32(len(lines)), ctypes.c_int32(iters)]
        if kern is krd:
            args.append(ctypes.c_uint64(out))
        kern.launch(grid, block, args)
        dev.sync()
        return float(np.median([kern.timed_launch(grid, block, args) for _ in range(reps)]))

    res = {"base": base, "raw": raw, "pages": npages, "rows": []}
    Ks = [40, 80, 160, 320]
    plan = {K: sets(K) for K in Ks}
    print("\npredictions (busiest unit's line count; slice-bound time is proportional to it):")
    for K in Ks:
        print(f"  K={K:3d}  " + "  ".join(f"{n} {busiest(v)}" for n, v in plan[K].items()), flush=True)
    res["pred"] = {K: {n: busiest(v) for n, v in plan[K].items()} for K in Ks}

    for kname, kern, iters in (("read", krd, 64), ("atomic", kat, 16)):
        print(f"\n== {kname}: ms per launch (median of {reps}), {grid} blocks x {block}; ratio vs balanced", flush=True)
        for K in Ks:
            for page in range(npages):
                t = {n: timeit(kern, page, v, iters) for n, v in plan[K].items()}
                res["rows"].append({"kind": kname, "K": K, "page": page, **t})
                b = t["balanced"]
                print(f"  K={K:3d} page {page:2d}  " + "  ".join(f"{n} {v:7.3f} ({v / b:4.2f}x)" for n, v in t.items()),
                      flush=True)
    (ROOT / "data" / "l2_place").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "data" / "l2_place" / "l2_place.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
