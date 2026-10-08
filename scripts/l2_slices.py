"""Map L2 slices by contention.

Block 0 (the probe) times every 128 B line of a 2 MB L2-resident buffer (ld.cg, clock -> load ->
wait -> clock). All other blocks hammer one reference line with ld.cg until the probe finishes.
Lines in the reference's slice queue behind the hammer and get slower. Repeating with a fresh
unassigned reference partitions the buffer into slices. Spin loops are capped: hung kernels are not
reset on this machine.

  pcslurm submit -- python scripts/l2_slices.py
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

SRC = r"""
extern "C" __global__ void k(const unsigned char* __restrict__ B, int nlines, int ref, int hammer,
                             volatile unsigned* done, unsigned short* lat)
{
  if (blockIdx.x == 0) {
    if (threadIdx.x != 0) return;
    unsigned acc = 0;
    for (int i = 0; i < nlines; ++i) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(B + (size_t)i * 128)); acc += v; }
    done[1] = 1u;                                                   // go: start hammering
    for (volatile int w = 0; w < 20000; ++w) { }                  // let the hammer ramp up
    for (int i = 0; i < nlines; ++i) {
      long long t0, t1 = 0;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
      const unsigned char* a = B + (size_t)i * 128 + ((unsigned long long)t0 >> 63);
      unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a));
      asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0x7fffffff; @p mov.u64 %0, %%clock64; }" : "+l"(t1) : "r"(v));
      acc += v;
      lat[i] = (unsigned short)min((long long)65535, t1 - t0);
    }
    *done = 1u + (acc == 0x12345u);
    return;
  }
  if (!hammer) return;
  const unsigned char* a = B + (size_t)ref * 128 + (threadIdx.x & 31) * 4;
  unsigned acc = 0;
  for (unsigned long long it = 0; it < 2000000000ull && done[1] == 0u; ++it) { }   // wait for go (capped)
  for (unsigned long long it = 0; it < 2000000000ull && done[0] == 0u; ++it) {     // hammer until done (capped)
    unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a)); acc += v;
  }
  if (acc == 0x12345u) lat[0] = 1;
}
"""


def main():
    dev = Device()
    k = Kernel.load(dev, toolchain.build(SRC).cubin, "k")
    size = 2 << 20
    n = size // 128
    B = dev.alloc(size)
    dev.memset(B, size)
    done = dev.alloc(8)
    lat = dev.alloc(n * 2)
    base_ptr = B

    def run(ref, hammer):
        dev.memset(done, 8)
        args = [ctypes.c_uint64(B), ctypes.c_int32(n), ctypes.c_int32(ref), ctypes.c_int32(hammer),
                ctypes.c_uint64(done), ctypes.c_uint64(lat)]
        k.launch(1 + 67 * 4, 256, args)
        dev.sync()
        return dev.dtoh(np.zeros(n, np.uint16), lat).astype(float)

    for _ in range(3):
        run(0, 0)                                     # warm clocks
    base = np.median([run(0, 0) for _ in range(5)], axis=0)
    label = -np.ones(n, int)
    refs = []
    rng = np.random.default_rng(0)
    for it in range(80):
        free = np.nonzero(label < 0)[0]
        if len(free) == 0:
            break
        ref = int(rng.choice(free))
        d = np.median([run(ref, 1) for _ in range(3)], axis=0) - base
        mad = np.median(np.abs(d - np.median(d)))
        thr = max(25.0, np.median(d) + 6 * 1.4826 * mad)
        if d[ref] < thr:                    # hammer must slow its own line, or the run is invalid
            print(f"iter {it:2d} ref {ref}: reference not slowed (+{d[ref]:.0f}), skipping", flush=True)
            continue
        members = np.nonzero(d > thr)[0]
        members = members[label[members] < 0]
        label[members] = it
        label[ref] = it
        refs.append({"ref": ref, "members": int(len(members)), "thr": float(thr),
                     "d_ref": float(d[ref]), "d_median": float(np.median(d))})
        print(f"iter {it:2d} ref line {ref:5d}: +{d[ref]:.0f} cyc at ref, {len(members)} lines over {thr:.0f}, "
              f"unassigned left {int((label < 0).sum())}", flush=True)
    np.save(ROOT / "results" / "l2_slice_labels_contention.npy", label)
    (ROOT / "results" / "l2_slices.json").write_text(json.dumps({"base": int(base_ptr), "refs": refs}, indent=1))
    sizes = np.bincount(label[label >= 0])
    print(f"slices found: {len(sizes)}; sizes min/median/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}; "
          f"unassigned {int((label < 0).sum())}")


if __name__ == "__main__":
    main()
