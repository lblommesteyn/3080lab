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
  unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  if (blockIdx.x == 0) {
    if (threadIdx.x != 0) return;
    done[2] = sm + 1;                                               // publish the probe's SM
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
  // wait for go, polling rarely (each poll is itself L2 traffic to the flag's slice)
  for (unsigned long long it = 0; it < 200000000ull; ++it) {
    if ((it & 1023) == 0 && done[1] != 0u) break;
  }
  // never share the probe's SM: its loads would queue behind ours in that SM's LSU, not in L2
  for (unsigned long long it = 0; it < 200000000ull && done[2] == 0u; ++it) { }
  if (done[2] == sm + 1) return;
  unsigned acc = 0;
  const unsigned char* line = B + (size_t)ref * 128;
  for (unsigned long long it = 0; it < 2000000000ull; ++it) {             // capped
    // varying word in the same line: loop-variant address, so ptxas cannot hoist the load
    unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(line + ((threadIdx.x + it) & 31) * 4));
    acc += v;
    if ((it & 1023) == 0 && done[0] != 0u) break;
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
    done = dev.alloc(16)
    lat = dev.alloc(n * 2)
    base_ptr = B

    def run(ref, hammer):
        dev.memset(done, 16)
        args = [ctypes.c_uint64(B), ctypes.c_int32(n), ctypes.c_int32(ref), ctypes.c_int32(hammer),
                ctypes.c_uint64(done), ctypes.c_uint64(lat)]
        k.launch(1 + 67 * 2, 256, args)
        dev.sync()
        return dev.dtoh(np.zeros(n, np.uint16), lat).astype(float)

    for _ in range(3):
        run(0, 0)                                     # warm clocks
    base = np.median([run(0, 0) for _ in range(5)], axis=0)
    # references: one line from each of the 40 latency clusters (results/l2_slice_labels.npy)
    lat_lab = np.load(ROOT / "results" / "l2_slice_labels.npy")
    rng = np.random.default_rng(0)
    refs = [int(rng.choice(np.nonzero(lat_lab == c)[0])) for c in range(lat_lab.max() + 1)]
    D = np.zeros((len(refs), n), np.float32)
    for r, ref in enumerate(refs):
        D[r] = np.median([run(ref, 1) for _ in range(3)], axis=0) - base
        print(f"ref {r:2d} line {ref:5d}: +{D[r, ref]:.0f} at ref; lines > +25: {(D[r] > 25).sum()}, > +200: {(D[r] > 200).sum()}, "
              f"> +2000: {(D[r] > 2000).sum()}", flush=True)
    np.save(ROOT / "results" / "l2_contention_D.npy", D)
    np.save(ROOT / "results" / "l2_contention_refs.npy", np.array(refs))
    (ROOT / "results" / "l2_slices.json").write_text(json.dumps({"base": int(base_ptr)}))


if __name__ == "__main__":
    main()
