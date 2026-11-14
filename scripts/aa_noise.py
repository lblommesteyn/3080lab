"""A/A noise: which part of the suite protocol makes identical binaries time differently?

Arms are all the SAME ptxas cubin. 2x2 factorial per case:
  module m0/m1: two separately loaded copies of the cubin (code at different addresses)
  buffers b0/b1: two separately allocated output-buffer sets (data at different addresses)
Each process times the 4 arms interleaved and shuffled with the suite's L2 flush, exactly like
suite_run.child_time. Several fresh processes per case show whether a gap is a stable property of
the allocation (flips sign across processes) or of the code/arm.

  pcslurm submit -- python scripts/aa_noise.py [cases=transpose,softmax,...] [procs=3] [rounds=30]
"""
import ctypes, json, random, statistics, subprocess, sys, zlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa
from lab import toolchain
import suite_run as S

ARMS = ["m0b0", "m0b1", "m1b0", "m1b1"]


def child(name, rounds, seed):
    from lab.gpu import Device, Kernel
    c = S.case_by_name(name)
    dev = Device()
    _, outs, in_ptrs, scal = S.setup(dev, c)
    cub = toolchain.build(c.src).cubin
    mods = [Kernel.load(dev, cub, "k") for _ in range(2)]
    bufs = [S.alloc_outs(dev, outs) for _ in range(2)]
    fl = Kernel.load(dev, toolchain.build(S.FLUSH_SRC).cubin, "flush")
    buf, sink = dev.alloc(S.FLUSH_BYTES), dev.alloc(4)
    dev.memset(buf, S.FLUSH_BYTES)
    fargs = [ctypes.c_uint64(buf), ctypes.c_int32(S.FLUSH_BYTES // 16), ctypes.c_uint64(sink)]
    rng = random.Random(seed)
    t = {a: [] for a in ARMS}
    for r in range(rounds + 2):
        order = ARMS[:]; rng.shuffle(order)
        for a in order:
            m, b = int(a[1]), int(a[3])
            fl.launch(dev.props["sms"] * 4, 256, fargs)
            x = mods[m].timed_launch(S.dim3(c.grid), S.dim3(c.block), S.args_for(in_ptrs, bufs[b], scal), c.smem)
            if r >= 2:
                t[a].append(x * 1e3)
    print("RESULT " + json.dumps({"times": t, "out_ptrs": [[int(p) for p in bb] for bb in bufs]}))


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:])
    cases = kv.get("cases", "transpose,softmax,nbody,layernorm,saxpy,reduce,gemv,jacobi2d").split(",")
    procs, rounds = int(kv.get("procs", 3)), int(kv.get("rounds", 30))
    out = {}
    for n in cases:
        out[n] = []
        for p in range(procs):
            r = subprocess.run([sys.executable, __file__, "_child", n, str(rounds), str(1000 + p)],
                               capture_output=True, text=True, timeout=1800)
            line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
            if not line:
                print(n, p, "FAILED", r.stderr[-400:], flush=True); continue
            d = json.loads(line[7:]); out[n].append(d)
            med = {a: statistics.median(v) for a, v in d["times"].items()}
            mod = (med["m0b0"] + med["m0b1"]) / (med["m1b0"] + med["m1b1"])
            bf = (med["m0b0"] + med["m1b0"]) / (med["m0b1"] + med["m1b1"])
            print(f"{n:12s} proc{p} " + " ".join(f"{a} {m:.1f}" for a, m in med.items()) +
                  f"  module m0/m1 {mod:.4f}  buffers b0/b1 {bf:.4f}", flush=True)
    dst = ROOT / kv.get("out", "data/optsel/aa_noise.json")
    dst.write_text(json.dumps(out)); print("wrote", dst)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_child"]:
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
