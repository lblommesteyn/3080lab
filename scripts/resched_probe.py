"""Launch each rescheduled random kernel in its own process (a fault poisons the CUDA context), and
compare its output bitwise with ptxas's.

  python scripts/resched_probe.py build      # offline: write cubins to results/resched_probe/
  pcslurm submit -- python scripts/resched_probe.py run
"""
import ctypes
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "resched_probe"


def build():
    from lab import toolchain
    from lab.experiments.randk import ReschedRandom
    OUT.mkdir(parents=True, exist_ok=True)
    e = ReschedRandom()
    meta = {}
    for v in e.variants({}):
        cub = e.transform(toolchain.build(e.source(v)).cubin, v)
        name = v.label.replace("/", "_")
        (OUT / f"{name}.cubin").write_bytes(cub)
        meta[name] = {"warps": v.params["warps"], "iters": v.params["iters"]}
    (OUT / "meta.json").write_text(json.dumps(meta))


SEEDS = (1036,)
BISECT = {"crit_noreuse": dict(policy="crit", reuse=False),
          "crit_loops": dict(policy="crit", only_loops=True),
          "crit_loops_noreuse": dict(policy="crit", only_loops=True, reuse=False)}


def bisect():
    """Variants of a few failing seeds to localise the bug."""
    from lab import resched, toolchain
    from lab.experiments.randk import ReschedRandom
    e = ReschedRandom()
    meta = json.loads((OUT / "meta.json").read_text())
    for v in e.variants({}):
        seed = v.params["seed"]
        if seed not in SEEDS or v.params["policy"] != "ptxas":
            continue
        cub = toolchain.build(e.source(v)).cubin
        base = v.label.split("/")[0]
        for tag, kw in BISECT.items():
            out, _ = resched.reschedule(cub, seed=seed, **kw)
            (OUT / f"{base}_{tag}.cubin").write_bytes(out)
            meta[f"{base}_{tag}"] = meta[f"{base}_ptxas"]
    (OUT / "meta.json").write_text(json.dumps(meta))


def one(name, iters):
    from lab.gpu import Device, Kernel
    meta = json.loads((OUT / "meta.json").read_text())[name]
    dev = Device()
    k = Kernel.load(dev, (OUT / f"{name}.cubin").read_bytes(), "k")
    w = meta["warps"]
    out, cyc, ns, fin, iin = (dev.alloc(32 * w * 4), dev.alloc(w * 8), dev.alloc(w * 8), dev.alloc(12), dev.alloc(12))
    dev.htod(fin, np.array([1.0, 1.0, 0.0], np.float32))
    dev.htod(iin, np.array([1, 1, 0], np.int32))
    args = [ctypes.c_uint64(x) for x in (out, cyc, ns, fin, iin)] + [ctypes.c_int32(iters)]
    k.launch(1, 32 * w, args)
    dev.sync()
    y = dev.dtoh(np.zeros(32 * w, np.float32), out)
    print(hashlib.sha1(y.tobytes()).hexdigest()[:12])


def stress(pat, reps=5):
    """Full iteration count (as timed), several launches each: races show up as unstable hashes."""
    meta = {k: v for k, v in json.loads((OUT / "meta.json").read_text()).items() if re.search(pat, k)}
    for name in sorted(meta):
        hs = []
        for _ in range(reps):
            r = subprocess.run([sys.executable, __file__, "one", name, str(meta[name]["iters"])], capture_output=True,
                               text=True, timeout=300)
            hs.append(r.stdout.strip() if r.returncode == 0 else "FAULT")
        print(f"{name:40s} {len(set(hs))} distinct: {sorted(set(hs))}", flush=True)


def run(pat=""):
    meta = {k: v for k, v in json.loads((OUT / "meta.json").read_text()).items() if re.search(pat, k)}
    res = {}
    for name in sorted(meta):
        for iters in (1, 3):
            r = subprocess.run([sys.executable, __file__, "one", name, str(iters)], capture_output=True, text=True,
                               timeout=120)
            res[f"{name}@{iters}"] = r.stdout.strip() if r.returncode == 0 else "FAULT " + r.stderr.strip()[-160:]
    bad = 0
    for key in sorted(res):
        if not key.split("@")[0].endswith("_ptxas"):
            base = re.sub(r"_(crit|random|identity)\w*@", "_ptxas@", key)
            ok = res[key] == res[base]
            bad += not ok
            if not ok:
                print(f"MISMATCH {key}: {res[key]} vs ptxas {res[base]}")
    print(f"{len(res)} launches, {bad} mismatches/faults")
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"build": build, "bisect": bisect, "run": lambda: run(sys.argv[2] if len(sys.argv) > 2 else ""), "stress": lambda: stress(sys.argv[2]), "one": lambda: one(sys.argv[2], int(sys.argv[3]))}[sys.argv[1]]()
