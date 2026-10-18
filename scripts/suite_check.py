"""CPU-only check of the frozen kernel suite (lab/suite.py): compile every case, print its resources and
SASS size, and run the scheduler (resched.reschedule, policy crit) on it. --ref also builds each
case's inputs and runs its numpy reference (shape/dtype sanity and host cost).

  python scripts/suite_check.py [--ref] [name-regex]
"""
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401  (puts the CUDA DLLs nvrtc needs on the search path)

from lab import resched, sass, toolchain  # noqa: E402
from lab.suite import CASES  # noqa: E402


def main():
    ref = "--ref" in sys.argv
    pat = next((a for a in sys.argv[1:] if not a.startswith("--")), "")
    fails = []
    print(f"{'case':14s} {'kind':8s} {'regs':>4s} {'smem':>6s} {'sass':>5s}  resched")
    for c in CASES:
        if not re.search(pat, c.name):
            continue
        try:
            b = toolchain.build(c.src)
        except Exception as e:                                        # noqa: BLE001
            fails.append((c.name, "build", str(e)))
            print(f"{c.name:14s} BUILD FAILED: {e}")
            continue
        r = toolchain.ptxas_resources(b.ptxas_log)
        n = len(sass.parse(b.sass))
        spill = f" spill {r['spill_stores']}/{r['spill_loads']}" if r.get("spill_stores") else ""
        try:
            new, log = resched.reschedule(b.cubin, "k", policy="crit")
            sched = sum(1 for _, m in log if m.startswith("scheduled"))
            moved = sum(int(re.search(r"(\d+) moved", m).group(1)) for _, m in log if m.startswith("scheduled"))
            kept = [m for _, m in log if m.startswith("kept order:")]
            status = f"ok: {sched} blocks scheduled, {moved} instrs moved, {len(kept)} kept, " \
                     f"{'identical' if new == b.cubin else 'changed'}"
        except Exception as e:                                        # noqa: BLE001
            fails.append((c.name, "resched", "".join(traceback.format_exception_only(e)).strip()))
            status = f"FAILED: {type(e).__name__}: {e}"
        print(f"{c.name:14s} {c.kind:8s} {r.get('registers', 0):4d} {r['static_smem'] + c.smem:6d} {n:5d}  "
              f"{status}{spill}", flush=True)
        if ref:
            t = time.perf_counter()
            ins, outs, sc = c.make(np.random.default_rng(0))
            want = c.reference(ins)
            assert len(want) == len(outs) and all(np.shape(w) == s for w, (s, _) in zip(want, outs)), c.name
            print(f"{'':14s} reference ok ({time.perf_counter() - t:.1f} s, "
                  f"{sum(a.nbytes for a in ins) / 2**20:.0f} MB in)", flush=True)
    print(f"{len(fails)} failures")
    for name, stage, err in fails:
        print(f"--- {name} ({stage})\n{err}")


if __name__ == "__main__":
    main()
