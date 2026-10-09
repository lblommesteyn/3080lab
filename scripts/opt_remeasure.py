"""Re-measure selected optimize-benchmark kernels with extra arms (noise checks, register caps; README).

  OPT_BENCH_EXTRA=cap128,cap168 OPT_BENCH_TAG=CAP pcslurm submit -- python scripts/opt_remeasure.py C <id> ...
"""
import os, subprocess, sys
suite, ids = sys.argv[1], sys.argv[2:]
for i in ids:
    r = subprocess.run([sys.executable, "scripts/opt_bench.py", "run1", suite, i], capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or ["?"])[-1], r.returncode, (r.stderr or "")[-200:] if r.returncode else "", flush=True)
