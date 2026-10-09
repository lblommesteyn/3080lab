"""Re-measure selected optimize-benchmark kernels with 300 interleaved launches (noise checks, README)."""
import os, subprocess, sys
os.environ["OPT_BENCH_TRIALS"] = "300"
os.environ["OPT_BENCH_TAG"] = "D300"
os.environ["OPT_BENCH_EXTRA"] = "dedicated"
IDS = [("A", "qo7/R8U2/u_major/default"), ("A", "qo7/R4U2/u_major/O1"), ("A", "qo7/R4U2/r_major/default"),
       ("A", "gu7/R8U2/u_major/default"), ("A", "gu7/R4U2/u_major/default"), ("A", "sq4k/R4U2/u_major/O1"),
       ("B", "dn8/R4U2/u_major/lb8"), ("A", "qo7/R4U2/u_major/default")]
for suite, i in IDS:
    r = subprocess.run([sys.executable, "scripts/opt_bench.py", "run1", suite, i], capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or ["?"])[-1], r.returncode, (r.stderr or "")[-200:] if r.returncode else "", flush=True)
