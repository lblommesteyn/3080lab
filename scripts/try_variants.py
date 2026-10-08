"""Launch each variant of an experiment once, each in a fresh process (a CUDA
illegal-instruction error poisons the context). Prints OK/FAIL per variant.

Usage (as a GPU job): python scripts/try_variants.py <experiment> [label-substring ...]
"""
import subprocess
import sys

if len(sys.argv) > 2 and sys.argv[1] == "--one":
    from lab import toolchain
    from lab.experiments import registry
    from lab.gpu import Device, Kernel
    exp = registry()[sys.argv[2]]
    v = next(x for x in exp.variants({"iters": 4096, "diag": True}) if x.label == sys.argv[3])
    b = toolchain.build(exp.source(v), ptxas_flags=exp.ptxas_flags)
    cubin = exp.transform(b.cubin, v)
    dev = Device()
    k = Kernel.load(dev, cubin, exp.kernel_name)
    st = exp.prepare(dev, v)
    L = st["launch"]
    k.launch(L["grid"], L["block"], L["args"], L.get("smem", 0))
    dev.sync()
    print("regs", k.attrs(L["block"])["registers"])
    sys.exit(0)

from lab.experiments import registry  # noqa: E402

name, subs = sys.argv[1], sys.argv[2:]
for v in registry()[name].variants({"iters": 4096, "diag": True}):
    if subs and not any(s in v.label for s in subs):
        continue
    r = subprocess.run([sys.executable, __file__, "--one", name, v.label], capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f"{v.label:<28} {'OK  ' if r.returncode == 0 else 'FAIL'} {tail[-1] if tail else ''}", flush=True)
