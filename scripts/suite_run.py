"""Run the frozen kernel suite (lab/suite.py) under ptxas's schedule and our rescheduled variants.

Per case: build the ptxas cubin and every variant (CPU, this process), then
  1. one subprocess per (case, variant) launches it once on the same seeded inputs (a fault poisons the
     CUDA context, so no variant can hide behind or break another); ptxas is checked against the numpy
     reference, every other variant must match ptxas's outputs bitwise (sha256);
  2. one subprocess per case times every variant that passed: rounds of shuffled order, each timed
     launch (CUDA events) preceded by a read-only sweep of a 64 MB buffer to flush L2.
Reports per-case medians, ratio ptxas/variant (>1 = variant faster) and the geomean over cases.

  pcslurm submit -- python scripts/suite_run.py [variants=ptxas,crit] [rounds=30] [out=suite] [cases=regex]
"""
import ctypes
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401  (puts the CUDA DLLs nvrtc needs on the search path)

from lab import resched, sass, toolchain  # noqa: E402
from lab.suite import CASES, compare  # noqa: E402

VARIANTS = {"sb": dict(policy="sb"), "sb_wide": dict(policy="sb_wide"), "crit": dict(policy="crit"), "crit_noreuse": dict(policy="crit", reuse=False),
            "crit_loops": dict(policy="crit", only_loops=True), "random": dict(policy="random"),
            "identity": dict(policy="identity"), "identity_late": dict(policy="identity", late_entry=True, sticky_waits=True),
            "crit_late": dict(policy="crit", late_entry=True, sticky_waits=True), "sb_late": dict(policy="sb", late_entry=True, sticky_waits=True),
            "sb_wide_late": dict(policy="sb_wide", late_entry=True, sticky_waits=True)}
FLUSH_BYTES = 64 << 20
FLUSH_SRC = r"""
extern "C" __global__ void flush(const float4* p, int n, float* sink) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float4 v = p[i]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (s.x + s.y + s.z + s.w == 1234.5f) sink[0] = s.x;
}"""


def case_by_name(name):
    return next(c for c in CASES if c.name == name)


def setup(dev, c):
    """Seeded inputs on the device; returns (inputs, outs spec, input ptrs, scalar ctypes)."""
    ins, outs, sc = c.make(np.random.default_rng(zlib.crc32(c.name.encode())))
    ptrs = []
    for a in ins:
        p = dev.alloc(a.nbytes)
        dev.htod(p, np.ascontiguousarray(a))
        ptrs.append(p)
    scal = [ctypes.c_int32(v) if t == "i32" else ctypes.c_float(v) for t, v in sc]
    return ins, outs, ptrs, scal


def alloc_outs(dev, outs):
    return [dev.alloc(int(np.prod(s)) * np.dtype(d).itemsize) for s, d in outs]


def dim3(t):
    return tuple(t) + (1,) * (3 - len(t))


def args_for(in_ptrs, out_ptrs, scal):
    return [ctypes.c_uint64(p) for p in in_ptrs + out_ptrs] + scal


def child_check(cubin_path, name, variant):
    from lab.gpu import Device, Kernel
    c = case_by_name(name)
    dev = Device()
    ins, outs, in_ptrs, scal = setup(dev, c)
    k = Kernel.load(dev, Path(cubin_path).read_bytes(), "k")
    out_ptrs = alloc_outs(dev, outs)
    for p, (s, d) in zip(out_ptrs, outs):
        dev.memset(p, int(np.prod(s)) * np.dtype(d).itemsize)
    k.launch(dim3(c.grid), dim3(c.block), args_for(in_ptrs, out_ptrs, scal), c.smem)
    dev.sync()
    got = [dev.dtoh(np.empty(s, d), p) for p, (s, d) in zip(out_ptrs, outs)]
    res = {"sha": hashlib.sha256(b"".join(g.tobytes() for g in got)).hexdigest()[:16],
           "attrs": k.attrs(int(np.prod(c.block)), c.smem)}
    if variant == "ptxas":
        cmp = [compare(g, r, c.tol, c.frac) for g, r in zip(got, c.reference(ins))]
        res["ref"] = {"ok": all(x["ok"] for x in cmp), "outputs": cmp}
    print("RESULT " + json.dumps(res))


def child_time(cubin_dir, name, variants, rounds):
    from lab.gpu import Device, Kernel
    c = case_by_name(name)
    dev = Device()
    _, outs, in_ptrs, scal = setup(dev, c)
    kern = {v: Kernel.load(dev, (Path(cubin_dir) / f"{name}.{v}.cubin").read_bytes(), "k") for v in variants}
    args = {v: args_for(in_ptrs, alloc_outs(dev, outs), scal) for v in variants}
    fl = Kernel.load(dev, toolchain.build(FLUSH_SRC).cubin, "flush")
    buf, sink = dev.alloc(FLUSH_BYTES), dev.alloc(4)
    dev.memset(buf, FLUSH_BYTES)
    fargs = [ctypes.c_uint64(buf), ctypes.c_int32(FLUSH_BYTES // 16), ctypes.c_uint64(sink)]
    rng = random.Random(zlib.crc32(name.encode()))
    times = {v: [] for v in variants}
    for r in range(rounds + 2):                                       # two warmup rounds
        order = list(variants)
        rng.shuffle(order)
        for v in order:
            fl.launch(dev.props["sms"] * 4, 256, fargs)
            t = kern[v].timed_launch(dim3(c.grid), dim3(c.block), args[v], c.smem)
            if r >= 2:
                times[v].append(t * 1e3)
    print("RESULT " + json.dumps(times))


def run_child(*argv, timeout=600):
    try:
        r = subprocess.run([sys.executable, __file__, *map(str, argv)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout} s"
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    if r.returncode or line is None:
        return None, (r.stderr.strip() or r.stdout.strip())[-600:]
    return json.loads(line[7:]), None


def main():
    opts = dict(a.split("=", 1) for a in sys.argv[1:])
    variants = opts.get("variants", "ptxas,crit").split(",")
    rounds, out, pat = int(opts.get("rounds", 30)), opts.get("out", "suite"), opts.get("cases", "")
    assert variants[0] == "ptxas" and all(v in VARIANTS for v in variants[1:]), variants
    results, ratios = {}, {v: [] for v in variants[1:]}
    with tempfile.TemporaryDirectory() as tmp:
        for c in CASES:
            if not re.search(pat, c.name):
                continue
            b = toolchain.build(c.src)
            res = {"kind": c.kind, "resources": toolchain.ptxas_resources(b.ptxas_log),
                   "sass_instrs": len(sass.parse(b.sass)), "variants": {}}
            ok = []
            for v in variants:
                try:
                    cub = b.cubin if v == "ptxas" else resched.reschedule(b.cubin, "k", **VARIANTS[v])[0]
                except Exception as e:                                # noqa: BLE001
                    res["variants"][v] = {"status": "build_failed", "error": f"{type(e).__name__}: {e}"[:600]}
                    continue
                path = Path(tmp) / f"{c.name}.{v}.cubin"
                path.write_bytes(cub)
                r, err = run_child("_check", path, c.name, v)
                info = {"identical_to_ptxas": cub == b.cubin}
                if err:
                    info.update(status="fault", error=err)
                else:
                    info.update(sha=r["sha"], registers=r["attrs"]["registers"])
                    if v == "ptxas":
                        info["ref"] = r["ref"]
                        info["status"] = "ok" if r["ref"]["ok"] else "ref_mismatch"
                    else:
                        base = res["variants"]["ptxas"].get("sha")
                        info["status"] = "ok" if r["sha"] == base else "bitwise_mismatch"
                res["variants"][v] = info
                if info["status"] == "ok":
                    ok.append(v)
            if "ptxas" in ok:
                t, err = run_child("_time", tmp, c.name, ",".join(ok), rounds, timeout=1800)
                if err:
                    res["timing_error"] = err
                else:
                    base = statistics.median(t["ptxas"])
                    for v in ok:
                        med = statistics.median(t[v])
                        res["variants"][v].update(us=t[v], median_us=med, ratio=base / med)
                        if v != "ptxas":
                            ratios[v].append(base / med)
            results[c.name] = res
            line = " ".join(f"{v}:{i['status']}" + (f" {i['median_us']:.1f}us x{i['ratio']:.3f}" if "ratio" in i else "")
                            for v, i in res["variants"].items())
            print(f"{c.name:14s} {line}" + (f"  TIMING ERROR {res['timing_error'][-160:]}" if "timing_error" in res else ""),
                  flush=True)
            for v, i in res["variants"].items():
                if i["status"] != "ok":
                    print(f"    {v}: {i['status']} {i.get('error', '')[-300:]}", flush=True)
    geo = {v: math.exp(statistics.fmean(map(math.log, r))) if r else None for v, r in ratios.items()}
    for v, g in geo.items():
        print(f"geomean ptxas/{v}: {g:.4f} over {len(ratios[v])} cases" if g else f"{v}: no timed cases")
    dst = ROOT / "data" / "suite" / f"{out}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"variants": variants, "rounds": rounds, "geomean_ratio": geo,
                               "geomean_cases": {v: len(r) for v, r in ratios.items()}, "cases": results}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_check":
        child_check(*sys.argv[2:5])
    elif len(sys.argv) > 1 and sys.argv[1] == "_time":
        child_time(sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5]))
    else:
        main()
