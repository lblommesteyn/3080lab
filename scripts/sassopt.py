"""One-command SASS rescheduling of an outside CUDA kernel, with compatibility checks and equivalence testing.

  python scripts/sassopt.py kernel.cu --grid 4096 --block 256 --n 1048576 [--scalar n=1048576 --scalar a=2.0]
                            [--kernel name] [--policy sb] [--seeds 4] [--rounds 20] [--int-range 0,1024]
                            [--out kernel.opt.cubin] [--cpu-only]
  (GPU stages: run it through pcslurm submit)

Stages, each of which can reject the rewrite (the output is then ptxas's own cubin, never a broken one):
  1. compatibility  exactly one extern "C" __global__ kernel; every parameter a pointer to a supported
                    element type or a supported scalar; builds for this GPU without spills or stack; the
                    rescheduler accepts it (it refuses blocks it cannot model: indirect branches, barriers
                    it does not understand, and so on) and keeps the register count.
  2. static         lab/verify.py: no read-after-write, write-after-write or write-after-read hazard that
                    ptxas's schedule does not already have (independent of the scheduler's own DAG).
  3. differential   one subprocess per (seed, variant) launches on identical inputs: random seeds plus an
                    edge seed (zeros, +-0, denormals, inf, NaN, extreme ints). Every non-const buffer must
                    match ptxas BITWISE after the launch. A fault or timeout in the candidate rejects it.
  4. timing         interleaved rounds, each launch after a 64 MB L2 flush; the rewrite is kept only if
                    it is >= MIN_GAIN faster by median (run-to-run CV of this harness is about 1%).
Every pointer gets an --n element buffer (or --n name=count per parameter); integer inputs are drawn
from --int-range so index arrays stay in bounds. Report: <out>.json next to the output cubin.
"""
import argparse
import ctypes
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401  (CUDA DLLs nvrtc needs)

from lab import resched, sass, toolchain, verify  # noqa: E402

MIN_GAIN = 1.02
FLUSH_BYTES = 64 << 20
FLUSH_SRC = r"""
extern "C" __global__ void flush(const float4* p, int n, float* sink) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float4 v = p[i]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (s.x + s.y + s.z + s.w == 1234.5f) sink[0] = s.x;
}"""
PTR_TYPES = {"float": np.float32, "double": np.float64, "int": np.int32, "unsigned": np.uint32,
             "unsigned int": np.uint32, "uint32_t": np.uint32, "int32_t": np.int32, "char": np.int8,
             "signed char": np.int8, "unsigned char": np.uint8, "uint8_t": np.uint8, "int8_t": np.int8,
             "short": np.int16, "unsigned short": np.uint16, "long long": np.int64, "int64_t": np.int64,
             "unsigned long long": np.uint64, "uint64_t": np.uint64, "__half": np.float16, "half": np.float16,
             "float2": (np.float32, 2), "float4": (np.float32, 4), "int2": (np.int32, 2), "int4": (np.int32, 4),
             "uint4": (np.uint32, 4)}
SCALAR_TYPES = {"float": ("f32", ctypes.c_float), "int": ("i32", ctypes.c_int32),
                "unsigned": ("u32", ctypes.c_uint32), "unsigned int": ("u32", ctypes.c_uint32),
                "double": ("f64", ctypes.c_double), "long long": ("i64", ctypes.c_int64),
                "size_t": ("u64", ctypes.c_uint64), "int64_t": ("i64", ctypes.c_int64)}
_SIG = re.compile(r'extern\s+"C"\s+__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(\w+)\s*\(([^)]*)\)', re.S)


class Incompatible(Exception):
    pass


def parse_signature(src: str, want: str | None):
    sigs = _SIG.findall(src)
    if want:
        sigs = [s for s in sigs if s[0] == want]
    if len(sigs) != 1:
        raise Incompatible(f"need exactly one extern \"C\" __global__ kernel{' named ' + want if want else ''}, "
                           f"found {[s[0] for s in _SIG.findall(src)]}")
    name, params = sigs[0]
    alias = {k: " ".join(v.split()) for k, v in re.findall(r"^\s*#\s*define\s+(\w+)[ \t]+([\w ]+?)\s*$", src, re.M)}
    alias.update({k: " ".join(v.split()) for v, k in re.findall(r"\btypedef\s+([\w ]+?)\s+(\w+)\s*;", src)})
    out = []
    for p in filter(None, (x.strip() for x in params.split(","))):
        p = re.sub(r"\b(__restrict__|restrict|volatile)\b", " ", p)
        m = re.fullmatch(r"(const\s+)?([\w ]+?)\s*(\*?)\s*(?:const\s+)?(\w+)", " ".join(p.split()))
        if not m:
            raise Incompatible(f"cannot parse parameter {p!r}")
        const, typ, star, pname = m.groups()
        typ = " ".join(typ.split())
        for _ in range(4):                            # resolve #define / typedef aliases (DATA_TYPE -> float)
            typ = alias.get(typ, typ)
        if star:
            if typ not in PTR_TYPES:
                raise Incompatible(f"pointer element type {typ!r} of {pname} not supported")
            out.append({"name": pname, "kind": "ptr", "type": typ, "const": bool(const)})
        else:
            if typ not in SCALAR_TYPES:
                raise Incompatible(f"scalar type {typ!r} of {pname} not supported")
            out.append({"name": pname, "kind": "scalar", "type": typ})
    return name, out


def elem(typ):
    t = PTR_TYPES[typ]
    return (t, 1) if not isinstance(t, tuple) else t


def make_inputs(params, counts, seed, int_range):
    """Host arrays for every pointer (in parameter order). seed -1 = edge values."""
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    arrs = []
    for p in params:
        if p["kind"] != "ptr":
            continue
        dt, w = elem(p["type"])
        n = counts[p["name"]] * w
        if np.issubdtype(dt, np.floating):
            if seed < 0:
                fi = np.finfo(dt)
                edge = np.array([0.0, -0.0, fi.tiny, -fi.tiny, fi.tiny / 4, fi.max, -fi.max, np.inf, -np.inf, np.nan,
                                 1.0, -1.0], dtype=np.float64)
                a = np.resize(edge, n).astype(dt)
            else:
                a = rng.uniform(-1, 1, n).astype(dt)
        else:
            lo, hi = int_range
            if seed < 0:
                ii = np.iinfo(dt)
                a = np.resize(np.array([lo, hi - 1, lo, max(lo, 0)], dtype=np.int64), n).astype(dt)
                if not p["const"]:                    # outputs/in-outs: extreme values are safe to start from
                    a = np.resize(np.array([ii.min, ii.max, 0, -1 if ii.min < 0 else 1], dtype=object), n).astype(dt)
            else:
                a = rng.integers(lo, hi, n).astype(dt)
        arrs.append(a)
    return arrs


def scalar_args(params, scalars):
    out = []
    for p in params:
        if p["kind"] == "scalar":
            if p["name"] not in scalars:
                raise Incompatible(f"no value for scalar {p['name']} (pass --scalar {p['name']}=...)")
            tag, ct = SCALAR_TYPES[p["type"]]
            v = scalars[p["name"]]
            out.append(ct(float(v) if tag.startswith("f") else int(v)))
    return out


def launch_args(params, ptrs, scal):
    a, ip, isc = [], iter(ptrs), iter(scal)
    for p in params:
        a.append(ctypes.c_uint64(next(ip)) if p["kind"] == "ptr" else next(isc))
    return a


# ---- GPU children (one process each, so a fault cannot poison another variant) --------------------------

def child_check(spec_path, cubin_path, seed):
    from lab.gpu import Device, Kernel
    spec = json.loads(Path(spec_path).read_text())
    params = spec["params"]
    dev = Device()
    arrs = make_inputs(params, spec["counts"], int(seed), spec["int_range"])
    ptrs = []
    for a in arrs:
        p = dev.alloc(max(a.nbytes, 4))
        dev.htod(p, np.ascontiguousarray(a))
        ptrs.append(p)
    k = Kernel.load(dev, Path(cubin_path).read_bytes(), spec["kernel"])
    k.launch(tuple(spec["grid"]), tuple(spec["block"]), launch_args(params, ptrs, scalar_args(params, spec["scalars"])),
             spec["smem"])
    dev.sync()
    h = hashlib.sha256()
    for p, a, prm in zip(ptrs, arrs, [x for x in params if x["kind"] == "ptr"]):
        if not prm["const"]:
            h.update(dev.dtoh(np.empty_like(a), p).tobytes())
    print("RESULT " + json.dumps({"sha": h.hexdigest()[:16], "registers": k.attrs(int(np.prod(spec["block"])),
                                                                                    spec["smem"])["registers"]}))


def child_time(spec_path, ptxas_path, opt_path, rounds):
    from lab.gpu import Device, Kernel
    spec = json.loads(Path(spec_path).read_text())
    params = spec["params"]
    dev = Device()
    scal = scalar_args(params, spec["scalars"])
    kern, args = {}, {}
    for v, path in (("ptxas", ptxas_path), ("opt", opt_path)):
        arrs = make_inputs(params, spec["counts"], 0, spec["int_range"])
        ptrs = []
        for a in arrs:
            p = dev.alloc(max(a.nbytes, 4))
            dev.htod(p, np.ascontiguousarray(a))
            ptrs.append(p)
        kern[v] = Kernel.load(dev, Path(path).read_bytes(), spec["kernel"])
        args[v] = launch_args(params, ptrs, scal)
    fl = Kernel.load(dev, toolchain.build(FLUSH_SRC).cubin, "flush")
    buf, sink = dev.alloc(FLUSH_BYTES), dev.alloc(4)
    dev.memset(buf, FLUSH_BYTES)
    fargs = [ctypes.c_uint64(buf), ctypes.c_int32(FLUSH_BYTES // 16), ctypes.c_uint64(sink)]
    rng = random.Random(1)
    times = {"ptxas": [], "opt": []}
    for r in range(int(rounds) + 2):
        order = ["ptxas", "opt"]
        rng.shuffle(order)
        for v in order:
            fl.launch(dev.props["sms"] * 4, 256, fargs)
            t = kern[v].timed_launch(tuple(spec["grid"]), tuple(spec["block"]), args[v], spec["smem"])
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


def dims(s):
    t = tuple(int(x) for x in s.split(","))
    return t + (1,) * (3 - len(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--kernel")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--smem", type=int, default=0, help="dynamic shared memory bytes")
    ap.add_argument("--n", action="append", required=True, help="elements per buffer: N, or name=N")
    ap.add_argument("--scalar", action="append", default=[], help="name=value")
    ap.add_argument("--int-range", default="0,1024")
    ap.add_argument("--policy", default="sb")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--cpu-only", action="store_true")
    a = ap.parse_args()
    src_path = Path(a.source)
    out = Path(a.out) if a.out else src_path.with_suffix(".opt.cubin")
    rep = {"source": str(src_path), "policy": a.policy, "stages": {}, "decision": None}

    def finish(cubin, decision):
        rep["decision"] = decision
        out.write_bytes(cubin)
        out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
        print(f"\n{decision}\nwrote {out} and {out.with_suffix('.json')}")

    src = src_path.read_text()
    # 1. compatibility
    try:
        name, params = parse_signature(src, a.kernel)
        default_n = next((int(x) for x in a.n if "=" not in x), None)
        counts = {p["name"]: default_n for p in params if p["kind"] == "ptr"}
        counts.update({k: int(v) for k, v in (x.split("=", 1) for x in a.n if "=" in x)})
        if any(v is None for v in counts.values()):
            raise Incompatible(f"no --n for {[k for k, v in counts.items() if v is None]}")
        scalars = dict(x.split("=", 1) for x in a.scalar)
        scalar_args(params, scalars)
        b = toolchain.build(src)
        res = toolchain.ptxas_resources(b.ptxas_log)
        if res.get("spill_stores") or res.get("stack_bytes"):
            raise Incompatible(f"ptxas build spills or uses a stack frame ({res}); not rescheduled")
        try:
            opt, log = resched.reschedule(b.cubin, name, policy=a.policy)
        except Exception as e:                                        # noqa: BLE001
            raise Incompatible(f"rescheduler refused: {type(e).__name__}: {e}") from e
        sched = [m for _, m in log if not m.startswith("kept order")]
        oi, ni = sass.parse(b.sass), sass.parse(toolchain.disassemble(opt))
        moved = sum(x.text.strip() != y.text.strip() for x, y in zip(oi, ni))
        ctrl = sum(x.text.strip() == y.text.strip() and x.hi != y.hi for x, y in zip(oi, ni))
        rep["stages"]["compat"] = {"ok": True, "kernel": name, "params": params, "resources": res,
                                   "sass_instrs": len(oi), "blocks_reordered": len(sched), "slots_new_instr": moved,
                                   "slots_new_control_only": ctrl, "identical": opt == b.cubin}
    except Incompatible as e:
        rep["stages"]["compat"] = {"ok": False, "reason": str(e)}
        print(f"INCOMPATIBLE: {e}")
        if "b" in locals():
            return finish(b.cubin, "kept ptxas: incompatible input")
        sys.exit(2)
    print(f"compat ok: {name}, {len(params)} params, {res.get('registers')} regs, {len(sched)} blocks reordered, {moved} slots new instr, {ctrl} control-only")
    if opt == b.cubin:
        return finish(b.cubin, "kept ptxas: the rescheduler produced the same code")
    # 2. static hazards
    haz = verify.new_hazards(opt, b.cubin, name)
    rep["stages"]["static"] = {"ok": not haz, "new_hazards": [str(h) for h in haz[:20]]}
    if haz:
        print(f"static check: {len(haz)} new hazards")
        return finish(b.cubin, "kept ptxas: static hazard check failed")
    print("static check ok")
    if a.cpu_only:
        return finish(opt, "rewrite passed CPU checks only (no differential test, no timing): use with care")
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"kernel": name, "params": params, "counts": counts, "scalars": scalars, "smem": a.smem,
                "grid": dims(a.grid), "block": dims(a.block), "int_range": [int(x) for x in a.int_range.split(",")]}
        sp, pp, op = Path(tmp) / "spec.json", Path(tmp) / "ptxas.cubin", Path(tmp) / "opt.cubin"
        sp.write_text(json.dumps(spec))
        pp.write_bytes(b.cubin)
        op.write_bytes(opt)
        # 3. differential
        diff = []
        for seed in [-1] + list(range(a.seeds)):
            r0, e0 = run_child("_check", sp, pp, seed)
            if e0:
                rep["stages"]["differential"] = {"ok": False, "reason": f"ptxas build fails on seed {seed}: {e0}"}
                return finish(b.cubin, "kept ptxas: the original kernel faults on these inputs (fix --n/--int-range)")
            r1, e1 = run_child("_check", sp, op, seed)
            row = {"seed": seed, "ptxas": r0["sha"], "opt": r1["sha"] if r1 else None, "error": e1,
                   "registers": [r0["registers"], r1["registers"] if r1 else None]}
            diff.append(row)
            print(f"seed {seed:2d}: ptxas {r0['sha']} opt {row['opt']} "
                  f"{'MATCH' if row['opt'] == r0['sha'] else 'MISMATCH ' + (e1 or '')[-200:]}", flush=True)
        ok = all(d["opt"] == d["ptxas"] for d in diff)
        rep["stages"]["differential"] = {"ok": ok, "seeds": diff}
        if not ok:
            return finish(b.cubin, "kept ptxas: differential test failed")
        # 4. timing
        t, err = run_child("_time", sp, pp, op, a.rounds, timeout=1800)
        if err:
            rep["stages"]["timing"] = {"ok": False, "reason": err}
            return finish(b.cubin, "kept ptxas: timing failed")
        mp, mo = statistics.median(t["ptxas"]), statistics.median(t["opt"])
        rep["stages"]["timing"] = {"ok": True, "ptxas_us": mp, "opt_us": mo, "speedup": mp / mo, "samples": t}
        print(f"timing: ptxas {mp:.2f} us, rescheduled {mo:.2f} us, x{mp / mo:.3f}")
        if mp / mo >= MIN_GAIN:
            return finish(opt, f"kept rewrite: x{mp / mo:.3f} faster, bitwise equal on {len(diff)} input sets")
        return finish(b.cubin, f"kept ptxas: rewrite x{mp / mo:.3f}, below the {MIN_GAIN} threshold")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_check":
        child_check(*sys.argv[2:5])
    elif len(sys.argv) > 1 and sys.argv[1] == "_time":
        child_time(*sys.argv[2:6])
    else:
        main()
