"""Cross-kernel optimizer (lab/xkernel.py) on the multi-kernel PolyBench/GPU programs (outside code).

  python scripts/xkernel_bench.py plan                         CPU: build programs, print fusion decisions
  pcslurm submit -- python scripts/xkernel_bench.py run [rounds=20]

Programs come from the PolyBench host code (scripts/ext_polybench.py fetches and extracts it): the kernel
launches in source order, buffer names from the host arguments, the benchmark's own time / panel loop
unrolled for a few iterations (fdtd-2d and jacobi t = 0..9, gramschmidt and lu k = 0..7, 3dconv i = 1..8).
Variants: base (one stream launch per kernel, the original program), graph (the same launches in one CUDA
graph), fused (lab/xkernel.plan, stream launches), fused_graph. Check: one subprocess per variant runs the
program once on identical seeded buffers; EVERY buffer must match base bitwise afterwards. Timing: one
subprocess interleaves the variants that passed, each program run preceded by a 64 MB L2 flush.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402

from lab import toolchain, xkernel  # noqa: E402

KDIR = ROOT / "vendor" / "ext_kernels"
HOST = ROOT / "vendor" / "polybenchGpu" / "CUDA"
OUT = ROOT / "data" / "xkernel"
LOOPS = {"FDTD-2D": ("t", range(10)), "JACOBI1D": (None, range(10)), "JACOBI2D": (None, range(10)),
         "GRAMSCHM": ("k", range(8)), "LU": ("k", range(8)), "3DCONV": ("i", range(1, 9))}
SKIP = {"ADI": "nested data-dependent inner loops", "2MM": "not extracted", "3MM": "not extracted",
        "DOITGEN": "not extracted"}
FLUSH_BYTES = 64 << 20
FLUSH_SRC = r"""
extern "C" __global__ void flush(const float4* p, int n, float* sink) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float4 v = p[i]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (s.x + s.y + s.z + s.w == 1234.5f) sink[0] = s.x;
}"""


def programs():
    man = {(m["bench"], m["kernel"]): m for m in json.loads((KDIR / "manifest.json").read_text())}
    progs = {}
    for d in sorted(p for p in HOST.iterdir() if p.is_dir()):
        if d.name in SKIP:
            continue
        cu = next(d.glob("*.cu"), None)
        calls = re.findall(r"(\w+)\s*<<<[^>]*>>>\s*\(([^)]*)\)", cu.read_text(errors="replace")) if cu else []
        if len(calls) < 2 and d.name not in LOOPS:
            continue
        var, iters = LOOPS.get(d.name, (None, [0]))
        seq, nbuf = [], 0
        for it in iters:
            for kname, args in calls:
                m = man.get((d.name, kname))
                if m is None:
                    break
                src = (KDIR / m["file"]).read_text()
                info = xkernel.analyze(src)
                argv = []
                for (pname, _, is_ptr), a in zip(info.params, [x.strip() for x in args.split(",")]):
                    if is_ptr:
                        argv.append(re.sub(r"_?[gG]pu$", "", a))
                    elif pname == var or a == var:
                        argv.append(it)
                    else:
                        argv.append(m["scalars"].get(pname, 1))
                seq.append(xkernel.Launch(src, tuple(m["grid"]), tuple(m["block"]), argv))
                nbuf = max(nbuf, m["n"])
        if len(seq) >= 2:
            progs[d.name.lower()] = {"launches": seq, "n": nbuf}
    return progs


def variants(prog):
    fused, log = xkernel.plan(prog["launches"])
    return {"base": (prog["launches"], False), "graph": (prog["launches"], True),
            "fused": (fused, False), "fused_graph": (fused, True)}, log


def _scal(info, a, pname):
    decl = next(d for p, d, _ in info.params if p == pname)
    return ctypes.c_float(float(a)) if re.search(r"\b(float|DATA_TYPE)\b", decl) else ctypes.c_int32(int(a))


class Runner:
    """Loads a launch list into one torch context; run() issues it on the current stream."""
    def __init__(self, launches, bufs):
        from qwen_e2e import check, cu
        self.check, self.cu = check, cu
        self.items, self.keep = [], []
        cache = {}
        for l in launches:
            if l.src not in cache:
                info = xkernel.analyze(l.src)
                mod = check(cu.cuModuleLoadData(toolchain.build(l.src).cubin))
                cache[l.src] = (info, check(cu.cuModuleGetFunction(mod, info.name.encode())))
            info, fn = cache[l.src]
            a = [ctypes.c_uint64(bufs[x].data_ptr()) if is_ptr else _scal(info, x, p)
                 for (p, _, is_ptr), x in zip(info.params, l.args)]
            ptrs = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
            self.keep.append((a, ptrs))
            g = tuple(l.grid) + (1,) * (3 - len(l.grid))
            b = tuple(l.block) + (1,) * (3 - len(l.block))
            self.items.append((fn, g, b, ptrs))

    def run(self):
        st = self.cu.CUstream(torch.cuda.current_stream().cuda_stream)
        for fn, g, b, ptrs in self.items:
            self.check(self.cu.cuLaunchKernel(fn, *g, *b, 0, st, ctypes.addressof(ptrs), 0))


def _setup(prog_name):
    from qwen_e2e import check, cu
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    prog = programs()[prog_name]
    names = sorted({a for l in prog["launches"] for a, (_, _, p) in
                    zip(l.args, xkernel.analyze(l.src).params) if p})
    return prog, names


def _init(names, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return {b: torch.rand(n, device="cuda", generator=g) * 2 - 1 for b in names}


def child_check(prog_name, variant):
    prog, names = _setup(prog_name)
    launches, use_graph = variants(prog)[0][variant]
    bufs = _init(names, prog["n"])
    r = Runner(launches, bufs)
    if use_graph:
        s = torch.cuda.Stream()
        gr = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(gr, stream=s):
            r.run()
        bufs2 = _init(names, prog["n"])               # capture does not execute; restore inputs, replay once
        for b in names:
            bufs[b].copy_(bufs2[b])
        gr.replay()
    else:
        r.run()
    torch.cuda.synchronize()
    h = hashlib.sha256()
    for b in names:
        h.update(bufs[b].cpu().numpy().tobytes())
    print("RESULT " + json.dumps({"sha": h.hexdigest()[:16], "launches": len(launches)}))


def child_time(prog_name, vlist, rounds):
    prog, names = _setup(prog_name)
    vs = variants(prog)[0]
    from qwen_e2e import check, cu
    fl_mod = check(cu.cuModuleLoadData(toolchain.build(FLUSH_SRC).cubin))
    fl = check(cu.cuModuleGetFunction(fl_mod, b"flush"))
    fbuf, sink = torch.zeros(FLUSH_BYTES // 4, device="cuda"), torch.zeros(1, device="cuda")
    fa = [ctypes.c_uint64(fbuf.data_ptr()), ctypes.c_int32(FLUSH_BYTES // 16), ctypes.c_uint64(sink.data_ptr())]
    fp = (ctypes.c_void_p * 3)(*[ctypes.addressof(z) for z in fa])
    run = {}
    for v in vlist:
        launches, use_graph = vs[v]
        bufs = _init(names, prog["n"])
        r = Runner(launches, bufs)
        if use_graph:
            s = torch.cuda.Stream()
            gr = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(gr, stream=s):
                r.run()
            run[v] = (gr.replay, bufs, r)
        else:
            run[v] = (r.run, bufs, r)
    rng = random.Random(0)
    times = {v: [] for v in vlist}
    for rd in range(rounds + 2):
        order = list(vlist)
        rng.shuffle(order)
        for v in order:
            st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
            check(cu.cuLaunchKernel(fl, 272, 1, 1, 256, 1, 1, 0, st, ctypes.addressof(fp), 0))
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            run[v][0]()
            e1.record()
            torch.cuda.synchronize()
            if rd >= 2:
                times[v].append(e0.elapsed_time(e1) * 1e3)
    print("RESULT " + json.dumps(times))


def run_child(*argv, timeout=900):
    try:
        r = subprocess.run([sys.executable, __file__, *map(str, argv)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout} s"
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    if r.returncode or line is None:
        return None, (r.stderr.strip() or r.stdout.strip())[-800:]
    return json.loads(line[7:]), None


def cmd_plan():
    for name, prog in programs().items():
        fused, log = xkernel.plan(prog["launches"])
        for l in fused:
            toolchain.build(l.src)                      # every fused kernel must compile
        print(f"{name:10s} {len(prog['launches'])} launches -> {len(fused)}")
        for x in log:
            if not x.startswith("boundary") or "--all" in sys.argv:
                print("    " + x)
        bounds = [x for x in log if x.startswith("boundary")]
        if bounds:
            print(f"    {len(bounds)} boundaries kept, e.g. {bounds[0][9:160]}")


def cmd_run(rounds):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, prog in programs().items():
        vs, log = variants(prog)
        row = {"launches": {v: len(l) for v, (l, _) in vs.items()}, "plan": log, "variants": {}}
        ok = []
        base = None
        for v in vs:
            r, err = run_child("_check", name, v)
            if err:
                row["variants"][v] = {"status": "fault", "error": err}
                continue
            if v == "base":
                base = r["sha"]
            st = "ok" if r["sha"] == base else "bitwise_mismatch"
            if v != "base" and vs[v][0] is vs["base"][0] and not vs[v][1]:
                st = "ok"
            row["variants"][v] = {"status": st, "sha": r["sha"]}
            if st == "ok":
                ok.append(v)
        if "base" in ok:
            t, err = run_child("_time", name, ",".join(ok), rounds, timeout=1800)
            if err:
                row["timing_error"] = err
            else:
                mb = statistics.median(t["base"])
                for v in ok:
                    m = statistics.median(t[v])
                    row["variants"][v].update(median_us=m, speedup=mb / m, us=t[v])
        res[name] = row
        print(f"{name:10s} " + "  ".join(f"{v}:{i['status']}" + (f" {i['median_us']:.0f}us x{i['speedup']:.3f}"
                                                                   if "speedup" in i else "")
                                         for v, i in row["variants"].items()), flush=True)
        for v, i in row["variants"].items():
            if i["status"] != "ok":
                print(f"    {v}: {i['status']} {i.get('error', '')[-300:]}", flush=True)
    (OUT / "polybench.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'polybench.json'}")


if __name__ == "__main__":
    if sys.argv[1] == "_check":
        child_check(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "_time":
        child_time(sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]))
    elif sys.argv[1] == "plan":
        cmd_plan()
    else:
        kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
        cmd_run(int(kv.get("rounds", 20)))
