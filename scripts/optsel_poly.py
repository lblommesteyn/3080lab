"""PolyBench/GPU SASS decisions under timing protocol v2 (data/optsel/PROTOCOL.md): 40 kernels from
vendor/ext_kernels (scripts/ext_polybench.py extract) x the suite's 7 transforms, plus an A/A arm.

Per kernel: compatibility (scripts/sassopt.parse_signature, no spills), transform, lab/verify static
hazard check, bitwise differential vs ptxas on an edge seed and 2 random seeds (sassopt's _check child),
then protocol-v2 timing of every passing arm (K arms over K input-buffer sets rotated per round, L2
flush before each launch, trimmed mean of paired log ratios, 2 fresh processes).

  pcslurm submit -- python scripts/optsel_poly.py [rounds=36] [procs=2] [only=regex] [vendor=PATH]
"""
import ctypes
import json
import math
import random
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401
import numpy as np  # noqa: E402

from lab import resched, toolchain, verify  # noqa: E402
import sassopt as SO  # noqa: E402
from optsel_suite import CANDS, tmean  # noqa: E402
from suite_run import VARIANTS  # noqa: E402


def transform(cubin, name, v):
    kw = dict(VARIANTS[v])
    extra = kw.pop("split", None)
    if extra:
        from lab import jointra
        cubin = jointra.split_crit(cubin, name, extra=extra)[0]
    return resched.reschedule(cubin, name, **kw)[0]


def child_time(spec_path, cdir, arms, rounds, seed):
    from lab.gpu import Device, Kernel
    spec = json.loads(Path(spec_path).read_text())
    params = spec["params"]
    dev = Device()
    scal = SO.scalar_args(params, spec["scalars"])
    K = len(arms)
    src = {a: ("ptxas" if a == "ptxas_dup" else a) for a in arms}
    kern = {a: Kernel.load(dev, (Path(cdir) / f"{src[a]}.cubin").read_bytes(), spec["kernel"]) for a in arms}
    sets = []
    for _ in range(K):
        ptrs = []
        for x in SO.make_inputs(params, spec["counts"], 0, spec["int_range"]):
            p = dev.alloc(max(x.nbytes, 4))
            dev.htod(p, np.ascontiguousarray(x))
            ptrs.append(p)
        sets.append(SO.launch_args(params, ptrs, scal))
    fl = Kernel.load(dev, toolchain.build(SO.FLUSH_SRC).cubin, "flush")
    buf, sink = dev.alloc(SO.FLUSH_BYTES), dev.alloc(4)
    dev.memset(buf, SO.FLUSH_BYTES)
    fargs = [ctypes.c_uint64(buf), ctypes.c_int32(SO.FLUSH_BYTES // 16), ctypes.c_uint64(sink)]
    rng = random.Random(seed)
    t = {a: [] for a in arms}
    for r in range(rounds + 2):
        order = list(range(K))
        rng.shuffle(order)
        for i in order:
            fl.launch(dev.props["sms"] * 4, 256, fargs)
            x = kern[arms[i]].timed_launch(tuple(spec["grid"]), tuple(spec["block"]), sets[(i + r) % K], spec["smem"])
            if r >= 2:
                t[arms[i]].append(x * 1e3)
    print("RESULT " + json.dumps(t))


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:])
    rounds, procs, only = int(kv.get("rounds", 36)), int(kv.get("procs", 2)), kv.get("only", "")
    kdir = Path(kv.get("vendor", ROOT / "vendor")) / "ext_kernels"
    man = json.loads((kdir / "manifest.json").read_text())
    res = {}
    for m in man:
        if not re.search(only, m["file"]):
            continue
        key = m["file"][:-3]
        row = {"manifest": m, "variants": {}}
        res[key] = row
        src = (kdir / m["file"]).read_text()
        try:
            name, params = SO.parse_signature(src, m["kernel"])
            b = toolchain.build(src)
            rsc = toolchain.ptxas_resources(b.ptxas_log)
            row["resources"] = rsc
            if rsc.get("spill_stores") or rsc.get("stack_bytes"):
                raise SO.Incompatible("spills")
        except Exception as e:  # noqa: BLE001
            row["status"] = f"incompatible: {e}"[:300]
            print(key, row["status"], flush=True)
            continue
        n = m["n"]
        spec = {"kernel": name, "params": params, "counts": {p["name"]: n for p in params if p["kind"] == "ptr"},
                "scalars": m["scalars"], "smem": 0, "grid": SO.dims(",".join(map(str, m["grid"]))),
                "block": SO.dims(",".join(map(str, m["block"]))), "int_range": [0, 1024]}
        cd = ROOT / "data/optsel/cubins_poly" / key
        cd.mkdir(parents=True, exist_ok=True)
        (cd / "spec.json").write_text(json.dumps(spec))
        (cd / "ptxas.cubin").write_bytes(b.cubin)
        base_sha = {}
        for seed in (-1, 0, 1):
            r0, _ = SO.run_child("_check", cd / "spec.json", cd / "ptxas.cubin", seed)
            base_sha[seed] = r0["sha"] if r0 else None
            if seed == 0 and r0:
                row["registers"] = r0["registers"]
        if None in base_sha.values():
            row["status"] = "ptxas faults on these inputs"
            print(key, row["status"], flush=True)
            continue
        ok = ["ptxas"]
        for v in CANDS:
            info = {}
            try:
                cub = transform(b.cubin, name, v)
            except Exception as e:  # noqa: BLE001
                row["variants"][v] = {"status": f"refused: {type(e).__name__}: {e}"[:200]}
                continue
            (cd / f"{v}.cubin").write_bytes(cub)
            info["identical_to_ptxas"] = cub == b.cubin
            haz = verify.new_hazards(cub, b.cubin, name) if cub != b.cubin else []
            if haz:
                info["status"] = f"static: {len(haz)} hazards"
                row["variants"][v] = info
                continue
            bad = None
            for seed in (-1, 0, 1):
                r1, e1 = SO.run_child("_check", cd / "spec.json", cd / f"{v}.cubin", seed)
                if not r1:
                    bad = f"seed {seed}: fault {(e1 or '')[-150:]}"
                    break
                if r1["sha"] != base_sha[seed]:
                    bad = f"seed {seed}: bitwise mismatch"
                    break
                info["registers"] = r1["registers"]
            info["status"] = bad or "ok"
            row["variants"][v] = info
            if not bad:
                ok.append(v)
        arms = ["ptxas", "ptxas_dup"] + ok[1:]
        times = {a: [] for a in arms}
        for p in range(procs):
            r = subprocess.run([sys.executable, __file__, "_time", str(cd / "spec.json"), str(cd), ",".join(arms),
                                str(rounds), str(zlib.crc32(key.encode()) + p)],
                               capture_output=True, text=True, timeout=1800)
            line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
            if not line:
                row["timing_error"] = r.stderr[-300:]
                break
            for a, xs in json.loads(line[7:]).items():
                times[a].append(xs)
        row["times"] = times
        if "timing_error" not in row:
            row["y"] = {a: tmean([math.log(x / y) for pb, pa in zip(times["ptxas"], times[a]) for x, y in zip(pb, pa)])
                        for a in arms[1:]}
        row["status"] = "ok"
        bad = " ".join(f"{v}:{i['status']}" for v, i in row["variants"].items() if i["status"] != "ok")
        print(key, " ".join(f"{a}:{v:+.4f}" for a, v in row.get("y", {}).items()), row.get("timing_error", "")[-200:],
              "|", bad, flush=True)
    dst = ROOT / "data/optsel" / kv.get("out", "poly_v2.json")
    dst.write_text(json.dumps(res))
    print("wrote", dst)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_time"]:
        child_time(sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
