"""Benchmark for `3080lab optimize`: ptxas vs source fix vs blind rewrite vs predictor-guided rewrite.

  python scripts/opt_bench.py build A 2          # CPU: compile + optimize suite A with cost model v2
  pcslurm submit -- python scripts/opt_bench.py run A
  python scripts/opt_bench.py report             # results/opt_bench/REPORT.md

Kernels: int4 GEMVs from lab/experiments/gemv3.py. Suite A: 9 shapes x 5 load layouts plus
register-cap (__launch_bounds__(128, 8) -> 64 regs, spills) and -O1 variants. Suite B: shapes never
used while building or revising the optimizer (Qwen2.5-3B, Llama-3-8B, small squares).

Model history (kept honest): cost model v1 was frozen before suite A ran; its decisions are the
held-out result on A. Suite A refuted one v1 hypothesis (re-fetches slow latency-bound kernels);
v2 fixes it from first principles. v2 on A is therefore post hoc; v2's held-out test is suite B.

Arms: orig (ptxas), srcfix (lane-contiguous source where the layout has one), blind (hoist+rename
whenever it validates), guided_v1 / guided_v2 (rewrite iff predicted speedup >= 1.02).
Outputs of rewritten arms must equal orig bitwise; every arm must match a float64 reference; two
random input seeds. Timing: in-kernel globaltimer span, which ticks in ~1.02 us steps, so the MEAN
over 60 launches (random tick phase makes it unbiased) is reported, not the median.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")     # parallel build workers: one BLAS thread each
os.environ.setdefault("OMP_NUM_THREADS", "1")
import ctypes
import hashlib
import json
import math
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "opt_bench"

SUITES = {
    "A": [("o15", 1536, 1536), ("gu15", 17920, 1536), ("dn15", 1536, 8960), ("qo7", 3584, 3584),
          ("gu7", 18944, 3584), ("dn7", 3584, 18944), ("sq4k", 4096, 4096), ("up11k", 11008, 4096),
          ("kv7", 512, 3584)],
    "B": [("qo3", 2048, 2048), ("gu3", 22016, 2048), ("dn3", 2048, 11008), ("gu8", 28672, 4096),
          ("dn8", 4096, 14336), ("sq1k", 1024, 1024), ("sq2k", 2048, 2048)],
}
VARIANT_SHAPES = {"A": ("gu15", "qo7", "sq4k"), "B": ("gu3", "dn8")}
LAYOUTS = [(8, 2, "u_major"), (4, 2, "u_major"), (4, 1, "u_major"), (8, 2, "lane_contig"), (4, 2, "r_major")]
# register cap: -maxrregcount is ignored when the source has __launch_bounds__, so the cap is applied
# as __launch_bounds__(128, 8) (8 blocks of 128 threads per SM -> at most 64 registers)
FLAG_SETS = {"lb8": [], "O1": ["-O1"]}
DEV = "gu7/R8U2/u_major/default"
GROUP = 128


def specs(suite):
    out = []
    for s, n, k in SUITES[suite]:
        for r, u, wo in LAYOUTS:
            out.append(dict(shape=s, N=n, K=k, R=r, U=u, worder=wo, flags="default"))
    for s, n, k in SUITES[suite]:
        if s in VARIANT_SHAPES[suite]:
            for r, u, wo in ((8, 2, "u_major"), (4, 2, "u_major")):
                for f in FLAG_SETS:
                    out.append(dict(shape=s, N=n, K=k, R=r, U=u, worder=wo, flags=f))
    for x in out:
        x["suite"] = suite
        x["id"] = f"{x['shape']}/R{x['R']}U{x['U']}/{x['worder']}/{x['flags']}"
        x["srcfix"] = (x["id"].replace(x["worder"], "lane_contig") if x["U"] >= 2 and x["worder"] != "lane_contig"
                       else None)
    return out


def _variant(sp):
    from lab.experiments.base import Variant
    return Variant(sp["id"], {"N": sp["N"], "K": sp["K"], "R": sp["R"], "U": sp["U"], "T": 128, "deq": "magic",
                              "worder": sp["worder"], "warps": 4, "iters": 1})


def _fname(i: str) -> str:
    return i.replace("/", "__")


def build_one(args):
    sp, ver = args
    f = _fname(sp["id"])
    if (OUT / f"{f}.report_v{ver}.json").exists():           # resumable
        rep = json.loads((OUT / f"{f}.report_v{ver}.json").read_text())
        return sp["id"], ver, rep["chosen"], rep["blind"], rep.get("predicted_speedup")
    from lab import costmodel, optimize, toolchain
    from lab.experiments.gemv3 import Gemv3
    if (OUT / f"{f}.orig.cubin").exists():
        orig = (OUT / f"{f}.orig.cubin").read_bytes()
    else:
        src = Gemv3().source(_variant(sp))
        if sp["flags"] == "lb8":
            assert "__launch_bounds__(128)" in src
            src = src.replace("__launch_bounds__(128)", "__launch_bounds__(128, 8)")
        b = toolchain.build(src, ptxas_flags=FLAG_SETS.get(sp["flags"]))
        orig = b.cubin
        (OUT / f"{f}.orig.cubin").write_bytes(orig)
        (OUT / f"{f}.ptxas.log").write_text(b.ptxas_log)
    grid = -(-sp["N"] // (4 * sp["R"]))
    trips = -(-(sp["K"] // 32) // (32 * sp["U"])) if ver >= 3 else None    # loop trips per warp (launch config)
    _, rep = optimize.optimize(orig, "k", costmodel.Launch(block=128, grid=grid, trips=trips), mode="guided",
                               model_version=ver)
    cubins = rep.pop("_cubins")
    (OUT / f"{f}.guided_v{ver}.cubin").write_bytes(cubins.get(rep["chosen"], orig))
    blind_name = "hoist+rename" if "hoist+rename" in cubins else None
    blind = cubins[blind_name] if blind_name else orig
    if (OUT / f"{f}.blind.cubin").exists() and (OUT / f"{f}.blind.cubin").read_bytes() != blind:
        raise RuntimeError(f"{sp['id']}: blind rewrite is not deterministic")
    (OUT / f"{f}.blind.cubin").write_bytes(blind)
    rep["blind"] = blind_name or "original"
    (OUT / f"{f}.report_v{ver}.json").write_text(json.dumps(rep, indent=1, default=str))
    return sp["id"], ver, rep["chosen"], rep["blind"], rep.get("predicted_speedup")


def build(suite, ver):
    OUT.mkdir(parents=True, exist_ok=True)
    sps = specs(suite)
    extra = [dict(sp, worder="lane_contig", id=sp["srcfix"], srcfix=None) for sp in sps
             if sp["srcfix"] and sp["srcfix"] not in {x["id"] for x in sps}]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(build_one, [(sp, ver) for sp in sps + extra]):
            print(*r, flush=True)


def prepare(dev, sp, seed):
    n, k = sp["N"], sp["K"]
    rng = np.random.default_rng(seed * 1_000_003 + n * 7 + k)
    q = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    packed = np.zeros((n, k // 8), np.uint32)
    for i in range(8):
        packed |= q[:, i::8].astype(np.uint32) << (4 * i)
    s = rng.uniform(0.002, 0.02, size=(n, k // GROUP)).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    ref = np.zeros(n)
    for r0 in range(0, n, 2048):                     # float64 reference, chunked to bound memory
        qq = q[r0:r0 + 2048].astype(np.float64) - 8
        ref[r0:r0 + 2048] = (qq * np.repeat(s[r0:r0 + 2048], GROUP, axis=1)) @ x.astype(np.float64)
    blocks = -(-n // (4 * sp["R"]))
    st_ = {"W": dev.alloc(packed.nbytes), "S": dev.alloc(s.nbytes), "X": dev.alloc(x.nbytes),
           "Y": dev.alloc(n * 4), "T": dev.alloc(blocks * 16), "ref": ref, "blocks": blocks}
    dev.htod(st_["W"], packed)
    dev.htod(st_["S"], s)
    dev.htod(st_["X"], x)
    st_["args"] = [ctypes.c_uint64(st_[z]) for z in ("W", "S", "X", "Y", "T")] + [ctypes.c_int32(n), ctypes.c_int32(k)]
    return st_


def run(suite):
    """One child process per kernel: a faulting rewrite cannot poison the other measurements."""
    import subprocess
    results = {}
    for sp in specs(suite):
        i = sp["id"]
        r = subprocess.run([sys.executable, __file__, "run1", suite, i], capture_output=True, text=True, timeout=900)
        f = OUT / f"{_fname(i)}.run_{suite}2.json"
        if r.returncode == 0 and f.exists():
            results[i] = json.loads(f.read_text())
            print(r.stdout.strip().splitlines()[-1], flush=True)
        else:
            results[i] = {"fault": (r.stderr or "")[-400:]}
            print(i, "FAULT", (r.stderr or "")[-200:], flush=True)
        (OUT / f"run_{suite}2.json").write_text(json.dumps(results, indent=1))


def run1(suite, i):
    from lab.gpu import Device, Kernel
    dev = Device()
    sp = {x["id"]: x for x in specs(suite)}[i]
    f = _fname(i)
    arms = {"orig": OUT / f"{f}.orig.cubin", "blind": OUT / f"{f}.blind.cubin",
            "guided_v1": OUT / f"{f}.guided_v1.cubin", "guided_v3": OUT / f"{f}.guided_v3.cubin",
            "blindD": OUT / f"{f}.blindD.cubin", "guided_v3D": OUT / f"{f}.guided_v3D.cubin"}
    if sp["srcfix"]:
        arms["srcfix"] = OUT / f"{_fname(sp['srcfix'])}.orig.cubin"
    for extra in os.environ.get("OPT_BENCH_EXTRA", "").split(","):
        if extra:
            arms[extra] = OUT / f"{f}.{extra}.cubin"
    arms = {a: p.read_bytes() for a, p in arms.items() if p.exists()}
    ks = {a: Kernel.load(dev, c, "k") for a, c in arms.items()}
    rec = {"times": {a: [] for a in arms}, "hash": {a: [] for a in arms}, "err": {a: [] for a in arms}}
    n_t = int(os.environ.get("OPT_BENCH_TRIALS", "60"))
    for seed, trials in ((1, n_t), (2, 10)):
        data = prepare(dev, sp, seed)
        for a, kern in ks.items():                       # correctness + warm-up
            for _ in range(3):
                kern.launch(data["blocks"], 128, data["args"])
            dev.sync()
            y = dev.dtoh(np.zeros(sp["N"], np.float32), data["Y"])
            rec["hash"][a].append(hashlib.sha1(y.tobytes()).hexdigest()[:12])
            rec["err"][a].append(float(np.max(np.abs(y - data["ref"])) / np.max(np.abs(data["ref"]))))
        for _ in range(trials):                          # interleaved timing
            for a, kern in ks.items():
                kern.launch(data["blocks"], 128, data["args"])
                dev.sync()
                t = dev.dtoh(np.zeros(2 * data["blocks"], np.uint64), data["T"]).reshape(-1, 2)
                rec["times"][a].append(int(t[:, 1].max() - t[:, 0].min()) / 1e3)
        for z in ("W", "S", "X", "Y", "T"):
            dev.free(data[z])
    mean = {a: st.mean(v) for a, v in rec["times"].items()}
    tag = os.environ.get("OPT_BENCH_TAG", f"{suite}2")
    (OUT / f"{f}.run_{tag}.json").write_text(json.dumps(rec))
    ok = all(rec["hash"][a] == rec["hash"]["orig"] for a in arms if a != "srcfix")
    print(i, {a: round(v, 2) for a, v in mean.items()}, "bitwise" if ok else "MISMATCH", flush=True)


def tmean(v, p=0.2):
    """20%-trimmed mean: unbiased under globaltimer quantization (random tick phase) and robust to the
    rare preempted launch (4 us kernels occasionally read 17 us). Chosen on identical-binary controls."""
    v = sorted(v)
    k = int(len(v) * p)
    return st.mean(v[k:len(v) - k])


def noise_bands():
    """Max |ratio - 1| between byte-identical arms (orig vs an unrewritten blind/guided), by duration."""
    bands = {}
    for f in OUT.glob("*.run_[AB]2.json"):
        n = f.name.split(".run_")[0]
        r = json.loads(f.read_text())
        o = (OUT / f"{n}.orig.cubin").read_bytes()
        for arm in ("blind", "guided_v1", "guided_v3"):
            pth = OUT / f"{n}.{arm}.cubin"
            if pth.exists() and pth.read_bytes() == o and arm in r["times"]:
                a, b = r["times"]["orig"][:60], r["times"][arm][:60]
                bands.setdefault(_dur_class(tmean(a)), []).append(abs(tmean(a) / tmean(b) - 1))
    return {k: max(v) for k, v in bands.items()}, {k: len(v) for k, v in bands.items()}


def _dur_class(us):
    return "<8us" if us < 8 else "8-20us" if us < 20 else ">20us"


def dedicate():
    """Derive the +dedicate arms from the existing rewrites (same decisions, same moves):
    blindD = dedicate_barrier(blind) when it validates, guided_v3D = blindD where v3 chose to rewrite."""
    from lab import optimize, schedule
    for suite in SUITES:
        for sp in specs(suite):
            f = _fname(sp["id"])
            o, b = (OUT / f"{f}.orig.cubin").read_bytes(), (OUT / f"{f}.blind.cubin").read_bytes()
            d, why = b, "no rewrite"
            if b != o:
                try:
                    c, why = schedule.dedicate_barrier(b, original=o)
                    if c is not b and optimize.validate(c, o) is None:
                        d = c
                    else:
                        why = f"not applied ({why})"
                except Exception as e:
                    why = f"failed {e!r}"[:120]
            (OUT / f"{f}.blindD.cubin").write_bytes(d)
            rep = json.loads((OUT / f"{f}.report_v3.json").read_text())
            (OUT / f"{f}.guided_v3D.cubin").write_bytes(d if rep["chosen"] != "original" else o)
            print(sp["id"], why, flush=True)


def _gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def report():
    L = ["# `3080lab optimize` benchmark", "",
         "int4 GEMV kernels (lab/experiments/gemv3.py). Times: 20%-trimmed mean of the in-kernel span over 60 "
         "interleaved launches (seed 1; seed 2 checks correctness). Speedup = ptxas time / arm time.", ""]
    all_fails = []
    bands, nb = noise_bands()
    L.append("Noise floor (max deviation between byte-identical binaries, 20%-trimmed mean): "
             + ", ".join(f"{k} {bands[k]:.1%} (n={nb[k]})" for k in sorted(bands)) + ".")
    L.append("")
    for suite in SUITES:
        f = OUT / f"run_{suite}2.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        rows, fails, arms_sp = [], [], {}
        dec, preds = {1: [], 2: [], 3: []}, {1: [], 2: [], 3: []}
        for sp in specs(suite):
            i = sp["id"]
            r = res.get(i)
            if r is None:
                continue
            if "fault" in r:
                fails.append(f"{i}: launch fault {r['fault'][-120:]}")
                continue
            mean = {a: tmean(v[:60]) for a, v in r["times"].items()}
            for a in r["hash"]:
                if a != "srcfix" and r["hash"][a] != r["hash"]["orig"]:
                    fails.append(f"{i} {a}: output differs from ptxas")
                if max(r["err"][a]) > 1e-3:
                    fails.append(f"{i} {a}: error vs float64 reference {max(r['err'][a]):.2e}")
            sp_ = {a: mean["orig"] / mean[a] for a in mean if a != "orig"}
            reps = {v: json.loads((OUT / f"{_fname(i)}.report_v{v}.json").read_text()) for v in (1, 2, 3)
                    if (OUT / f"{_fname(i)}.report_v{v}.json").exists()}
            held = i != DEV
            if held:
                for a, x in sp_.items():
                    arms_sp.setdefault(a, []).append(x)
            for v, rep in reps.items():
                c = rep["candidates"].get("hoist+rename", {})
                if "predicted_speedup" in c and held:
                    target = sp_["blind"]                             # the transform guided mode applies (hoist)
                    preds[v].append((c["predicted_speedup"], target))
                    band = bands[_dur_class(mean["orig"])]
                    truth = "helps" if target >= 1 + band else "hurts" if target <= 1 - band else "neutral"
                    dec[v].append((truth, rep["chosen"] != "original", target))
            rows.append((i, mean["orig"], sp_, {v: reps[v]["chosen"] for v in reps},
                         {v: reps[v]["candidates"].get("hoist+rename", {}).get("predicted_speedup") for v in reps},
                         sum(p["streaming"] for p in reps[1]["split_pairs"]) if 1 in reps else None))
        all_fails += [f"[{suite}] {x}" for x in fails]
        tag = {"A": "suite A (v1 held out; v3 revised on A, post hoc)", "B": "suite B (v1 and v3 both held out)"}[suite]
        L += [f"## {tag}: {len(rows)} kernels", ""]
        L.append("| arm | kernels | geomean speedup | best | worst | regressions (<0.98) |")
        L.append("|---|---|---|---|---|---|")
        for a in ("srcfix", "blind", "guided_v1", "guided_v3", "blindD", "guided_v3D"):
            xs = arms_sp.get(a, [])
            if xs:
                L.append(f"| {a} | {len(xs)} | {_gm(xs):.3f} | {max(xs):.3f} | {min(xs):.3f} | {sum(x < 0.98 for x in xs)} |")
        L.append("")
        for v in (1, 2, 3):
            d = dec[v]
            if not d:
                continue
            sig = [(t, did) for t, did, _ in d if t != "neutral"]
            ok = sum((t == "helps") == did for t, did in sig)
            fp = sum(t == "hurts" and did for t, did in sig)
            fn = sum(t == "helps" and not did for t, did in sig)
            errs = [abs(p - m) / m for p, m in preds[v]]
            L.append(f"- model v{v}: decision accuracy {ok}/{len(sig)} on kernels where the rewrite measurably helps or "
                     f"hurts (beyond the identical-binary noise band; {len(d) - len(sig)} neutral); wrongly applied "
                     f"{fp}, wrongly skipped {fn}; speedup prediction error median {st.median(errs):.1%}, "
                     f"max {max(errs):.1%}")
        L.append("")
        # regret vs a per-kernel oracle (best of ptxas and the rewrite), differences inside the
        # identical-binary noise band counted as ties
        L.append("| policy | geomean regret vs oracle | worst outcome |")
        L.append("|---|---|---|")
        for pol, rw in (("never rewrite", None), ("blind (hoist)", "blind"), ("guided v1 (hoist)", "guided_v1"),
                        ("guided v3 (hoist)", "guided_v3"), ("ablation: blind hoist+dedicate", "blindD")):
            reg, worst, nk = 0.0, 1.0, 0
            for sp in specs(suite):
                i = sp["id"]
                r = res.get(i)
                if r is None or "fault" in r or i == DEV:
                    continue
                o = tmean(r["times"]["orig"][:60])
                band = bands[_dur_class(o)]
                eff = lambda x: 1.0 if abs(x - 1) <= band else x  # noqa: E731
                best = max(1.0, eff(o / tmean(r["times"]["blind"][:60])))       # oracle over the shipped transform
                got = 1.0 if rw is None else eff(o / tmean(r["times"][rw][:60]))
                reg += math.log(best / got)
                worst = min(worst, got)
                nk += 1
            L.append(f"| {pol} | {math.exp(reg / nk) - 1:.2%} | {worst:.3f} |")
        L.append("")
        L.append("| kernel | ptxas us | srcfix | blind (hoist) | guided v1 | guided v3 | ablation: hoist+dedicate | predicted v1 / v3 | split pairs |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        for i, o, s_, ch, pr, npairs in rows:
            L.append(f"| {i} | {o:.2f} | {fmt(s_.get('srcfix'))} | {fmt(s_.get('blind'))} | "
                     f"{fmt(s_.get('guided_v1'))} | {fmt(s_.get('guided_v3'))} | {fmt(s_.get('blindD'))} | "
                     f"{fmt(pr.get(1))} / {fmt(pr.get(3))} | {npairs} |")
        L.append("")
    L.insert(4, f"**Correctness failures: {len(all_fails)}**" + ("" if not all_fails else "\n\n" + "\n".join(f"- {x}" for x in all_fails)))
    L.insert(5, "")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], int(sys.argv[3]))
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "dedicate":
        dedicate()
    elif cmd == "run1":
        run1(sys.argv[2], sys.argv[3])
    else:
        report()
