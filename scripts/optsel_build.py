"""Build the optimization-selection corpus (data/optsel/corpus_<family>.jsonl). CPU only.

One row per decision: family, group (kernel/program), transform, measured y = log(t_base/t_cand) with its
source, and lab/optsel features (static SASS + launch configuration; never a candidate's measured time).

  CUDA_VISIBLE_DEVICES=-1 python scripts/optsel_build.py rand|suite|poly|fusion
    rand    rand_dev (seeds 1000-1039, job 2211) and rand_heldout (seeds 3000-3039, job 2212), clock64
    suite   data/optsel/suite_v2.json (scripts/optsel_suite.py, protocol v2)
    poly    data/optsel/poly_v2.json (scripts/optsel_poly.py, protocol v2)
    fusion  data/xkernel/polybench.json + data/optsel/launch_gap.json
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401
from lab import optsel, toolchain  # noqa: E402
from lab.experiments import randk  # noqa: E402

POLS = ["identity_late", "sb_late", "split_identity_late", "split_sb_late"]


def gap_constants(g):
    """Per-launch gap from launch_gap.json: slope of total time vs launch count for 1- and 64-block
    kernels (stream and graph), i.e. what one more dependent launch costs beyond its work."""
    out = {}
    for mode in ("stream", "graph"):
        sl = []
        for blocks in (1, 64):
            pts = sorted((int(k.split("x")[1]), v[mode]) for k, v in g.items() if int(k.split("x")[0]) == blocks)
            (l0, t0), (l1, t1) = pts[0], pts[-1]
            sl.append((t1 - t0) / (l1 - l0))
        out[mode] = sum(sl) / len(sl)
    return out


def parse_log(txt):
    d = {}
    for m in re.finditer(r"^\[(seed\d+)/(\w+)\]\n((?:  .*\n)*)", txt, re.M):
        b = m.group(3)
        c = re.search(r"cycles/op:\s*([\d.]+)\s*\[p05 ([\d.]+), p95 ([\d.]+)", b)
        r = re.search(r"registers:\s*(\d+)", b)
        w = re.search(r"= (\d+) warps/SM", b)
        ws = re.search(r"warps:\s*(\d+)", b)
        if c:
            d[(m.group(1), m.group(2))] = dict(cyc=float(c[1]), p05=float(c[2]), p95=float(c[3]),
                                                regs=int(r[1]), occ=int(w[1]), warps=int(ws[1]))
    return d


def rand(key, log, fam):
    meas = parse_log((ROOT / log).read_text(encoding="utf-8", errors="replace"))
    e = randk.registry()[key]
    base, rows = {}, []
    for v in e.variants({"n": 40, "policies": ",".join(POLS)}):
        s, pol = v.label.split("/")
        if pol == "ptxas":
            base[s] = toolchain.build(e.source(v)).cubin
            continue
        if (s, pol) not in meas or (s, "ptxas") not in meas:
            continue
        cand = e.transform(base[s], v)
        mb, mc = meas[(s, "ptxas")], meas[(s, pol)]
        f = optsel.sass_decision(base[s], cand, warps_sm=mb["warps"], regs_base=mb["regs"], regs_cand=mc["regs"],
                                 occ_base=mb["occ"], occ_cand=mc["occ"], ctx={})
        sp = 0.5 * (math.log(mc["p95"] / mc["p05"]) + math.log(mb["p95"] / mb["p05"]))
        rows.append({"family": fam, "group": f"{fam}:{s}", "transform": pol, "y": math.log(mb["cyc"] / mc["cyc"]),
                     "spread": sp, "source": log, "features": f})
        print(rows[-1]["group"], pol, round(rows[-1]["y"], 4), round(f["hw_pred"], 4), flush=True)
    return rows


def v2_rows(fam, path, cubin_of, kernel_of, threads_of, grid_of, smem_of):
    """Rows from a protocol-v2 run (optsel_suite / optsel_poly JSON). Every timed candidate becomes a row;
    the ptxas_dup A/A arm is kept as transform 'aa' (its y is pure noise)."""
    d = json.loads((ROOT / path).read_text())
    rows = []
    for name, r in d.items():
        if "y" not in r:
            continue
        base = cubin_of(name, "ptxas").read_bytes()
        thr, gb, sm = threads_of(name, r), grid_of(name, r), smem_of(name, r)
        regs_b = r.get("registers") or r["variants"]["ptxas"]["registers"]
        occ_b = optsel.resident_warps(regs_b, thr, sm, gb)
        tb = statistics.median(x for xs in r["times"]["ptxas"] for x in xs)
        for a, y in r["y"].items():
            if a == "ptxas_dup":
                cand, regs_c, tr = base, regs_b, "aa"
            else:
                cand, regs_c, tr = cubin_of(name, a).read_bytes(), r["variants"][a]["registers"], a
            occ_c = optsel.resident_warps(regs_c, thr, sm, gb)
            f = optsel.sass_decision(base, cand, warps_sm=occ_b, regs_base=regs_b, regs_cand=regs_c,
                                     occ_base=occ_b, occ_cand=occ_c, ctx={}, kernel=kernel_of(name, r))
            rows.append({"family": fam, "group": f"{fam}:{name}", "transform": tr, "y": y, "t_base_us": tb,
                         "source": path, "features": f})
        print(name, len(rows), flush=True)
    return rows


def fusion_rows(gap):
    """polybench_fusion: data/xkernel/polybench.json (scripts/xkernel_bench.py run)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import xkernel_bench as X
    from lab import xkernel
    progs = X.programs()
    d = json.loads((ROOT / "data/xkernel/polybench.json").read_text())
    rows = []
    for p, x in d.items():
        vb, lb = x["variants"]["base"], x["launches"]["base"]
        tb = statistics.median(vb["us"])
        # fraction of pointer arguments that another launch of the program also touches: the data reuse
        # fusion can turn into cache hits (static, from the launch list)
        args = [set(a for a, (_, _, isp) in zip(l.args, xkernel.analyze(l.src).params) if isp)
                for l in progs[p]["launches"]]
        tot = sum(len(a) for a in args)
        shared = sum(len(a & set().union(*(args[:i] + args[i + 1:]))) for i, a in enumerate(args))
        for v in ("graph", "fused", "fused_graph"):
            vc = x["variants"][v]
            if vc.get("status") != "ok":
                continue
            lc = x["launches"][v]
            y = math.log(tb / statistics.median(vc["us"]))
            f = optsel.fusion_decision(tb, lb, lc, v.endswith("graph"), gap, shared / max(tot, 1) if lc < lb else 0.0)
            rows.append({"family": "polybench_fusion", "group": f"polybench_fusion:{p}", "transform": v, "y": y,
                         "t_base_us": tb, "source": "data/xkernel/polybench.json", "features": f})
            print(p, v, round(y, 4), round(f["hw_pred"], 4), flush=True)
    return rows


def write(name, rows):
    p = ROOT / "data" / "optsel" / f"corpus_{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("wrote", p, len(rows))


def main():
    what = sys.argv[1]
    (ROOT / "data" / "optsel").mkdir(parents=True, exist_ok=True)
    if what == "rand":
        for key, log, fam in [("resched_rand", "data/ra/ra-rand-2211.log", "rand_dev"),
                              ("resched_rand_b", "data/ra/ra-rand-b-2212.log", "rand_heldout")]:
            write(fam, rand(key, log, fam))
    elif what == "suite":
        from lab import suite
        cs = {c.name: c for c in suite.CASES}
        write("suite", v2_rows("suite", "data/optsel/suite_v2.json",
                               lambda n, a: ROOT / "data/optsel/cubins" / f"{n}.{a}.cubin", lambda n, r: "k",
                               lambda n, r: int(np.prod(cs[n].block)), lambda n, r: int(np.prod(cs[n].grid)),
                               lambda n, r: cs[n].smem))
    elif what == "poly":
        write("polybench_sass", v2_rows("polybench_sass", "data/optsel/poly_v2.json",
                                        lambda n, a: ROOT / "data/optsel/cubins_poly" / n / f"{a}.cubin",
                                        lambda n, r: r["manifest"]["kernel"],
                                        lambda n, r: int(np.prod(r["manifest"]["block"])),
                                        lambda n, r: int(np.prod(r["manifest"]["grid"])), lambda n, r: 0))
    elif what == "fusion":
        gap = gap_constants(json.loads((ROOT / "data/optsel/launch_gap.json").read_text()))
        print("gap", gap)
        write("polybench_fusion", fusion_rows(gap))


if __name__ == "__main__":
    main()
