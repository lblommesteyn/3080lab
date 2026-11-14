"""Build the optimization-selection corpus (data/optsel/corpus_<family>.jsonl). CPU only.

One row per decision: family, group (kernel/program), transform, measured y = log(t_base/t_cand) with its
source, and lab/optsel features. Families:
  rand_dev   random kernels seeds 1000-1039, job 2211 (clock64 cycles/op)
  rand_heldout  seeds 3000-3039, job 2212
  (suite, polybench_sass, polybench_fusion are added by their own runs, see PROTOCOL.md)

  CUDA_VISIBLE_DEVICES=-1 python scripts/optsel_build.py rand
"""
import json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa
from lab import optsel, toolchain
from lab.experiments import randk

POLS = ["identity_late", "sb_late", "split_identity_late", "split_sb_late"]


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
    vs = e.variants({"n": 40, "policies": ",".join(POLS)})
    base = {}
    rows = []
    for v in vs:
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
        y = math.log(mb["cyc"] / mc["cyc"])
        # within-run spread: half-width of the p05-p95 ranges, in log units
        sp = 0.5 * (math.log(mc["p95"] / mc["p05"]) + math.log(mb["p95"] / mb["p05"]))
        rows.append({"family": fam, "group": f"{fam}:{s}", "transform": pol, "y": y, "spread": sp,
                     "source": log, "features": f})
        print(json.dumps({k: rows[-1][k] for k in ("group", "transform", "y")} | {"hw": round(f["hw_pred"], 4)}), flush=True)
    return rows


def main():
    what = sys.argv[1]
    out = ROOT / "data" / "optsel"
    out.mkdir(parents=True, exist_ok=True)
    if what == "rand":
        for key, log, fam in [("resched_rand", "data/ra/ra-rand-2211.log", "rand_dev"),
                              ("resched_rand_b", "data/ra/ra-rand-b-2212.log", "rand_heldout")]:
            rows = rand(key, log, fam)
            (out / f"corpus_{fam}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print("wrote", fam, len(rows))


if __name__ == "__main__":
    main()
