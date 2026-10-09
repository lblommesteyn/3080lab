"""3080lab command line.

  3080lab list
  3080lab sass dependent_ffma          # compile only, no GPU
  3080lab run dependent_ffma           # queues through pcslurm, waits, prints report
  3080lab run dependent_ffma --local   # run in this process (what the queued job does)
  3080lab optimize k.cubin --block 128 # split-sector analysis, prediction, safe rewrite
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import runner, sass, toolchain
from .experiments import registry

ROOT = Path(__file__).resolve().parents[1]


def _ints(s: str | None):
    return [int(x) for x in s.split(",")] if s else None


def _sizes(s: str | None):
    if not s:
        return None
    mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    return [int(float(x[:-1]) * mult[x[-1].lower()]) if x[-1].lower() in mult else int(x) for x in s.split(",")]


def cmd_list(_):
    for name, exp in registry().items():
        print(f"{name:<24} {exp.description}")


def cmd_sass(a):
    exp = registry()[a.experiment]
    if a.ptxas is not None:
        exp.ptxas_flags = a.ptxas.split()
    v = exp.variants({})[0]
    b = toolchain.build(exp.source(v), ptxas_flags=exp.ptxas_flags)
    ins = sass.parse(b.sass)
    print(b.ptxas_log.strip())
    print(sass.format_listing(ins if a.all else sass.loop_body(ins), raw=not a.no_raw))


def _submit(a) -> int:
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    argv = [str(py), "-m", "lab.cli", "run", *a.experiment, "--local", "--trials", str(a.trials)]
    for flag in ("warps", "iters", "seed", "lock_clock", "ptxas", "sizes"):
        val = getattr(a, flag)
        if val:
            argv += [f"--{flag.replace('_', '-')}={val}"]
    if a.force:
        argv.append("--force")
    if a.cold:
        argv.append("--cold")
    r = subprocess.run(["pcslurm", "submit", "--shared", "-p", str(ROOT), "-J", f"3080lab-{a.experiment[0]}" + (f"+{len(a.experiment) - 1}" if len(a.experiment) > 1 else ""), "--", *argv],
                       capture_output=True, text=True)
    sys.stderr.write(r.stdout + r.stderr)
    m = re.search(r"(\d{2,})", r.stdout + r.stderr)
    if r.returncode or not m:
        print("submit failed", file=sys.stderr)
        return 1
    job = m.group(1)
    print(f"queued job {job}; waiting ...", file=sys.stderr)
    while True:
        st = subprocess.run(["pcslurm", "job", job], capture_output=True, text=True).stdout
        if re.search(r"COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL", st):
            break
        time.sleep(3)
    print(subprocess.run(["pcslurm", "logs", job], capture_output=True, text=True).stdout)
    return 0 if "COMPLETED" in st else 1


def cmd_run(a):
    if not a.local:
        return _submit(a)
    reg = registry()
    unknown = [e for e in a.experiment if e not in reg]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    failed = 0
    for name in a.experiment:
        exp = reg[name]
        if a.ptxas is not None:
            exp.ptxas_flags = a.ptxas.split()
        opts = {"force": a.force, "trials": a.trials, "warps": _ints(a.warps), "iters": a.iters,
                "seed": a.seed, "lock_clock": a.lock_clock, "sizes": _sizes(a.sizes), "cold": a.cold}
        try:
            rec = runner.run(exp, opts)
            print(runner.report(rec), flush=True)
        except (SystemExit, Exception) as e:  # refusal or crash: report and continue the batch
            print(f"{name}: {e}", flush=True)
            failed += 1
        print("=" * 100, flush=True)
    return 1 if failed else 0


def cmd_show(a):
    rec = json.loads((Path(a.path) / "record.json").read_text())
    print(runner.report(rec))


def cmd_table(a):
    """Latest record per experiment -> one markdown table (also written to results/TABLE.md)."""
    latest = {}
    for f in sorted(runner.RESULTS.glob("*/record.json")):
        rec = json.loads(f.read_text())
        latest[rec["experiment"]] = rec
    rows = ["| experiment | variant | cycles/op | CV | warp-instr/cyc/SM | correct | warnings |",
            "|---|---|---:|---:|---:|---|---|"]
    for name in sorted(latest):
        rec = latest[name]
        for label, s in rec["summary"].items():
            m = s["metrics"]
            ok = {True: "yes", False: "NO", None: ""}[s.get("correctness")]
            if "lost_fraction" in m:
                ok = f"lost {m['lost_fraction']['median']:.2%}"
            nw = sum(w.startswith(f"[{label}]") or not w.startswith("[") for w in rec["warnings"])
            rows.append(f"| {name} | {label} | {m['cycles_per_op']['median']:.3f} | {m['cycles_per_op']['cv']:.2%} | "
                        f"{m['warp_ops_per_cycle']['median']:.3f} | {ok} | {nw or ''} |")
    text = "\n".join(rows)
    (runner.RESULTS / "TABLE.md").write_text(text + "\n")
    print(text)


def cmd_optimize(a):
    from . import costmodel, optimize
    src = Path(a.cubin).read_bytes()
    out, rep = optimize.optimize(src, a.kernel, costmodel.Launch(block=a.block, grid=a.grid, trips=a.trips),
                                 mode=a.mode)
    rep.pop("_cubins", None)
    print(optimize.summary(rep))
    dst = Path(a.output or (Path(a.cubin).with_suffix("").as_posix() + ".opt.cubin"))
    dst.write_bytes(out)
    print(f"wrote {dst}")
    if a.report:
        rep2 = json.loads(json.dumps(rep, default=lambda o: None))
        Path(a.report).write_text(json.dumps(rep2, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="3080lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("sass")
    s.add_argument("experiment")
    s.add_argument("--all", action="store_true", help="whole kernel, not just the timed loop")
    s.add_argument("--no-raw", action="store_true")
    s.add_argument("--ptxas")
    s.set_defaults(fn=cmd_sass)
    r = sub.add_parser("run")
    r.add_argument("experiment", nargs="+")
    r.add_argument("--local", action="store_true", help="run here instead of queueing via pcslurm")
    r.add_argument("--trials", type=int, default=30)
    r.add_argument("--warps", help="comma list, e.g. 1,4,8")
    r.add_argument("--iters", type=int, help="ops per chain per thread (default 10M)")
    r.add_argument("--seed", type=int)
    r.add_argument("--sizes", help="working sets for chase experiments, e.g. 4k,64k,1m,64m")
    r.add_argument("--cold", action="store_true", help="chase: skip the warm walk")
    r.add_argument("--lock-clock", type=int, help="try nvidia-smi -lgc MHz (needs admin)")
    r.add_argument("--ptxas", help="override ptxas flags, e.g. --ptxas=-O3")
    r.add_argument("--force", action="store_true", help="measure even if SASS validation fails")
    r.set_defaults(fn=cmd_run)
    sh = sub.add_parser("show")
    sh.add_argument("path")
    sh.set_defaults(fn=cmd_show)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    o = sub.add_parser("optimize", help="detect, predict, rewrite and validate split-sector loads in a cubin")
    o.add_argument("cubin")
    o.add_argument("--kernel", default="k")
    o.add_argument("--block", type=int, default=128, help="threads per block (occupancy for the cost model)")
    o.add_argument("--grid", type=int, help="blocks per launch (default: 4 full waves)")
    o.add_argument("--trips", type=float, help="main-loop trips per warp (whole-kernel time incl. fixed cost)")
    o.add_argument("--mode", choices=("guided", "blind"), default="guided")
    o.add_argument("-o", "--output")
    o.add_argument("--report", help="write the full decision report as JSON")
    o.set_defaults(fn=cmd_optimize)
    a = p.parse_args(argv)
    if a.cmd == "run" and not a.local and os.environ.get("LAB_FORCE_LOCAL"):
        a.local = True
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
