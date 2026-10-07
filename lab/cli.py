"""3080lab command line.

  3080lab list
  3080lab sass dependent_ffma          # compile only, no GPU
  3080lab run dependent_ffma           # queues through pcslurm, waits, prints report
  3080lab run dependent_ffma --local   # run in this process (what the queued job does)
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
    argv = [str(py), "-m", "lab.cli", "run", a.experiment, "--local", "--trials", str(a.trials)]
    for flag in ("warps", "iters", "seed", "lock_clock", "ptxas"):
        val = getattr(a, flag)
        if val:
            argv += [f"--{flag.replace('_', '-')}={val}"]
    if a.force:
        argv.append("--force")
    r = subprocess.run(["pcslurm", "submit", "--shared", "-p", str(ROOT), "-J", f"3080lab-{a.experiment}", "--", *argv],
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
    exp = registry()[a.experiment]
    if a.ptxas is not None:
        exp.ptxas_flags = a.ptxas.split()
    opts = {"force": a.force, "trials": a.trials, "warps": _ints(a.warps), "iters": a.iters, "seed": a.seed, "lock_clock": a.lock_clock}
    rec = runner.run(exp, opts)
    print(runner.report(rec))
    return 0


def cmd_show(a):
    rec = json.loads((Path(a.path) / "record.json").read_text())
    print(runner.report(rec))


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
    r.add_argument("experiment")
    r.add_argument("--local", action="store_true", help="run here instead of queueing via pcslurm")
    r.add_argument("--trials", type=int, default=30)
    r.add_argument("--warps", help="comma list, e.g. 1,4,8")
    r.add_argument("--iters", type=int, help="ops per chain per thread (default 10M)")
    r.add_argument("--seed", type=int)
    r.add_argument("--lock-clock", type=int, help="try nvidia-smi -lgc MHz (needs admin)")
    r.add_argument("--ptxas", help="override ptxas flags, e.g. --ptxas=-O3")
    r.add_argument("--force", action="store_true", help="measure even if SASS validation fails")
    r.set_defaults(fn=cmd_run)
    sh = sub.add_parser("show")
    sh.add_argument("path")
    sh.set_defaults(fn=cmd_show)
    a = p.parse_args(argv)
    if a.cmd == "run" and not a.local and os.environ.get("LAB_FORCE_LOCAL"):
        a.local = True
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
