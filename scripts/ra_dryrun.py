"""CPU dry run of the lab/jointra variants: which kernels transform, which are refused (and why),
and whether the static verifier finds new hazards. usage: python scripts/ra_dryrun.py [out=...]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401
from lab import jointra, suite, toolchain, verify  # noqa: E402
from lab.experiments import randk  # noqa: E402

kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
rows = []


def one(name, cub):
    r = {"name": name}
    for extra in (8, 24):
        try:
            s, info = jointra.split_crit(cub, "k", extra=extra)
            r[f"split{extra}"] = {k: info[k] for k in ("cp_before", "cp_after", "moved", "top_before", "top_after")}
            if s is not cub:
                hz = verify.new_hazards(s, cub)
                r[f"split{extra}"]["hazards"] = len(hz)
                out, _ = jointra.resched.reschedule(s, "k", policy="sb", late_entry=True, sticky_waits=True)
                r[f"split{extra}"]["sb_hazards"] = len(verify.new_hazards(out, cub))
        except Exception as e:                           # refused: recorded, the variant falls back
            r[f"split{extra}"] = {"refused": repr(e)[:160]}
    rows.append(r)
    print(json.dumps(r), flush=True)


for c in suite.CASES:
    one("suite:" + c.name, toolchain.build(c.src).cubin)
for key in ("resched_rand", "resched_rand_b"):
    e = randk.registry()[key]
    for v in e.variants({"n": 40, "policies": "ptxas"}):
        one(f"{key}:{v.label.split('/')[0]}", toolchain.build(e.source(v)).cubin)
out = ROOT / kv.get("out", "data/ra/dryrun.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=1))
print("wrote", out)
