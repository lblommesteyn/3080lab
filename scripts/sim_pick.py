"""Does lab/model.simulate pick the faster schedule? Predicted vs measured (job 2054) per seed."""
import sys, re, math, json
from pathlib import Path
WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT)); sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa
from lab import resched, sass, toolchain, model
from lab.experiments.randk import ReschedRandom
log = open(WT / "data/ablation/job2054_exit_cycles.txt", encoding="utf-8", errors="replace").read()
meas = {}
for m in re.finditer(r"^\[(seed\d+)/(\w+)\]\n((?:  .*\n)*)", log, re.M):
    c = re.search(r"cycles/op:\s*([\d.]+)", m.group(3)); o = re.search(r"= (\d+) warps/SM", m.group(3))
    if c: meas[(m.group(1), m.group(2))] = (float(c[1]), int(o[1]))
exp = ReschedRandom()
want = set(sys.argv[1:])
pols = [("identity_late", "identity"), ("sb_late", "sb"), ("crit_late", "crit")]
out = {}
for v in exp.variants({"n": 40}):
    s = v.label.split("/")[0]
    if v.params["policy"] != "ptxas" or (want and s not in want): continue
    cub = toolchain.build(exp.source(v)).cubin
    wsm = meas[(s, "ptxas")][1]
    pred = {}
    for name, pol in [("ptxas", None)] + pols:
        o = cub if pol is None else resched.reschedule(cub, "k", policy=pol, late_entry=True, sticky_waits=True)[0]
        body = sass.loop_body(sass.parse(toolchain.disassemble(o)))
        pred[name] = model.simulate(body, warps=wsm, iters=1000, sim_iters=4)["per_iter"]
    row = {n: (meas[(s, "ptxas")][0] / meas[(s, n)][0], pred["ptxas"] / pred[n]) for n, _ in pols}
    out[s] = row
    print(s, wsm, " ".join(f"{n}: meas {a:.3f} pred {b:.3f}" for n, (a, b) in row.items()), flush=True)
json.dump(out, open(WT / "data/ablation/sim_pick_2054.json", "w"))
