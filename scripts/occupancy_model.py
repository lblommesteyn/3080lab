"""lab/gemm_model.py's per-stage cycles for the configs of scripts/occupancy_sweep.py (CPU only), and,
given the sweep's JSON, measured vs simulated side by side.

  python scripts/occupancy_model.py [tiles=A,D] [abl=,noload,nocompute] [meas=data/occupancy_sweep.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy_sweep import TILES  # noqa: E402

from lab import gemm_model as GM  # noqa: E402


def one(job):
    name, cfg, abl = job
    t = GM.Tile(*cfg, abl)
    b = GM.build(t)
    nat = GM.blocks_per_sm(b["regs"], t.nthr, b["smem"])["blocks"]
    out = []
    for nb in range(1, nat + 1):
        r = GM.simulate_sm(b["body"], nblocks=nb, wpb=t.nthr // 32, stage_lanes=t.stage_lanes(),
                           prologue_groups=t.NSTG - 1, stages=GM.SIM_STAGES)
        out.append({"tile": name, "abl": abl, "blocks_sm": nb, "warps_sm": nb * t.nthr // 32,
                    "sim_cyc": round(r["stage_cyc"], 1)})
    return out


def main():
    cfg = {"tiles": "A,D,B,E,H,T", "abl": ",noload,nocompute", "meas": "", "j": "8"}
    for a in sys.argv[1:]:
        k, v = a.split("=", 1)
        cfg[k] = v
    jobs = [(n, TILES[n] if n in TILES else tuple(int(x) for x in n.split(".")), abl)
            for n in cfg["tiles"].split(",") for abl in cfg["abl"].split(",")]
    sim = [r for j in jobs for r in one(j)]          # serial: ~1 s per job (a spawn Pool hung on Windows)
    meas = {}
    if cfg["meas"] and Path(cfg["meas"]).exists():
        meas = {(r["tile"], r["abl"], r["blocks_sm"]): r for r in json.loads(Path(cfg["meas"]).read_text())}
    print(f"{'tile':5s} {'abl':10s} {'blk':>3s} {'warps':>5s} {'sim':>8s} {'meas':>8s} {'meas/sim':>8s} {'MHz':>5s}")
    for r in sim:
        m = meas.get((r["tile"], r["abl"], r["blocks_sm"]))
        ms = f"{m['stage_cyc']:8.1f} {m['stage_cyc'] / r['sim_cyc']:8.2f} {m['mhz']:5d}" if m else ""
        print(f"{r['tile']:5s} {r['abl'] or 'full':10s} {r['blocks_sm']:3d} {r['warps_sm']:5d} {r['sim_cyc']:8.1f} {ms}")
    Path("data/occupancy_model.json").write_text(json.dumps(sim, indent=1))


if __name__ == "__main__":
    main()
