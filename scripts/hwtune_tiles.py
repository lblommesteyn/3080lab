"""Addendum 2 of data/hwtune/PROTOCOL.md: prefill GEMM tiles, cost-model-ordered search vs autotuning.

  python scripts/hwtune_tiles.py    # data/hwtune/tiles.txt, tiles.json
"""
import itertools
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ["qkv", "o", "gu", "down"]
DEFAULT = (2, 2, 2, 4, 2, 3, 3)


def main():
    M = json.loads((ROOT / "data/rank_fresh/measured.json").read_text())
    P = {tuple(x["tile"]): x for x in json.loads((ROOT / "data/rank_fresh/preds.json").read_text())["all"]}
    T = {tuple(r["tile"]): r for r in M["rows"]}
    tiles = [x for x in T if x in P]                  # the 18 test tiles (anchors were calibration points)
    assert len(tiles) == 18
    assert all(t in P for t in tiles), [t for t in tiles if t not in P]
    assert DEFAULT in T
    occ = max(tiles, key=lambda t: (P[t]["warps_sm"], P[t]["blocks_sm"]))  # max() keeps the first on ties
    rng = random.Random(0)
    res = {s: {} for s in SHAPES}
    for s in SHAPES:
        t = {x: T[x][s] for x in tiles}
        best = min(t.values())
        r = res[s]
        r["default"] = T[DEFAULT][s] / best
        r["max_occupancy"] = t[occ] / best
        for m in ("old", "queue"):
            order = sorted(tiles, key=lambda x: P[x][m][s])
            for k in (1, 2, 4):
                r[f"{m}({k})"] = min(t[x] for x in order[:k]) / best
        ts = list(t.values())
        for k in (1, 2):
            r[f"autotune({k})"] = statistics.mean(min(c) for c in itertools.combinations(ts, k)) / best
        r["autotune(4)"] = statistics.mean(min(rng.sample(ts, 4)) for _ in range(20000)) / best
    gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))  # noqa: E731
    keys = list(res[SHAPES[0]])
    summ = {k: gm([res[s][k] for s in SHAPES]) for k in keys}
    lines = [f"18 blind test tiles x 4 shapes; regret = t / t(best of 18); max_occupancy picks {list(occ)}",
             f"{'policy':16s} " + " ".join(f"{s:>7s}" for s in SHAPES) + "  geomean"]
    for k in keys:
        lines.append(f"{k:16s} " + " ".join(f"{res[s][k]:7.3f}" for s in SHAPES) + f"  {summ[k]:7.3f}")
    ok = any(all(summ[f"{m}({k})"] < summ[f"autotune({k})"] and summ[f"{m}({k})"] < min(summ["default"],
                 summ["max_occupancy"]) for k in (1, 2)) for m in ("old", "queue"))
    lines.append(f"\nH2 on the tile space: {'SUPPORTED' if ok else 'REFUTED'}")
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "data/hwtune/tiles.txt").write_text(txt + "\n")
    (ROOT / "data/hwtune/tiles.json").write_text(json.dumps({"shapes": res, "summary": summ, "h2": ok}, indent=1))


if __name__ == "__main__":
    main()
