"""Score copy-path variants of lab/gemm_model.py against every measured prefill GEMM timing: the 11
in-sample tiles (scripts/eval_gemm_model.py) and the 11 held-out tiles (data/gemm_model_heldout_preds.json).

  python scripts/gemm_model_variants.py            (CPU only)

A variant sets CP_LANES_PER_CYC (copy port rate), CP_INFLIGHT (lanes in flight per SM) and the copy
latency. Reports median |error| and, per GEMM, Spearman rank correlation over all 22 tiles.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402,F401  (puts the CUDA DLLs nvrtc needs on the search path)
sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"))
from eval_gemm_model import MEASURED, P, SHAPES  # noqa: E402

from lab import gemm_model as GM  # noqa: E402

VARIANTS = {
    "v1 (rate 1.0, lat 570)": (1.0, None, 570),
    "rate 0.75": (0.75, None, 570),
    "rate 0.70": (0.70, None, 570),
    "cap 400, lat 450": (1.0, 400, 450),
    "cap 500, lat 570": (1.0, 500, 570),
    "cap 600, lat 700": (1.0, 600, 700),
}


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def main():
    held = json.loads(Path("data/gemm_model_heldout_preds.json").read_text())["measured"]["rows"]
    tiles = [(t, {g: statistics.median(v) for g, v in m.items()}, "in") for t, m in MEASURED.values() if not t.ABL]
    tiles += [(GM.Tile(*r["tile"]), {g: r[g] for g, _, _ in SHAPES}, "held") for r in held]
    built = {t: GM.build(t) for t, _, _ in tiles}
    for name, (rate, cap, lat) in VARIANTS.items():
        GM.CP_LANES_PER_CYC, GM.CP_INFLIGHT = rate, cap
        GM.simulate_sm.__kwdefaults__["copy_lat"] = lat
        errs = {"in": [], "held": []}
        per_g = {g: ([], []) for g, _, _ in SHAPES}
        for t, meas, split in tiles:
            for g, N, K in SHAPES:
                pr = GM.predict(t, N, K, P, built[t])["us"]
                errs[split].append(abs(pr / meas[g] - 1))
                per_g[g][0].append(pr)
                per_g[g][1].append(meas[g])
        rho = {g: round(spearman(*v), 2) for g, v in per_g.items()}
        print(f"{name:24s} median |err| in {statistics.median(errs['in']):.3f} held {statistics.median(errs['held']):.3f}"
              f" | within 20%: {sum(e < .2 for e in errs['in'] + errs['held'])}/88 | Spearman {rho}", flush=True)


if __name__ == "__main__":
    main()
