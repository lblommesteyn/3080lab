"""Search strategies on the measured tile table (data/autotune_bench/PROTOCOL.md).

  python scripts/autotune_sim.py            writes data/autotune_bench/sim.json and results.txt
  python scripts/autotune_sim.py plots      writes docs/reports/figures/autotune_*.png

Every number here follows the frozen protocol: true time = median of the 3 rounds, a tuner observes
one random round's value per measurement, regret = true(incumbent) / true(best).
"""
import json
import os
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from autotune_bench import DEFAULT, STAGING, WARPS  # noqa: E402

OUT = Path(os.environ.get("AB_OUT", ROOT / "data" / "autotune_bench"))
BUDGETS = (1, 2, 4, 8, 16, 32, 64)
HORIZON, CENSOR = 64, 128
N_RANDOM, N_SEEDS, N_BOOT = int(os.environ.get("AB_NR", 5000)), int(os.environ.get("AB_NS", 20)), 10000
gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))  # noqa: E731


def load():
    d = json.loads((OUT / "preds.json").read_text())
    rows = {tuple(r["tile"]): r for r in d["rows"] if r["built"]}
    chk = json.loads((OUT / "check.json").read_text())["rows"]
    passed = {(c["shape"], tuple(c["tile"]) if c["tile"] != "AA" else "AA") for c in chk if c["pass"]}
    rounds = sorted((OUT / "raw").glob("round*.json"))
    obs, wall = {}, {}
    for f in rounds:
        for r in json.loads(f.read_text())["rows"]:
            t = "AA" if r["tile"] == "AA" else tuple(r["tile"])
            obs.setdefault((r["shape"], t), []).append(r["us"])
            wall.setdefault((r["shape"], t), []).append(r["wall_s"])
    shapes = [s for s, *_ in d["shapes"]]
    W = {}
    for s in shapes:
        cand = [t for t in rows if s in rows[t]["pred"] and (s, t) in passed
                and len(obs.get((s, t), [])) == len(rounds) and None not in obs[(s, t)]]
        W[s] = {"cand": cand, "obs": {t: obs[(s, t)] for t in cand},
                "true": {t: statistics.median(obs[(s, t)]) for t in cand},
                "pred": {t: rows[t]["pred"][s] for t in cand},
                "cost": {t: rows[t]["build_s"] + statistics.median(wall[(s, t)]) for t in cand},
                "aa": obs.get((s, "AA")), "default_obs": obs.get((s, DEFAULT)),
                "n_built": sum(s in rows[t]["pred"] for t in rows),
                "n_fail_check": sum(1 for c in chk if c["shape"] == s and c["tile"] != "AA" and not c["pass"])}
        W[s]["best"] = min(W[s]["true"].values())
    return d, rows, shapes, W, len(rounds)


def trace(w, seq, rng):
    """Regret of the incumbent after each measurement of seq (list of tiles), observing random rounds."""
    best_obs, inc, out = math.inf, None, []
    for t in seq:
        o = w["obs"][t][rng.randrange(len(w["obs"][t]))]
        if o < best_obs:
            best_obs, inc = o, t
        out.append(w["true"][inc] / w["best"])
    return out


def m5(tr):
    for i, r in enumerate(tr):
        if r <= 1.05:
            return i + 1
    return None


def tpe_run(w, seed, seeds_first=()):
    import optuna
    from optuna.distributions import CategoricalDistribution as C
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    dists = {"w": C(list(range(len(WARPS)))), "s": C(list(range(len(STAGING)))), "m": C([1, 2, 3])}
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    for t in seeds_first:
        study.enqueue_trial({"w": WARPS.index(t[:4]), "s": STAGING.index(t[4:6]), "m": t[6]})
    cand, rng = set(w["cand"]), random.Random(10_000 + seed)
    seq, seen, idle, t_s = [], {}, 0, 0.0
    while len(seq) < HORIZON:
        t0 = time.perf_counter()
        tr = study.ask(dists)
        t_s += time.perf_counter() - t0
        t = (*WARPS[tr.params["w"]], *STAGING[tr.params["s"]], tr.params["m"])
        if idle >= 200:   # protocol: fall back to a random unmeasured candidate (counted)
            t = rng.choice(sorted(cand - set(seen)))
        if t not in cand:
            study.tell(tr, state=optuna.trial.TrialState.FAIL)
            idle += 1
            continue
        if t in seen:
            study.tell(tr, seen[t])
            idle += 1
            continue
        o = w["obs"][t][rng.randrange(len(w["obs"][t]))]
        seen[t], idle = o, 0
        seq.append((t, o))
        t0 = time.perf_counter()
        study.tell(tr, o)
        t_s += time.perf_counter() - t0
    return seq, t_s


def seq_trace(w, seq):
    best_obs, inc, out = math.inf, None, []
    for t, o in seq:
        if o < best_obs:
            best_obs, inc = o, t
        out.append(w["true"][inc] / w["best"])
    return out


def summarize_runs(w, runs):
    """runs: list of (trace, tiles measured in order). Per-budget mean regret, per-run m5, cost to m5."""
    reg = {k: statistics.mean(tr[k - 1] for tr, _ in runs) for k in BUDGETS}
    ms, costs = [], []
    for tr, tiles in runs:
        m = m5(tr)
        ms.append(m)
        costs.append(sum(w["cost"][t] for t in tiles[:m]) if m else None)
    return reg, ms, costs


_G = {}


def _init():
    _G["d"], _G["rows"], _G["shapes"], _G["W"], _G["nr"] = load()


def sim_shape(si):
    rows, s = _G["rows"], _G["shapes"][si]
    w = _G["W"][s]
    cand = w["cand"]
    r = {"n_cand": len(cand), "n_built": w["n_built"], "n_fail_check": w["n_fail_check"], "best_us": w["best"]}
    order = sorted(cand, key=lambda t: w["pred"][t])
    occ = max(cand, key=lambda t: (rows[t]["warps_sm"], rows[t]["blocks_sm"], -cand.index(t)))
    r["k0"] = {"default": w["true"][DEFAULT] / w["best"] if DEFAULT in w["true"] else None,
               "max_occupancy": w["true"][occ] / w["best"], "model_top1": w["true"][order[0]] / w["best"]}
    # A/A
    if w["aa"] and w["default_obs"]:
        r["aa_ratio"] = statistics.median(w["aa"]) / statistics.median(w["default_obs"])
        r["aa_rounds"] = [a / b for a, b in zip(w["aa"], w["default_obs"])]
    best_t = min(cand, key=lambda t: w["true"][t])
    r["best_tile"] = list(best_t)
    r["best_round_spread"] = max(w["obs"][best_t]) / min(w["obs"][best_t])
    r["model_rank_of_best"] = order.index(best_t) + 1
    r["n_within5"] = sum(w["true"][t] <= 1.05 * w["best"] for t in cand)
    # spearman
    tr_ = [w["true"][t] for t in cand]
    pr_ = [w["pred"][t] for t in cand]
    r["spearman"] = spearman(pr_, tr_)
    # model-ranked (20 noise seeds)
    runs = []
    for sd in range(N_SEEDS):
        rng = random.Random(sd * 7919 + si)
        runs.append((trace(w, order[:HORIZON], rng), order[:HORIZON]))
    r["model"] = pack(*summarize_runs(w, runs))
    # random search (Monte Carlo)
    rng = random.Random(si)
    runs = []
    for _ in range(N_RANDOM):
        seq = rng.sample(cand, HORIZON)
        runs.append((trace(w, seq, rng), seq))
    r["random"] = pack(*summarize_runs(w, runs))
    # TPE and model-seeded TPE
    for name, first in (("tpe", ()), ("seeded_tpe", tuple(order[:4]))):
        runs, ts = [], []
        for sd in range(N_SEEDS):
            seq, t_s = tpe_run(w, sd, first)
            runs.append((seq_trace(w, seq), [t for t, _ in seq]))
            ts.append(t_s)
        r[name] = pack(*summarize_runs(w, runs))
        r[name]["sampler_s"] = statistics.mean(ts)
    r["model"]["overhead_s"] = sum(rows[t]["build_s"] for t in cand) + sum(rows[t]["predict_s"] for t in cand) / 32
    r["model"]["overhead_amortized_s"] = (sum(rows[t]["build_s"] for t in cand) / 32
                                         + sum(rows[t]["predict_s"] for t in cand) / 32)
    print(f"{s:26s} n={len(cand)} best {w['best']:8.1f} us  model k0 {r['k0']['model_top1']:.3f}  "
          f"m5 model {r['model']['m5_gm']:5.1f} tpe {r['tpe']['m5_gm']:5.1f} seeded {r['seeded_tpe']['m5_gm']:5.1f} "
          f"random {r['random']['m5_gm']:5.1f}", flush=True)
    return s, r


def simulate():
    from multiprocessing import Pool
    _init()
    with Pool(int(os.environ.get("AB_J", 8)), initializer=_init) as pool:
        res = dict(pool.map(sim_shape, range(len(_G["shapes"]))))
    (OUT / "sim.json").write_text(json.dumps({"rounds": _G["nr"], "shapes": res}, indent=1))
    report(res)


def pack(reg, ms, costs):
    c128 = [m if m else CENSOR for m in ms]
    c65 = [m if m else HORIZON + 1 for m in ms]
    ok = [c for c in costs if c is not None]
    return {"regret": reg, "m5_gm": gm(c128), "m5_gm_c65": gm(c65), "censored": sum(m is None for m in ms) / len(ms),
            "m5_median": statistics.median(c128), "cost_to_m5_median_s": statistics.median(ok) if ok else None}


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rk(a), rk(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def boot(vals_fn, shapes, n=N_BOOT, seed=0):
    rng = random.Random(seed)
    xs = sorted(vals_fn([rng.choice(shapes) for _ in shapes]) for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def report(res):
    shapes = list(res)
    L = []
    strat = ("model", "tpe", "seeded_tpe", "random")
    L.append(f"{len(shapes)} workloads; candidates per workload {min(r['n_cand'] for r in res.values())}-"
             f"{max(r['n_cand'] for r in res.values())}; correctness failures "
             f"{sum(r['n_fail_check'] for r in res.values())}")
    aa = [abs(math.log(r["aa_ratio"])) for r in res.values() if "aa_ratio" in r]
    L.append(f"A/A (twin / default, median of rounds): median |log ratio| {statistics.median(aa) * 100:.2f}%, "
             f"max {max(aa) * 100:.2f}%; best-tile round spread median "
             f"{statistics.median(r['best_round_spread'] for r in res.values()):.4f}")
    L.append("")
    L.append("geomean regret over workloads [95% bootstrap]")
    for k0 in ("default", "max_occupancy", "model_top1"):
        f = lambda ss, k0=k0: gm([res[s]["k0"][k0] for s in ss])  # noqa: E731
        lo, hi = boot(f, shapes)
        L.append(f"  k=0  {k0:14s} {f(shapes):.4f} [{lo:.4f}, {hi:.4f}]")
    for k in BUDGETS:
        parts = []
        for st in strat:
            f = lambda ss, st=st: gm([res[s][st]["regret"][k] for s in ss])  # noqa: E731
            lo, hi = boot(f, shapes, n=2000)
            parts.append(f"{st} {f(shapes):.4f} [{lo:.3f},{hi:.3f}]")
        L.append(f"  k={k:<3d} " + "  ".join(parts))
    L.append("")
    L.append("measurements to within 5% of best (geomean over workloads; censored at 64 counted as 128)")
    for st in strat:
        g = gm([res[s][st]["m5_gm"] for s in shapes])
        g65 = gm([res[s][st]["m5_gm_c65"] for s in shapes])
        cen = statistics.mean(res[s][st]["censored"] for s in shapes)
        L.append(f"  {st:11s} {g:6.2f}  (censor=65: {g65:6.2f}; censored runs {cen * 100:4.1f}%)")
    L.append("")

    def ratio(a, b):
        return lambda ss: gm([res[s][a]["m5_gm"] for s in ss]) / gm([res[s][b]["m5_gm"] for s in ss])
    E = ratio("tpe", "model")(shapes)
    lo, hi = boot(ratio("tpe", "model"), shapes)
    verdict = "SUPPORTED" if lo >= 4 else "PARTIAL" if E >= 4 else "REFUTED"
    L.append(f"PRIMARY  E = m5(TPE) / m5(model-ranked) = {E:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  -> {verdict}")
    for a, b in (("random", "model"), ("tpe", "seeded_tpe"), ("random", "tpe")):
        e = ratio(a, b)(shapes)
        l2, h2 = boot(ratio(a, b), shapes)
        L.append(f"  secondary m5({a}) / m5({b}) = {e:.2f} [{l2:.2f}, {h2:.2f}]")
    L.append("")
    L.append("E by group (TPE / model)")
    groups = {}
    for s in shapes:
        f, l, p = s.split("/")
        for key in (f, l, p):
            groups.setdefault(key, []).append(s)
    for key, ss in groups.items():
        L.append(f"  {key:12s} n={len(ss):2d}  E {ratio('tpe', 'model')(ss):6.2f}   m5 model "
                 f"{gm([res[s]['model']['m5_gm'] for s in ss]):5.2f}  tpe {gm([res[s]['tpe']['m5_gm'] for s in ss]):5.2f}")
    L.append("")
    L.append("smallest budget with geomean regret <= 1.05")
    for st in strat:
        ks = [k for k in BUDGETS if gm([res[s][st]["regret"][k] for s in shapes]) <= 1.05]
        L.append(f"  {st:11s} {ks[0] if ks else '>64'}")
    L.append("")
    L.append("wall-clock to within 5% (median over workloads of the per-workload median run, seconds)")
    for st in strat:
        c = [res[s][st]["cost_to_m5_median_s"] for s in shapes if res[s][st]["cost_to_m5_median_s"] is not None]
        L.append(f"  {st:11s} measurements+builds {statistics.median(c):7.2f} s" if c else f"  {st:11s} n/a")
    ov = statistics.median(res[s]["model"]["overhead_s"] for s in shapes)
    ova = statistics.median(res[s]["model"]["overhead_amortized_s"] for s in shapes)
    L.append(f"  model overhead: build+predict all candidates {ov:.1f} s per workload alone, {ova:.1f} s amortized "
             f"over 32 workloads; TPE sampler {statistics.median(res[s]['tpe']['sampler_s'] for s in shapes):.2f} s per run")
    L.append("")
    L.append("per workload: n, best us, model k0 regret, model rank of best, tiles within 5%, spearman, "
             "m5 model / tpe / seeded / random, regret@4 model / tpe")
    for s in shapes:
        r = res[s]
        L.append(f"  {s:26s} {r['n_cand']:3d} {r['best_us']:9.1f} {r['k0']['model_top1']:6.3f} "
                 f"{r['model_rank_of_best']:4d} {r['n_within5']:3d} {r['spearman']:+.2f}  "
                 f"{r['model']['m5_gm']:5.1f} {r['tpe']['m5_gm']:5.1f} {r['seeded_tpe']['m5_gm']:5.1f} "
                 f"{r['random']['m5_gm']:5.1f}  {r['model']['regret'][4]:.3f} {r['tpe']['regret'][4]:.3f}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "results.txt").write_text(txt + "\n")


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.loads((OUT / "sim.json").read_text())["shapes"]
    shapes = list(res)
    figd = ROOT / "docs" / "reports" / "figures"
    figd.mkdir(parents=True, exist_ok=True)
    col = {"model": "#2a6fdb", "seeded_tpe": "#7b4fd6", "tpe": "#d9822b", "random": "#8a8f98"}
    lab = {"model": "copy-cap model ranked", "seeded_tpe": "model-seeded TPE", "tpe": "Optuna TPE", "random": "random"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for st in ("random", "tpe", "seeded_tpe", "model"):
        ys = [gm([res[s][st]["regret"][k] for s in shapes]) for k in BUDGETS]
        ax.plot(BUDGETS, ys, marker="o", ms=4, lw=2, color=col[st], label=lab[st])
    for k0, mk in (("model_top1", "*"), ("max_occupancy", "s"), ("default", "D")):
        ax.scatter([0.7], [gm([res[s]["k0"][k0] for s in shapes])], marker=mk, s=50, color="#333",
                   label=f"{k0.replace('_', ' ')} (0 measurements)")
    ax.axhline(1.05, color="#aaa", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.7, *BUDGETS], ["0", *map(str, BUDGETS)])
    ax.set_xlabel("GPU measurements")
    ax.set_ylabel("geomean regret over 32 workloads")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figd / "autotune_regret.png", dpi=150)
    fig, ax = plt.subplots(figsize=(5, 5))
    x = [res[s]["model"]["m5_gm"] for s in shapes]
    y = [res[s]["tpe"]["m5_gm"] for s in shapes]
    ax.scatter(x, y, color=col["tpe"], s=22)
    lim = [0.8, 140]
    ax.plot(lim, lim, color="#aaa", lw=1, label="equal")
    ax.plot(lim, [4 * v for v in lim], color="#2a6fdb", ls="--", lw=1, label="TPE needs 4x more")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measurements to within 5%: copy-cap model ranked")
    ax.set_ylabel("measurements to within 5%: Optuna TPE")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figd / "autotune_m5.png", dpi=150)
    print("plots written")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plots":
        plots()
    else:
        simulate()
