"""Score optimization-selection predictors on the frozen split (data/optsel/PROTOCOL.md). CPU only.

  python scripts/optsel_eval.py [split=frozen|lofo] [out=data/optsel/eval_<split>.json]

Every predictor gives, per decision row, a score s (predicted log speedup) and P(win = y > tau). Rows
whose transform is 'aa' (A/A arm) are excluded from decisions and reported as noise. Hyper-parameters
(ridge/logistic strength) and every predictor's probability calibration are fit on the training
families only: for split=frozen that is rand_dev + suite; for split=lofo all families but the one scored.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
from lab.optsel import FEATURES  # noqa: E402

TAU = 0.01
DEV = ["rand_dev", "suite"]
TEST = ["rand_heldout", "polybench_sass", "polybench_fusion"]
LAMBDAS = [0.1, 1, 10, 100, 1000]


def load():
    rows = []
    for p in sorted((ROOT / "data/optsel").glob("corpus_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def X_of(rows):
    return np.array([[r["features"].get(k, 0.0) for k in FEATURES] for r in rows], float)


def y_of(rows):
    return np.array([r["y"] for r in rows], float)


# ---- models --------------------------------------------------------------------------------------------

class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y, w=None):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.ones(len(y)) if w is None else w
        A = Z.T @ (Z * w[:, None]) + self.lam * np.eye(Z.shape[1])
        self.b0 = float(np.average(y, weights=w))
        self.b = np.linalg.solve(A, Z.T @ (w * (y - self.b0)))
        return self

    def predict(self, X):
        return self.b0 + ((X - self.mu) / self.sd) @ self.b


class Logistic:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, t):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        b = np.zeros(Z.shape[1])
        R = self.lam * np.eye(Z.shape[1]); R[0, 0] = 0
        for _ in range(100):                                   # Newton
            p = 1 / (1 + np.exp(-Z @ b))
            g = Z.T @ (p - t) + R @ b
            H = Z.T @ (Z * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(len(b))
            step = np.linalg.solve(H, g)
            b -= step
            if np.abs(step).max() < 1e-8:
                break
        self.b = b
        return self

    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return 1 / (1 + np.exp(-Z @ self.b))


def platt(s_train, t_train):
    """1-D logistic calibration of a score into P(win); fit on training rows only."""
    m = Logistic(1e-3).fit(np.asarray(s_train, float)[:, None], np.asarray(t_train, float))
    return lambda s: m.predict(np.asarray(s, float)[:, None])


# ---- predictors ----------------------------------------------------------------------------------------

def heuristic_score(r):
    f = r["features"]
    if f["kind_fusion"]:
        return 0.02 if (f["d_launch_frac"] > 0 or r["transform"].endswith("graph")) else -0.02
    if not f["changed"] or f["d_occ"] < 0:
        return -0.02 if f["d_occ"] < 0 else 0.0
    return -0.5 * (f["d_stall"] + f["d_depth"])


def lofo_lambda(train_rows, fams):
    """Choose ridge/logistic strength by leave-one-family-out inside the training families."""
    best = None
    for lam in LAMBDAS:
        err, ll = 0.0, 0.0
        for f in fams:
            tr = [r for r in train_rows if r["family"] != f]
            te = [r for r in train_rows if r["family"] == f]
            if not tr or not te:
                continue
            Xt, yt = X_of(tr), y_of(tr)
            s = Ridge(lam).fit(Xt, yt).predict(X_of(te))
            err += float(np.mean((s - y_of(te)) ** 2))
            p = np.clip(Logistic(lam).fit(Xt, (yt > TAU).astype(float)).predict(X_of(te)), 1e-4, 1 - 1e-4)
            t = (y_of(te) > TAU)
            ll += float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
        if best is None or (err, ll) < best[1]:
            best = (lam, (err, ll))
    return best[0] if best and len(fams) > 1 else 10.0


def predictors(train, test):
    """name -> (scores, probs) on test rows, everything fit on train rows."""
    yt = y_of(train)
    win_t = (yt > TAU).astype(float)
    base_rate = float(win_t.mean())
    out = {}
    n = len(test)
    out["never"] = (np.full(n, -1.0), np.full(n, base_rate))
    out["always"] = (np.full(n, 1.0), np.full(n, base_rate))
    hs_tr = np.array([heuristic_score(r) for r in train])
    hs = np.array([heuristic_score(r) for r in test])
    out["heuristic"] = (hs, platt(hs_tr, win_t)(hs))
    hw_tr = np.array([r["features"]["hw_pred"] for r in train])
    hw = np.array([r["features"]["hw_pred"] for r in test])
    out["hw_model"] = (hw, platt(hw_tr, win_t)(hw))
    lam = lofo_lambda(train, sorted({r["family"] for r in train}))
    Xtr, Xte = X_of(train), X_of(test)
    s = Ridge(lam).fit(Xtr, yt).predict(Xte)
    p = Logistic(lam).fit(Xtr, win_t).predict(Xte)
    out["feature_model"] = (s, p)
    # feature model without the hardware model's prediction: does the model add anything?
    keep = [i for i, k in enumerate(FEATURES) if k != "hw_pred"]
    s2 = Ridge(lam).fit(Xtr[:, keep], yt).predict(Xte[:, keep])
    p2 = Logistic(lam).fit(Xtr[:, keep], win_t).predict(Xte[:, keep])
    out["features_no_hw"] = (s2, p2)
    y = y_of(test)
    out["oracle"] = (y, (y > TAU).astype(float))
    return out, lam


# ---- metrics -------------------------------------------------------------------------------------------

def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def decide(s, p, mode, c):
    """Apply? mode 'score': s > tau; 'prob': P(win) >= c (abstain = keep baseline otherwise)."""
    return (s > TAU) if mode == "score" else (p >= c)


def group_choice(rows, s, applied):
    """Per group: among candidates the predictor would apply, take the highest score; else baseline."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["group"], []).append(i)
    chosen, regret = [], []
    for g, idx in by.items():
        best = max(0.0, max(rows[i]["y"] for i in idx))
        app = [i for i in idx if applied[i]]
        yc = rows[max(app, key=lambda i: s[i])]["y"] if app else 0.0
        chosen.append(yc)
        regret.append(best - yc)
    return np.array(chosen), np.array(regret), list(by)


def ece(p, t, bins=5):
    e, rel = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - t[m].mean())
            rel.append([round(float(lo), 2), int(m.sum()), round(float(p[m].mean()), 3), round(float(t[m].mean()), 3)])
    return e, rel


def boot_ci(groups_vals, n=2000, seed=0):
    rng = random.Random(seed)
    v = list(groups_vals)
    if not v:
        return (float("nan"), float("nan"))
    ms = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def score(rows, preds):
    y = y_of(rows)
    t = (y > TAU).astype(float)
    nz = np.abs(y) > TAU
    res = {"n": len(rows), "groups": len({r["group"] for r in rows}), "wins": int((y > TAU).sum()),
           "losses": int((y < -TAU).sum()), "neutral": int((~nz).sum()), "predictors": {}}
    for name, (s, p) in preds.items():
        app = decide(s, p, "score", None)
        d = {"dir_acc": float(np.mean(app[nz] == (y[nz] > 0))) if nz.any() else float("nan"),
             "spearman": spearman(s, y) if name not in ("never", "always") else float("nan"),
             "applied": int(app.sum()), "applied_wins": int((app & (y > TAU)).sum()),
             "applied_losses": int((app & (y < -TAU)).sum())}
        ch, rg, _ = group_choice(rows, s, app)
        d.update(mean_regret=float(rg.mean()), realised_geomean=float(math.exp(ch.mean())),
                 realised_ci=[math.exp(x) for x in boot_ci(ch)])
        e, rel = ece(np.asarray(p, float), t)
        d.update(ece=e, reliability=rel, brier=float(np.mean((np.asarray(p) - t) ** 2)))
        ab = []
        for c in (0.5, 0.6, 0.7, 0.8, 0.9):
            a2 = decide(s, p, "prob", c)
            ch2, rg2, _ = group_choice(rows, s, a2)
            ab.append({"c": c, "coverage": float(a2.mean()), "applied": int(a2.sum()),
                       "losses": int((a2 & (y < -TAU)).sum()), "wins": int((a2 & (y > TAU)).sum()),
                       "realised_geomean": float(math.exp(ch2.mean())), "mean_regret": float(rg2.mean())})
        d["abstention"] = ab
        res["predictors"][name] = d
    return res


def run(split):
    rows = load()
    aa = [r for r in rows if r["transform"] == "aa"]
    rows = [r for r in rows if r["transform"] != "aa"]
    fams = sorted({r["family"] for r in rows})
    out = {"tau": TAU, "split": split, "families_present": fams,
           "aa_noise": {f: {"n": len(v), "abs_q95": float(np.quantile(np.abs(v), 0.95)), "max_abs": float(np.max(np.abs(v)))}
                        for f in sorted({r["family"] for r in aa})
                        for v in [[r["y"] for r in aa if r["family"] == f]]},
           "scores": {}}
    if split == "frozen":
        train = [r for r in rows if r["family"] in DEV]
        plan = [(f, [r for r in rows if r["family"] == f]) for f in TEST if any(r["family"] == f for r in rows)]
        plan.append(("test_pooled", [r for r in rows if r["family"] in TEST]))
        plan.append(("dev_lofo", None))
    else:
        plan = [(f, [r for r in rows if r["family"] == f]) for f in fams]
    for name, test in plan:
        if name == "dev_lofo":
            # in-sample-free view of the dev families: each scored by a model fit on the other
            parts = []
            for f in DEV:
                tr = [r for r in rows if r["family"] in DEV and r["family"] != f]
                te = [r for r in rows if r["family"] == f]
                if tr and te:
                    pr, lam = predictors(tr, te)
                    out["scores"][f"dev_lofo:{f}"] = score(te, pr) | {"lambda": lam}
            continue
        if not test:
            continue
        tr = train if split == "frozen" else [r for r in rows if r["family"] != name]
        pr, lam = predictors(tr, test)
        out["scores"][name] = score(test, pr) | {"lambda": lam, "train_families": sorted({r['family'] for r in tr})}
    return out


def table(out):
    lines = []
    for fam, sc in out["scores"].items():
        lines.append(f"\n## {fam}  n={sc['n']} groups={sc['groups']} wins={sc['wins']} losses={sc['losses']} "
                     f"neutral={sc['neutral']}  lambda={sc.get('lambda')}")
        lines.append(f"{'predictor':16s} {'dir_acc':>7s} {'rho':>6s} {'apply':>5s} {'W/L':>7s} {'regret':>7s} "
                     f"{'realised':>8s} {'95% CI':>15s} {'ECE':>5s} {'Brier':>6s}  abstain c=0.7: cov/W/L/realised")
        for n, d in sc["predictors"].items():
            a7 = next(a for a in d["abstention"] if a["c"] == 0.7)
            lines.append(f"{n:16s} {d['dir_acc']:7.3f} {d['spearman']:6.2f} {d['applied']:5d} "
                         f"{d['applied_wins']:3d}/{d['applied_losses']:<3d} {d['mean_regret']:7.4f} {d['realised_geomean']:8.4f} "
                         f"[{d['realised_ci'][0]:.3f},{d['realised_ci'][1]:.3f}] {d['ece']:5.3f} {d['brier']:6.3f}  "
                         f"{a7['coverage']:.2f}/{a7['wins']}/{a7['losses']}/{a7['realised_geomean']:.4f}")
    return "\n".join(lines)


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    split = kv.get("split", "frozen")
    global TAU
    TAU = float(kv.get("tau", TAU))                    # sensitivity runs only; the protocol's tau is 0.01
    out = run(split)
    dst = ROOT / kv.get("out", f"data/optsel/eval_{split}.json")
    dst.write_text(json.dumps(out, indent=1))
    txt = table(out)
    print("A/A noise:", json.dumps(out["aa_noise"]))
    print(txt)
    dst.with_suffix(".txt").write_text("A/A noise: " + json.dumps(out["aa_noise"]) + "\n" + txt + "\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
