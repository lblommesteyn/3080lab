"""Analyze results/l2map_<k>.npy: per-SM L2 latency for every 128 B line of a 2 MB buffer.

1. Pool all latencies, split into clusters (gaps in the sorted histogram).
2. Label each (SM, line) near/far by the main split; group SMs with identical
   label vectors (= SMs on the same side of the L2).
3. For each SM group, fit the label as an XOR (parity) of address bits over
   GF(2) by Gaussian elimination; report how many lines the linear hash explains.
"""
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results"


def gf2_fit(X: np.ndarray, y: np.ndarray):
    """Solve X w = y over GF(2) (least-violations by elimination on a consistent subset)."""
    A = np.concatenate([X, y[:, None]], 1).astype(np.uint8) % 2
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols - 1):
        p = next((i for i in range(r, rows) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    w = np.zeros(cols - 1, np.uint8)
    for i, c in enumerate(piv):
        w[c] = A[i, -1]
    pred = (X.astype(np.uint8) @ w) % 2
    return w, float(np.mean(pred == y))


def main():
    k = sys.argv[1] if len(sys.argv) > 1 else max(int(p.stem.split("_")[1]) for p in R.glob("l2map_*.npy") if "base" not in p.stem)
    lat = np.load(R / f"l2map_{k}.npy").astype(np.float64)
    base = int(np.load(R / f"l2map_{k}_base.npy")[0])
    got = lat.max(1) > 0
    sms = np.nonzero(got)[0]
    L = lat[got]
    print(f"SMs measured: {len(sms)}  lines: {L.shape[1]}  base {base:#x}")
    v = np.sort(L.ravel())
    pct = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    print("latency percentiles 1/5/25/50/75/95/99:", np.round(pct).astype(int))
    hist, edges = np.histogram(v[(v > pct[0] - 20) & (v < pct[-1] + 20)], bins=60)
    print("histogram (cycles: count):")
    for h, e in zip(hist, edges):
        if h:
            print(f"  {e:6.0f}: {'#' * max(1, int(60 * h / hist.max()))} {h}")
    # main split: largest gap between the 5th and 95th percentile region
    mid = v[(v > pct[1]) & (v < pct[5])]
    gaps = np.diff(mid)
    thr = mid[np.argmax(gaps)] + gaps.max() / 2 if len(gaps) else np.median(v)
    print(f"split threshold ~{thr:.0f} cycles (largest gap {gaps.max() if len(gaps) else 0:.0f})")
    far = (L > thr).astype(np.uint8)
    print(f"fraction far: {far.mean():.3f}; per-SM far fraction min/median/max "
          f"{far.mean(1).min():.3f}/{np.median(far.mean(1)):.3f}/{far.mean(1).max():.3f}")
    # group SMs by label vector similarity
    groups = []
    for i in range(len(sms)):
        for gidx, g in enumerate(groups):
            if np.mean(far[i] == far[g[0]]) > 0.95:
                g.append(i)
                break
        else:
            groups.append([i])
    print(f"SM groups with matching near/far maps (>95% agreement): {len(groups)}")
    addr = base + np.arange(L.shape[1], dtype=np.uint64) * 128
    bits = np.array([(addr >> np.uint64(b)) & np.uint64(1) for b in range(7, 21)], dtype=np.uint8).T
    X = np.concatenate([bits, np.ones((len(addr), 1), np.uint8)], 1)
    for gidx, g in enumerate(groups[:6]):
        y = (far[g].mean(0) > 0.5).astype(np.uint8)
        w, acc = gf2_fit(X, y)
        used = [7 + b for b in range(14) if w[b]]
        print(f"  group {gidx}: {len(g):2d} SMs (smid {sorted(int(sms[i]) for i in g)[:8]}...), far frac {y.mean():.2f}, "
              f"XOR of address bits {used} (+const {int(w[-1])}) explains {acc:.1%} of lines")


if __name__ == "__main__":
    main()
