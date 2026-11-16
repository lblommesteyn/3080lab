# Does hardware-informed optimization find wins a heuristic or autotuning misses? (Nov 16 2026)

**Answer: yes, but only where the space is too large to search; not in small spaces.** On a
212-tile GEMM space for shapes neither model had seen (Llama-3-8B, addendum 3), the copy-cap model's
first pick, with zero measurements, lands within 1.3% of the best of all 212 tiles (geomean over 4
shapes). The production default tile is 22.6% off, the max-occupancy rule 23.6%, and 16 random
measurements 7.0%. Both models pass the pre-registered rule on all 4 shapes (`bigtile/score.txt`).

**In small spaces the answer is no.** Across four smaller optimization spaces on kernels the models were not
built on, the hardware-informed side either found nothing outside an autotuner's reach (H1) or saved
measurements only where a simple rule saved the same ones (H2). The one pre-registered "supported"
result (GEMM tiles) is a 1.3% edge at two measurements that is gone by four, and its one-measurement
edge flips sign in one of three timing rounds.

Protocol, pre-registered and amended only before each new number: `PROTOCOL.md`. Scripts:
`scripts/hwtune_gemv.py`, `hwtune_h2.py`, `hwtune_fusion.py`, `hwtune_tiles.py`. No new GPU runs;
every time is from earlier measured runs, every prediction was written before its timing run.

## H1: candidates outside the autotuning space

| space (held out) | autotuned best vs heuristic | + hardware-informed rewrite vs autotuned best | shapes improved beyond noise |
|---|---:|---:|---:|
| int4 GEMV, suites B + C (12 shapes, 4-9 configs each) | 1.131x | **1.003x** | **0 / 12** |
| SASS transforms, random kernels + PolyBench (`data/optsel/`) | n/a | oracle headroom 0.47-0.64% | below A/A noise |

The split-sector rewrite's held-out 1.074x over ptxas comes from configurations an autotuner would
not keep (u_major R8U2 and R4U2, which have split pairs). The autotuned best is almost always R4U1,
which has no split pairs, so there is nothing left for the rewrite to fix.

## H2: fewer measurements to reach the best

Regret = time of the chosen configuration / time of the best in the space (geomean; 1.000 = optimal).

| space | heuristic (0 measured) | model top-1 (0) | random (1) | model top-2 (2) | random (2) | rule learned on dev (2) |
|---|---:|---:|---:|---:|---:|---:|
| int4 GEMV, held out | 1.131 | 1.071 | 1.334 | 1.010 | 1.147 | **1.010** |
| fusion / graph, held-out family | 1.226x (always fused_graph)* | 1.229x* | 1.108x* | 1.232x* | 1.188x* | n/a |
| fusion / graph, unseen 1/4 size | 1.619x* | 1.609x* | 1.282x* | 1.632x* | 1.508x* | n/a |
| GEMM tiles, 18 blind tiles x 4 shapes | 1.155 (max occupancy), 1.160 (default) | 1.129 | 1.136 | 1.078 | 1.091 | n/a |

\* speedup over base (higher is better), not regret; oracle 1.233x and 1.642x.

* GEMV: the model's top-1 equals the coalescing heuristic's pick on 10 of 12 shapes. Its top-2 is
  within 1.04% of exhaustive search, but "measure the heuristic's pick and the smallest-R, U = 1
  configuration", a rule read off the dev suite A only, gives the identical 1.0104 and the same 11/12.
  Pre-registered verdict: REFUTED (threshold 1.010).
* Fusion: the model beats random trials but ties the fixed rule "fuse what is legal, then capture a
  graph" at every budget that matters.
* Tiles: pre-registered verdict SUPPORTED for the copy-cap model (k = 1: 1.129 vs 1.136; k = 2: 1.078 vs
  1.091), but with 4 shapes the k = 1 edge reverses in timing round 1 of 3 (1.145 vs 1.143), and at
  k = 4 random search (1.054) beats both models (1.078, 1.100). The queue model fails at k = 2.

## H2 in a large space: 212 GEMM tiles, unseen Llama-3-8B shapes (addendum 3)

| policy (measurements) | qkv | o | gate_up | down | geomean regret |
|---|---:|---:|---:|---:|---:|
| default tile (0) | 1.188 | 1.239 | 1.151 | 1.334 | 1.226 |
| max occupancy (0) | 1.207 | 1.189 | 1.198 | 1.356 | 1.236 |
| **copy-cap model top-1 (0)** | 1.030 | 1.022 | 1.000 | 1.000 | **1.013** |
| queue model top-1 (0) | 1.073 | 1.036 | 1.130 | 1.082 | 1.080 |
| random (4) | 1.150 | 1.130 | 1.144 | 1.177 | 1.150 |
| random (16) | 1.075 | 1.055 | 1.056 | 1.093 | 1.070 |
| copy-cap model top-16 (16) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Round-to-round noise of the best tile: 0.1-1.7%. Spearman, predicted vs measured: 0.71-0.83 (copy-cap).
No random budget up to 16 matches the copy-cap model's zero-measurement pick.

Caveat: the output check was void. Under the harness's random inputs, the default tile's
output is non-finite (also with zero residuals, job 2241), so tile correctness was not
verified here. These are the same kernel generator and inputs as `rank_fresh`. Timing should not
depend on the values, but a wrong-yet-fast tile cannot be ruled out (addendum 4).

## What this says

The hardware models are good at explaining *why* a configuration wins (split sectors, launch gaps,
occupancy), and that explanation is what a good heuristic encodes. They are not accurate enough to
rank close candidates (rank correlation 0.4-0.7, prediction errors 5-30% against differences of
1-10% among good options), so once a heuristic and two or four measurements are allowed, they add
nothing measurable. The remaining place a model could pay is a space too large to sample where its
errors are smaller than the spread of candidates; the 212-tile space is that, and there the model wins by a wide margin.

## Limits

* Spaces are small (4-21 candidates). Exhaustive autotuning was affordable everywhere.
* Only one GPU and one model generation per family; tile result rests on 4 shapes.
* Pre-registration was amended twice (each before the next number) and had one deviation (anchor
  tiles lacked predictions); all three are recorded in `PROTOCOL.md`.
* Exploratory analyses (dev-learned rule, per-round tile check) were not pre-registered and are
  labelled as such.
