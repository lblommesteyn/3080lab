# Optimization-selection benchmark: protocol (written Nov 14 2026, before any model was fit)

Question: given a baseline binary and a candidate optimization, does the measured hardware model (or a
feature model built on it) predict whether the candidate is faster on the GPU, by how much relative to
other candidates, and how sure it is? Absolute runtime is not the target.

## Decision unit
One row = (family, group, transform). group = one kernel or one multi-kernel program. y = log(t_base /
t_candidate) measured on the RTX 3080 (> 0 means the optimization helps). Baseline is ptxas's binary (SASS
families) or the original launch sequence (fusion family).

## Families and split (frozen)
| family | decisions | role |
|---|---|---|
| rand_dev: random kernels seeds 1000-1039 x {identity_late, sb_late, split_identity_late, split_sb_late}, job 2211 | 160 | development |
| suite: 19 textbook kernels x 7 SASS transforms, protocol v2 (job 2217) | ~130 | development |
| rand_heldout: seeds 3000-3039, same 4 transforms, job 2212 | 160 | test |
| polybench_sass: 40 PolyBench/GPU kernels x {sb, sb_wide}, Oct 18 sassopt runs | 80 | test (held-out family) |
| polybench_fusion: 12 multi-kernel PolyBench programs x {graph, fused, fused_graph}, xkernel run | 36 | test (held-out family, held-out transform type) |

Model class, features, regularisation and thresholds are chosen using dev families only (leave-one-family-
out between rand_dev and suite). Test families are scored once with everything frozen. A secondary leave-
one-family-out over all five families is reported with the same frozen settings.

## Noise and labels
Neutral band tau = 0.01 (1%), fixed now. Basis: the event timer quantum is 1.024 us (1% of a 100 us kernel)
and A/A job 2216 found a 1.0% stable buffer-placement gap; protocol v2 removes the placement bias. A row is
a win if y > tau, a loss if y < -tau, neutral otherwise. The suite A/A arm (ptxas_dup) gives the empirical
noise distribution, reported alongside.

## Predictors
- never: keep baseline. always: apply every candidate.
- heuristic: SASS: apply if the hot loop's stall sum and dependence depth do not grow and registers do not
  cost occupancy; fusion: apply if the launch count falls or a graph is used.
- hw_model: the measured hardware model alone: lab/model.simulate on the hot loop at the launch's
  warps/SM (SASS), launch-gap microbenchmark (scripts/launch_gap.py) for fusion/graph. Apply if hw_pred > tau.
- feature model: ridge regression on y plus a logistic model for P(y > tau), on lab/optsel.FEATURES.
- oracle: apply iff measured y > 0 (and per group, pick the measured best).

## Metrics
directional accuracy on rows with |y| > tau; Spearman rank correlation of prediction vs y (per family);
regret per group = max(0, best y in group) - y of the chosen option (choice among baseline + candidates);
realised geomean speedup of the chosen options; calibration (reliability bins, ECE) of P(win); abstention:
when confidence < c keep the baseline, report realised gain and risk (losses taken) vs coverage.

## Rules
Never tune on test families. Every run's raw data stays in data/optsel/. Negative results are reported.
