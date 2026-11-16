# Does hardware-informed optimization find wins that a heuristic or autotuning misses?

Written and committed before any number below was computed. Nov 16 2026.

## Question

On kernels the hardware-informed tools were not built on, does hardware-informed optimization
(the measured-latency model, the split-sector SASS rewrite, the launch-cost model) produce speedups
that (a) a straightforward heuristic or (b) conventional autotuning over the kernel's ordinary
parameters does not reach?

Autotuning with an unlimited budget measures every candidate in its space and so finds that space's
best. The hardware-informed side can only beat it in two ways, tested separately:

* **H1 (reach):** it produces candidates outside the autotuning space (a SASS rewrite) that beat the
  autotuned best by more than measurement noise.
* **H2 (budget):** at a small measurement budget k, ordering candidates by the hardware model reaches a
  better result than measuring k random candidates, or the heuristic's choice.

## Stage 1: int4 GEMV (existing data, offline)

Source: the per-kernel tables of `docs/optimize_benchmark.md` (job outputs of `scripts/opt_bench.py`;
arms timed in the same interleaved run, so all times within a suite are comparable). Each shape
(suite, shape id) is one tuning problem. Its autotuning space is every measured source configuration of
that shape: rows/threads R, unroll U, weight layout (u_major, r_major, lane_contig), compile option
(default, `__launch_bounds__` lb8, -O1). Time of a configuration under ptxas: the `ptxas us` column.
Time after the hardware-informed rewrite: ptxas us / `blind` speedup.

Held-out status: suite B shapes were never used while building or revising the optimizer; suite C
(high register pressure) is held out for v1/v3. Suite A was used to revise the model (post hoc).
Results are reported per suite; the verdict uses B and C only.

Policies, per shape:

* `heuristic`: the lane-contiguous configuration with the largest R (the coalescing rule a GPU
  programmer applies first); if the shape has no lane_contig configuration, its first listed one.
* `autotune(k)`: the best of k configurations drawn uniformly without replacement, ptxas times;
  expected value computed exactly over all k-subsets. `autotune(all)` = the space's best.
* `hw_reach`: the best over the union of ptxas configurations and their rewrites (exhaustive).
* `hw_guided(k)`: configurations ordered by the guided-v1 predicted speedup applied to ptxas time
  is NOT available (the cost model predicts rewrite speedup, not cross-configuration runtime), so H2 is
  not testable on this data and is not claimed.

Primary metric for H1: per shape, g = t(autotune(all)) / t(hw_reach). A shape counts as a win when
g exceeds the identical-binary noise bound for its time bucket from `docs/optimize_benchmark.md`
(<8 us: 7.9%, 8-20 us: 1.2%, >20 us: 1.5%). Report geomean g, wins / shapes, and the same for
hw_reach vs heuristic and autotune(all) vs heuristic.

Decision rule: H1 is supported on GEMV if, on held-out suites B and C, geomean g >= 1.02 and at least
half the shapes are wins. Otherwise it is refuted for this family.

## Stage 2: optimization-selection corpus (existing data, offline)

`data/optsel/` already measures, per kernel, every SASS transform (SASS families) and every launch
variant (fusion). Here exhaustive autotuning over the transforms is the oracle row of
`eval_frozen.txt`. H1 for these families is answered by the existing oracle headroom; H2 by the
realised geomean of the hw_model at k = 1 (apply its top pick only) vs the best fixed rule at k = 1.
No refitting; numbers are read from `data/optsel/eval_frozen.json` and `demo_eval.json`.

## Stage 3 (only if stage 1 or 2 leaves H1 or H2 open): new kernels on the GPU

Defined after stages 1-2 are reported, in an addendum to this file, before any run.
