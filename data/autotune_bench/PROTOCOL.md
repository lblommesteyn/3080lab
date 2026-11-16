# Budgeted autotuning benchmark: frozen protocol (Nov 16 2026)

Committed together with `preds.json` before any GPU timing of these shapes. Any later change is an
addendum at the end of this file, dated, and states whether it was written before or after the numbers
it could affect.

## Question

Does the existing copy-cap hardware cost model find near-optimal int4 x int8 prefill GEMM tiles with at
least **4x fewer GPU measurements than Optuna TPE** on unseen workloads?

## Frozen inputs

* **Model:** `lab/gemm_model.py` with `CP_QUEUE = None` (copy-cap), sha256 recorded in `preds.json`
  (`gemm_model_sha256`). Constants were fitted on Qwen2.5-1.5B timings only. No refit, no constant
  change, no new feature after this commit. The second model (`queue`) is not part of this benchmark.
* **Workloads (32):** 4 model families (Qwen2.5-7B, Gemma-2-9B, Phi-3-mini, Qwen2.5-14B) x 4 linear
  layers (fused qkv, attention out, fused gate+up, down) x 2 prompt lengths (P = 256, 1024). None
  of these (N, K, P) has been timed in this repo before. Listed in `scripts/autotune_bench.py`.
* **Candidate space:** `scripts/rank_fresh.py`'s grid (17 warp layouts x 7 (KB, NSTG) x 3 MINB = 357),
  keeping tiles that build without spills and divide the shape: 212 per shape (all 32 shapes).
  Minus any tile that fails the correctness check below (excluded, counted, reported).
* **Predictions:** `preds.json` (`pred[shape]`, microseconds), with per-tile build time and
  prediction time.
* **Seeds:** timing order rounds 1000 + r; observation noise draws seed 0; random-search Monte Carlo seed
  0 (5,000 runs per shape per budget); Optuna seeds 0-19 per shape; bootstrap seed 0 (10,000 resamples).

## Correctness (before any timing is accepted)

For each workload: a random bf16 activation is quantized by the production quantizer (`QUANT_Q8`), every
candidate tile computes Y, and Y is compared with an fp32 reference X @ dequant(W). A tile passes on that
workload if max |Y - ref| / max |ref| < 1e-2. Failing tiles are not timed and are reported per workload.

## Measurement

One pcslurm job (`autotune_bench.py measure rounds=3`). Each round times every (workload, passing
tile) pair once, plus an **A/A twin** of the default tile (separately loaded copy of the same kernel)
on every workload, in a fresh random order across all workloads and tiles (fully interleaved). Timing is
the repo's harness (`batched_gemv_bench.timeit`: CUDA graph of 20 launches, 0.3 s warm-up, median of 7
replays). Raw per-round files: `raw/round{r}.json`, with the wall time of each measurement.

* **True time** of a (tile, workload): median over the 3 rounds. **Best** = the minimum true time.
* **Observed time** (what a tuner sees when it measures a tile): one of the 3 rounds' values, drawn at
  random (seeded). The tuner picks its incumbent by observed time; regret is scored on true time.
* **A/A:** per workload, true(A/A twin) / true(default) and its per-round spread; reported as the noise
  floor. Differences below it are not called wins.

## Strategies (budget k = number of GPU measurements)

| strategy | k | definition |
|---|---|---|
| default | 0 | the production tile (2,2,2,4,2,3,3) |
| max-occupancy | 0 | most resident warps per SM; ties by more blocks per SM, then first in grid order |
| model top-1 | 0 | lowest predicted time |
| random | 1-64 | k distinct tiles uniformly; keep best observed (Monte Carlo) |
| Optuna TPE | 1-64 | `optuna 4.1.0`, `TPESampler(seed=s)` defaults (n_startup_trials = 10, univariate); search space = three categoricals: warp layout (17), staging (7), MINB (3). Proposals outside the shape's candidate set are told as FAIL and cost nothing (validity is known without the GPU). A repeated proposal returns its cached observation and costs nothing. After 200 non-measuring proposals in a row, a uniformly random unmeasured candidate is measured instead (counted). 20 seeds. |
| model-ranked | 1-64 | measure the k lowest-predicted tiles; keep best observed (20 noise seeds) |
| model-seeded TPE | 1-64 | the model's top 4 are enqueued as the first 4 trials (`enqueue_trial`), then TPE as above; 20 seeds |

Budgets reported: 0, 1, 2, 4, 8, 16, 32, 64.

## Metrics

* **Regret** at budget k = true(incumbent) / true(best). Per workload: mean over seeds/draws. Summary:
  geometric mean over the 32 workloads, with a 95% bootstrap interval over workloads.
* **m5** (measurements to within 5%) for one run: the smallest number of measurements after which the
  incumbent's regret is <= 1.05. Runs not reaching it by 64 are censored and counted as 128 in the
  primary metric (reported also with 65, and the censored fraction). Per workload: geometric mean over
  seeds (model-ranked: over noise seeds). A strategy whose budget-0 pick is already within 5% does not
  get m5 = 0; m5 counts measurements only.
* **Wall-clock tuning cost** to reach within 5% = sum over measured tiles of (that tile's build time
  from `preds.json` + its median measurement wall time) + strategy overhead (model: prediction time of
  all 212 candidates; TPE: sampler time, recorded in the simulation).
* Also: per-workload regret tables, model Spearman per workload, and the smallest budget at which each
  strategy's geomean regret is <= 1.05.

## Decision rule (primary)

E = geomean over workloads of m5(TPE) / geomean of m5(model-ranked). 95% bootstrap interval over
workloads (workloads resampled, both strategies' per-workload values kept paired).

* **SUPPORTED** if the interval's lower bound is >= 4.
* **PARTIAL** if E >= 4 but the lower bound is < 4.
* **REFUTED** if E < 4. Then the report states where it fails: per family, per layer, per P, and
  which of (model ranking error, TPE strength, noise floor) accounts for it.

Secondary (no verdict attached): the same ratio for model-ranked vs random search and for model-seeded TPE
vs TPE. A success claim may not rest on the random-search comparison.

## Not allowed

Changing the model, the space, the shapes, the seeds, or the rules after timing starts, except by a
dated addendum that says why and that is reported in the write-up.

## Addendum 1 (Nov 16, during timing round 0, before any timing result was read): exact seeds, code

The simulator `scripts/autotune_sim.py` is committed with this addendum, before any timing file exists.
It was tested only on synthetic data (in a scratch directory, not committed). Exact seed scheme, which
replaces the shorthand above: random search uses `random.Random(i)` for workload index i (for both
draws and observations), 5,000 runs. Model-ranked observation noise uses `random.Random(7919 * s + i)`
for s = 0..19. TPE and model-seeded TPE use `TPESampler(seed=s)`, with observations and the fallback from
`random.Random(10000 + s)`. Bootstrap uses `random.Random(0)`: 10,000 resamples for the k = 0 and m5
ratios, and 2,000 for the per-budget regret intervals. Correctness: all 32 x 212 candidates passed
(job 2243 log), so no tile is excluded.

## Addendum 2 (Nov 16 ~19:00; job 2243 cancelled during round 0, no timing written or read)

Job 2243 passed the correctness check (all 32 x 212; `job2243_cancelled/check.json`), but then measured at
more than 3 s per (tile, workload). The profile in job 2244 (`job2243_cancelled/autotune-profile-2244.log`)
shows why. The repo harness's warm-up calls `g.replay()` in a loop for 0.3 s of *CPU* time without waiting
for the GPU, so it queues hundreds of replays. For a 3 ms kernel that is 16 s of GPU work per measurement,
which would make about 20 h per round.

Change: `autotune_bench.timeit` is the same harness, except that it synchronizes after each warm-up replay,
so the warm-up lasts 0.3 s of real time. The CUDA graph of 20 launches and the median of 7 timed replays
are unchanged. Nothing else changes; the measurement job is resubmitted from the start, including the
correctness check.
