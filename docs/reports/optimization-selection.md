# Can the hardware model pick optimizations?

Summary. The Nov 14 benchmark asked whether the measured hardware model predicts whether an
optimization helps, rather than how long a kernel runs. Split, threshold, predictors and metrics were
frozen in `data/optsel/PROTOCOL.md` before any model was fit. For block-local SASS transforms the
decision is not worth making: the oracle's ceiling is 0.47-0.64% and the best held-out predictor
realises 1.0007. For launch structure (fusion and CUDA graphs) the launch-cost model picks within 0.2%
of the oracle (1.229 vs 1.231) against 1.159 for the best fixed rule. On a never-measured 1/4-size
PolyBench, the feature model realises 1.631 against an oracle of 1.642, but its 0.7% margin over
"always fused_graph" (1.619) is inside the A/A noise.

## Question / hypothesis

Given a baseline binary and a candidate optimization, does the hardware model (or a feature model
built on it) predict the sign of y = log(t_base / t_candidate), rank candidates, and know when it is
sure? Hypothesis: a model calibrated from microbenchmarks should transfer to unseen kernels better than
fixed rules. Earlier evidence pointed both ways (below).

## Earlier predictors that bear on this

- SM simulator (`lab/model.py`, Phase 5): median error 3.4% (68% within 5%) on 60 held-out random
  kernels for v0; on a second held-out set v0 5.9%, v1 7.0%.
- Whole-GPU predictor (`lab/model_gpu.py`, `scripts/eval_gpu_model.py`): T = max(SM simulation, bytes /
  710 GB/s) + fixed overhead, with flow-sensitive address provenance and an L1 port charged per 128 B
  line. On 131 GEMV variants: median 13.6% (73% within 20%), then 12.8% with the split-sector rule and
  11.5% (79% within 20%) after simulating prologue/epilogue once. The later figure supersedes.
- Occupancy cost model for the multistage GEMM (`lab/gemm_model.py`): held out on 11 never-run tiles
  (`data/gemm_model_heldout_preds.json`), median error 12.7% but rank correlation only 0.5-0.55. After
  the occupancy sweep (`scripts/occupancy_sweep.py`, `data/occupancy_sweep.json`) a copy-lane cap of 500
  in flight at 570 cycles gave 9.8% / 6.4% median error on all 88 timings, but was chosen on those
  points, and ranking did not improve.
- Blind tile ranking (`data/rank_fresh/`, `data/rank_direction/`): 195 never-measured tiles, median
  error 5-15%, pairwise accuracy 0.57 (copy-cap model) and 0.46 (queue model), Spearman -0.49 to 0.49
  per shape. Right on single-variable directions for occupancy (15/16) and k-chunk (29/30), wrong for
  launch bounds (2-4 of 12) and ring depth (4-8 of 13).
- `3080lab optimize` guided mode (`sass-optimization.md`): guided was not better than blind; decision
  accuracy fell to 5/9 under high register pressure.

The pattern going in: absolute error is acceptable, but differences between good options are smaller
than the error.

## Method

- Step 1, noise (`scripts/aa_noise.py`, job 2216, `data/optsel/aa-noise-2216.log`): a 2x2 design over
  module copy and output-buffer set with 3 fresh processes. Buffer placement gave transpose a stable
  1.0% gap; module copy contributed 0. The CUDA event timer reads in steps of 1.024 us.
- Protocol v2: K arms rotate over K buffer sets every round, an A/A arm (`ptxas_dup`) rides in every
  case, and the statistic is a 10%-trimmed mean of paired per-round log ratios pooled over 2 processes.
- Corpus (`data/optsel/corpus_*.jsonl`, built by `scripts/optsel_build.py`):
  - rand_dev / rand_heldout: 40 + 40 random kernels x 4 SASS transforms (jobs 2211/2212, clock64);
  - suite: 19 kernels x 7 SASS transforms, protocol v2 (job 2217, `scripts/optsel_suite.py`);
  - polybench_sass: 40 PolyBench kernels x 7 transforms, protocol v2, every candidate bitwise checked
    on 3 input sets (job 2219, `scripts/optsel_poly.py`);
  - polybench_fusion: 12 programs x {graph, fused, fused_graph} (`data/xkernel/polybench.json`,
    `lab/xkernel.py`, `scripts/xkernel_bench.py`).
- Features (`lab/optsel.py`) from static SASS and launch configuration only: occupancy and its change,
  registers and their change, hot-loop instruction mix, changes in stall sum, scoreboard waits,
  dependence depth and reuse flags, launch-count change, data shared between launches, and the hardware
  model's prediction (`lab/model.simulate` at resident warps/SM for SASS; a launch-cost model with
  constants from `scripts/launch_gap.py` for fusion).
- Predictors: never, always, heuristic, hw_model (apply if hw_pred > tau), feature model (ridge on y
  plus logistic for P(y > tau)), oracle.
- Training on rand_dev and suite only; each test family scored once (`scripts/optsel_eval.py
  split=frozen`). Win or loss means |y| > tau = 1%.

## Results

### A/A noise after protocol v2

Suite A/A is at most 0.69% (q95 0.47%) over 20 kernels. PolyBench A/A q95 is 2.8% (max 2.98% in
`data/optsel/eval_frozen.txt`).

### Frozen held-out evaluation (`data/optsel/eval_frozen.txt`)

| held-out family | predictor | directional acc. | Spearman | applied W/L | regret | realised (oracle) |
|---|---|---|---|---|---|---|
| rand_heldout (160) | never / always | 0.65 / 0.35 | | 0/0, 27/50 | 0.0062 | 1.000 (1.0047) |
| | heuristic | 0.75 | 0.40 | 13/5 | 0.0081 | 0.998 |
| | hw_model | 0.74 | 0.43 | 14/7 | 0.0081 | 0.998 |
| | feature model | 0.73 | 0.49 | 6/0 | 0.0054 | 1.0007 |
| polybench_sass (276) | always | 0.43 | | 29/38 | 0.0138 | 0.994 (1.0064) |
| | hw_model | 0.57 | 0.16 | 0/0 | 0.0079 | 1.000 |
| | feature model | 0.75 | 0.09 | 13/1 | 0.0070 | 1.0007 |
| polybench_fusion (36) | always | 0.94 | | 17/1 | 0.0612 | 1.159 (1.231) |
| | heuristic (launches fall or graph) | 1.00 | 0.53 | 17/0 | 0.0612 | 1.159 |
| | hw_model (launch cost) | 0.83 | 0.65 | 14/0 | 0.0026 | 1.229 |
| | feature model | 0.94 | 0.65 | 17/1 | 0.0033 | 1.229 |
| | features without hw_pred | 0.94 | 0.28 | 17/1 | 0.0612 | 1.159 |

- SASS transforms: oracle ceiling 0.47-0.64%; the best predictor captures 0.07% of it. Spearman on
  held-out seeds 0.43-0.49 (sign better than chance); on PolyBench 0.09-0.16 because noise exceeds the
  effects.
- In-family vs held-out: on rand_dev (leave-one-family-out, `data/optsel/eval_lofo.txt`) the hardware
  model has directional accuracy 0.90, Spearman 0.75, and realises 1.005 against 1.006. On unseen seeds
  of the same generator it falls to 0.74 and realises 0.998, because it approves sb_late where sb_late
  loses.
- Fusion and graphs: the launch-cost model chooses within 0.2% of the oracle. Its magnitudes are badly
  wrong (it predicts e^3 on 3dconv, because a Python stream launch costs 8.8 us alone but overlaps with
  kernel work in sequence); only its ranking is right. Without hw_pred the feature model falls to the
  fixed-rule result, so all of the value is the hardware model's.

### Fusion and graph optimizer itself (`lab/xkernel.py`)

Vertical and horizontal fusion of adjacent launches plus graph capture on 12 PolyBench programs, all
bitwise identical; geomean of fused + graph 1.226x (jacobi1d 3.92x graph only, mvt 1.36x, bicg 1.34x,
3dconv 1.23x, fdtd-2d 1.13x, jacobi2d 1.10x, lu 1.04x, other 5 1.00x).

### Calibration and abstention

After Platt scaling on the training families, held-out ECE is 0.09-0.16, not reliable enough to act on
alone. Abstention (apply only when P(win) >= 0.7) never takes a loss, but covers 0-7% of SASS decisions.
On fusion it covers 28-31% and realises 1.169 against 1.229 without abstention.

### tau sensitivity (post hoc)

At tau = 3% (above PolyBench's A/A q95; post hoc and labelled as such, `data/optsel/eval_frozen_tau03.txt`)
the SASS families have almost no real wins left, 4 of 160 and 5 of 276, and every predictor ties with
"never".

### Demo: 1/4-size PolyBench (job 2222, `data/optsel/demo_eval.json`)

Plan committed before the run (addendum in `data/optsel/PROTOCOL.md`). Predictors fit on the whole
benchmark; per program each picks base, graph, fused or fused_graph.

| policy | realised | programs slower than base |
|---|---|---|
| never (base) | 1.000 | 0 |
| always graph / heuristic | 1.479 | 3 |
| hw_model | 1.602 | 2 |
| always fused_graph | 1.619 | 1 |
| feature model | 1.631 | 2 |
| oracle | 1.642 | 0 |

The feature model is within 0.7% of the oracle and beats always fused_graph by 0.7%, but base vs
base_dup reaches 5% on jacobi1d and 3.3% on corr in this run. The "slower" picks are graph on atax and
gemver at -1.1% to -1.2%, near noise.

## Negative results and limits

- Block-local SASS selection is not worth doing on this corpus: headroom (< 0.7%) is below both the
  noise and the model's held-out error.
- The hardware model generalizes within a family but not to new seeds of the same generator (0.90 to
  0.74 directional accuracy).
- PolyBench noise has two characterized causes left: 4-5 us kernels are only 4-5 timer steps long (adi,
  lu1, gramschm2, jacobi1d/2d), and in-place accumulating kernels (atax2, gramschm3) have 9-15% CV.
  Back-to-back launches per sample would fix the first and were not built.
- Bootstrap intervals that resample rounds were too narrow for the earlier suite A/A (softmax 1.035 vs
  0.997 on identical binaries), the motivation for protocol v2.
- The launch-cost model's magnitudes are uncalibrated (additive gap). The notebook's suggested next step
  is an overlap-aware max(issue time, work) model, which would need a new held-out set.
- The demo shows parity with "fuse what is legal, then capture a graph", not a demonstrated win.
- Earlier tile-ranking and occupancy models estimate time reasonably but cannot judge between two tiles
  within about 10% of each other.

## Reproduce

```
pcslurm submit -- python scripts/aa_noise.py
pcslurm submit -- python scripts/optsel_suite.py
pcslurm submit -- python scripts/optsel_poly.py
pcslurm submit -- python scripts/launch_gap.py
python scripts/optsel_build.py rand      # also: suite, poly, fusion
python scripts/optsel_eval.py split=frozen
python scripts/optsel_eval.py split=lofo
python scripts/run_env.py XK_AA=1 XK_KDIR=... -- python scripts/xkernel_bench.py   # 1/4-size run
python scripts/optsel_demo.py
python scripts/eval_gpu_model.py ; python scripts/gemm_model_sweep.py ; python scripts/gemm_model_variants.py
python scripts/rank_fresh.py ; python scripts/rank_direction.py
```

Data: `data/optsel/` (protocol, corpora, `eval_frozen*.txt/json`, `eval_lofo.*`, `demo_small.json`,
`demo_eval.json`, job logs), `data/xkernel/polybench.json`, `data/occupancy_sweep.json`,
`data/occupancy_model.json`, `data/gemm_model_heldout_preds.json`, `data/rank_fresh/`,
`data/rank_direction/`.

## Notebook sections

From [the notebook](../notebook.md):
- Predictor (Phase 5)
- Whole-GPU predictor for real kernels
- Open questions closed (Oct 8): Predictor
- Occupancy cost model for the multistage GEMM (Oct 10)
- Occupancy sweep: the bias was the copy path (Oct 11)
- `3080lab optimize` (decision accuracy and regret)
- Oct 18-21: "Can the hardware model pick optimizations? (Nov 14)", "Cross-kernel optimizer", "Blind
  tile ranking"
