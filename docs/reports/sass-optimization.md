# SASS-level optimization against ptxas

Summary. Three kinds of binary rewrite were tested against ptxas on the RTX 3080, all with bitwise
output checks. Register re-allocation for bank conflicts gave 1.11-1.33x on FFMA:SHF mixes. A
targeted pass that re-pairs split 32 B sector loads (Phase 7, generalized as `3080lab optimize`)
took the 7B gate_up R8U2 GEMV from 94.3 to 62.7 us (1.50x) and gave held-out geomeans of 1.036 /
1.074 / 1.055 on 120 int4 GEMV kernels with 0 correctness failures. A general block scheduler
(Phase 8) reached parity with ptxas (1.003 on random kernels, 0.994-1.003 on a held-out suite) but
not a win; an oracle over three orders reaches only 1.005, and joint register allocation plus
scheduling and software pipelining (Nov 14) gave no measurable gain.

## Question / hypothesis

With the hazard, bank and reuse rules measured (see `microarchitecture.md`), can a pass that edits
the cubin directly beat ptxas? Sub-hypotheses: (1) bank-aware register allocation removes conflicts
ptxas leaves; (2) ptxas's load order wastes DRAM bandwidth by splitting sectors; (3) a list scheduler
with regenerated control words beats ptxas's order; (4) renaming registers to remove false WAR/WAW
edges, or software pipelining, frees the scheduler.

## Method

- Register re-allocation (`lab/regalloc.py`, experiments `realloc_*` in `lab/experiments/realloc.py`):
  live-range (web) splitting over the CFG, interference from liveness, annealed colouring for bank
  conflict cost, operand fields from `lab/operands.py`. Checked three ways: disassembly matches the
  substitution, reaching definitions of every use are unchanged, GPU outputs are bit-identical.
- Phase 7 load-order pass (`lab/schedule.py`, `gemv_load_order` sassfix variants): find split-sector
  pairs, retarget the guarding ISETP to a free predicate and its 87 guards, hoist late halves to their
  partners with legality checks, and repair blocked moves (rename destination webs, carry default
  defs, land after the guard, slide past close address producers).
- `3080lab optimize` (`lab/optimize.py`): analyze with an affine abstract interpretation of LDG
  addresses (`lab/sectors.py`); predict per trip `max(T_sm, unique / BW(in flight), (unique +
  refetched) / peak)` with m(D) read off `mem_split_mech` (`lab/costmodel.py`); rewrite
  (`lab/schedule.py`); validate (nvdisasm decode, hazard verifier with no new hazards, <= 255
  registers); in guided mode keep the rewrite only if predicted gain >= 1.02x.
- Benchmark (`scripts/opt_bench.py`, `docs/optimize_benchmark.md`, kernels in
  `lab/experiments/gemv3.py`): 120 int4 GEMV kernels. Suite A 57 kernels (9 shapes x 5 layouts plus
  -O1 and 64-register-cap variants); suite B 43 kernels (Qwen2.5-3B, Llama-3-8B, small squares), never
  used while building; suite C 20 high-register-pressure kernels (R16U2, R8U4, R16U1). Two input seeds,
  60 launches per arm each after a read-only 64 MB L2 flush, arm order shuffled every round,
  20%-trimmed mean of the in-kernel span. Noise floor from byte-identical binaries: 1.2-1.5% at
  8 us and above, 7.9% below.
- Phase 8 block scheduler (`lab/resched.py`): list-schedules every basic block from a dependence DAG
  on top of ptxas's register allocation and regenerates stalls, wait masks, write barriers and reuse
  flags. Independent hazard checker `lab/verify.py` walks execution order with loops unrolled 3x.
  Random kernels from `lab/experiments/randk.py` (experiment `resched_rand`), hardware rules from
  `pair_latency` and `mufu_latency`.
- Held-out tests (Oct 18-21): 20 textbook kernels (`lab/suite.py`, `scripts/suite_run.py`,
  `data/suite/`) with `scripts/suite_bisect.py` to localise failing blocks; 40 PolyBench/GPU kernels at
  a pinned commit (`scripts/sassopt.py`, `scripts/ext_polybench.py`, `data/ext_polybench/`).
- Joint RA + scheduling (Nov 14, `lab/jointra.py`, `data/ra/`): `jointra.split_crit` moves the later
  definition's web of each WAR/WAW edge on a loop's critical path into a spare register (8 extra),
  repeats while the path shrinks, then sb_late rebuilds the DAG. Every rename must pass
  `regalloc.prove` and `verify.new_hazards`. Static headroom from `scripts/ra_headroom.py`,
  `scripts/ra_reuse_probe.py`, `scripts/swp_headroom.py`. A kernel counts as improved only if its
  p05-p95 range clears ptxas's by more than 1%.

## Results

### Register re-allocation

| FFMA:SHF mix | ptxas | re-allocated | speedup |
|---|---:|---:|---:|
| 3/8 | 2.43 | 2.97 | 1.22x |
| 4/8 | 2.64 | 3.52 | 1.33x |
| 5/8 | 2.87 | 3.75 | 1.31x |
| 7/8 | 3.52 | 3.91 | 1.11x |

(warp-instr/cyc/SM.) A hand operand patch on the 4/8 mix reached 3.91, 1.48x over ptxas.

### Phase 7 on R8U2 GEMVs

| R8U2 shape | ptxas u_major | sassfix (gap>=64) | sassfix (gap>=3) | source fix (lane_contig) |
|---|---|---|---|---|
| gate_up 17920x1536 | 94.2 us | 62.5 us (1.51x) | 63.5 us | 54.3 us |
| down 1536x8960 | 66.6 us | 63.5 us | 65.5 us | 59.4 us |
| q_o 1536x1536 | 15.4 us | 14.3 us | 15.4 us | 13.3 us |

Outputs are bitwise identical (SHA-1 of y). About 15% remains to the source-level fix, which also
changes which L1 lines each warp touches.

### `3080lab optimize` (held-out GEMV benchmark)

Cause, from microbenchmarks: L2 reuse distance. `mem_split_mech` gives full speed with halves at most
2 loads apart and 0.83 / 0.61 / 0.50 of peak at G = 4 / 8 / >= 16. Once the bytes requested GPU-wide
between halves exceed the 5 MB L2, the second half comes from DRAM again.

| held-out geomean speedup vs ptxas | A (56) | B (43) | C (20) |
|---|---|---|---|
| source fix (lane-contiguous, where a layout has one) | 1.098 | 1.188 | 1.369 |
| blind rewrite | 1.036 | 1.074 | 1.055 |
| guided v1 | 1.037 | 1.075 | 1.053 |
| guided v3 | 1.036 | 1.072 | 1.050 |

- Correctness: 0 failures; every rewrite is bitwise equal to ptxas and every arm matches a float64
  reference.
- Unseen kernels: Llama-3-8B gate_up 1.62x (R8U2), 1.59x (R4U2); Llama-7B up 1.57x; Qwen2.5-3B
  gate_up 1.47x and 1.43x, 1.33x at R16U2; Llama-3-8B down 1.14x; spilling 64-register builds 1.17x to
  1.38x.
- Where it wins: multi-wave grids 1.3x to 1.6x; long-K sub-wave grids 1.1x to 1.2x; short sub-wave
  grids (qo7, sq4k) nothing.
- Decision accuracy (v3, kernels where the rewrite measurably helps or hurts): 14/15 on A, 13/15 on B,
  5/9 on C. Median speedup prediction error 4.1% (A), 4.9% (B), 24% (C) for v3 against 11-34% for v1.

Regret against a per-kernel oracle:

| policy | A | B | C |
|---|---|---|---|
| never rewrite | 3.49% | 7.38% | 7.13% |
| blind rewrite | 0.07% | 0.04% | 1.37% |
| guided v1 | -0.07% | 0.02% | 1.57% |
| guided v3 | -0.01% | 0.44% | 1.95% |

Note: suite A was not fully held out for v3 (v3 was revised on A, post hoc, per
`docs/optimize_benchmark.md`); B and C are held out for both v1 and v3.

### Phase 8 scheduler, first results (40 random kernels)

All 120 rescheduled kernels (identity, crit, random orders) are bitwise identical to ptxas over 5
launches each. Geomean of ptxas/ours: identity 0.964, critical path 0.932, random legal 0.846. On
memory-bound R8U2 GEMVs it loses (gate_up 94.2 to 99.3 us, down 66.6 to 94.2, q_o 15.4 to 20.5), so
the targeted Phase 7 pass stays the right tool there.

### Oct 18-21 held-out tests and wait placement

- The suite exposed four rules the random kernels never hit (fixed-latency producer to SHFL needs 5
  cycles; BRA/EXIT predicate needs 13; memory ops at least 4 apart; BAR.SYNC.DEFER_BLOCKING to first
  memory op at least 6). With them, all 19 timed cases are bitwise identical under every policy
  (black_scholes excluded: ptxas's own build misses the CPU reference). Geomean crit 0.999, identity
  0.999, sb 1.000 (`data/suite/imad.json`).
- PolyBench: 0 static failures, 0 bitwise failures. sb 0.992x raw, 1.000x after keep/revert; sb_wide
  1.003x raw, 1.008x after keep/revert. An earlier 1.011x relied on timings the four rules now forbid.
- Ablation on random kernels (`data/ablation/`): identity 0.959, with ptxas reuse flags 0.965, no reuse
  flags 0.939, crit 0.920, renaming then crit 0.917, sb 0.980, random 0.845. The loss was where
  scoreboard waits sat: waits on values from earlier blocks were hoisted to block entry and barriers
  were drained at block exit.
- Wait placement fix (`late_entry` + `sticky_waits`): wait words match ptxas on 12,172 of 12,200
  instructions (11,138 before). Job 1882: identity 0.959 to 0.987, crit 0.920 to 0.976, sb 0.980
  unchanged. Suite (job 1993, `data/suite/late2.json`): 0.996-1.003 across variants.
- Reuse model and stall fixes (commits d7dae46, 682a8d2):

| random kernels (geomean vs ptxas) | job 1882 | + reuse model (2050) | + stall fixes (2054) |
|---|---|---|---|
| identity_late | 0.987 | 0.993 | 1.003 |
| crit_late | 0.976 | 0.982 | 0.989 |
| sb_late | 0.980 | 0.985 | 0.995 |

  Suite (`data/suite/reuse.json`, `data/suite/exit.json`): 0.994 to 0.998, within noise.

### Why scheduling stops at parity (Oct 24, `scripts/sim_pick.py`)

- sb_late loses 2 to 5% against identity_late on high-occupancy kernels where ptxas already issues
  about 3 warp-instructions per cycle per SM, and changes nothing at low occupancy.
- Picking the simulator's predicted-fastest of three orders per kernel measures 1.001 against 1.003
  for identity_late alone (sign right on 16 of 24 seeds that differ by more than 0.5%).
- A perfect oracle over the three orders reaches only 1.005. The wins that exist (1.02 to 1.04 on
  low-occupancy seeds) come from control words, for example ptxas stalling 6 cycles on FFMA to MUFU
  where 4 is enough.

### Joint RA + scheduling and software pipelining (Nov 14)

Static headroom:

| lever | headroom |
|---|---|
| bank-aware recolouring | 6 of 51 loops improve, by 1 cycle each; ptxas within 3-5% of a rough floor |
| more reuse hits through order | about 3% fewer reads at best, only by ignoring latency |
| removing false WAR/WAW edges | 6% geomean; large on scan_block (2.07x), gemv, nbody, spmv_csr |
| software pipelining | random kernels 0.97 (dev) and 1.006 (held-out); suite loops with room are DRAM-bound or run 1-2 trips |

GPU (jobs 2210-2214; `data/suite/ra_suite*.json`, `data/suite/ra_scan.json`, logs in `data/ra/`):

| geomean vs ptxas | random dev (40) | random held-out, seed 3000+ (40) | suite (16-19) |
|---|---|---|---|
| identity_late (existing best) | 1.003 | 1.000 | 0.999 |
| sb_late | 0.995 | 0.966 | 0.999 |
| renaming only (split_identity_late) | 1.003 | 0.998 | 0.998 |
| joint (split_sb_late) | 0.996 | 0.967 | 1.000 |
| renaming vs identity_late, improved / regressed | 0 / 0 | 0 / 1 | - |

0 correctness failures across all 80 random kernels and every timed suite case. CPU dry run over
100 kernels: 0 refusals, 0 new hazards.

## Negative results and limits

- Phase 7 gap>=3 (fix all pairs) renames more and raises registers 103 to 151 on gate_up, no better
  than fixing only long gaps. A load-only reorder cannot reach the source fix.
- Prediction-guided is not better than blind on the GEMV benchmark: on ordinary GEMVs the hoist never
  hurts, so blind is already at the oracle. Under high register pressure it hurts: R8U4 at 1536 columns
  runs at 0.81x after renaming takes it from 96 to 195 registers (20 to 8 resident warps per SM), and
  neither model predicts it.
- Failed cost-model hypotheses: v2 (re-fetches only cost at the bandwidth roof) changed nothing; v3
  fixed a 2x unit mismatch in m(D) (error 11% to 4%) and dropped an SM-simulator latency-hiding gain
  that did not exist on the GPU; cold start (first trip immune) refuted (m = 0.96 at 1 trip); v4 (post
  hoc on C) left C error at 24% and is opt-in; in-flight bytes by barrier tracking raised A/B error to
  24%; m(D) does not depend only on D (at D about 4 MB, 0.44 at 136 warps vs 0.18 at 1088).
- Register cost of the rewrite cannot be compacted: R8U2 needs >= 128 registers (renaming used 131),
  R8U4 >= 160 (used 195). Register caps soften losses (0.82x to 0.92x) but never create wins.
- `schedule.dedicate_barrier` (false scoreboard dependence) nets -1% geomean with clean timing; opt-in.
- Measurement artifact: with a fixed arm order each arm inherited the previous arm's L2 contents, which
  produced apparent 4-8% losses on qo7. A memset flush raised noise to 14-28%; a read-only flush and
  shuffled order fixed it.
- The Qwen2.5-1.5B decode kernels have nothing to fix: every streaming split pair is already adjacent.
- Phase 7 latent bug: `hoist()` checked producer distance for registers, not guard predicates; the
  verifier caught it. `-maxrregcount` is ignored with `__launch_bounds__`.
- Register re-allocation soundness bugs: `CS2R Rn` writes Rn:Rn+1; only whitelisted pure 32-bit
  opcodes are renamable. `regalloc.webs` put a predicated write and its implicit read in different
  webs (found by the Nov 14 proof, fixed). The build cache keyed on source + `build_key()` once
  silently measured the unpatched cubin.
- Phase 8 scheduler: no variant beats ptxas without timed keep/revert. Run-to-run noise on the suite is
  about 1% (softmax moved 176 to 183 us); an A/A pair on softmax measured 1.035 vs 0.997, so suite
  differences under about 4% on that kernel are noise.
- sb_late does not generalize: developed on seeds 1000-1039, it loses 16-18 of 40 held-out kernels,
  down to 0.75x.
- Joint RA: scan_block's critical path drops 87 to 47 cycles yet measures 1.000 / 0.996 (32 warps hide
  it). 24 spare registers lose 3% on the suite (0.969) through occupancy. Recommendation in the
  notebook: abandon joint RA + scheduling and SASS software pipelining on this corpus. Beating ptxas by
  1.05x would need something block-local reordering cannot reach.
- An LDS left without a scoreboard by ptxas must keep memory order (sb_wide broke conv_row and
  bitonic_block, `data/suite/bisect_wide_late.json`); fixed.

## Reproduce

```
3080lab optimize k.cubin --block 128 --grid 560 --trips 2 --report rep.json
3080lab optimize k.cubin --mode blind
python scripts/opt_bench.py build A 3 && pcslurm submit -- python scripts/opt_bench.py run A
python scripts/opt_bench.py report                      # writes docs/optimize_benchmark.md
python scripts/suite_run.py variants=ptxas,identity_late,sb_late,split_identity_late,split_sb_late out=ra_suite
python -m lab.cli run resched_rand_b --policies identity_late,sb_late,split_identity_late,split_sb_late
python scripts/ra_headroom.py
python scripts/swp_headroom.py
python scripts/ra_report.py ; python scripts/rand_report.py
python scripts/sim_pick.py
python scripts/sassopt.py
```

Data: `docs/optimize_benchmark.md`, `data/suite/`, `data/ablation/` (cycles files
`job2050_reuse_cycles.txt`, `job2054_exit_cycles.txt`, `sim_pick_2054.json`), `data/ext_polybench/`,
`data/ra/`.

## Notebook sections

From [the notebook](../notebook.md):
- The FP32/INT32 "no overlap" ceiling was a ptxas register-allocation artifact
- Automatic register re-allocation beats ptxas
- Open questions closed (Oct 8): the R8U2 GEMV mystery
- Round 4: load order, launch cost, L2 slices (Oct 8)
- Phase 7: automatic SASS pass beats ptxas on R8U2
- `3080lab optimize`: a general split-sector optimizer (Phase 7 generalized)
- Phase 8: our own block scheduler
- Oct 18-21: held-out tests, cross-kernel fusion, and why the scheduler stops at parity (through
  "Joint register allocation + scheduling: negative")
