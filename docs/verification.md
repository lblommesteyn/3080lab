# What a 3080lab SASS rewrite guarantees, and what it does not

The SASS tools here (`lab/resched.py`, `lab/jointra.py`, `lab/optimize.py`, `scripts/sassopt.py`) rewrite
ptxas's cubins in place. This page states exactly which properties are checked statically, which are only
tested, and which rest on hardware measurements of one GPU. Short version: **nothing here is a proof of
semantic equivalence.** A rewrite is accepted when a static hazard check finds nothing new relative to
ptxas's own schedule and, where a harness exists, when the outputs match ptxas's bitwise on a handful of
input sets. Both are conditional on a supported subset and on a timing model measured on one RTX 3080.

## 1. Supported subset

A rewrite only ever reorders instructions and patches control words, register fields and guard predicates.
It never changes opcodes, immediates, addressing or the instruction count (`verify.match` checks this by
matching every rewritten instruction to an original one with only those fields masked).

**Target.** sm_86 (GA102), cubins from the pip-wheel ptxas of CUDA 13.x, one kernel per cubin. Encodings
of other architectures are not handled.

**Instructions the scheduler may move** (`lab/resched.py`). A basic block is rescheduled only if every
instruction in it has a modelled operand map and none is in `UNSAFE`. Otherwise the block keeps ptxas's
order and only gets the exit rule (section 2). Specifically, a block keeps its order if it contains:

| refused | why |
|---|---|
| any opcode whose register fields `lab/operands.py` cannot map by perturbation (`_resources` returns `None`) | defs/uses unknown |
| uniform predicates (`UP0`-`UP6`) | not modelled |
| `ST*`, `ATOM*`, `RED`, `SUATOM`, `SURED`, `SUST` | memory order. With `wide=True` (`sb_wide`) stores/atomics may move, but any pair where either side writes memory keeps its order (no alias analysis: all memory is assumed to alias) |
| `MEMBAR`, `FENCE`, `ERRBAR`, `CCTL`, `DEPBAR`, `LDGSTS`, `LDGDEPBAR`, `ARRIVES` | ordering / async copy semantics |
| `S2R`, `CS2R` (except `CS2R Rx, SRZ`), `B2R`, `R2B`, `LEPC` | special registers, clocks |
| `VOTE`, `MATCH` | warp-collective |

Control instructions (`BRA`, `BRX`, `JMP`, `CALL`, `RET`, `EXIT`, `BSSY`, `BSYNC`, `WARPSYNC`, `BAR`, ...)
always end a block and never move.

**Register renaming** (`lab/regalloc.py`, used by `lab/jointra.py` and `lab/optimize.py` hoist+rename) only
recolours webs whose every operand is in the operand map. Opcodes with register pairs/quads or implicit
registers (`operands.WIDE_HINTS`: 64/128-bit forms, `DFMA`, `HMMA`, `IMMA`, `LDSM`, `SHFL`, `ATOM`, `TEX`, ...)
are never renamed.

**Outside-kernel harness** (`scripts/sassopt.py`): exactly one `extern "C" __global__` kernel; every
parameter a pointer to a supported element type or a supported scalar; no spills, no local stack; the
rewrite must keep the register count.

## 2. What is checked statically, on every rewrite

| property | where | strength |
|---|---|---|
| The cubin still decodes | `toolchain.disassemble` (nvdisasm) after every rewrite | encoding validity only |
| Same multiset of instructions, only register/predicate/control fields changed | `verify.match` | exact |
| Renaming preserves dataflow: every use keeps the same reaching definitions, and no non-register bit changed | `regalloc.apply` | exact over the modelled operand map; unmodelled operands block the rename |
| No RAW, WAW or WAR hazard the original schedule did not already have | `verify.new_hazards` | sound *relative to the timing model below*, on the modelled path (see limits) |
| Memory issue gap: no two memory ops closer than 4 cycles, unless ptxas already had that pair that close | `verify.new_hazards` (`MEM_GAP`) | empirical rule |
| At least 6 cycles from `BAR.SYNC` to the next memory op | `verify.new_hazards` (`BAR_MEM`) | empirical rule |
| No fixed-latency result is still in flight at a block exit | `resched.reschedule` (exit padding) | by construction, every block |
| Reuse flags only on forms where ptxas itself set that bit | `resched._reuse_block` | by construction |

`lab/verify.py` is deliberately independent of the scheduler: it rebuilds defs/uses itself and walks the
code in execution order (prologue, the largest loop body (`sass.loop_body`) three times so loop-carried hazards are seen,
then the epilogue). The scheduler's own dependence DAG is never trusted by the check.

### The timing model the hazard check relies on

A hazard is "a reader issues before the writer's result is guaranteed". Whether that is a hazard depends on
latencies, which come from measurement, not from documentation:

* FMA-pipe or ALU-pipe producer to a consumer in the same pipe: 4 cycles. Across pipes, or into `SHFL`: 5.
  (`pair_latency`; `SHFL` at 4 gave wrong reductions on two suite kernels, so it is treated as cross-pipe.)
* FP32 producer into `MUFU`: 4 (`mufu_latency`; ptxas uses exactly 4 on 91 such pairs, never less).
* A predicate read by a branch or `EXIT`: 13 (ptxas never leaves less on 240 pairs; 12 broke `spmv_csr`).
* Any other fixed-latency producer: the smallest producer-to-consumer distance ptxas itself left in this
  cubin (clamped to 5..15), else 15. This trusts ptxas never to under-stall.
* Scoreboarded (variable-latency) producers: correct by wait-mask bookkeeping, plus a measured 2-cycle
  arm delay for barriers.

Every number above was measured on one RTX 3080 at the clocks in the notebook. Several were tightened
only after a rewrite produced wrong answers (Oct 18 2026; the notebook records each case).

## 3. What is only tested (empirical)

| test | where | coverage |
|---|---|---|
| Bitwise output equality with ptxas's cubin | `scripts/sassopt.py` stage 3 | 4 random seeds plus 1 edge seed (zeros, signed zero, denormals, inf, NaN, extreme ints) by default; every non-const buffer compared bitwise; a fault or timeout rejects |
| Bitwise equality on the benchmark corpora | `scripts/optsel_poly.py`, suite runs, `resched_rand` | 3 seeds per PolyBench kernel; the random kernels compare an output hash (`y_hash`) of every variant to ptxas |
| Held-out correctness of the optimizer | `docs/optimize_benchmark.md` | 0 correctness failures over the GEMV suites |

Differential tests find bugs in the inputs they exercise. They say nothing about data-dependent paths the
seeds did not reach (for example a branch taken only on a rare value), about race conditions that need a
specific warp interleaving, or about launch shapes other than the one tested.

## 4. Known gaps

1. **Unmodelled instructions are skipped by the checker, not rejected.** When `_resources` cannot model an
   instruction, `verify.check` advances time past it without checking its operands. The scheduler never
   moves such a block, so a hazard there can only come from moves in a *neighbouring* block. That case is
   covered by the exit rule, not by the checker.
2. **One path.** The checker follows the fall-through path and only the largest loop (three iterations).
   Taken forward branches, every other loop and nested-loop back edges are not walked. Every block exit is
   drained by the exit rule, so cross-block fixed-latency hazards are excluded by construction rather than
   checked.
3. **Hardware rules are inferred.** The latencies, `MEM_GAP` and `BAR_MEM` are measurements and "ptxas never
   does less" observations on one GPU. Another sm_86 part at other clocks, or a driver that changes
   scheduling behaviour, is not covered.
4. **No memory model.** All memory is assumed to alias. That is conservative for ordering, but nothing
   reasons about inter-thread visibility; blocks with fences or atomics are simply not touched.
5. **Encoding validity is nvdisasm's opinion.** A word nvdisasm decodes may still be invalid on hardware
   (the notebook found that the stall field must be 1..11 with the yield bit set; `_set_stall` encodes that).

## 5. How to read a result

* "Rewrite kept" from `sassopt.py` means: static check clean, bitwise equal on N input sets, faster by the
  timing rule. It does not mean equivalent on all inputs.
* `lab/optimize.py` results in `docs/optimize_benchmark.md` were validated statically on every kernel, and
  dynamically on the benchmark's inputs.
* Anything outside section 1's subset is refused, and the output is ptxas's cubin unchanged.
