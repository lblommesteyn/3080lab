# 3080lab

An empirical microarchitecture lab for one RTX 3080 (GA102, sm_86).
Goal: measure the machine well enough to predict kernel runtime from SASS,
then schedule better than ptxas, then make Qwen faster.

## Current results: Qwen2.5-1.5B, llama.cpp's own Q4_0 GGUF, same RTX 3080

| workload | ours | llama.cpp b11485 | ratio | measured | section |
|---|---:|---:|---:|---|---|
| decode, 1 sequence (tok/s) | 514.0 | 365.8 | **1.41x** | Oct 14 | Night run, Oct 13 |
| batched decode, 2 / 4 / 8 / 16 / 32 sequences (tok/s) | 950 / 1792 / 3395 / 5578 / 7135 | 625 / 972 / 1416 / 2489 / 3597 | **1.52 / 1.84 / 2.40 / 2.24 / 1.98x** | Oct 14 | Night run, Oct 13 |
| prefill, 128 / 512 / 2048-token prompt (tok/s) | 21,888 / 30,767 / 28,114 | 10,937 / 18,502 / 17,639 | **2.00 / 1.66 / 1.59x** | Oct 14 | Night run, Oct 13 |

Both sides best of repeated runs (`scripts/night_summary.py`). Correctness: last-token argmax
matches a token-by-token reference; perplexity on README text within 0.1% of an fp16 cuBLAS path.
The sections below are a lab notebook in the order the work happened; later sections supersede
earlier numbers.

## Quick start

```
.venv\Scripts\3080lab list                    # experiments
.venv\Scripts\3080lab sass dependent_ffma     # compile only: timed-loop SASS + control bits + raw bytes
.venv\Scripts\3080lab run dependent_ffma      # queues via pcslurm, waits, prints report
.venv\Scripts\3080lab run independent_ffma --warps 1,2,4,8,16,32
.venv\Scripts\3080lab run dependent_iadd --ptxas=-O3 --force   # see what folding does
.venv\Scripts\3080lab show results\<dir>      # re-print a saved record
```

`run` submits itself through `pcslurm submit --shared` so it never competes
with other GPU jobs; `--local` runs in-process (that is what the queued job runs).

## Toolchain

No CUDA toolkit install. Everything comes from pip wheels (CUDA 13.x):
NVRTC (CUDA C to PTX), `ptxas` (PTX to cubin, flags under our control),
`nvdisasm -hex` (SASS + 128-bit encodings), `cuda-python` driver API (load the
cubin, launch), NVML (clocks, temperature, throttle). No host compiler is
involved; kernels are loaded as cubins, which is also the path SASS patching
will take later.

## Measurement protocol

Every run:

1. **Static validation before any GPU time.** The timed loop is pinned with
   `#pragma unroll 1` around an explicitly generated body. The runner parses
   the SASS, isolates the loop, and refuses to measure unless the loop holds
   exactly the expected count of the target opcode and nothing spills.
2. **Warmup** for a fixed wall time (default 500 ms) so the GPU leaves idle P-state.
3. **Shuffled, interleaved trials** across all variants (seed recorded), 30 per variant by default.
4. **Per-trial environment**: NVML SM/memory clock, temperature, power, P-state, throttle reasons.
5. **In-kernel clocks**: `clock64()` cycles and `%globaltimer` ns bracket the
   loop, giving cycles/op (frequency independent) and the true in-kernel SM clock.
6. **Correctness**: integer chains are checked against a host reference.
7. **Warnings** for throttling, clock drift, temperature, CV > 2%, SASS mismatch, wrong results.

Artifacts per run go in `results/<timestamp>_<experiment>/`: `.cu`, `.ptx`,
`.cubin`, ptxas log, raw nvdisasm, decoded SASS listing, SASS JSON
(per-instruction raw bytes + stall/yield/barriers/wait mask/reuse), and
`record.json` with config, environment, every trial, and summary distributions.

## Control-word decoding

Bits [105:125] of each 128-bit instruction: stall (4), yield (1), write
barrier (3), read barrier (3), wait mask (6), reuse (4). Listing format:
`S04 - W- R- wait:- reuse:0` means stall 4, no yield bit, no scoreboards.

## Findings so far (RTX 3080, boost ~1965 MHz, cycles from clock64)

### Instruction table (`3080lab table` regenerates `results/TABLE.md`)

| op | dependent latency (cyc) | throughput, warp-instr/cyc/SM (32 warps) | lanes/SM |
|---|---:|---:|---:|
| FFMA / FADD / FMUL | 4 | 3.94-3.97 | 128 |
| HFMA2 | 4 | 2.00 | 64 (x2 halves) |
| IMAD | 4 | 2.00 | 64 |
| SHF | 4 | 1.99 | 64 |
| IADD3 | 4 (stall-probe) | n/a yet (see -O0 note) | |
| MUFU.RSQ / MUFU.EX2 | 17 | 0.50 | 16 |
| FMUL.RZ + MUFU.SIN | 23 (pair) | 0.50 | 16 |
| SHFL.IDX | 26 | 0.50 | 16 |
| DFMA | 55 | 0.0625 | 2 (1/64 rate) |
| LDS (shared) | 23 | | |

### The hardware trusts the compiler (control-bit patching)

Patching the stall field of a dependent FFMA/FADD/IMAD/IADD3 chain:
stall 4+ gives correct results; stall 1-3 gives **wrong answers in 20/20
trials** with exactly half the increments lost (each op reads the register
before the previous write lands, so pairs collapse). There is no interlock
for fixed-latency ops: latency = 4 cycles, measured by breaking it.
**Stall 0 is not "zero"**: it is correct but costs ~32.8 cycles/op. That is
why `-O0` code (which emits S00 on IADD3/LOP3) runs dependent chains at
33.7 cycles; `-O0` timings are invalid as latency measurements.
A single warp cannot issue dependent ALU ops faster than 1 per 2 cycles
(stall 1 and 2 both give ~2.06), while 8 independent chains in one warp
reach ~0.95 FFMA/cycle.

### Hazards: data is the compiler's job, structure is the hardware's (`issue_*`, `scoreboard_*`)

One warp, k independent chains, every op's stall patched to s:

- FFMA: correct iff the real time between producer and consumer is >= 4
  cycles (k=2,s=1 runs 1.55 cyc/instr -> 3.1 cycles apart -> wrong;
  k=4,s=1 -> 4.2 -> correct). Legal code issues FFMA at ~1/cycle from one
  warp. The "2-cycle floor" only appears in *illegal* schedules (a 1-cycle
  bubble when an op reads a register with a write still pending), so it
  costs nothing in real code.
- IMAD never issues faster than 1 per 2 cycles from one warp, whatever the
  stall field says: a 16-lane pipe takes 2 cycles per warp, and that
  **structural** hazard is interlocked by hardware. Same for DFMA (16
  cyc/instr, one shared FP64 pipe), MUFU (8), SHFL (4).
- Scoreboards are expensive and necessary. With waits stripped, an isolated
  SHFL is correct at 12-cycle spacing (wrong at 8) vs 26 cycles through the
  scoreboard; MUFU.SIN is correct at 15 (wrong at 12) vs ~19 through the
  scoreboard. But with 2-4 SHFLs in flight, 16 cycles is no longer enough,
  and DFMA is wrong even 64 cycles apart (correct only with 8 chains): issue
  runs ahead of variable-latency pipes, so no static stall can stand in for
  the barrier.
- Stripping a SHFL barrier also corrupted the kernel's *clock* registers
  (negative cycle counts): the late SHFL write landed on a register ptxas
  had already reused. Write-after-write is a scoreboard job too.
- nvdisasm rejects yield=1 combined with stall 0 or stall >= 12 as illegal
  encodings, so the control field is not fully free.

### L1: size, carveout, replacement (`chase_l1_carveout`, `l1_replacement`)

| preferred carveout | L1 flat up to | fully missing at |
|---|---|---|
| default / 0 | 104 KB | > 136 KB (degrades from 112) |
| 50 | 56 KB | 96 KB |
| 100 | 16 KB | 32 KB |

The carveout attribute moves L1 exactly as expected. Even at carveout 0,
L1 holds ~100-108 KB of data, not 128.
Per-access scripted timing (L1 hit ~54 cycles including timing overhead,
miss ~250): after filling N lines, a forward re-read thrashes past ~100 KB
(hit 0.9% at 128 KB), while a reverse re-read keeps ~780 lines (97 KB).
**Replacement is recency-based (LRU-family), not FIFO and not random**:
lines re-touched after the fill survive 100% even though they were inserted
first, and untouched lines die oldest-first (survival 25% oldest vs 79%
newest when streaming 384 new lines). Not strict LRU; some old lines
survive, which fits pseudo-LRU with uneven set load.

### Register file: 2 banks, 1 read each, and the reuse cache is mandatory (`regbank_ffma`)

Every FFMA in an 8-chain loop rewritten in the cubin to `FFMA S, A, S, C`
with chosen register parity (operand fields patched directly; register count
raised by patching both `sh_info` and EIATTR_REGCOUNT):

| operand banks | no reuse flags | reuse on A and C |
|---|---:|---:|
| all three in one bank | 0.32 issue/cyc/partition (3 cycles) | 0.94 |
| two in one bank | 0.49 (2 cycles) | 0.94 |

Two banks (register parity), one read per bank per cycle. A 3-source FFMA
always has two operands in one bank, so **without the reuse cache FFMA can
never exceed 1 issue per 2 cycles**; full FP32 rate depends on reuse.
(Consistent with Huerta et al. MICRO'25; the costs here are measured.)
Also: with a register count of 64, R62 and R63 fault with
CUDA_ERROR_ILLEGAL_INSTRUCTION; only R0-R61 are usable.

### The FP32/INT32 "no overlap" ceiling was a ptxas register-allocation artifact (`mixbank_*`)

In the FFMA:SHF mixes, the interleaved SHF evicts one FFMA operand from the
reuse cache, and ptxas had put the two remaining FFMA register reads in the
same bank. Moving each FFMA's chain register to the opposite bank (operand
patch only, same instructions, same schedule):

| FFMA fraction | ptxas | opposite bank | same bank |
|---|---:|---:|---:|
| 3/8 | 2.45 | 3.15 | 2.44 |
| 4/8 | 2.65 | **3.91** | 2.65 |
| 6/8 | 3.17 | 3.92 | 3.17 |

FP32 and INT32 do run concurrently: the fixed kernel tracks
min(4, 2/(1-p)) warp-instr/cyc/SM, i.e. only the 16-lane INT pipe limits it.
**A register renumbering alone makes the 50/50 mix 1.48x faster than ptxas.**
IMAD does not benefit (it shares the fmaheavy pipe with FFMA; real contention).

### Reuse cache and yield semantics

- `.reuse` only takes effect when the raw yield bit is 1 (warp does not
  yield); nvdisasm hides `.reuse` otherwise (85/512 IMADs in one test). This
  fixes the yield polarity the literature disagrees on: raw 1 = stay.
- Entries are per (bank, operand slot). An instruction that does not read a
  slot (RZ, immediate, constant) leaves that entry alone, so a value can stay
  cached across an intervening instruction. A read of the same bank and slot
  either refreshes the entry (reuse flag) or evicts it.
- IMAD (half rate) follows the same bank rules: 3 operands in one bank cost 3
  cycles; 2 in one bank cost nothing extra because the read overlaps the
  2-cycle pipe. Each bank has its own read port.

### Predictor (Phase 5)

`lab/model.py` simulates one SM from the control bits plus measured tables.
Out-of-sample on 60 random kernels it never saw: v0 median error 3.4%, 68%
within 5%. On a second held-out set: v0 5.9%, v1 (fma/alu pipes, per-bank
register ports, reuse cache) 7.0%. v1 fixes register-bank kernels (7.9% ->
2.6%) and the HFMA2 mix, but still mis-handles some IMAD+FFMA mixes and
IMAD/SHF-heavy multi-warp kernels. Cache-capacity edges need a partial-hit model.

### Automatic register re-allocation beats ptxas (`lab/regalloc.py`, `realloc_*`)

Live-range (web) splitting over the CFG, interference from liveness, annealed
coloring for bank-conflict cost, operand fields discovered empirically per
instruction form (`lab/operands.py`). Every result is checked three ways:
disassembly text must match the substitution, reaching definitions of every
use must be unchanged, and GPU outputs must be bit-identical.

| FFMA:SHF mix | ptxas | re-allocated | speedup |
|---|---:|---:|---:|
| 3/8 | 2.43 | 2.97 | 1.22x |
| 4/8 | 2.64 | 3.52 | 1.33x |
| 5/8 | 2.87 | 3.75 | 1.31x |
| 7/8 | 3.52 | 3.91 | 1.11x |

Two soundness bugs found on the way (both now pinned): plain `CS2R Rn` writes
the pair Rn:Rn+1, and a renamed read of the hidden Rn+1 corrupted the kernel's
clock while outputs still matched. Lesson: only whitelisted pure-32-bit
opcodes are renamable, and the runner now cross-checks clock64 against
globaltimer. The proof is only as good as the operand model, so GPU checks stay.

### int4 GEMV baseline (Phase 9 start, `gemv_int4`)

One-warp-per-row W4 (group-128) GEMV at Qwen2.5-7B layer shapes reaches only
161-260 GB/s of ~760 GB/s. Batch-1 decode here is instruction/latency bound,
not DRAM bound: per 32 weights ptxas emits 32 FFMA + 84 integer ALU ops
(including 32 IADD3 for the "-8") + 32 I2FP. Bank re-allocation does nothing
for it (no conflicts in the hot loop). This is the target for Phase 9.

### int4 GEMV: 2.4-2.5x from memory-level parallelism, then dequant count (`gemv_int4_v2`)

| shape (N x K) | naive R1 | best | speedup | GB/s |
|---|---:|---:|---:|---:|
| gate_up 18944 x 3584 | 135.2 us | 56.3 us (R4, magic) | 2.40x | 642 (~84% of peak) |
| down 3584 x 18944 | 151.6 us | 60.9 us (R4, magic) | 2.49x | 594 |
| q_o 3584 x 3584 | 31.7 us | 13.3 us (R4, magic) | 2.38x | 515 |
| qkv_kv 512 x 3584 | 6.1 us | 5.1 us | 1.20x | 194 (launch/tail bound) |

At one row per warp the dequant scheme does not matter (latency bound, too
few loads in flight). Four rows per warp fixes that, and then the kernel is
instruction bound, where the exact "magic" fp32 dequant (`0x4B000000|q`,
one FADD instead of IADD3 + I2F) adds ~14% (561 -> 642 GB/s). The half2
(Marlin-style) variant currently computes wrong results (harness catches it);
not yet debugged.

### Phase 9: Qwen2.5-1.5B decode, end to end

| decode (batch 1, greedy, ~40-token prompt, 256 new tokens) | tok/s |
|---|---:|
| PyTorch eager-style ops + tinygemm int4 linears, one CUDA graph | 207 |
| llama.cpp b11485, official Q4_0 GGUF, `-fa 0` | 316 |
| llama.cpp b11485, official Q4_0 GGUF, `-fa 1` | 340.5 |
| **ours, llama.cpp's exact Q4_0 weights** (`scripts/qwen_gguf.py`) | **437** |
| ours, own int4 g128 weights incl. int4 lm_head (`scripts/qwen_fused.py`) | 501 |

The win is fusion, not the GEMV: tinygemm is already ~94% of the 705 GB/s
practical roofline at 7B shapes, and our split-K GEMV ties it. The PyTorch
decode launches ~1,100 kernels/token; ours ~200 (fused RMSNorm, GEMVs with
bias / residual / SiLU*mul epilogues, one RoPE + KV-write + GQA attention
kernel, argmax + position update), all captured in one CUDA graph.

The GGUF run is the fair one: every tensor comes from llama.cpp's file
(Q4_0 linears repacked bit-exactly, Q6_K output stored exactly as int8 +
fp32 scale per 16, which reads 292 MB/token vs llama.cpp's 191 MB). Both
engines produce coherent answers that diverge after 8 words (llama.cpp
quantizes activations to Q8_1; we keep bf16 + fp32 accumulation). The
official Qwen GGUF has per-channel scales folded into its norm weights
(W*diag(norm) matches HF to 9.6% while the norms differ by 65%), so all
tensors must come from the same file. At 2.29 ms/token we are at ~64% of the
bandwidth bound (1.03 GB/token at 705 GB/s); the rest is launch overhead and
small kernels.

### Where the remaining 2.29 ms/token goes (GGUF run)

- GEMVs ~1.65 ms: layers 1.235 ms (per layer qkv 4.1 + o 3.1 + gate/up 23.6 +
  down 13.3 us; 432-657 GB/s) + exact-int8 lm_head ~0.42 ms (292 MB at 697 GB/s).
  A Q6_K-native head (191 MB) would save ~0.15 ms.
- Attention ~0.2 ms (v1: 4.4 us at pos 40 -> 11.5 us at 300). A one-launch
  flash-decoding version (`ATTN2`, last-block combine) is exact but only wins
  past ~300 positions: the cross-block combine costs what the parallelism saves.
- Norms, embed, argmax, and ~1 us per kernel inside a graph for ~200 kernels.
- Kernel cost in a graph: ~1.0 us empty, growing with grid size (4,480 blocks:
  4.8 us; 37,984 blocks: 27.5 us, ~49 ns/block/SM). With real work per block the
  dispatch overlaps execution: G = 1..8 row-groups per block made no difference.

### Memory system under load (`mem_*`, `smem_banks`, `l1_wavefronts`, `l2_latency_map`, `tlb_reach`)

- **Transfer granularity is the 32 B sector** for L2 and DRAM: useful bandwidth
  halves per stride doubling up to 32 B. L2 serves ~2.0-2.2 TB/s, DRAM ~710-720
  GB/s regardless of 4/8/16 B per-lane width. At a 256 B lane stride DRAM drops
  to ~60% of the sector rate (row/bank locality; 256 B chunks per bank/row).
- **Little's law**: one SM caps at ~63 GB/s; DRAM saturates at 710 GB/s with
  >= 4 warps/SM on 68 SMs or >= 16 warps/SM on 17 SMs. Loaded DRAM latency
  (calibrated through the SM simulator): 570 cycles = 292 ns.
- **L1 wavefronts**: an L1-hit load costs one SM cycle per distinct 128 B line
  the warp touches (floor ~3): 1 line 2.8, 4 lines 4.3, 8 -> 8.0, 32 -> 32.0 cycles.
  ld.global.nc behaves identically. This, not DRAM latency, was why the naive
  one-row-per-warp GEMV was slow (each fp32 x load touched 32 lines).
- **Shared memory**: 32 banks x 4 B, but one request per 2 cycles, so 2-way
  conflicts are free; n-way costs n/2 (4-way 0.25, ..., 32-way 0.031 warp-instr/cycle).
- **L2 map** (every line of a 2 MB buffer timed from all 68 SMs, trial-to-trial
  r = 0.997): no A100-style bimodal split. Variance = SM position 69% (45-cycle
  range; SMs 2k/2k+1 identical = TPC pairs; blocks of 6 = GPC placement), line
  21%, and a rank-1 SM x line interaction 9%. That interaction is a two-way L2
  half selected by **XOR of address bits {8, 11, 14, 15, 16, 17, 19}**, which
  explains 100.0% of 16,384 lines; near/far costs only ~3-12 cycles.
- **TLB**: cudaMalloc uses 2 MB pages; a 16-entry first level (knee between 32 and
  48 MB), and only +10-15 cycles beyond it even at 2 GB.

### Whole-GPU predictor for real kernels (`lab/model_gpu.py`, `scripts/eval_gpu_model.py`)

T = max(SM simulation of the real SASS loop, bytes / 710 GB/s) + fixed overhead.
The SM simulation uses the calibrated DRAM latency, scoreboards, pipes, register
banks, and an SM-wide L1 port charged per 128 B line. Which pointer each load
reads is found by **flow-sensitive address provenance** (reaching definitions
back to kernel parameters); lines per instruction come from each kernel
family's access pattern, scaled by active lanes. Every constant comes from
microbenchmarks. On 131 measured GEMV variants: **median error 13.6%, 73% within
20%** (23.1% without the L1 term). Remaining misses: register-heavy R8U2
kernels (~2x slower than predicted, unexplained) and epilogue cost on 3-4 us kernels.

### Tensor cores (`tc_*`)

| mma.sync | latency (cyc) | peak / SM / cycle | at 1.95 GHz |
|---|---:|---:|---:|
| FP16 or BF16 -> FP32 acc | 32.9 | 512 FLOP | 67.9 TFLOPS |
| FP16 -> FP16 acc | 24.0 | 1,024 FLOP | 135.5 TFLOPS |
| TF32 -> FP32 | 32.9 | 256 FLOP | 33.9 TFLOPS |
| INT8 / INT4 -> INT32 | 24.0 | 2,044 / 4,088 ops | 271 / 542 TOPS |
| binary AND+POPC | 24.0 | 16,351 ops | 2,168 TOPS |

FP32 accumulation is half rate (GeForce segmentation). One warp per SM
partition saturates the tensor core: the issue interval (32 cycles FP32-acc, 16
FP16-acc) is about its latency, so chains within one warp do not overlap.
Register fragment layouts for m16n8k16 f16 and m16n8k32 s8 verified exactly
against numpy (`tc_correct`).

### Open questions closed (Oct 8)

**L1 capacity** (`l1_capacity`, `l1_carveout_fine`): capacity is constant in
128 B *lines* (~830) across slot strides 32-128 B, and one formula fits all
eight measured carveouts:

    L1 data = 128 KB - max(16 KB, smem carveout) - 8 KB

(0/8/16% -> 104 KB, 24/32% -> 88, 50% -> 56, 100% -> ~20). The 16 KB floor and
the constant 8 KB are inferred from the data, not documented.

**L2 slices** (`l2_latency_map`): per-line latency profiles across 68 SMs are
2-dimensional and discrete; equal-size groups of ~410 lines -> ~40 slices of
128 KB (4 per 32-bit memory controller), interleaved at 256 B. Some slice pairs
have identical latency profiles, so the full slice hash (non power-of-two) is
not recoverable from latency alone; needs an eviction-set method.

**Instruction cache** (`icache`): full issue rate up to 64 KB of loop code; one
warp per partition drops at 96 KB (1.34/cycle) and 192 KB (0.82).

**The R8U2 GEMV mystery** (`gemv_l1_pollution`, `gemv_ablate`, `mem_split_gap`,
`gemv_load_order`): with U=2 each lane's two 16 B loads split every 32 B sector
across two instructions; under register pressure ptxas emits the second-half
loads ~1,000 instructions after the first; L1 merges a second half-sector request
only within ~2 loads (microbenchmark: gap 4 -> 609 GB/s, gap 8 -> 441 vs 723), so
each sector is fetched twice. Refuted on the way: instruction cache, L1
pollution, split sectors at any occupancy, partial trips, per-warp MLP.
Fix (predicted, then measured): lane-contiguous U loads -> gate_up R8U2
94.2 -> 55.3 us (1.70x), q_o 15.4 -> 12.3, down 65.5 -> 59.4.

**Predictor**: adding the split-sector rule (efficiency 0.61 from the
microbenchmark) takes the GEMV median error to 12.8% (77% within 20%); simulating
each kernel's prologue/epilogue once (later waves' straight-line code overlaps
other warps' loops) brings it to **11.5% median, 79% within 20%** on 131 variants
(7B split-K family 6.8%). Residual: ~1.3 us on 4 us kernels (block-dispatch ramp
and the __syncthreads/shared-memory reduction, not modeled).

**Predictor-guided Qwen work** (`scripts/qwen_kernel_times.py`): per-kernel
cost vs bandwidth floor (with clocks warmed; the first attempt ran at the 210 MHz
idle clock and was 10x off). Sum 2.136 ms/token vs 2.29 measured; floor 1.468;
recoverable 0.668 ms: attention 31%, RMSNorm launches 19%, qkv 15%, gate/up 15%,
down 11%, o 8%. Attention ablation (fixed 3.0 us, scores 2.5, PV 1.5 at pos 160)
showed iterations not overlapping across shuffle reductions; attention v4
(4-way ILP in scores and PV) is exact and 17-21% faster. **End to end on
llama.cpp's Q4_0 weights: 446 -> 463 tok/s, 1.36x llama.cpp (340.5).**

### Qwen decode, round 3 (same bytes as llama.cpp)

| change (GGUF weights, Qwen2.5-1.5B) | tok/s |
|---|---:|
| attention v1 | 446 |
| attention v4 (ILP) | 463 |
| + Q6_K-exact lm_head (191 MB, llama.cpp's own format and bytes; checked vs float64, 9e-8) | 492 |
| + gate/up split-K 2, single-pass RMSNorm | **509 (1.50x llama.cpp's 340.5)** |

**Re-measured Oct 11 with the desktop idle** (Night run section): llama.cpp 368.7 tok/s, ours 509.6, so **1.38x**.

Refuted: folding RMSNorm into the residual GEMVs with a last-block-done tail
(396 tok/s): the fenced, serialized tail costs ~8.7 us per site against 2.25 us
for a separate kernel in the graph. Also refuted: prefetching the next GEMVs'
weights into L2 from idle SMs during attention (505/504/502 tok/s for 1/3/6 MB
vs 512 without), even though `prefetch.global.L2` (CCTL.E.PF2) demonstrably fills
L2 (chase after prefetch 279 cycles vs 495). A software grid barrier costs ~1.0 us
with 68-136 co-resident blocks (1.3 us at 272, 2.3 at 544), the same as a kernel
boundary in a graph, so a persistent megakernel would not remove the per-kernel
cost. Hung kernels are not reset by TDR on this machine: spin loops need caps. Remaining gap to the bandwidth floor (0.52
ms/token) is per-kernel fixed cost: attention 32%, RMSNorm 18%, GEMV tails.

### Round 4: load order, launch cost, L2 slices (Oct 8)

- **Production Q4_0 GEMVs are clean**: weights never split a sector; only the
  L1-resident activation vector has split pairs, 1-6 instructions apart (inside
  the merge window). No ptxas load-deferral problem there.
- **Kernel launch cost in a CUDA graph** (`scripts/launch_cost.py`): 1.07-1.28 us
  for an empty kernel regardless of block shape (1-68 blocks, 128-1024 threads) or
  parameter count; +0.2 us when it declares 5.7 KB of static shared memory;
  dispatch grows past ~500 blocks (4,480 blocks: 3.46 us, ~34 ns per block per SM).
  This is most of the "~1.3 us residual" on tiny kernels and most of attention's
  fixed cost. Splitting a head across blocks needs a cross-block combine (>= 2 us
  measured), so one block per head is near the floor on Ampere.
- **L2 = 40 slices x 128 KB, mapped by contention** (`scripts/l2_slices.py`):
  67 SMs hammer one line while a probe SM times every line of a 2 MB buffer.
  Getting this right took fixing three of my own bugs: ptxas hoisted the
  hammer's loop-invariant load, flag polling was itself a hammer, and hammer
  blocks on the probe's SM slowed everything through that SM's load pipeline.
  With 57 adaptive references every line has a unit: 37 units of 402-428 lines
  (16,384/40 = 409.6), the rest one unsplit pair. Slices interleave at 256 B
  (98% of 256 B chunks in one slice). The hash has a **linear part of rank 3**:
  XOR of bits {12,13,15,16,17,18}, {8,11,14,15,16,17,19} (the near/far half),
  {10,14,16,17,20} are constant within every slice, giving 8 classes of 5 slices.
  The 5-way selector is not a simple modulo of address bits below 2^21 (best
  0.36 vs 0.24 chance); it likely uses higher physical bits or a table.

### Phase 7: automatic SASS pass beats ptxas on R8U2 (`lab/schedule.py`, `gemv_load_order` sassfix*)

ptxas splits each 32 B weight sector into two 16 B loads and, under register pressure, defers the second half
by ~1000 instructions; L1 only merges halves issued within ~2 loads, so every sector goes to L2 twice. The pass
re-joins the halves directly in the cubin, with no source change:

1. Find split-sector pairs (same base register, offsets 16 B apart in one 32 B sector).
2. The late halves are guarded (`@!P0`) by an `ISETP` placed just before them, and P0 is reused as a carry flag
   elsewhere. Retarget that ISETP to a free predicate (P4: dest field hi[17:20]), retarget the 87 guards that
   consume it (guard field lo[12:15]), then hoist the ISETP up to the first half.
3. Hoist each late half up to its partner. Legality checks: no barriers, memory ops or control flow in range;
   no register RAW/WAR/WAW; predicates conflict only when one side defines them; a fixed-latency producer
   is at least 6 cycles away; a scoreboard producer's barrier is added to the wait mask.
4. Repairs when a move is blocked:
   - rename the load's destination web to fresh registers (regcount raised);
   - carry along the constant default ptxas writes before a guarded load (the not-taken value);
   - land after the guard def;
   - slide the landing point down past a too-close address producer.

| R8U2 shape | ptxas u_major | sassfix (gap>=64) | sassfix (gap>=3) | source fix (lane_contig) |
|---|---|---|---|---|
| gate_up 17920x1536 | 94.2 us | **62.5 us (1.51x)** | 63.5 us | 54.3 us |
| down 1536x8960 | 66.6 us | 63.5 us | 65.5 us | 59.4 us |
| q_o 1536x1536 | 15.4 us | 14.3 us | 15.4 us | 13.3 us |

The outputs are bitwise identical to the unpatched ptxas kernel (SHA-1 of y) for every shape and variant.
Fixing all pairs (gap>=3) renames more and raises regcount further (103 -> 151 on gate_up), so it is no better
than fixing only the long gaps. About 15% remains to the source-level fix. lane_contig also changes which
lines each warp touches (fewer L1 lines per request), and a load-only reorder cannot do that.
Gotcha: the runner caches builds by source + `build_key()`, so a transform-only variant must override
`build_key`. The first run silently measured the unpatched cubin.

### `3080lab optimize`: a general split-sector optimizer (Phase 7 generalized)

```
3080lab optimize k.cubin --block 128 --grid 560 --trips 2 --report rep.json   # guided (default)
3080lab optimize k.cubin --mode blind                                        # rewrite whenever valid
python scripts/opt_bench.py build A 3 && pcslurm submit -- python scripts/opt_bench.py run A
python scripts/opt_bench.py report                                           # docs/optimize_benchmark.md
```

Pipeline for one kernel:
1. **Analyze** (`lab/sectors.py`). An affine abstract interpretation of the SASS gives every LDG
   address as `root + const + coef * lane`. It follows `S2R`, constant banks, `IADD3`, `IMAD(.WIDE)`,
   `LEA`, `SHF`, lane masks, spill slots (`STL`/`LDL`) and the loop back edge. A pair is two loads
   whose sectors overlap, given each address's lane stride, base, offsets and width; this is exact
   for any base alignment. A load "streams" if its address depends on the warp or block id.
   Nothing is R8U2-specific.
2. **Predict** (`lab/costmodel.py`). Per loop trip:
   `max(T_sm, unique / BW(in flight), (unique + refetched) / peak)`.
   - Re-fetched bytes = overlap x bytes x m(D).
   - D = streaming bytes requested between the halves x concurrent warps.
   - m(D) is read off the `mem_split_mech` microbenchmark.
   - Whole kernel = fixed overhead + trips x per trip.
3. **Rewrite** (`lab/schedule.py`). Hoist each later half next to its partner. Guard predicates are
   retargeted, destination webs renamed, default defs carried, and loads slide past producers
   that are too close. Global loads may cross local spills, and `.CONSTANT` loads may cross any
   store.
4. **Validate**. nvdisasm must decode the result. The hazard verifier must find no hazard the
   original lacks, matching instructions by encoding so register renames do not hide anything.
   Registers must stay at 255 or fewer.
5. **Decide**. Guided mode keeps the rewrite only if the predicted gain is at least 1.02x.

**What the hardware does (microbenchmarks, not the evaluation kernels):**

| test | result |
|---|---|
| `mem_split_mech`, halves G loads apart | full speed at G <= 2; 0.83 / 0.61 / 0.50 of peak at G = 4 / 8 / >= 16 |
| second half issued only after the first half's data arrived | no help: not a pending-miss merge problem |
| `.nc` (L1) vs `.cg` (L1 bypass) | the same: L1 does not rescue it |
| 1, 2, 4, 32 trips per warp | m = 0.96 to 0.99: not a steady-state effect (hypothesis refuted) |
| `mem_split_phase`: pairing done by the optimizer itself | 1.3x to 1.9x at 0.25 to 4 waves, 2 or 8 trips |

The cause is **L2 reuse distance**. Once the bytes requested GPU-wide between the two halves
exceed the 5 MB L2, the second half is fetched from DRAM again.

**Benchmark** (`docs/optimize_benchmark.md`). 120 int4 GEMV kernels:
- **Suite A, 57 kernels:** 9 shapes x 5 load layouts, plus -O1 and 64-register-cap variants, where
  the cap makes ptxas spill.
- **Suite B, 43 kernels:** Qwen2.5-3B, Llama-3-8B and small-square shapes, never used while
  building or revising anything.
- **Suite C, 20 kernels:** high register pressure (R16U2, R8U4, R16U1).

Each kernel ran on two random input seeds. Timing: 60 launches per arm, each after a read-only
64 MB L2 flush, with arm order shuffled every round; I report the 20%-trimmed mean of the in-kernel
span. Byte-identical binaries give the noise floor: 1.2-1.5% at 8 us and above, 7.9% below.

- **Correctness: 0 failures.** Every rewritten kernel's output is bitwise equal to ptxas's. Every
  arm matches a float64 reference.

| held-out geomean speedup vs ptxas | A (56) | B (43) | C (20) |
|---|---|---|---|
| source fix (lane-contiguous, where a layout has one) | 1.098 | 1.188 | 1.369 |
| blind rewrite | 1.036 | 1.074 | 1.055 |
| guided v1 | 1.037 | 1.075 | 1.053 |
| guided v3 | 1.036 | 1.072 | 1.050 |

- **Phase 7 recovered automatically:** 7B gate_up R8U2 goes 94.3 -> 62.7 us (1.50x).
- **Unseen kernels improved:**
  - Llama-3-8B gate_up: 1.62x (R8U2), 1.59x (R4U2);
  - Llama-7B up: 1.57x;
  - Qwen2.5-3B gate_up: 1.47x and 1.43x; at R16U2, 1.33x;
  - Llama-3-8B down: 1.14x;
  - spilling 64-register builds: 1.17x to 1.38x.
- **Decision accuracy** (v3), counting kernels where the rewrite measurably helps or hurts:
  14/15 on A, 13/15 on B, 5/9 on C.
- **Speedup prediction error** (median): 4.1% on A, 4.9% on B and 24% on C for v3, against 11-34%
  for v1.

**Prediction-guided vs blind: not better.** I measured regret against a per-kernel oracle (the
better of ptxas and the rewrite, with differences inside the noise band counted as ties):

| policy | A | B | C |
|---|---|---|---|
| never rewrite | 3.49% | 7.38% | 7.13% |
| blind rewrite | 0.07% | 0.04% | 1.37% |
| guided v1 | -0.07% | 0.02% | 1.57% |
| guided v3 | -0.01% | 0.44% | 1.95% |

On ordinary GEMVs the hoist essentially never hurts, so blind is already at the oracle. Under high
register pressure it does hurt: R8U4 at 1536 columns runs at 0.81x after renaming takes it from 96
to 195 registers (20 -> 8 resident warps per SM). Neither model predicts that, so suite C is
where a better occupancy model would pay off.

**When and why it beats ptxas.** Under register pressure ptxas issues the second 16 B halves of a
lane's 32 B sector 20 to 30 loads after the first (u_major order). When the reuse distance of
those halves exceeds the L2, the halves are fetched from DRAM twice, and re-pairing them gains:
- multi-wave grids: 1.3x to 1.6x;
- long-K sub-wave grids: 1.1x to 1.2x;
- short sub-wave grids (qo7, sq4k): nothing.

**Failed hypotheses and corrections** (all kept in the code or this record):

| item | outcome |
|---|---|
| v2: re-fetches only cost time at the bandwidth roof (latency regime) | made zero difference: the mispredicted kernels never sat in that regime |
| v3 fix: m(D) curve in useful bytes vs D in sector bytes (a 2x unit mismatch) | fixed; prediction error 11% -> 4% |
| v3 fix: SM simulator's predicted latency-hiding gain from moved loads | did not exist on the GPU; v3 keeps the original's issue time |
| cold start (first trip immune) | refuted: m = 0.96 at 1 trip |
| **measurement artifact**: with a fixed arm order, each arm inherited the previous arm's L2 contents | byte-identical binaries differed by a full 1 us tick on 6 MB weights. The apparent 4-8% losses on qo7, and my 300-launch "confirmation" of them, came from this. Fixed by a cold L2 plus shuffled order. A first flush by memset raised noise to 14-28% (write-backs of dirty lines landing inside the timed kernel); a read-only flush fixed that. |
| false scoreboard dependence (hoisted loads keep ptxas's barrier) | plausible, but the only supporting gains (1.08x on qo7) came from the biased runs. With clean timing, `schedule.dedicate_barrier` nets -1% geomean, so it is opt-in and unconfirmed |
| v4, post hoc on suite C: SM term at the rewrite's occupancy, partial-lane trips and reuse distance | principled and harmless on A and B (3.8% / 4.8% error) but suite C error unchanged (24%): gu15 R8U4 still predicted 1.02x against 0.81x measured. Not held-out validated, so opt-in (`version=4`); v3 stays the default |
| in-flight bytes from barrier tracking (Little's law per outstanding load) | refuted: A and B median error 4% -> 24%, r_major wins predicted as losses |
| register cost of the rewrite (rename compaction) | not possible: the predicated loads have no default values, so their old destination registers are live around the whole loop and interfere with every value. The minimum cost is inherent anyway (each hoisted value lives through the window): R8U2 needs >= 128 registers (renaming used 131), R8U4 >= 160 (used 195) |
| register caps at occupancy cliffs (partial hoisting, `max_regs`) | soften losses (gu15 R8U4 0.82x -> 0.92x) but never create wins; on the winners the full hoist is best (up11k R8U4: 1.25x full vs 1.14x / 1.11x capped). Register pressure is not the main cause of the losses |
| m(D) depends only on D = bytes between x warps | refuted (`mem_split_mech`, G x warps grid): at equal D, fewer warps re-fetch MORE (D ~ 4 MB: 0.44 at 136 warps vs 0.18 at 1088). Small GEMV grids gain nothing because they never run at the DRAM roof (qo7: about 455 of 710 GB/s), which the model's in-flight estimate cannot see |
| `mem_split_mech` with ~70 variants | the runner keeps every variant's buffer alive; 70 x 256 MB overflowed the 10 GB card and WDDM paged buffers to host memory (25 GB/s readings). Buffers are now 64 MB |
| the Qwen2.5-1.5B decode kernels (Q4_0 GEMVs, Q6_K head, RMSNorm, embedding, attention) | nothing to fix: every streaming split pair is already adjacent (gap 0, at most 1 in the Q6_K head); the hand-written kernels are lane-contiguous |
| Phase 7 latent bug | `hoist()` checked producer distance for registers, not guard predicates. The verifier caught a load placed 1 cycle after its 13-cycle ISETP; fixed |
| `-maxrregcount` | ptxas ignores it when the source has `__launch_bounds__`; capped builds use `__launch_bounds__(128, 8)` |

### Batched decode kernels (`lab/qwen_batched.py`, `scripts/batched_gemv_bench.py`)

llama.cpp b11485 batched decode (`llama-batched-bench`, Qwen2.5-1.5B Q4_0, flash attention, prompt 128,
128 generated per sequence), in tok/s summed over sequences: B=1 336, 2 614, 4 964, 8 1420, 16 2424,
32 3566. Prefill is about 17-18K tok/s.

At B = 1 our decode is 509 tok/s, but a token step re-reads all weights for each sequence. Two
batched Q4_0 GEMVs read each weight once for B input vectors:

- **CUDA cores** (`gemv_batched_source`). Dequantize one 32-bit word (8 weights) and FMA it against
  the matching 8 values of all B vectors. Correct (bitwise equal to the single-vector kernel), but
  only about 2.3x over B separate launches at B = 4. At B = 8, the per-weight FMAs and 244
  registers make it compute- and occupancy-bound; capping at 128 registers makes it spill.
- **Tensor cores** (`gemv_mma_source`, `mma.sync` m16n8k16 fp16 -> fp32). A warp owns 16 rows; Q4_0
  nibble pairs become `half2` via the 0x6400 exponent trick, scaled by the block's fp16 scale; the
  B input columns are converted bf16 -> fp16. Each A fragment is reused across ceil(B/8) n-tiles.
  The error against the fp32 single-vector kernel is 2e-4 to 5e-3 (weights rounded to fp16).
  Memory parallelism matters most: one block of weights in flight per warp left gate_up at
  220 GB/s. With 8 split-K warps and 4 blocks in flight (S = 8, U = 4), per launch:

| us per launch | B = 1 | 4 | 8 | 16 | 32 | production single-vector, per vector |
|---|---|---|---|---|---|---|
| qkv 2048 x 1536 | 5.4 | 5.9 | 7.3 | 11.0 | 20.3 | 5.3 |
| o 1536 x 1536 | 5.3 | 8.8 | 7.6 | 10.9 | 19.0 | 4.7 |
| gate_up 17920 x 1536 | 30.3 | 34.7 | 46.8 | 80.4 | 146 | 25.4 |
| down 1536 x 8960 | 32.3 | 37.0 | 38.8 | 52.7 | 92 | 14.5 |

`down` (long K, only 96 row tiles of 16) lacks parallelism at small B.

At large B the B fragments (inputs) are the expensive part: each is converted bf16 -> fp16 and
loaded once per 16-row tile. Giving each warp MT m-tiles (16·MT rows) reuses every B fragment MT
times. Split-K partials go through one shared [16·MT][B] buffer, accumulated warp by warp in a
fixed order (deterministic, and under the 48 KB static shared-memory limit). Best per shape:

| us per launch, tuned (S, U, MT) | B = 8 | 16 | 32 |
|---|---|---|---|
| qkv | 7.2 (8,2,2) | 11.0 | 16.7 (8,2,2) |
| o | 7.5 | 10.9 (8,4,2) | 16.6 (8,4,2) |
| gate_up | 37.3 (8,2,2) | 59.7 (8,2,2) | 90.8 (4,2,4) |
| down | 29.5 (16,2,2) | 42.5 (16,2,2) | 59.5 (16,2,2) |

At B = 32 the four GEMVs drop from 277 to about 184 us per layer.

**End to end** (`scripts/qwen_batched_decode.py`). Tensor-core GEMVs; embedding, RMSNorm, attention
(separate KV cache per sequence) and argmax are the single-sequence kernels, with a batch index from
`blockIdx.y`; the Q6_K head is dequantized once to fp16 for a cuBLAS matmul. The whole step is a
CUDA graph. B copies of one prompt give identical, coherent sequences. With the tuned (S, U, MT)
per shape (`CONFIG` in the script, bucketed by B), they match the validated single-sequence decode
for 84 of 128 tokens and then diverge, presumably at a near-tie flipped by the fp16 weight rounding
of the tensor-core path. The B = 32 config (gate_up with S = 4, MT = 4) changes the split-K
summation order and diverges earlier, at token 25, still into coherent text; with the B = 16
config at B = 32 it matches for 84 tokens again, at 4607 tok/s.

| decode tok/s (128 generated per sequence) | B = 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| ours, S = 8, U = 4, MT = 1 | 342 | 1258 | 2133 | 2989 | 3556 |
| ours, tuned per shape | | 1278 | 2501 | 3618 | **5002** |
| ours, tuned + Q6_K tensor-core head | | 1384 | 2680 | 3717 | (cuBLAS head) |
| ours, v2 GEMVs (packed weights, shared-memory inputs) | | 1753 | 3248 | 5191 | 6595 |
| ours, v2 GEMVs + v2 Q6_K head | 468 | **1821** | **3420** | **5573** | **7106** |
| llama.cpp b11485 | 336 | 964 | 1420 | 2424 | 3566 |
| ratio, best | 1.39 | **1.89** | **2.41** | **2.30** | **1.99** |

**Re-measured Oct 11 with the desktop idle** (Night run section): 1.50 / 1.88 / 2.41 / 2.27 / 1.97x at B = 2-32.

B = 2: 930 tok/s against llama.cpp's 614 (1.51x). B = 1 through this path: 468 tok/s; the
single-sequence decode (509) is still better there.

**v2 GEMVs** (`pack_q4_mma`, `gemv_v2_source`). Three changes to the tensor-core kernel:
1. Weights are repacked once into fragment order: per 16 x 64 tile each lane loads one uint4
   (a warp reads 512 contiguous bytes), and `(w >> 4j) & 0x000F000F` is A register j as a nibble
   pair. v1 had all 4 lanes of a row group load the same 16 bytes and pick their nibbles.
   Scales are 8 bytes per row group, read as a broadcast; total bytes equal Q4_0.
2. Each block stages its input chunk in shared memory once (bf16 -> fp16), and warps read B
   fragments with `ldmatrix.x4`. v1 re-read and re-converted the whole input per 16-row tile.
   At B = 32 that was 7x the weight bytes in cache traffic.
3. For short-N shapes (down: 96 row tiles on 68 SMs), K is also split across KS blocks. Partials
   go to a workspace, and the last block per tile (self-resetting counter) sums them in a fixed
   order. The result stays deterministic.

| us per launch, best (WM, WK, MT, KS) | B = 8 | 16 | 32 |
|---|---|---|---|
| qkv 2048 x 1536 | 7.8 (1,4,1,4) | 8.9 (2,4,1,2) | 14.0 (2,4,1,2) |
| o 1536 x 1536 | 7.1 (1,8,1,2) | 8.7 (2,4,1,2) | 13.3 (2,4,1,1) |
| gate_up 17920 x 1536 | 28.6 (4,2,2,1) | 35.4 (2,4,1,1) | 55.3 (2,4,1,1) |
| down 1536 x 8960 | 20.3 (1,4,1,4) | 26.0 (1,4,1,4) | 41.6 (2,2,1,4) |
| sum per layer, v2 (v1 tuned) | 64 (81.5) | 79 (124) | 124 (184) |

**What did not help after v2** (gate_up at B = 32 is 55 us, the fp32-accumulate tensor-core floor is 30 us):

| hypothesis | test | result |
|---|---|---|
| tensor-core rate binds (fp32 accumulate is half rate on GA102) | `ACC16`: fp16 accumulation per 64-k chunk, flushed to fp32 | 53.5-57.9 us, no gain: not MMA-bound |
| weight-load latency binds (constant weights: 29.6 -> 18.3 us at B = 8) | register prefetch 2-3 iterations ahead (`PF`) | slower (gu B = 32 69-84 us), more registers and partial groups |
| input staging binds (skipping it: 58 -> 39 us at B = 32) | `gemv_v3_source`: producers write fp16, X and weights via `cp.async` through a 2-4 stage shared-memory ring, one barrier per iteration | gu 54.5 us, qkv/o 1-2 us faster, down slower: the ring's shared memory (up to 47 KB per block) cuts occupancy; not integrated |

Dequantization is not a cost (replacing it: 58.1 -> 54.1 us). Removing the MMAs is not a valid
ablation: the compiler then deletes the dequantization and weight loads that only fed them.

With v2, every B (32 included) matches the single-sequence decode for 73-74 of 128 tokens before
diverging at a near-tie. `--v1` selects the old kernels.

**LM head** (`head_q6_mma_source`, `scripts/head_bench.py`). The fp16-dequantized head reads 467 MB
per step; Q6_K is 191 MB. The tensor-core kernel keeps each sub-block's raw q - 32 in the A
fragment (exact in fp16), multiplies into a zeroed accumulator, and applies D x SC in fp32 per
row. Its logit error against fp32 is 3e-7 to 6e-7 of the largest logit, against 4e-4 for the fp16
head (which got one argmax of 8 wrong on random inputs). Speed, real head 151936 x 1536:

| us | B = 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Q6_K tensor cores, best (S, U, MT) | 342 (6,4,1) | 385 (6,4,1) | 434 (6,4,1) | 516 (8,2,2) | 1106 (8,2,2) |
| cuBLAS on fp16 head | 661 | 677 | 683 | 690 | 704 |

**v2 head** (`pack_q6_mma`, `head_v2_source`): the v2 GEMV structure on Q6_K repacked into fragment
order (low nibbles as Q4_0, high 2-bit fields in the same slots with two k-steps per word, int8
sub-block scales and fp16 super-block scales broadcast per row group), with the exact fp32 scaling
above. With (WM, WK, MT) = (8, 1, 1): 286 / 290 / 292 / 311 / 368 us at B = 1 / 4 / 8 / 16 / 32,
671 GB/s at B = 1, against cuBLAS's 660-705 us on the fp16 head. Logit error 4e-7 to 6e-7.

At B = 32 the v1 head kernel is input-bound: with 16 rows per tile it re-reads and re-converts the whole
[32, 1536] input for each of 9496 tiles (about 0.9 GB from cache), so B = 32 keeps cuBLAS.

At B = 1 the single-vector path (509 tok/s) is the better choice. At B = 32 the step is 6.4 ms,
of which the GEMVs are about 5.2 ms (28 layers x 184 us); they still run far below tensor-core
peak, since per n-tile they convert bf16 inputs to fp16 and decode weights.

### Prefill (`scripts/qwen_prefill.py`, `scripts/prefill_gemm_bench.py`)

One sequence, P prompt tokens per pass. Per layer: RMSNorm, QKV, a K/V-write pass for all chunk
tokens, causal attention, O, RMSNorm, gate_up, down. Correctness: the KV cache and last-token
logits against token-by-token prefill through the validated decode step; perplexity on README text.

| prefill tok/s | P = 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| v2 GEMVs, chunks of 32, per-token attention | | | 7,171 | | |
| cuBLAS on per-layer fp16-dequantized weights, one chunk (chunks of 512 beyond) | 7,369 | | 9,342 | 10,291 | 10,924 |
| int8 GEMM, per-token attention | | | 14,606 | | |
| int8 GEMM (register weights) + tensor-core flash attention | 16,261 | 19,636 | 22,056 | | |
| **int8 GEMM (multistage) + flash attention, chunks of 512 beyond** | **17,290** | **20,253** | **21,424** | **20,910** | **19,277** |
| llama.cpp b11485 (`llama-bench -fa 1`, about +-1.5K) | 10,225 | 14,379 | 17,727 | 16,756 | 15,946 |
| ratio | **1.69** | **1.41** | **1.21** | **1.25** | **1.21** |

Prompts longer than 512 run in chunks of 512 against a `maxlen`-sized KV cache (`maxlen=` argument,
default max(512, P)); the last-token argmax matches the token-by-token reference at 1024 and 2048.
llama.cpp's pp512 measured 15,910 +- 1,718 in the later session, so the 512 ratio is between 1.21 and 1.35.
**Re-measured Oct 11 with the desktop idle** (Night run section): 1.64 / 1.50 / 1.40 / 1.35 / 1.33x at P = 128-2048; with the new down tile 1.41x at 512 and 1.36x at 2048; with v3 attention too **1.44x and 1.39x**; with the fused quantize passes (Oct 12) **1.68x and 1.59x**.

Where the 55 ms of the cuBLAS path went at P = 512 (ablation): matmuls 27 ms (at the fp16 x fp16 ->
fp32 tensor-core peak), attention 13 ms, dequantization 5 ms, torch epilogues and the rest 10 ms.

- **int8 GEMM** (`pack_q4_i8`, `QUANT_Q8`, `gemm_q4i8_ms_source`; `gemm_q4i8_source` is the first version), llama.cpp MMQ-style: activations
  quantized to int8 per 32 values (scale, sum), `mma.m16n8k32.u8.s8` on the raw Q4_0 nibbles, then
  y += dw * dx * (dot - 8 sum) in fp32 per 32-k block. The MMA's C operand carries
  0x4B400000 - 8 sum, so the int32 result is already the float bit pattern of 2^23 + 2^22 + (dot - 8 sum):
  no I2F, no integer correction. Bias, residual and SwiGLU are fused into the epilogue, and no
  dequantized weights are written. It runs at 55-80 TOPS of the measured 271 (see below for what
  bounds it). Tiles per GEMM: 128 rows x 64 tokens, 8 warps, 2 blocks per SM for qkv and gate_up;
  64 x 64, 4 warps, 3 blocks per SM for o and down (N = 1536 leaves too few 128-row blocks).
  Per layer at P = 512: 865 us against cuBLAS's 948 us plus dequantization.
- **Flash attention** (`flash_prefill_source`): one block per head and 64 tokens; Q (RoPE, bf16)
  is held in A fragments, K/V tiles of 64 positions are shared by the 4 warps through shared memory,
  S = Q K^T and O += P V on bf16 tensor cores (`ldmatrix`, `ldmatrix.trans` for V), online softmax
  with the causal mask. The per-token kernel re-read the whole prefix per (head, token) block,
  about 22 GB of cache traffic per 512-token prefill.
- **Quality**: on 512 tokens of this README, perplexity is 26.910 with fp16 weights and fp32
  accumulation (cuBLAS path) and 26.940 with int8 activations (+0.1%). Last-token argmax matches
  the token-by-token reference at every P tested.

Tested hypotheses on the int8 GEMM: quarter-rate I2F as the bottleneck (magic-number conversion:
no change); register pressure (forced 2 blocks per SM at 128 registers: no change); smaller
blocks with more blocks per SM: +15%.

What bounds the int8 GEMM (`prefill_gemm_bench.py`, P = 512, per layer):

| change | per layer (us) | gate_up (us) |
|---|---|---|
| first version (register weights, 64 x 64 tiles) | 865 | 448 |
| issue diet: 31 -> 19 instructions per MMA (ldmatrix B, one 128-bit C-operand load) | 861 | 448 |
| token tiles fastest in the grid (blocks sharing weights run together) | 837-853 | 431-435 |
| multistage: weights through shared memory, 3-stage cp.async ring | 813-830 | 399-405 |
| ablation: no MMA (integer op on the same operands) | 817 | 400 |
| ablation: no per-block fp32 scaling (one add per output) | 779 | 383 |
| 128-row tiles, 8 warps, 2 blocks per SM (qkv, gate_up) | 763 | 357 |

- **The prefetch had distance zero.** The first version loaded the next stage's weight fragments
  into registers one stage ahead. Global loads complete out of order, so they get no counted wait:
  ptxas put both stages' loads on one scoreboard (SB5) and the first wait for this stage's weights,
  about 20 instructions later, also waited for the prefetch. A fake data dependency
  (`wq & (K >> 30)`, zero but opaque) moved the loads after the wait, but at the 3-blocks-per-SM
  register cap (168) two live stages of weights do not fit: ptxas then sank the loads to the end of
  the stage or spilled them. A branch merge point also makes ptxas wait on every outstanding load.
  The fix is the CUTLASS one: weights go global -> shared with cp.async (counted groups,
  `wait_group NSTG - 2`), which also frees the weight registers (168 -> 151).
- **Not compute.** Replacing every MMA with an integer op leaves the time unchanged, and removing
  the fp32 scaling saves 4%. The loop is bound by moving X and W: with 64-row tiles the 1.1 MB
  of int8 X and scales is re-read by each of the 280 gate_up row-blocks (about 430 MB from L2 per
  call, ~1.1 TB/s). 128-row tiles cut that to 275 MB but gained 11%, so traffic is not the whole
  bound either; every 16-warp-per-SM configuration beat every 8-warp one.
- **The benchmark gains did not reach the full model.** End to end the multistage kernel prefills
  512 tokens at 21.4-21.7K tok/s, level with the committed first version (22.1K); the 128-row tiles
  changed nothing end to end. The first version after the issue diet and grid swap ran 3.8-6.4K
  tok/s end to end (80-135 ms at P = 512, varying run to run) while the isolated benchmark timed it
  steady at ~850 us per layer: unexplained, and the reason the multistage kernel is the default.
  **Update (Oct 10, later):** most of this is the desktop. The 3080 also drives the display, and
  with no job queued it shows 38% utilization (browsers, Teams, Photos, a computer-use agent);
  pcslurm serializes CUDA jobs, not graphics contexts. The same build then measured 5.5-25K tok/s
  run to run. `qwen_prefill.py` now warms for 0.5 s and reports the minimum of 21 graph replays
  (contention only adds time) with the median beside it: the default build prefills 512 tokens in
  20.6-21.4 ms (24.0-24.9K tok/s; medians 23.7-29.0 ms) and 2048 in 110-116 ms (17.6-18.6K).
  The earlier "erratic" kernel results above were measured before this fix.

### Occupancy cost model for the multistage GEMM (`lab/gemm_model.py`, Oct 10)

`lab/model.py` simulates the warps of one block running a straight loop. The int8 GEMM's tile choice
trades things that model does not have: several blocks per SM (registers and shared memory decide
how many), a block-wide barrier every stage, tensor-core and shared-memory ports, and the cp.async
copy path. `gemm_model.py` simulates a whole SM on the kernel's real SASS ring loop with all of
those, per wave of blocks, and adds the epilogue. Six new microbenchmarks supply the constants:

| microbenchmark | finding |
|---|---|
| `smem_width` | shared memory is 128 B/cycle for LDS.64, LDS.128 and LDSM.x4, 64 B/cycle for LDS.32 (one request per 2 cycles); 4 warps reach 70-75% of it, 8 warps ~96% |
| `l2_cpasync` | **cp.async costs one SM cycle per active lane, not per byte**: half the lanes take half the time, 8 B and 4 B copies take as long as 16 B. At 16 B per lane that is 2.1-2.2 TB/s GPU-wide, the familiar "L2 bandwidth", but it is an SM-side limit (one block per SM already reaches it). A lane alone in its 32 B sector costs 2 (16 B pieces one per line: 1.0 TB/s). Sharing is free: 8 or 34 blocks reading the same lines, or the same DRAM stream (misses merge in L2), all run at ~2.1 TB/s |
| `mem_epilogue` | fp32 read-modify-write of a 64 MB output: 562 GB/s (16 warps/SM) to 660; plain stores ~690 |
| `tc_tput_u8s8_s32`, `tc_lat_u8s8_s32` | the kernel's mixed-sign u8 x s8 MMA runs at the s8 x s8 rate (0.25 warp-MMAs per cycle per SM) |
| NVML during the GEMM | **sustained int8 GEMM is power-capped**: 1725-1785 MHz at ~365 W (throttle reason SW power cap), not the 1.95 GHz light kernels run at |

Ablations of the 64 x 64 tile (`ABL=noload` computes on whatever the ring holds, `nocompute` only
copies, `noloop` runs prologue and epilogue only) and slopes over K at a fixed grid (fixed costs
cancel) split each stage into its parts:

| cycles per stage (1 wave, slope over K) | 64 x 64, 3 blocks/SM | 128 x 64, 2 blocks/SM |
|---|---|---|
| compute only, measured / simulated | 1445 / 1382 | 4390 / 3611 |
| copies only, measured / simulated | 2047 / 1704 | 2545 / 2743 |
| full kernel, measured / simulated | 2330 / 1779 | 4910 / 3847 |

What this changes in the account above:
- **The benchmark's gate_up time is ~40% epilogue.** `prefill_gemm_bench.py` uses the residual
  epilogue for every GEMM: 73 MB of fp32 read-modify-write for gate_up, 160-180 us of the
  ~360-400 (`noloop`). The full model's gate_up writes 1 B per output (SwiGLU), so the benchmark
  overstates it there.
- **"Bound by moving X and W" is the cp.async lane rate.** Copies alone run each stage in 2047
  cycles for the 64 x 64 tile; the loop overlaps compute and copies well (full = copies + 14%).
  Per output, the 128 x 64 tile needs 40% fewer copy lanes (X is re-read by half as many row
  blocks), which is why it wins on gate_up.
- **Two simulator bugs found on the way, both worth knowing.** The loop to simulate must be found by
  its DEPBAR (in the copy-only build the epilogue loop is longer). Greedy-then-oldest scheduling lets
  the lowest block race ahead and finish, after which the others run alone: steady state must be
  measured while every block is still running, not extrapolated from the last stages (that read
  2.9x too fast for 2 blocks of 8 warps).

**Accuracy.** In sample (the 11 tiles above, 44 GEMM timings used for diagnosis, no constant fitted
to them): median error 14.9%, 32/44 within 20%, and it picks the measured-best tile for gate_up, o
and down (qkv: 4% off the best). **Held out** (`scripts/gemm_model_sweep.py` ranked 67 compiled
tiles; 11 never-run tiles chosen across the predicted range, predictions written to
`data/gemm_model_heldout_preds.json` before the GPU ran them): median error 12.7%, 35/44 within
20%, but rank correlation only 0.5-0.55 (0.83 for down). The misses are one systematic bias:
**tiles with 8 warps per SM run ~18% slower than predicted, 16-warp tiles ~5%**. Its top pick
(128 x 64 with 4 warps, 255 registers) measured 716 us per layer against the 641 of the best
held-out tile (128 x 32, 16 warps), so the model still under-charges low occupancy, the very
effect ("16 warps beat 8") it was built to explain. Its best find, a 128 x 32 / 16-warp down tile
(predicted 172.1 us, measured 172.4 vs 192 for the default), did not resolve end to end under
desktop contention (two of three runs each way) and is not the default. **Update (Oct 11):** with
the desktop idle it won all 8 interleaved end-to-end pairs (+2.4% at P = 512, +2.6% at 2048) and is
now the default; see the next section.

### Occupancy sweep: the bias was the copy path (`scripts/occupancy_sweep.py`, Oct 11)

To separate occupancy from everything else that differs between tiles, the sweep runs the SAME
compiled kernel at 1, 2, 3 blocks per SM: occupancy is lowered by launching with unused dynamic
shared memory, the grid is one wave of exactly b blocks per SM, and per-stage cycles are the slope
over K times the SM clock read by NVML during the timing (the clock moves with occupancy: 1935 MHz
copying only, 1655-1890 with the tensor cores busy). Six tiles x {full, compute only, copies only}
x every reachable occupancy, 41 points (`data/occupancy_sweep.json`), against
`scripts/occupancy_model.py`:

- **Compute only is right at one block** (measured / simulated 0.98-1.15) and 11-20% slow from two
  blocks up (one exception: the 64 x 64 tile at 2 blocks, 0.97), for 4- and 8-warp blocks alike:
  an inter-block cost the simulator does not have,
  not a latency-hiding one.
- **Copies only are where the model was wrong**: the GEMM's copies run at ~0.7 lanes per cycle per
  SM (1.34-1.65 cycles per lane at 2-3 blocks), not the 1.0 that `l2_cpasync` measured with a
  plain copy loop. A per-SM limit on copy lanes in flight (Little's law: in flight / latency) fits
  this better than a slower port: with at most 500 lanes in flight and the 570-cycle latency,
  multi-block copy-only points land at 0.88-1.13 and full kernels at 0.99-1.15
  (`lab/gemm_model.py: CP_INFLIGHT`). Still wrong: one block with a 3-stage ring runs 25-39%
  slower than simulated (one block and its barrier per stage cannot keep copies and math going
  at once); one block with a 2-stage ring copies faster than the cap allows.

Scored on all 88 GEMM timings (11 in-sample tiles, the 11 held-out ones; `scripts/gemm_model_variants.py`):

| copy model | median error, in sample | held out | within 20% | Spearman qkv / o / gu / down |
|---|---|---|---|---|
| v1: 1 lane / cycle, no cap | 15.5% | 10.9% | 60/88 | 0.66 / 0.58 / 0.49 / 0.68 |
| 0.75 lanes / cycle | 12.3% | 6.8% | 66/88 | 0.74 / 0.32 / 0.48 / 0.59 |
| **cap 500 lanes in flight, latency 570** | **9.8%** | **6.4%** | **71/88** | 0.74 / 0.42 / 0.47 / 0.71 |

The variant was chosen on these same 88 points, so the "held out" column is no longer a blind test
(the sweep, not the 88, suggested the mechanism). Absolute error drops by a third; ranking does
not improve, because the remaining errors (the inter-block compute cost, one-block rings) are as
large as the differences between the good tiles. The model is now a fair estimator of a tile's
time and still not a reliable judge between two tiles within ~10% of each other.

### Night run, Oct 11: idle desktop, everything re-measured (`data/night_oct11/`, `scripts/night_summary.py`)

GPU at start: 1% utilization, 692 MiB used, no compute processes (desktop apps left open but idle).
Three interleaved rounds of ours and llama.cpp b11485 on the same GGUF; best of everything on both
sides (llama.cpp: best of 15 `llama-bench` repetitions per round for prefill, 10 for decode, best
`llama-batched-bench` round; ours: minimum of 21 graph replays per run, best run). Our runs agree to
within 1% round to round; llama.cpp's best round mean is within 2-4% of its best sample.

| | ours | llama.cpp | ratio |
|---|---:|---:|---:|
| decode B = 1 (`qwen_gguf.py` vs `llama-bench -n 256`, best of -fa 0 / 1) | 509.6 | 368.7 | **1.38x** |
| batched decode B = 2 / 4 / 8 / 16 / 32 (`qwen_batched_decode.py` vs `llama-batched-bench` S_TG) | 943 / 1851 / 3467 / 5674 / 7216 | 630 / 982 / 1437 / 2499 / 3664 | **1.50 / 1.88 / 2.41 / 2.27 / 1.97x** |
| prefill P = 128 / 256 / 512 / 1024 / 2048 (`qwen_prefill.py modes=i8` vs `llama-bench -fa 1`) | 17,949 / 22,794 / 25,868 / 25,063 / 23,515 | 10,921 / 15,173 / 18,528 / 18,503 / 17,637 | **1.64 / 1.50 / 1.40 / 1.35 / 1.33x** |

llama.cpp decodes faster on an idle desktop too (368.7 vs the 340.5 measured in September), so the
B = 1 decode lead is 1.38x, not 1.50x. Prefill gained more on our side (25.9K at 512 against the
21-25K measured under contention), so its lead grows. Prefill above is with the old down tile;
the new default adds 2.4-2.6% (next).

**Down tile A/B** (P = 512 and 2048, 4 rounds, default and candidate interleaved, min of 21 each):
128 x 32 with 16 warps per SM (`tile_down=4,2,2,2,4,2,1`) won all 8 pairs, 19.65-19.67 ms vs
20.07-20.19 at 512 and 85.31-85.47 vs 86.85-87.66 at 2048, with the same K/V and logit errors and the
same argmax. It is the default now: **26,032 tok/s at P = 512, 23,981 at 2048** (1.41x and 1.36x).

**Where long prompts lose** (ablations: `skip=fa`, `skip=kvw`, `skip=gemm`, round 2):

| P | total | attention | K/V write | everything else per token |
|---|---:|---:|---:|---:|
| 512 | 19.67 ms | 1.22 ms (6%) | 0.32 ms | 36.0 us |
| 1024 | 40.23 ms | 3.18 ms (8%) | 0.67 ms | 36.2 us |
| 2048 | 84.97 ms | 10.48 ms (12%) | 1.01 ms | 36.4 us |

(The GEMM column here includes the four int8 quantize passes, which `skip=gemm` also skips: see
"Fused small ops" below.) Without attention the cost per token is flat, so the drop from 26K to 24K tok/s is attention, whose
work grows with the context. At P = 2048 it already runs at 34 TFLOPS (361 GFLOP), about 55% of the
62 TFLOPS bf16-into-fp32 peak at the power-capped clock: at most ~4 ms (5%) to win.

**Prefill attention v3** (`flash_prefill_v3_source`, `scripts/flash_bench.py`). v1 runs one block of
4 warps per head and 64 tokens: 96 blocks for a 512-token chunk, so most SMs hold one warp per
partition, and the tensor core idles through every warp's softmax (one warp per partition saturates
it only while it issues MMAs back to back). v2 only overlapped the K/V loads (K and V as separate
cp.async groups) and scheduled the longest causal blocks first: 3.5-7% per chunk. v3 gives each block
8 warps in two groups that own the same 16-row query slices and split the keys (group 0 takes key
tiles 0, 64, 128, ..., group 1 tiles 32, 96, ...; own K/V buffers, named barriers per group); group 1
hands its (o, m, l) to group 0 through shared memory at the end. Two warps per partition, no extra
global traffic, no combine kernel, 166 registers instead of 226. One trap: masked scores must be
-inf, not v1's -1e30, because a group can meet a tile entirely in a row's future before it has seen
any key for that row (exp2(-1e30 - (-1e30)) = 1 would count it).

| chunk of 512 at position | v1 | v2 | v3 | v3 TFLOPS |
|---|---:|---:|---:|---:|
| 0 | 42.5 us | 40.7 | **27.9** | 28.9 |
| 512 | 81.5 | 75.9 | **64.3** | 37.6 |
| 1024 | 120.6 | 114.4 | **101.4** | 39.7 |
| 1536 | 161.0 | 152.7 | **138.5** | 40.7 |

Same error against an fp32 reference as v1 (1.5-3.1e-3). End to end (4 rounds interleaved with v1,
`fa=1` selects it): all 8 pairs won, 19.21-19.37 ms vs 19.60-19.68 at P = 512 and 83.31-83.62 vs
85.05-85.40 at 2048, same K/V errors and argmax. Perplexity on README text moves by +0.01 to +0.10%
(i8 at 2048: 23.591 to 23.613; the P values are rounded to bf16 against each group's own running
max), the size of the int8-vs-fp16 difference itself. It is the default.

**Prefill now: 26,647 tok/s at P = 512 (1.44x llama.cpp's best 18,528) and 24,584 at 2048 (1.39x
its 17,637).** Attention at P = 2048 is ~8.6 ms of 83.3; at 41 TFLOPS against the ~62 peak the most
left there is ~3 ms.

### Fused small ops: the int8 quantize passes were 12% of prefill (Oct 12, `data/fuse_oct12/`)

The `skip=gemm` ablation above also skipped the quantize kernels (they run inside `gemm8`), so it hid
them. Timing every kernel of one layer alone in a CUDA graph (`scripts/prefill_kernel_profile.py`,
P = 512, x 28 layers): the four QUANT_Q8 passes that turn each GEMM's bf16 input into per-32 int8
took **2.44 ms of 19.7 (12%)**, the one before `down` (512 x 8960 values) 1.52 ms by itself, and the
K/V-write pass another 0.43 ms. Each of them is now done by the kernel that produces its input
(`fuse=` flag, default all four, `fuse=0` the old path):

| fusion | replaces | how |
|---|---|---|
| `rms` | RMSNorm, bf16 x, QUANT_Q8 | `rmsnorm_q8_source`: H / 8 threads per token, 8 values each, so a 32-value block is one quad; 5.9-7.9 us vs 5.8 + 10-12 |
| `swiglu` | bf16 xm (P x 8960), QUANT_Q8 | gate_up epilogue `swiglu_q8`: 8 lanes per (token, 32 outputs), writes `down`'s int8 input to a second buffer set; gate_up 310.8 us vs 314.2 |
| `attn` | bf16 attention output, QUANT_Q8 | attention v3 `q8=True`: each 32-dim block of a row already sits in one quad; 32.4 us vs 27.3 + 10.4 |
| `kv` | K/V-write pass (RoPE on K, cache store) | QKV epilogue `biasf_kv`: a block is one 128-row head, so K's RoPE partner row d +- 64 is in the same tile; 41.5 us vs 39.8 + 15.3 |

Two of these were slow on the first try, both from the epilogue's latency in a GEMM that runs in
about one wave: one warp per 32 SwiGLU outputs (5-step shuffles, 16 rounds per warp) cost gate_up
+31 us, and the K-head blocks of the QKV GEMM re-loaded bias and cos/sin after every store (they may
alias), +9.5 us. Four outputs per lane and hoisted, batched loads fixed both.

In-process A/B (`scripts/prefill_ab.py`: both configs captured in one process, graphs replayed round
robin in shuffled order, so desktop load hits both alike; it was not idle):

| P | `fuse=0` min / p10 | fused min / p10 | p10 change | fused tok/s (min) |
|---|---:|---:|---:|---:|
| 512 | 19.25 / 19.43 ms | 16.43 / 16.56 ms | **-14.8%** | **31,164** |
| 2048 | 84.69 / 85.25 ms | 73.21 / 73.96 ms | **-13.2%** | **27,973** |

The `swiglu`, `attn` and `kv` fusions compute the same bits as the passes they replace (K/V cache and
hidden state identical to `fuse=0`); the new RMSNorm adds the sum of squares in another order, so the
scale can differ in the last bit. Against the token-by-token reference: same K/V errors (6.3e-3,
7.3e-2), last-token logits 2.2e-2 vs 1.9e-2, same argmax. Perplexity on README text (i8):
26.955 vs 26.968 at P = 512, 23.596 vs 23.613 at 2048 (fp16 cuBLAS 26.920 / 23.594).

**Prefill now: 31,164 tok/s at P = 512 (1.68x llama.cpp's idle-desktop best 18,528) and 27,973 at
2048 (1.59x its 17,637).** The GEMMs are now 92% of the kernel time at P = 512 (gate_up alone 51%);
what is left outside them is attention and two 6-8 us norms per layer.

Short prompts too (in-process A/B, one chunk): **-11.6% at P = 128 (21,842 tok/s), -13.4% at 256
(27,066)**; the last-token argmax matches the reference at 128, 256 and 1024. (Separate-process runs
had shown the fused build 2x slower at 128 and 256: desktop noise, which is why `prefill_ab.py` exists.)

**Is the prefill GEMM issue-bound? No.** The gate_up main loop issues 609 instructions per 32 IMMA
(19 issue cycles per IMMA against 16 tensor cycles; 384 of them are the per-block fp32 rescale) but
runs at ~42 cycles per IMMA per partition. The model's copy count says why: per stage an SM copies
2 x 1,280 16-byte lanes (X 704, weights 576 per block) for 512 IMMA, ~3,660 cycles at the in-place
0.7 lanes / cycle against ~2,050 tensor cycles. Pre-expanding weights to int8 would cut ~11% of the
instructions and double the weight lanes, so it is not worth building. Would plain loads beat
cp.async for X? `scripts/ldg_vs_cpasync.py` (same L2-resident addresses, 1-3 blocks per SM):
`ld.global.cg.v4` moves **1.0 lanes / cycle / SM** with or without the `st.shared`, flat with
occupancy; cp.async 1.3-2.1 (noisy, above `l2_cpasync`'s 1.0). The L2 -> SM path is not faster
through registers, so the lever left is fewer copied bytes per IMMA (bigger tiles), which the tile
sweeps already searched.

### Decode B = 1: where the 1.95 ms goes (Oct 12, `scripts/decode_kernel_profile.py`, `scripts/decode_ab.py`)

Each role's 28 launches (distinct weights, so DRAM-cold as in a real step) captured and timed
alone, position 128:

| role | per step | per launch | weight bytes | effective GB/s |
|---|---:|---:|---:|---:|
| gate_up GEMV | 680 us (35%) | 24.3 us | 15.5 MB | 638 |
| down GEMV | 395 us (20%) | 14.1 us | 7.7 MB | 549 |
| LM head (Q6_K) | 289 us (15%) | | 191 MB | 662 |
| attention | 155 us (8%) | 5.5 us | | |
| qkv GEMV | 136 us (7%) | 4.8 us | 1.8 MB | 365 |
| o GEMV | 115 us (6%) | 4.1 us | 1.3 MB | 324 |
| 2 RMSNorms | 105 us (5%) | 1.8-1.9 us | | |
| argmax (FINISH, one block), embed | 32 us | | | |

928 MB of weights per token in 1.95 ms is 63% of the ~760 GB/s peak. In-process A/B (300 rounds,
shuffled): a 64-block argmax with a last-block combine (`FINISH_MB`) is -0.3 to -0.4% (identical
logits; now the default, `--fin1` for the old one); more split-K warps lose everywhere (down, o, qkv
x4: +1.4 to +1.9%; gate_up x4: +19.7%; down x1: +0.3%), so the current splits stand. What is left
outside the big GEMVs (norms, attention, small-GEMV tails, ~0.5 ms) is the per-kernel fixed cost that
round 3 already attacked: folding the norm into the GEMVs, L2 prefetch and a persistent megakernel
all measured slower.

### Two negative results, Oct 15: a decode megakernel and bigger prefill tiles (`data/oct15_tiles_mega/`)

**Decode megakernel** (`lab/qwen_mega.py`, `scripts/qwen_mega.py`). One persistent kernel for the whole
decode step: 68 blocks, one per SM, each with a producer warp that streams that block's slice of the
weights (703 MiB in total, balanced to within 0.1% per block) through a shared-memory ring with
`cp.async` and mbarriers, while consumer warps do the int4 GEMV units, attention and the norms, with
a grid barrier between ops. The aim was to remove the gaps between the ~200 kernels of the CUDA
graph. It compiles to 148 registers with no spills and runs without timeouts, but it is not correct
yet (it picks token 1 where the reference picks 89141), and even so it is slower than the graph:

| pos = 128, 200 rounds | p10 (us) | tok/s |
|---|---:|---:|
| CUDA graph, 200 kernels (current) | 1,910.8 | 523.3 |
| megakernel | 2,238.5 | 446.7 (17% slower) |

Since it is 17% behind before the correctness bug is found, it is parked. Two ring bugs found on the
way, for anyone reusing the pattern: a consumer warp must not release a chunk it skipped before that
chunk is full (its empty-barrier arrive lands in the slot's previous phase and the producer
overwrites live data), and it must wait and release one chunk at a time (waiting for all skipped
chunks first deadlocks once a skip is longer than the ring).

**Bigger prefill tiles** (`scripts/prefill_ab.py P=512`). Tiles whose shared memory goes over 48 KB
now use dynamic shared memory, which allowed 256-row and 128 x 128 GEMM tiles to cut L2-to-SM copies.
All outputs are bit-identical to the default, and all are slower:

| config | min (ms) | tok/s | vs default |
|---|---:|---:|---:|
| default | 16.625 | 30,798 | |
| `tile_qkv=4,4,2,4,2,3,1` | 16.920 | 30,261 | -1.7% |
| `tile_gu=8,2,2,4,4,3,1` | 17.837 | 28,704 | -6.8% |
| `tile_gu=8,2,2,4,4,2,1` | 17.937 | 28,544 | -7.3% |
| `tile_gu=8,2,2,4,2,4,1` | 19.312 | 26,513 | -13.9% |
| `tile_gu=8,2,2,4,2,3,1` | 19.317 | 26,506 | -13.9% |
| `tile_gu=4,4,2,4,2,3,1` | 19.489 | 26,272 | -14.7% |

The defaults stay. Neither result changes the numbers in the results table.

### Night run, Oct 13: everything re-measured after the Oct 12 fusions (`data/night_oct13/`)

Same script and settings as the Oct 11 night run (`scripts/night_summary.py dir=data/night_oct13`).
It queued behind other GPU jobs and ran in the afternoon of Oct 14 (17:11-17:29), not overnight.
GPU at start: 13% utilization, 2,213 MiB used, SM clock 210 MHz (idle P-state), only desktop apps.
Something did load the machine part way through: our prefill was 25-40% slower in rounds 2 and 3
than in round 1, while round 1 matches the in-process A/B of Oct 12 to within 1.3%. The best-of rule
keeps round 1 for both sides; llama.cpp's numbers are within 1% of Oct 11 either way.

| | ours | llama.cpp | ratio |
|---|---:|---:|---:|
| decode B = 1 (`qwen_gguf.py` vs `llama-bench -n 256`, best of -fa 0 / 1) | 514.0 | 365.8 | **1.41x** |
| batched decode B = 2 / 4 / 8 / 16 / 32 (`qwen_batched_decode.py` vs `llama-batched-bench` S_TG) | 950 / 1792 / 3395 / 5578 / 7135 | 625 / 972 / 1416 / 2489 / 3597 | **1.52 / 1.84 / 2.40 / 2.24 / 1.98x** |
| prefill P = 128 / 256 / 512 / 1024 / 2048 (`qwen_prefill.py modes=i8` vs `llama-bench -fa 1`) | 21,888 / 26,755 / 30,767 / 29,840 / 28,114 | 10,937 / 15,141 / 18,502 / 18,607 / 17,639 | **2.00 / 1.77 / 1.66 / 1.60 / 1.59x** |

Prefill is up 19-22% on our side since Oct 11 (the new down tile, attention v3 and the fused quantize
passes), so the lead at P = 128 is now 2x. Decode and batched decode did not change on our side (the
Oct 12 decode changes were worth 0.3-0.4%) and are within 1-2% of Oct 11 on both.

### Phase 8: our own block scheduler (`lab/resched.py`, `lab/verify.py`, `pair_latency`, `mufu_latency`, `resched_rand`)

Every basic block is list-scheduled from a dependence DAG on top of ptxas's register allocation,
and every control word is regenerated: stalls, wait masks, write barriers and reuse flags.

**Correctness.** Tested on 40 seeded random kernels (FFMA, IMAD, SHF, MUFU, SHFL; 1 to 32 warps).
Each kernel was run with three schedules:
- ptxas order with my control words (identity);
- my critical-path order (crit);
- a random legal order, as a stress test.

All 120 rescheduled kernels are bitwise identical to ptxas, stable over 5 full-length launches
each. The independent hazard checker (`lab/verify.py`) walks the code in execution order with
the loop unrolled 3x and finds no hazard that ptxas's own code does not also have. Getting
there required these hardware rules, each found from a wrong answer on the GPU:

| rule | evidence |
|---|---|
| forwarding latency 4 cycles within a pipe, 5 across the FMA/ALU pipes, 4 into SHFL, 4 FFMA/FADD -> MUFU | `pair_latency`, `mufu_latency` (stall patched below the minimum gives wrong values) |
| FMA pipe = FFMA, FADD, FMUL, IMAD, IMAD.IADD; ALU pipe = IADD3, LOP3, SHF, FMNMX | same |
| a scoreboard arms one cycle after issue: an instruction waiting on the previous instruction's barrier needs stall >= 2, including across a block boundary | races in random orders |
| waits are sticky: ptxas protects a value from an earlier block by any earlier wait in the block, so moved code must wait at block entry | race in a pre-loop block |
| a wait on an op's read barrier resolves its operand reads, not its result | loop-carried SHFL race |
| an op with only a read barrier has a stall-timed result | MUFU with R set, W unset |
| a variable-latency op with no barriers (e.g. a dead MUFU) is protected only by distance: keep ptxas's distance to the next write of its register | deterministic wrong answer at 32 warps |
| encoding: with the yield bit set the stall must be 1..11; 12..15 need yield clear | nvdisasm rejects the opex value |

Ptxas itself sometimes gives MUFU no scoreboard at all and relies on a 9 to 25 cycle stall. Ptxas
also turns some `add.s32` into `IMAD.IADD` (FMA pipe), apparently to balance pipes.

**Speed vs ptxas** (median cycles, 40 kernels; geomean of ptxas/ours):

| schedule | geomean | worst | best |
|---|---|---|---|
| identity order, my control words | 0.964 | 0.846 | 1.023 |
| critical-path order | 0.932 | 0.642 | 1.040 |
| random legal order | 0.846 | 0.483 | 1.023 |

Critical-path order picks instructions by EXPECTED completion (MUFU 17, SHFL 26, LDG 498
cycles). The encoded stalls come from a replay that uses only the hard requirements, so the
waits do the blocking. Before this split the critical-path order was 0.85x, because it put
consumers right behind MUFU/SHFL producers and the single warp sat at the wait. Write barriers
are reallocated in the new order, except producers still in flight at ptxas's block exit.

Remaining gap to ptxas:
- reuse flags: I set them only between neighbours, while the cache keeps an entry until another
  read hits the same bank and slot (seed1014: ptxas 84 flags vs mine 52);
- a conservative 5-cycle latency for unmeasured ops;
- ptxas's register allocation was made for ptxas's order, so WAR edges limit how far I can
  move anything.

**On the int4 GEMV (R8U2, memory-bound) the scheduler loses.** Outputs are bitwise identical
throughout:

| shape | ptxas | resched | Phase 7 sassfix | sassfix + resched | lane_contig + resched |
|---|---|---|---|---|---|
| gate_up | 94.2 us | 99.3 | **62.5** | 74.8 | 61.4 (55.3 without) |
| down | 66.6 | 94.2 | 63.5 | 82.9 | 92.2 (59.4 without) |
| q_o | 15.4 | 20.5 | 15.4 | 19.5 | 20.5 (13.3 without) |

The targeted Phase 7 pass is still the right tool for these kernels. Why the general scheduler
loses on memory-bound code is open; suspects are the entry-wait placement and its single-warp
model of load latency.

### Oct 18-21: held-out tests, cross-kernel fusion, and why the scheduler stops at parity

Every test in this round was fixed before it was measured: the kernel suite, the outside benchmark
and the tile-ranking predictions were committed before any GPU timing.

**Held-out kernel suite** (`lab/suite.py`, `scripts/suite_run.py`, `data/suite/`). 20 textbook kernels
(saxpy, reduce, transpose, stencils, softmax, layernorm, scan, bitonic sort, sgemm, nbody, ...),
checked bitwise against ptxas and timed interleaved with an L2 flush. It exposed four hardware rules
the random kernels never hit; each broke at least one kernel under every policy, including ptxas's
own order, and `scripts/suite_bisect.py` found the block by rescheduling one block at a time on the GPU:

| rule | evidence |
|---|---|
| a fixed-latency producer -> SHFL needs 5 cycles, not 4 | wrong reductions in spmv_csr, layernorm |
| a predicate read by BRA/EXIT needs 13 cycles (ptxas floor over 240 pairs) | 12 broke spmv_csr |
| memory ops issue at least 4 cycles apart | a 1-cycle gap broke transpose |
| BAR.SYNC.DEFER_BLOCKING -> first memory op at least 6 cycles (66 barriers) | 2 broke layernorm |

`lab/verify.py` checks all four. With them, all 19 timed cases are bitwise identical under every
policy (black_scholes is excluded: ptxas's own build misses the CPU reference). Geomean vs ptxas:
crit 0.999, identity 0.999, sb 1.000 (`data/suite/imad.json`). Parity, not a win.

**Outside benchmark** (`scripts/sassopt.py`, `data/ext_polybench/`). 40 PolyBench/GPU kernels at a
pinned commit, rescheduled with a static hazard check, a multi-seed bitwise differential and a timed
keep/revert decision. 0 static failures, 0 bitwise failures. On the corrected scheduler, sb changes
28 kernels and is 0.992x raw, 1.000x after keep/revert; sb_wide is 1.003x raw, 1.008x after
keep/revert. An earlier 1.011x relied on timings the four rules above now forbid.

**Where the gap comes from** (`lab/experiments/randk.py`, `data/ablation/`). Same 40 random kernels as
above, geomean of ptxas/ours:

| variant | geomean |
|---|---|
| ptxas order, my control words (identity) | 0.959 |
| same, with ptxas's own reuse flags | 0.965 |
| same, no reuse flags | 0.939 |
| critical-path order | 0.920 |
| register renaming (`lab/regalloc.py`), then critical path | 0.917 |
| scoreboard-aware order (sb) | 0.980 |
| random legal order | 0.845 |

Reuse flags are worth about 0.5%. Renaming registers first does not free the scheduler. Giving
IMAD its own issue port removed most of the extra static stall cycles and changed nothing measured:
the hardware interlocks structural conflicts, so stall slack is not the loss. A control-word diff of
the worst kernel (0.85x in ptxas's own order) shows stalls and reuse matching ptxas almost
everywhere; the difference is where scoreboard waits sit. Each block is encoded on its own, so waits
on values from earlier blocks are hoisted to the block's first instruction, and barriers still in
flight are drained before block exit or a CALL. Ptxas waits only at the consumer.

**Wait placement fix (Oct 21-22, `late_entry` + `sticky_waits`, `*_late` variants).** Two changes:
- a wait on a value from an earlier block stays at ptxas's position, not the block's first instruction;
- a barrier already cleared by an earlier wait in the block is not waited on again.

The second was most of the loss: a repeated wait on a barrier that a new MUFU or SHFL had re-armed
also waited for that op. In ptxas order my wait words now match ptxas on 12,172 of 12,200
instructions (11,138 before), with 0 static hazards.

| random kernels (job 1882) | before | after |
|---|---|---|
| identity | 0.959 | **0.987** (worst 0.848 -> 0.920) |
| crit | 0.920 | **0.976** (worst 0.588 -> 0.823) |
| sb | 0.980 | 0.980 |

Suite (job 1993, `data/suite/late2.json`): all 19 bitwise ok under every variant; geomean
identity_late 0.996, crit_late 1.002, sb 0.996, sb_late 1.003, sb_wide 1.003, sb_wide_late 0.997.
That is still parity: run-to-run noise on these kernels is about 1% (ptxas's own softmax time moved
from 176 to 183 us between runs). No variant beats ptxas without the timed keep/revert yet.

The same runs found one more correctness rule. Ptxas leaves some LDS without a scoreboard and
covers the result with the next load's barrier, because shared loads return in order. sb_wide moved
such a load behind the next one (conv_row wrong output, bitonic_block illegal address, bisect in
`data/suite/bisect_wide_late.json`). Memory order is now kept around barrier-less loads, and their
readers wait on the covering load's barrier.

**Reuse and stall model (Oct 22, commits d7dae46 and 682a8d2).** Reuse flags now come from a model of
the operand reuse cache:
- one entry per (register bank, operand slot);
- only FMA/ALU-pipe instructions hit or evict, so MUFU, SHFL and memory ops do not;
- FADD's second operand uses the third slot;
- a cleared yield bit flushes the cache.

In ptxas order this reproduces ptxas's reuse field on 12,185 of 12,200 instructions. Two stall fixes
followed:
- FFMA/FADD/FMUL into MUFU is 4 cycles, as ptxas uses on all 91 such pairs.
- The exit rule counts the terminator's own stall, which removes 4 wasted cycles per loop iteration.

| random kernels (geomean vs ptxas) | job 1882 | + reuse model (2050) | + stall fixes (2054) |
|---|---|---|---|
| identity_late | 0.987 | 0.993 | **1.003** |
| crit_late | 0.976 | 0.982 | 0.989 |
| sb_late | 0.980 | 0.985 | 0.995 |

Suite (`data/suite/reuse.json`, `data/suite/exit.json`): every rescheduled variant is bitwise ok
(ptxas's own black_scholes baseline trips the reference check as in every run). Geomeans are 0.994 to
0.998, which is within noise. Copying ptxas's order is now at parity on random kernels. The
reordering policies are still about 1% behind it, so the order the scheduler picks is now the loss,
not the control words. Cycles: `data/ablation/job2050_reuse_cycles.txt`, `job2054_exit_cycles.txt`.

**Reordering has almost no headroom left (Oct 24, `scripts/sim_pick.py`).**
- **Where sb_late loses** (2 to 5% vs identity_late): high-occupancy kernels where ptxas already
  issues about 3 warp-instructions per cycle per SM. A latency-driven reorder cannot help there.
- **Where it changes nothing:** low-occupancy kernels.
- **Model pick:** I simulated identity_late, sb_late and crit_late in `lab/model.simulate` at the
  measured occupancy and kept the predicted-fastest per kernel. That measures **1.001**, against
  1.003 for identity_late alone. The simulator gets the sign right on 16 of the 24 seeds where the
  orders differ by more than 0.5%.
- **The ceiling:** a perfect oracle over the three orders reaches only **1.005**. ptxas's order is
  close to optimal on these kernels.

The wins that do exist (1.02 to 1.04 on low-occupancy seeds) come from control words. One example:
ptxas sometimes stalls 6 cycles on FFMA -> MUFU where 4 is enough. Beating ptxas by 1.05x needs
something block-local reordering cannot reach, such as register allocation or cross-block motion.

**Cross-kernel optimizer** (`lab/xkernel.py`, `data/xkernel/polybench.json`). Vertical (thread-local)
and horizontal (grid-concatenated) fusion of adjacent launches, plus CUDA graph capture, on the 12
multi-kernel PolyBench programs. All outputs bitwise identical to the original. Geomean of fused +
graph: **1.226x**.

| program | launches | fused + graph |
|---|---|---|
| jacobi1d | 20 | 3.92x (graph only) |
| mvt | 2 -> 1 | 1.36x (horizontal) |
| bicg | 2 -> 1 | 1.34x (horizontal) |
| 3dconv | 8 | 1.23x (graph only) |
| fdtd-2d | 30 -> 20 | 1.13x (vertical) |
| jacobi2d | 20 | 1.10x |
| lu | 16 | 1.04x |
| other 5 | | 1.00x |

**Blind tile ranking** (`data/rank_fresh/`, `data/rank_direction/`). Both prefill GEMM cost models
predicted 195 never-measured tiles before timing. Median error was 5-15%, but ranking was near
chance: pairwise accuracy 0.57 (copy-cap model) and 0.46 (queue model), Spearman -0.49 to 0.49 per
shape. The prediction error is larger than the spread among good tiles. Single-variable tests show
which directions the model gets right: occupancy alone 15/16 and k-chunk 29/30, but launch bounds
2-4 of 12 and ring depth 4-8 of 13. Not usable for picking tiles.

**L2 placement** (`data/l2_place/`). Address sets aimed at one inferred L2 slice are 3-12x slower on
every page of a fresh allocation, so the slice map partly transfers. Spreading a working set
evenly across slices does not reliably beat a contiguous layout (0.69-1.34x by page and K). Negative.

### FP32/INT32 sharing (superseded by Phase 8 above)

GA102 has 16 FP32 + 16 FP32/INT32 lanes per partition. If INT ops simply
borrowed the shared half, a 50/50 FFMA:SHF mix would reach 4 warp-instr/cyc/SM.
Measured: 2.65. Most mix points fit T = p*1 + (1-p)*2 cycles per warp-instr
per partition (p = FFMA fraction): **no overlap between FFMA and INT issue**,
as if a warp FFMA occupies both halves for one cycle and an INT op holds the
shared half for two while the other idles. f=1/8 and 2/8 beat the model.
ptxas does not keep the generated interleave (it front-loads FFMAs), but an
additive per-instruction cost model is order-independent, so ordering alone
does not explain it. Next: pin the order by swapping instructions in the cubin.

### Memory hierarchy (one thread, dependent 64-bit loads, 128 B stride)

| level | latency (cyc) | ~ns | capacity edge |
|---|---:|---:|---|
| L1 hit | 35 | 18 | flat to 96 KB, degrading 104-128 KB (0 B smem kernel) |
| L2 hit | 238 | 121 | flat to 4.5 MB, 353 at 5 MB, miss at 5.5 MB (5 MB L2) |
| DRAM | 498 | 254 | +9 cyc at 256 MB random (TLB), not for sequential |
| shared | 23 | 12 | |

`ld.global.cg` bypasses L1 (238 at 2 KB). Past L1 capacity, a sequential
sweep is worse than random (128 KB: 192 vs 144), the LRU-thrash signature;
replacement experiments next. Variance spikes at capacity edges (CV 5-18%).

### Toolchain behaviour

- ptxas is an optimizing assembler; `asm volatile` pins PTX, not SASS. At
  -O1+ `add` chains fuse into 3-input IADD3, `xor` chains with a repeated
  operand fold away, uniform `shfl.idx` chains are deleted. The validator
  refuses to measure in all three cases.
- Inline `ld.global.ca` -> `LDG.E.64.STRONG.SM`; `ld.global.cg` ->
  `STRONG.GPU`; a C deref of an int-derived pointer -> generic `LD.E.64`.
  All three hit L1 at 35 cycles.
- ptxas encodes fixed-latency ops (FFMA, FADD, FMUL, IMAD, SHF, HFMA2) with
  stall 4 and no scoreboard; DFMA, MUFU, SHFL use scoreboard barriers.

## Not yet available

- **Hardware counters** (memory/cache counters, warp-stall reasons, executed
  instruction counts) need Nsight Compute (`ncu`), which is not on pip. It
  ships with the CUDA toolkit installer (needs admin), plus "allow access to
  GPU performance counters to all users" in the NVIDIA Control Panel.
  Until then, the instruction count is static loop body x trip count.
- **Clock locking** (`--lock-clock 1710`) calls `nvidia-smi -lgc` and needs an
  elevated shell; without it, clocks are recorded rather than controlled.
