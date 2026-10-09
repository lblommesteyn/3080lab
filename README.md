# 3080lab

An empirical microarchitecture lab for one RTX 3080 (GA102, sm_86).
Goal: measure the machine well enough to predict kernel runtime from SASS,
then schedule better than ptxas, then make Qwen faster.

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
| Phase 7 latent bug | `hoist()` checked producer distance for registers, not guard predicates. The verifier caught a load placed 1 cycle after its 13-cycle ISETP; fixed |
| `-maxrregcount` | ptxas ignores it when the source has `__launch_bounds__`; capped builds use `__launch_bounds__(128, 8)` |

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

### FP32/INT32 sharing (superseded by the section above)

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
