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
