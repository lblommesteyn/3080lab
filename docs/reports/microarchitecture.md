# GA102 microarchitecture characterization (RTX 3080, sm_86)

Summary. Microbenchmarks on one RTX 3080 (boost about 1965 MHz, cycles from `clock64`) measured
instruction latency and throughput, the control-word semantics, register banks, the reuse cache,
the memory hierarchy and the tensor cores. Three results carry most of the later work. First, the
hardware does not interlock fixed-latency data hazards: patching a dependent FFMA chain's stall
below 4 cycles gives wrong answers in 20/20 trials, so the compiler's control bits are the only
protection. Second, the register file has 2 banks with one read each, so without the reuse cache a
3-source FFMA can never exceed 1 issue per 2 cycles; a register renumbering alone made a 50/50
FFMA:SHF mix 1.48x faster than ptxas. Third, split 32 B sectors and L2 reuse distance, not DRAM
latency, explain several slow GEMV variants; DRAM saturates at about 710-720 GB/s and loaded DRAM
latency is 570 cycles (292 ns).

## Question / hypothesis

Can the machine be measured well enough to predict kernel runtime from SASS, and then to schedule
and allocate registers better than ptxas? This requires numbers the vendor does not publish:
fixed-op latencies, which hazards the hardware interlocks, register-bank and reuse-cache rules,
how FP32 and INT32 share lanes, cache capacities, replacement, the L2 hash, TLB reach, tensor-core
rates and launch cost.

## Method

- Toolchain: no CUDA toolkit install. NVRTC (CUDA C to PTX), `ptxas` with flags under our control,
  `nvdisasm -hex` for SASS and 128-bit encodings, the `cuda-python` driver API to load cubins, and
  NVML for clocks, temperature and throttle reasons. Experiments live in `lab/experiments/`
  (for example `alu.py`, `regfile.py`, `cache.py`, `mem.py`, `smem.py`, `tensor.py`, `l2map.py`,
  `pairs.py`, `probes.py`) and are driven by `lab/cli.py`.
- Measurement protocol (applies to every run):
  1. Static validation before GPU time: the timed loop is pinned with `#pragma unroll 1`, the runner
     parses the SASS, isolates the loop and refuses to measure unless it holds exactly the expected
     count of the target opcode and nothing spills.
  2. Warmup for a fixed wall time (default 500 ms) to leave the idle P-state.
  3. Shuffled, interleaved trials across variants (seed recorded), 30 per variant by default.
  4. Per-trial NVML environment: SM/memory clock, temperature, power, P-state, throttle reasons.
  5. In-kernel `clock64()` and `%globaltimer` bracket the loop (cycles/op and the true SM clock).
  6. Integer chains are checked against a host reference.
  7. Warnings for throttling, clock drift, temperature, CV > 2%, SASS mismatch, wrong results.
- Control-word decoding: bits [105:125] of each 128-bit instruction are stall (4), yield (1), write
  barrier (3), read barrier (3), wait mask (6), reuse (4). Patching is done directly in the cubin
  (`lab/patch.py`, `lab/sass.py`); operand fields were discovered empirically per instruction form
  (`lab/operands.py`, `data/operand_fields.json`).
- Runs go through `pcslurm submit --shared` so they never compete with other GPU jobs.

## Results

### Instruction table

| op | dependent latency (cyc) | throughput, warp-instr/cyc/SM (32 warps) | lanes/SM |
|---|---:|---:|---:|
| FFMA / FADD / FMUL | 4 | 3.94-3.97 | 128 |
| HFMA2 | 4 | 2.00 | 64 (x2 halves) |
| IMAD | 4 | 2.00 | 64 |
| SHF | 4 | 1.99 | 64 |
| IADD3 | 4 (stall-probe) | not measured (see the -O0 note) | |
| MUFU.RSQ / MUFU.EX2 | 17 | 0.50 | 16 |
| FMUL.RZ + MUFU.SIN | 23 (pair) | 0.50 | 16 |
| SHFL.IDX | 26 | 0.50 | 16 |
| DFMA | 55 | 0.0625 | 2 (1/64 rate) |
| LDS (shared) | 23 | | |

ptxas encodes fixed-latency ops (FFMA, FADD, FMUL, IMAD, SHF, HFMA2) with stall 4 and no scoreboard;
DFMA, MUFU and SHFL use scoreboard barriers.

### The hardware trusts the compiler (control-bit patching)

- Stall 4 or more on a dependent FFMA/FADD/IMAD/IADD3 chain is correct. Stall 1-3 gives wrong
  answers in 20/20 trials with exactly half the increments lost. There is no interlock for
  fixed-latency ops: latency is 4 cycles, measured by breaking it.
- Stall 0 is not zero: it is correct but costs about 32.8 cycles/op. `-O0` code emits S00 on
  IADD3/LOP3 and runs dependent chains at 33.7 cycles, so `-O0` timings are invalid as latency data.
- One warp cannot issue dependent ALU ops faster than 1 per 2 cycles (stall 1 and 2 both give about
  2.06); 8 independent chains in one warp reach about 0.95 FFMA/cycle.

### Hazards and scoreboards

- FFMA is correct iff the real time between producer and consumer is at least 4 cycles (k=2, s=1
  runs 1.55 cyc/instr, 3.1 cycles apart, wrong; k=4, s=1 gives 4.2, correct). Legal code issues
  FFMA at about 1/cycle from one warp; the "2-cycle floor" appears only in illegal schedules.
- Structural hazards are interlocked: IMAD never issues faster than 1 per 2 cycles from one warp
  whatever the stall says (16-lane pipe); DFMA 16 cyc/instr, MUFU 8, SHFL 4.
- Scoreboards are necessary. With waits stripped, an isolated SHFL is correct at 12-cycle spacing
  (wrong at 8) against 26 through the scoreboard; MUFU.SIN correct at 15 (wrong at 12) against about
  19. With 2-4 SHFLs in flight 16 cycles is not enough, and DFMA is wrong even 64 cycles apart.
  Write-after-write is also a scoreboard job: stripping a SHFL barrier corrupted the clock registers.
- nvdisasm rejects yield=1 with stall 0 or stall >= 12.
- Later rules found by the Phase 8 scheduler (see `sass-optimization.md`): forwarding is 4 cycles
  within a pipe and 5 across the FMA/ALU pipes; FMA pipe = FFMA, FADD, FMUL, IMAD, IMAD.IADD; ALU
  pipe = IADD3, LOP3, SHF, FMNMX; a scoreboard arms one cycle after issue; a fixed-latency producer
  into SHFL needs 5 cycles (superseding the earlier 4); a predicate read by BRA/EXIT needs 13 cycles;
  memory ops issue at least 4 cycles apart; BAR.SYNC.DEFER_BLOCKING to the first memory op needs at
  least 6 cycles; FFMA/FADD/FMUL into MUFU is 4 cycles.

### Register banks and the reuse cache

FFMAs rewritten to `FFMA S, A, S, C` with chosen register parity (`regbank_ffma`):

| operand banks | no reuse flags | reuse on A and C |
|---|---:|---:|
| all three in one bank | 0.32 issue/cyc/partition (3 cycles) | 0.94 |
| two in one bank | 0.49 (2 cycles) | 0.94 |

- Two banks (register parity), one read per bank per cycle; without reuse FFMA cannot exceed
  1 issue per 2 cycles. Consistent with Huerta et al. MICRO'25; the costs here are measured.
- `.reuse` only takes effect when the raw yield bit is 1 (warp does not yield); raw 1 = stay.
- Entries are per (bank, operand slot). An instruction that does not read a slot leaves the entry
  alone; a read of the same bank and slot refreshes (reuse flag) or evicts it. The Oct 22 model
  adds: only FMA/ALU-pipe instructions hit or evict (MUFU, SHFL, memory ops do not), FADD's second
  operand uses the third slot, and a cleared yield bit flushes the cache. In ptxas order this model
  reproduces ptxas's reuse field on 12,185 of 12,200 instructions.
- IMAD follows the same bank rules: 3 operands in one bank cost 3 cycles, 2 in one bank cost nothing
  extra (the read overlaps the 2-cycle pipe).
- With a register count of 64, R62 and R63 fault with CUDA_ERROR_ILLEGAL_INSTRUCTION.

### FP32/INT32 sharing

The first measurement (50/50 FFMA:SHF at 2.65 warp-instr/cyc/SM, fit by T = p + 2(1-p), "no
overlap") is superseded. The ceiling was a ptxas register-allocation artifact: the interleaved SHF
evicts one FFMA operand from the reuse cache and ptxas put the other two reads in one bank.
Moving each FFMA's chain register to the opposite bank (`mixbank_*`, same instructions and schedule):

| FFMA fraction | ptxas | opposite bank | same bank |
|---|---:|---:|---:|
| 3/8 | 2.45 | 3.15 | 2.44 |
| 4/8 | 2.65 | 3.91 | 2.65 |
| 6/8 | 3.17 | 3.92 | 3.17 |

FP32 and INT32 do run concurrently: the fixed kernel tracks min(4, 2/(1-p)). IMAD does not benefit
because it shares the fmaheavy pipe with FFMA.

### L1, L2, TLB and DRAM

Single-thread dependent 64-bit loads, 128 B stride:

| level | latency (cyc) | ~ns | capacity edge |
|---|---:|---:|---|
| L1 hit | 35 | 18 | flat to 96 KB, degrading 104-128 KB (0 B smem kernel) |
| L2 hit | 238 | 121 | flat to 4.5 MB, 353 at 5 MB, miss at 5.5 MB (5 MB L2) |
| DRAM | 498 | 254 | +9 cyc at 256 MB random (TLB), not for sequential |
| shared | 23 | 12 | |

- L1 capacity: constant in 128 B lines (about 830) across slot strides 32-128 B. One formula fits
  all eight measured carveouts: L1 data = 128 KB - max(16 KB, smem carveout) - 8 KB (0/8/16% give
  104 KB, 24/32% 88, 50% 56, 100% about 20). The 16 KB floor and 8 KB constant are inferred.
- L1 replacement is recency-based (LRU family), not FIFO or random: re-touched lines survive 100%;
  untouched lines die oldest first (25% oldest vs 79% newest survive when streaming 384 new lines).
  Not strict LRU.
- L1 wavefronts: an L1-hit load costs one SM cycle per distinct 128 B line the warp touches (floor
  about 3): 1 line 2.8, 4 lines 4.3, 8 lines 8.0, 32 lines 32.0 cycles. `ld.global.nc` is identical.
- Shared memory: 32 banks x 4 B, one request per 2 cycles, so 2-way conflicts are free and n-way
  costs n/2. Bandwidth (`smem_width`) is 128 B/cycle for LDS.64, LDS.128 and LDSM.x4, 64 B/cycle for
  LDS.32.
- Transfer granularity is the 32 B sector for L2 and DRAM. L2 serves about 2.0-2.2 TB/s, DRAM about
  710-720 GB/s regardless of 4/8/16 B per-lane width; a 256 B lane stride drops DRAM to about 60%.
- Little's law: one SM caps at about 63 GB/s; DRAM saturates with >= 4 warps/SM on 68 SMs or
  >= 16 warps/SM on 17 SMs. Loaded DRAM latency 570 cycles = 292 ns (calibrated through the SM
  simulator).
- L2 structure: 40 slices x 128 KB, interleaved at 256 B (98% of 256 B chunks in one slice), mapped by
  contention (`scripts/l2_slices.py`). The hash has a linear part of rank 3 (XOR of bits
  {12,13,15,16,17,18}, {8,11,14,15,16,17,19}, {10,14,16,17,20}), giving 8 classes of 5 slices. The
  near/far half is the second of those and explains 100.0% of 16,384 lines in the latency map;
  near/far costs only about 3-12 cycles. SM position explains 69% of latency variance (TPC pairs,
  GPC blocks of 6).
- TLB: cudaMalloc uses 2 MB pages; a 16-entry first level (knee between 32 and 48 MB), and only
  +10-15 cycles beyond it even at 2 GB.
- cp.async (`l2_cpasync`) costs one SM cycle per active lane, not per byte: 2.1-2.2 TB/s GPU-wide at
  16 B per lane, an SM-side limit. Inside the multistage GEMM copies run at about 0.7 lanes/cycle/SM,
  which fits a per-SM cap of about 500 lanes in flight at the 570-cycle latency. Plain
  `ld.global.cg.v4` moves 1.0 lanes/cycle/SM (`scripts/ldg_vs_cpasync.py`).
- Instruction cache: full issue rate up to 64 KB of loop code; one warp per partition drops at 96 KB
  (1.34/cycle) and 192 KB (0.82).
- `prefetch.global.L2` (CCTL.E.PF2) fills L2 (chase after prefetch 279 cycles vs 495).

### Tensor cores (`tc_*`)

| mma.sync | latency (cyc) | peak / SM / cycle | at 1.95 GHz |
|---|---:|---:|---:|
| FP16 or BF16 -> FP32 acc | 32.9 | 512 FLOP | 67.9 TFLOPS |
| FP16 -> FP16 acc | 24.0 | 1,024 FLOP | 135.5 TFLOPS |
| TF32 -> FP32 | 32.9 | 256 FLOP | 33.9 TFLOPS |
| INT8 / INT4 -> INT32 | 24.0 | 2,044 / 4,088 ops | 271 / 542 TOPS |
| binary AND+POPC | 24.0 | 16,351 ops | 2,168 TOPS |

FP32 accumulation is half rate. One warp per SM partition saturates the tensor core. Mixed-sign
u8 x s8 MMA runs at the s8 x s8 rate (0.25 warp-MMAs per cycle per SM). Fragment layouts for
m16n8k16 f16 and m16n8k32 s8 were verified against numpy (`tc_correct`). Sustained int8 GEMM is
power-capped at 1725-1785 MHz around 365 W, not the 1.95 GHz light kernels run at.

### Launch cost

- In a CUDA graph (`scripts/launch_cost.py`): 1.07-1.28 us for an empty kernel regardless of block
  shape (1-68 blocks, 128-1024 threads) or parameter count; +0.2 us with 5.7 KB of static shared
  memory; dispatch grows past about 500 blocks (4,480 blocks: 3.46 us, about 34 ns per block per SM).
  An earlier measurement in the same notebook gave 4.8 us at 4,480 blocks and 27.5 us at 37,984
  blocks (about 49 ns/block/SM); the Round 4 number is the later one.
- A software grid barrier costs about 1.0 us with 68-136 co-resident blocks (1.3 us at 272, 2.3 at
  544), the same as a kernel boundary in a graph.
- A Python stream launch costs 8.8 us when issued alone but overlaps with kernel work in a real
  sequence (`scripts/launch_gap.py`, `data/optsel/launch_gap.json`).

## Negative results and limits

- Cache-capacity edges are noisy (CV 5-18%) and the SM simulator needs a partial-hit model there.
- The 5-way L2 slice selector is not a simple modulo of address bits below 2^21 (best 0.36 vs 0.24
  chance); some slice pairs have identical latency profiles, so the full hash is not recoverable from
  latency alone.
- L2 placement (`data/l2_place/`): address sets aimed at one slice are 3-12x slower on every page of a
  fresh allocation, but spreading a working set across slices does not reliably beat a contiguous
  layout (0.69-1.34x). Negative.
- The R8U2 GEMV mystery: refuted causes include instruction cache, L1 pollution, split sectors at any
  occupancy, partial trips and per-warp MLP. The real cause (L2 reuse distance of split halves) is in
  `sass-optimization.md`.
- Hung kernels are not reset by TDR on this machine; spin loops need caps.
- No hardware counters: Nsight Compute is not available from pip, so instruction counts are static
  loop body x trip count. Clock locking needs an elevated shell, so clocks are recorded, not controlled.
- The 3080 also drives the desktop; background graphics load changed several end-to-end results
  (see `llm-inference.md`).
- Toolchain traps: ptxas is an optimizing assembler (`asm volatile` pins PTX, not SASS); at -O1+ `add`
  chains fuse into IADD3, repeated-operand `xor` chains fold, uniform `shfl.idx` chains are deleted.
  The validator refuses to measure these.

## Reproduce

```
.venv\Scripts\3080lab list
.venv\Scripts\3080lab sass dependent_ffma
.venv\Scripts\3080lab run dependent_ffma
.venv\Scripts\3080lab run independent_ffma --warps 1,2,4,8,16,32
.venv\Scripts\3080lab table          # regenerates results/TABLE.md
python scripts/launch_cost.py
python scripts/l2_slices.py
python scripts/ldg_vs_cpasync.py
python scripts/launch_gap.py
```

GPU scripts go through `pcslurm submit`. Run artifacts are written to `results/<timestamp>_<experiment>/`
(not tracked). Data in the repo: `data/dram_latency.json`, `data/operand_fields.json`,
`data/l2_place/`, `data/decode_oct12/ldg_vs_cpasync.txt`.

## Notebook sections

From [the notebook](../notebook.md):
- Measurement protocol; Control-word decoding; Toolchain
- Instruction table
- The hardware trusts the compiler (control-bit patching)
- Hazards: data is the compiler's job, structure is the hardware's
- L1: size, carveout, replacement
- Register file: 2 banks, 1 read each, and the reuse cache is mandatory
- The FP32/INT32 "no overlap" ceiling was a ptxas register-allocation artifact
- Reuse cache and yield semantics
- Memory system under load
- Tensor cores
- Open questions closed (Oct 8)
- Round 4: load order, launch cost, L2 slices (Oct 8)
- Where the remaining 2.29 ms/token goes (launch cost bullet)
- Occupancy cost model for the multistage GEMM (microbenchmark table)
- Phase 8 and Oct 18-21 (hazard rules, reuse model)
- FP32/INT32 sharing (superseded by Phase 8 above)
- Memory hierarchy; Toolchain behaviour; Not yet available
