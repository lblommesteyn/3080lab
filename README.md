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

## Findings so far

- ptxas is an optimizing assembler; `asm volatile` pins PTX, not SASS. At
  -O1 and above, `add` chains fuse pairwise into 3-input IADD3 (half the ops),
  `xor` chains with a repeated operand fold to identity, and a uniform-lane
  `shfl.idx` chain is deleted as idempotent. The validator catches all three.
  iadd/lop3 default to `-O0`; shfl uses a per-lane source.
- `-O0` keeps one SASS op per PTX op but emits **stall 0** on dependent IADD3
  chains, while -O3 builds encode stall 4 on dependent FFMA. Whether S00 code
  is correct, and what it costs, is a Phase 2 question.
- MUFU.SIN is always preceded by FMUL.RZ (range reduction by 1/2pi), so the
  sin "latency" is a two-instruction chain.

## Not yet available

- **Hardware counters** (memory/cache counters, warp-stall reasons, executed
  instruction counts) need Nsight Compute (`ncu`), which is not on pip. It
  ships with the CUDA toolkit installer (needs admin), plus "allow access to
  GPU performance counters to all users" in the NVIDIA Control Panel.
  Until then, the instruction count is static loop body x trip count.
- **Clock locking** (`--lock-clock 1710`) calls `nvidia-smi -lgc` and needs an
  elevated shell; without it, clocks are recorded rather than controlled.
