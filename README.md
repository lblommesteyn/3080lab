# 3080lab

An empirical microarchitecture lab for one RTX 3080 (GA102, sm_86). It measures the machine below
CUDA and PTX, builds performance models from those measurements, rewrites ptxas's SASS, and tests
on held-out workloads where those models do and do not pay off.

```
  microbenchmarks ──> latency / port / memory tables ──> SM simulator + whole-GPU model
  (lab/experiments)        (docs/reports/microarchitecture.md)       (lab/model.py, lab/model_gpu.py)
                                                                          │
         cubin from ptxas ──> SASS decode + operand map ──> rewrite ──────┤ predict: apply or not?
         (lab/toolchain.py)   (lab/sass.py, operands.py)   (resched,      │ (lab/optsel.py)
                                                            optimize,     ▼
                                                            jointra)   static hazard check + bitwise
                                                                       differential test + timing
                                                                       (lab/verify.py, sassopt.py)
  Qwen2.5-1.5B kernels (lab/qwen_*.py) use what the tables say: memory-level parallelism, fusion, graphs
```

## Three results

**1. Split-sector load optimizer: 1.074x on held-out GEMV kernels.** `3080lab optimize` finds loads whose
32 B sectors overlap a later load by affine address analysis of the SASS, hoists the pair together, and
validates the rewrite. Blind rewriting gives held-out geomeans of 1.036x / 1.074x / 1.055x on three int4
GEMV suites (56 / 43 / 20 kernels), with 0 correctness failures. A source-level layout fix does better
(1.10-1.37x) where one exists; the rewrite needs no source. Details: [SASS optimization](docs/reports/sass-optimization.md),
[benchmark](docs/optimize_benchmark.md).

**2. The hardware model picks launch structure, not instruction order.** On a pre-registered held-out
benchmark (771 measured decisions), the launch-cost model chose fusion and CUDA-graph variants for
PolyBench at 1.229x against an oracle of 1.231x and 1.159x for the best fixed rule. For SASS reordering
the oracle's whole headroom is 0.5-0.6%, below measurement noise, and no predictor beats "never". A
general block scheduler reproduces ptxas but does not beat it, and the notebook explains why.
Details: [optimization selection](docs/reports/optimization-selection.md).

**3. Qwen2.5-1.5B on llama.cpp's own Q4_0 weights: 1.41x decode, up to 2.40x batched, up to 2.00x prefill**
against llama.cpp b11485 on the same GPU (Oct 14). These used a best-of-repeated-runs rule on both sides;
a paired re-measurement with central estimates is in [LLM inference](docs/reports/llm-inference.md).

| workload | ours | llama.cpp b11485 | ratio |
|---|---:|---:|---:|
| decode, 1 sequence (tok/s) | 514.0 | 365.8 | 1.41x |
| batched decode, 2 / 8 / 32 sequences (tok/s) | 950 / 3395 / 7135 | 625 / 1416 / 3597 | 1.52 / 2.40 / 1.98x |
| prefill, 128 / 2048-token prompt (tok/s) | 21,888 / 28,114 | 10,937 / 17,639 | 2.00 / 1.59x |

## Reports

| report | what it covers |
|---|---|
| [Microarchitecture](docs/reports/microarchitecture.md) | latency and throughput table, control bits, scoreboards, register banks and reuse cache, L1/L2/TLB, tensor cores, launch cost |
| [SASS optimization](docs/reports/sass-optimization.md) | register re-allocation, split-sector optimizer, block scheduler, joint RA and scheduling, why scheduling stops at parity |
| [LLM inference](docs/reports/llm-inference.md) | Qwen2.5-1.5B decode, batched decode and prefill; where the time goes; negative results |
| [Optimization selection](docs/reports/optimization-selection.md) | A/A noise, the held-out decision benchmark, fusion and graph selection, calibration, the PolyBench demo |
| [Verification scope](docs/verification.md) | what a SASS rewrite guarantees statically, what is only tested, known gaps |
| [Lab notebook](docs/notebook.md) | the full chronological record every report is distilled from |

## Quick start

```
.venv\Scripts\3080lab list                    # experiments
.venv\Scripts\3080lab sass dependent_ffma     # compile only: timed-loop SASS + control bits + raw bytes
.venv\Scripts\3080lab run dependent_ffma      # queues via pcslurm, waits, prints report
.venv\Scripts\3080lab run independent_ffma --warps 1,2,4,8,16,32
.venv\Scripts\3080lab show results\<dir>      # re-print a saved record
```

`run` submits itself through `pcslurm submit --shared` so it never competes with other GPU jobs;
`--local` runs in-process. Scripts under `scripts/` find the project venv through `scripts/_site.py`
(set `LAB_VENV` to point elsewhere).

No CUDA toolkit install: NVRTC, `ptxas`, `nvdisasm` and the driver API all come from pip wheels
(CUDA 13.x). Kernels are loaded as cubins, which is also how rewritten SASS is run. The measurement
protocol (static validation of the timed loop, warmup, shuffled interleaved trials, in-kernel clocks,
correctness checks) is described at the top of the [notebook](docs/notebook.md#measurement-protocol).

## Limits

One GPU, one architecture. Every latency and hazard rule was measured on this RTX 3080; none has been
checked on another part. SASS rewrites are validated by a static hazard check and bitwise differential
tests, not proven equivalent ([verification scope](docs/verification.md)).
