# Qwen2.5-1.5B inference on one RTX 3080

Summary. On llama.cpp's own Q4_0 GGUF of Qwen2.5-1.5B, the custom engine (CUDA graph of about 200
fused kernels, int4 GEMVs, tensor-core batched GEMVs, int8 tensor-core prefill GEMM and flash
attention) measured against llama.cpp b11485 on the same card in the Oct 13 night run (executed
Oct 14): decode B = 1 at 514.0 vs 365.8 tok/s (1.41x), batched decode 1.52 / 1.84 / 2.40 / 2.24 /
1.98x at B = 2 / 4 / 8 / 16 / 32, and prefill 2.00 / 1.66 / 1.59x at 128 / 512 / 2048 tokens. Those
headline numbers use a best-of-repeated-runs rule on both sides (`scripts/night_summary.py`), not a
paired statistic; a paired re-measurement is pending (last section). Decode at B = 1 spends 1.95 ms
per token, 63% of the about 760 GB/s peak on 928 MB of weights.

## Question / hypothesis

Can microarchitecture-driven kernels make an LLM faster end to end than a mature engine on identical
weight bytes? The working hypotheses: batch-1 decode is limited by per-kernel fixed cost and memory
parallelism rather than raw GEMV speed (so fusion wins); batched decode should read each weight once
for B vectors on tensor cores; prefill should run int8 MMA on raw Q4_0 nibbles without writing
dequantized weights.

## Method

- Engines: `scripts/qwen_gguf.py` (decode B = 1 on GGUF weights, `lab/gguf.py`, `lab/qwen_kernels.py`),
  `scripts/qwen_fused.py` (own int4 g128 weights), `scripts/qwen_batched_decode.py` with
  `lab/qwen_batched.py`, `scripts/qwen_prefill.py`. Kernel benches: `scripts/batched_gemv_bench.py`,
  `scripts/head_bench.py`, `scripts/prefill_gemm_bench.py`, `scripts/flash_bench.py`,
  `scripts/attn_bench.py`. Profiles: `scripts/qwen_kernel_times.py`, `scripts/decode_kernel_profile.py`,
  `scripts/prefill_kernel_profile.py`. In-process A/B: `scripts/decode_ab.py`, `scripts/prefill_ab.py`.
- Baseline: llama.cpp b11485 with the official Q4_0 GGUF (`llama-bench`, `llama-batched-bench`). Every
  tensor comes from llama.cpp's file: Q4_0 linears repacked bit-exactly, Q6_K head kept in llama.cpp's
  format. The official GGUF folds per-channel scales into norm weights, so all tensors must come from
  the same file.
- Correctness: last-token argmax matches a token-by-token reference; KV cache and logits against
  token-by-token prefill; perplexity on README text within 0.1% of an fp16 cuBLAS path. Both engines
  give coherent answers that diverge after 8 words (llama.cpp quantizes activations to Q8_1).
- Headline protocol (`scripts/night_summary.py`, `data/night_oct11/`, `data/night_oct13/`): three
  interleaved rounds of ours and llama.cpp; best of everything on both sides (llama.cpp best of 15
  `llama-bench` repetitions per round for prefill, 10 for decode, best `llama-batched-bench` round;
  ours minimum of 21 graph replays per run, best run). `qwen_prefill.py` warms 0.5 s and reports the
  minimum of 21 replays because desktop contention only adds time.
- Change-level decisions after Oct 11 use in-process A/B: both configs captured in one process, graphs
  replayed round robin in shuffled order.

## Results

### Headline (Oct 13 night run, `data/night_oct13/`)

| | ours | llama.cpp | ratio |
|---|---:|---:|---:|
| decode B = 1 | 514.0 | 365.8 | 1.41x |
| batched decode B = 2 / 4 / 8 / 16 / 32 | 950 / 1792 / 3395 / 5578 / 7135 | 625 / 972 / 1416 / 2489 / 3597 | 1.52 / 1.84 / 2.40 / 2.24 / 1.98x |
| prefill P = 128 / 256 / 512 / 1024 / 2048 | 21,888 / 26,755 / 30,767 / 29,840 / 28,114 | 10,937 / 15,141 / 18,502 / 18,607 / 17,639 | 2.00 / 1.77 / 1.66 / 1.60 / 1.59x |

The run queued behind other jobs and ran Oct 14 17:11-17:29 with the GPU at 13% utilization and
desktop apps open. Our prefill was 25-40% slower in rounds 2 and 3 than round 1; the best-of rule keeps
round 1 for both sides. These supersede the Oct 11 night run (decode 1.38x, prefill 1.64-1.33x) and the
September 1.50x decode ratio, which was against a llama.cpp measured on a busy desktop (340.5 tok/s).

### Decode B = 1: history

| change (GGUF weights) | tok/s |
|---|---:|
| PyTorch eager-style ops + tinygemm int4, one CUDA graph | 207 |
| llama.cpp `-fa 0` / `-fa 1` (September, busy desktop) | 316 / 340.5 |
| ours, first GGUF run | 437 |
| attention v1 / v4 (4-way ILP, 17-21% faster attention) | 446 / 463 |
| + Q6_K-exact lm_head (191 MB) | 492 |
| + gate/up split-K 2, single-pass RMSNorm | 509 |
| Oct 13 night run (adds the 64-block argmax, -0.3 to -0.4%) | 514.0 |

The win is fusion, not the GEMV: tinygemm is already about 94% of the 705 GB/s practical roofline at
7B shapes and our split-K GEMV ties it. PyTorch launches about 1,100 kernels per token; ours about 200
(fused RMSNorm, GEMVs with bias/residual/SiLU*mul epilogues, one RoPE + KV-write + GQA attention kernel,
argmax + position update), in one CUDA graph.

### Where decode time goes (Oct 12, position 128, `data/decode_oct12/`)

| role | per step | per launch | weight bytes | effective GB/s |
|---|---:|---:|---:|---:|
| gate_up GEMV | 680 us (35%) | 24.3 us | 15.5 MB | 638 |
| down GEMV | 395 us (20%) | 14.1 us | 7.7 MB | 549 |
| LM head (Q6_K) | 289 us (15%) | | 191 MB | 662 |
| attention | 155 us (8%) | 5.5 us | | |
| qkv GEMV | 136 us (7%) | 4.8 us | 1.8 MB | 365 |
| o GEMV | 115 us (6%) | 4.1 us | 1.3 MB | 324 |
| 2 RMSNorms | 105 us (5%) | 1.8-1.9 us | | |
| argmax, embed | 32 us | | | |

Total 1.95 ms for 928 MB of weights, 63% of peak. What is left outside the big GEMVs (about 0.5 ms) is
per-kernel fixed cost: about 1.07-1.28 us per empty kernel in a graph (see `microarchitecture.md`).

### Batched decode

Two kernel families read each weight once for B vectors. The CUDA-core version is correct but only
about 2.3x over B launches at B = 4 and is compute- and occupancy-bound at B = 8 (244 registers). The
tensor-core version (`mma.sync` m16n8k16 fp16 -> fp32, Q4_0 nibbles to `half2` via the 0x6400 trick)
went through v1 (tuned S, U, MT) and v2 (weights repacked into fragment order, inputs staged once in
shared memory and read with `ldmatrix.x4`, extra K split with deterministic last-block combine):

| sum of the four GEMVs per layer (us) | B = 8 | 16 | 32 |
|---|---|---|---|
| v1 tuned | 81.5 | 124 | 184 |
| v2 | 64 | 79 | 124 |

The v2 Q6_K head runs 286 / 290 / 292 / 311 / 368 us at B = 1 / 4 / 8 / 16 / 32 (671 GB/s at B = 1)
against cuBLAS's 660-705 us on the fp16 head, logit error 4e-7 to 6e-7. With v2 every B matches the
single-sequence decode for 73-74 of 128 tokens before diverging at a near-tie (fp16 weight rounding).
At B = 1 the single-vector path stays better (509 vs 468 tok/s).

### Prefill

| step | P = 512 tok/s | ratio vs llama.cpp |
|---|---|---|
| cuBLAS on per-layer fp16-dequantized weights | 9,342 | |
| int8 GEMM (multistage) + flash attention (busy desktop) | 21,424 | 1.21 |
| Oct 11 idle desktop | 25,868 | 1.40 |
| + 128 x 32 / 16-warp down tile | 26,032 | 1.41 |
| + flash attention v3 (two warp groups split keys) | 26,647 | 1.44 |
| + fused int8 quantize passes (Oct 12) | 31,164 | 1.68 |
| Oct 13 night run | 30,767 | 1.66 |

- int8 GEMM (`gemm_q4i8_ms_source`): activations quantized to int8 per 32 values,
  `mma.m16n8k32.u8.s8` on raw Q4_0 nibbles, with the C operand carrying 0x4B400000 - 8 sum so the int32
  result is already the float bit pattern; bias, residual and SwiGLU fused; 3-stage cp.async ring.
  It runs at 55-80 TOPS of the measured 271.
- What bounds it: not compute (replacing every MMA with an integer op leaves time unchanged), not issue
  (609 instructions per 32 IMMA, 19 issue cycles per IMMA vs about 42 measured). It is the cp.async
  copy rate: about 3,660 cycles of copies per stage at 0.7 lanes/cycle against about 2,050 tensor
  cycles. The GEMMs are 92% of kernel time at P = 512 after fusion (gate_up alone 51%).
- Flash attention v3 (`flash_prefill_v3_source`): 27.9 us vs 42.5 for v1 on the first 512-token chunk,
  up to 40.7 TFLOPS at position 1536. Masked scores must be -inf, not -1e30.
- Fusions (Oct 12, `data/fuse_oct12/`): the four QUANT_Q8 passes took 2.44 ms of 19.7 (12%) at P = 512.
  Folding them into their producers (`rms`, `swiglu`, `attn`, `kv`) gave -14.8% at P = 512 and -13.2% at
  2048 (p10), -11.6% at 128 and -13.4% at 256. Perplexity 26.955 vs 26.968 at P = 512 (fp16 cuBLAS
  26.920).
- Long prompts: per-token cost outside attention is flat (36.0-36.4 us); the drop from 512 to 2048 is
  attention, which at 2048 runs at 41 TFLOPS against about 62 peak at the power-capped clock.

## Negative results and limits

- Decode megakernel (`lab/qwen_mega.py`, `scripts/qwen_mega.py`, `data/oct15_tiles_mega/`): one
  persistent 68-block kernel with producer warps streaming 703 MiB through cp.async/mbarrier rings. It
  is not correct (picks token 1 vs the reference's 89141) and is 17% slower anyway: p10 2,238.5 us
  (446.7 tok/s) vs 1,910.8 us (523.3 tok/s) for the graph. Parked. A software grid barrier costs about
  1.0 us, the same as a kernel boundary in a graph, which predicted this.
- Bigger prefill tiles (256-row, 128 x 128 through dynamic shared memory): bit-identical but all slower,
  -1.7% to -14.7% against a 16.625 ms default at P = 512.
- Folding RMSNorm into residual GEMVs with a last-block tail: 396 tok/s (about 8.7 us per site vs 2.25
  us for a separate kernel). L2 prefetch of next weights during attention: 505/504/502 vs 512 tok/s.
- One-launch flash-decoding attention (`ATTN2`) only wins past about 300 positions.
- More split-K warps in decode GEMVs lose everywhere (gate_up x4 +19.7%).
- Batched GEMV after v2: fp16 accumulation (`ACC16`) no gain (not MMA-bound); register prefetch slower;
  `gemv_v3_source` cp.async ring cuts occupancy and was not integrated. Dequantization is not a cost.
- Prefill int8 GEMM: pre-expanding weights to int8 not worth building; plain loads do not beat cp.async
  for X; the multistage kernel's benchmark gains did not reach the full model (21.4-21.7K vs 22.1K
  end to end on a busy desktop).
- The half2 (Marlin-style) int4 GEMV variant computed wrong results and was not debugged.
- Measurement limits: the 3080 drives the display, and pcslurm serializes CUDA jobs, not graphics
  contexts. One build measured 5.5-25K tok/s run to run under contention. The headline table is
  best-of-runs on both sides, so it reports each engine's best case, not a paired or typical
  comparison. llama.cpp's pp512 once measured 15,910 +- 1,718, an example of its run-to-run spread.
  Batched sequences match the single-sequence decode only for 73-84 tokens (near-tie divergence), and
  the B = 32 v1 config diverged at token 25.
- Our Q6_K head path at B = 1 is exact in llama.cpp's format; the earlier int8 head read 292 MB/token
  vs llama.cpp's 191 MB, which the 492 tok/s step removed.

## Reproduce

```
pcslurm submit -- python scripts/qwen_gguf.py
pcslurm submit -- python scripts/qwen_batched_decode.py
pcslurm submit -- python scripts/qwen_prefill.py modes=i8
python scripts/night_summary.py dir=data/night_oct13
python scripts/decode_kernel_profile.py ; python scripts/decode_ab.py
python scripts/prefill_kernel_profile.py ; python scripts/prefill_ab.py P=512
python scripts/qwen_mega.py
```

llama.cpp b11485 binaries are not in the repo (`vendor/` holds a different build, `llamacpp_b12428`).
Data: `data/night_oct11/`, `data/night_oct13/`, `data/decode_oct12/`, `data/fuse_oct12/`,
`data/oct15_tiles_mega/`.

## Paired re-measurement (Nov 16)

Pending: job 2233, scripts/decode_paired.py.

## Notebook sections

From [the notebook](../notebook.md):
- Current results (header table)
- int4 GEMV baseline; int4 GEMV: 2.4-2.5x from memory-level parallelism
- Phase 9: Qwen2.5-1.5B decode, end to end
- Where the remaining 2.29 ms/token goes (GGUF run)
- Open questions closed (Oct 8): predictor-guided Qwen work
- Qwen decode, round 3 (same bytes as llama.cpp)
- Batched decode kernels
- Prefill
- Night run, Oct 11; Fused small ops (Oct 12); Decode B = 1: where the 1.95 ms goes (Oct 12)
- Two negative results, Oct 15: a decode megakernel and bigger prefill tiles
- Night run, Oct 13
