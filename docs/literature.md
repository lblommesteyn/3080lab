# Literature survey (Oct 2026)

Compiled by a web-research pass. Items marked (unverified) could not be confirmed.

## Closest prior work

**Huerta et al., "Dissecting and Modeling the Architecture of Modern GPU Cores", MICRO'25 (arXiv 2503.20481).**
Tested RTX 3080 / 3080 Ti / 3090 / A6000. Already established:
- no hardware interlock for fixed-latency RAW hazards (matches our stall probes)
- 6 dependence counters (SB0-5, up to 63 each); producer increments at issue,
  decrements at write-back; `DEPBAR.LE SBx, N`
- issue: 1 instr/cycle per sub-core, "greedy then youngest" warp selection
- register file: 2 banks per sub-core, one 1024-bit read port each; register
  file cache with 1 entry per bank x 3 operand slots; bank conflicts add 0-2 cycles
- 2 intermediate stages between issue and operand read
- LDS RAW 23 / WAR 9; LDG RAW 29 (uniform addressing); per-sub-core memory
  queue ~5 entries; instruction prefetch modeled as a 16-entry stream buffer
- their simulator: ~10-14% MAPE on A6000 (stock Accel-Sim ~32%); code release unverified

## Other microbenchmark work

- Jia et al., Volta (arXiv 1804.06826), Turing T4 (arXiv 1903.07486): control word
  layout, register banks, reuse cache, scheduler = warp_id % 4. Turing: L1
  replacement "randomly evicts 4 consecutive lines"; detectable L1 ~7 KiB below
  nominal, unexplained. HFMA2 latency 6 on Turing (we measure 4 on GA102).
- Abdelkhalik et al., Ampere A100, HPEC'22 (arXiv 2208.11174): PTX-level; L1 33,
  L2 200, global 290; HMMA.16816 8 cycles.
- Mei & Chu, memory hierarchy P-chase, TPDS'17 (arXiv 1509.02308): Fermi-Maxwell only.
- Luo et al., Hopper, IPDPS'24 (arXiv 2402.13499): A100 L1 37.9 / L2 261.5 / DRAM 466;
  RTX 4090 L1 43.4 / L2 273 / DRAM 541.5.
- Sun et al., tensor cores, TPDS'23 (arXiv 2206.02874).
- Jarmusch et al., Blackwell (arXiv 2507.10789).

## Control bits

- Layout (Jia): reuse 4 | wait mask 6 | read bar 3 | write bar 3 | yield 1 | stall 4.
- Yield polarity is described inconsistently between Jia and Huerta; consistent
  if the raw bit is the inverse of the displayed flag. maxas (Maxwell): "stalls
  12-15 require yield", which matches nvdisasm rejecting our yield+stall>=12.
- **stall=0 on Volta+ is undocumented** (on Maxwell it meant dual issue). Our
  ~33-cycle cost appears novel.
- GA10x pipes (NVIDIA forum): fmaheavy = FP32 + FP16 + IMAD + IDP(dp4a);
  fmalite = FP32 + FP16. Explains FFMA 4, IMAD 2, HFMA2 2 per SM per cycle.
- DeepGEMM: clears reuse whenever it sets yield ("registers cannot be reused if
  the warp is yielded").

## Tools

- CuAssembler (cloudcores): sm_60-sm_86, the only maintained-ish sm_86 assembler.
- **DocumentSASS (0xD0GF00D): extracts ISA + latency tables embedded as strings in
  nvdisasm.** Highest-leverage next step: diff sm_86 tables against our numbers.
- nv_isa_solver (sm_89/90a), F2Asm (sm_90-107, arXiv 2608.20532), TuringAs,
  maxas (wiki = best control-code semantics), EoSS (ASPLOS'27, repo unverified).

## Beating ptxas

- Jia: +15% Volta SGEMM, +12% Turing via register-bank assignment + reuse flags.
- Yan, Wang, Chu (IPDPS'20): Turing HGEMM in SASS, 1.73x cuBLAS on RTX 2070.
- DeepGEMM: yield/reuse interleave on FFMA, "10%+ in some cases" (Hopper FP8);
  NVCC 12.9 now does this itself.
- SIP (arXiv 2403.16863): annealing memory-instruction placement, -6% to -12% latency.
- CuAsmRL (CGO'25, arXiv 2501.08071): RL reorders only memory instructions on A100,
  geomean 1.09x over Triton, up to +26%; never edits control bits.

## Performance prediction

Accel-Sim (has SM86_RTX3070 config), PPT-GPU (<16% MAPE), GCoM, GPUMech,
NeuSight (2.3% end-to-end on GPT-3/H100), SynPerf (6.1% kernel MAPE).
Gap: nothing predicts at basic-block level from explicit control bits;
real-hardware kernel errors are typically 10-30%.

## Memory system

- TLB (TunneLs, CCS'23, RTX 3080): 16-entry fully-assoc L1 dTLB shared by a TPC;
  8-way L2-uTLB per GPC; 8-way L3-uTLB shared. Tool: github.com/0x5ec1ab/gpu-tlb.
- DRAM (GPUHammer, USENIX Sec'25, A6000=GA102): 256 B chunks per bank/row, XOR bank
  hash learned as a lookup table; L2 bypass via `discard.global.L2`; RTX 3080 showed no flips.
- No published GA102 L2 set hash, L2 near/far partitioning, or L1 replacement policy.

## Our findings vs literature

| finding | status |
|---|---|
| ALU latency 4, no data-hazard interlock | agrees; published (Huerta) |
| stall=0 correct but ~33 cyc | likely novel |
| HFMA2 latency 4 on GA102 | differs from Turing (6); no Ampere figure found |
| FFMA:INT 50/50 mix = 2.65 not 4 | unexplained; likely novel (register-port pressure is a candidate) |
| SHFL 26 via scoreboard / 12 safe without | likely novel |
| scoreboard-free safe spacing (MUFU 15, DFMA none) | likely novel |
| L1 ~100-108 KB at carveout 0 | same kind of shortfall as Turing's, larger; unexplained |
| L1 pseudo-LRU on GA102 | novel; differs from Turing's random-4-line |
| memory latencies, L2 5 MB | agree |

## LLM decode kernels (Phase 9)

Marlin (W4A16, ~3.9x FP16 at small batch on A10/sm_86), FLUTE, QServe, ExLlamaV2,
llama.cpp MMVQ (dp4a -> IDP on fmaheavy, half rate). **No published SASS
control-bit/scheduling work on GEMV/dequant decode kernels**: all SASS-level
wins are compute-bound GEMM/attention on A100/H100. Clear gap for this project.

## Ranked open questions (feasible here)

1. Extract sm_86 latency tables from nvdisasm (DocumentSASS method); diff vs measured.
2. Yield polarity and stall=0 semantics on Ampere.
3. Explain the 2.65 FFMA:INT ceiling (register bank parity, reuse flags, LOP3/IADD3/IMAD partners).
4. Register bank-conflict cost and reuse-cache invalidation (validate Huerta's 2-bank / 3-slot model).
5. Safe minimum spacing without scoreboards, per op; savings in real kernels.
6. Basic-block cycle predictor with greedy-then-youngest + register ports; validate on Marlin / llama.cpp cubins.
7. GA102 L1 pseudo-LRU structure (tree-PLRU patterns) and carveout x capacity.
8. L2 near/far partitions (latency per address per SM) and set hash via `discard`.
9. TLB reach in our harness (reproduce TunneLs timing-only).
10. Yield/reuse + memory-reorder edits on a 4-bit GEMV for Qwen shapes.
11. IDP/dp4a vs HFMA2 dequant pipe balance on fmaheavy/fmalite.
12. GDDR6 bank XOR function (lower priority).

## References

Jia Volta https://arxiv.org/abs/1804.06826 · Jia Turing https://arxiv.org/abs/1903.07486 ·
Abdelkhalik https://arxiv.org/abs/2208.11174 · Mei & Chu https://arxiv.org/abs/1509.02308 ·
Huerta https://arxiv.org/abs/2503.20481 · Luo https://arxiv.org/abs/2402.13499 ·
Sun https://arxiv.org/abs/2206.02874 · Jarmusch https://arxiv.org/abs/2507.10789 ·
CuAsmRL https://arxiv.org/abs/2501.08071 · SIP https://arxiv.org/abs/2403.16863 ·
Yan https://home.cse.ust.hk/~weiwa/papers/yan-ipdps20.pdf · DeepGEMM https://github.com/deepseek-ai/DeepGEMM ·
maxas https://github.com/NervanaSystems/maxas/wiki/Control-Codes · CuAssembler https://github.com/cloudcores/CuAssembler ·
DocumentSASS https://github.com/0xD0GF00D/DocumentSASS · nv_isa_solver https://github.com/kuterd/nv_isa_solver ·
F2Asm https://arxiv.org/abs/2608.20532 · Accel-Sim https://github.com/accel-sim/accel-sim-framework ·
NeuSight https://arxiv.org/abs/2407.13853 · SynPerf https://arxiv.org/pdf/2601.14910 ·
TunneLs https://github.com/0x5ec1ab/gpu-tlb · GPUHammer https://arxiv.org/abs/2507.08166 ·
Spy in the GPU-box https://arxiv.org/abs/2203.15981 · Marlin https://arxiv.org/abs/2408.11743 ·
FLUTE https://arxiv.org/abs/2407.10960 · QServe https://arxiv.org/abs/2405.04532 ·
GA10x pipes forum https://forums.developer.nvidia.com/t/is-there-a-way-i-can-tell-whether-im-getting-concurrent-floating-point-instructions-on-cc86-cc89/309713
