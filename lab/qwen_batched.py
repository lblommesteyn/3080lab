"""Batched (B sequences) decode kernels for Qwen2.5 Q4_0 weights.

At batch 1, decode is bound by streaming the weights once per token. With B sequences the same
weight bytes serve B dot products, so a kernel that dequantizes each weight once and multiplies
it against all B input vectors approaches B x the throughput until it becomes compute-bound.

Weight layout is the one lab/qwen_kernels.gemv_source uses (int4 packed 8 per uint32, one uint4 =
one Q4_0 block of 32 weights, fp16 scale per block). X is [B, K] bf16, row-major.
"""
from __future__ import annotations


def gemv_batched_source(S_: int, B: int, epi: str = "store", unroll1: bool = True, minb: int = 1) -> str:
    """Split-K Q4_0 GEMV over B input vectors. A block = one 4-row group x S warps over K.
    Per uint4 of weights (32 values) a lane dequantizes 8 weights at a time (one 32-bit word) and
    multiplies them against the matching 8 values of all B vectors: registers stay ~8B + 8 + 4B.
    epi: store (bf16 Y[B, N]), biasf (bf16 Y = acc + fp32 AUX[N]), resid (fp32 Y[B, N] += acc),
         swiglu (rows interleaved g, u: bf16 Y[B, N/2] = silu(g) * u)."""
    L = []
    if unroll1:
        L.append("#pragma unroll 1")                   # keep one block of weights in flight per lane: registers
    L.append("for (int j = kb + lane; j < ke; j += 32) {")
    for r in range(4):
        L.append(f"  uint4 w{r} = (row0 + {r} < rowEnd) ? W[(size_t)(row0 + {r}) * KW + j]"
                 f" : make_uint4(0x88888888u,0x88888888u,0x88888888u,0x88888888u);")
        L.append(f"  float s{r} = (row0 + {r} < rowEnd) ? __half2float(reinterpret_cast<const __half*>(S)"
                 f"[(size_t)(row0 + {r}) * (K / 32) + j]) : 0.f;")
    L.append("  float part[4][B_];")
    L.append("  #pragma unroll")
    L.append("  for (int r = 0; r < 4; ++r)")
    L.append("    for (int b = 0; b < B_; ++b) part[r][b] = 0.f;")
    for k in range(4):
        # x values for word k of block j, all B vectors: 8 bf16 = one uint4 per vector
        L.append("  {")
        L.append(f"    float xv[B_][8];")
        L.append("    #pragma unroll")
        L.append("    for (int b = 0; b < B_; ++b) {")
        L.append(f"      uint4 t = X[(size_t)b * (K / 8) + (size_t)j * 4 + {k}];")
        L.append("      xv[b][0] = __uint_as_float(t.x << 16); xv[b][1] = __uint_as_float(t.x & 0xffff0000u);")
        L.append("      xv[b][2] = __uint_as_float(t.y << 16); xv[b][3] = __uint_as_float(t.y & 0xffff0000u);")
        L.append("      xv[b][4] = __uint_as_float(t.z << 16); xv[b][5] = __uint_as_float(t.z & 0xffff0000u);")
        L.append("      xv[b][6] = __uint_as_float(t.w << 16); xv[b][7] = __uint_as_float(t.w & 0xffff0000u);")
        L.append("    }")
        for r in range(4):
            q = f"w{r}.{'xyzw'[k]}"
            L.append("    {")
            for i in range(8):
                L.append(f"      float d{i} = __int_as_float((({q} >> {4 * i}) & 15) | 0x4B000000) - 8388616.0f;")
            L.append("      #pragma unroll")
            L.append("      for (int b = 0; b < B_; ++b) {")
            L.append(f"        float p = part[{r}][b];")
            for i in range(8):
                L.append(f"        p = fmaf(d{i}, xv[b][{i}], p);")
            L.append(f"        part[{r}][b] = p;")
            L.append("      }")
            L.append("    }")
        L.append("  }")
    L.append("  #pragma unroll")
    L.append("  for (int b = 0; b < B_; ++b) {")
    for r in range(4):
        L.append(f"    acc[{r}][b] = fmaf(s{r}, part[{r}][b], acc[{r}][b]);")
    L.append("  }")
    L.append("}")
    body = "\n      ".join(L)
    if epi == "swiglu":
        fin = """
  if (sub < 2 * B_) {
    int b = sub >> 1, h = sub & 1;
    float g = 0.f, u = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) { g += red[s][2 * h][b]; u += red[s][2 * h + 1][b]; }
    ((__nv_bfloat16*)Y)[(size_t)b * (N / 2) + row0 / 2 + h] = __float2bfloat16_rn(g / (1.f + __expf(-g)) * u);
  }"""
    else:
        store = {
            "store": "((__nv_bfloat16*)Y)[(size_t)b * N + r] = __float2bfloat16_rn(a);",
            "biasf": "((__nv_bfloat16*)Y)[(size_t)b * N + r] = __float2bfloat16_rn(a + ((const float*)AUX)[r]);",
            "resid": "((float*)Y)[(size_t)b * N + r] += a;",
        }[epi]
        fin = f"""
  if (sub < 4 * B_) {{
    int rr = sub & 3, b = sub >> 2, r = row0 + rr;
    if (r < rowEnd) {{
      float a = 0.f;
      #pragma unroll
      for (int s = 0; s < SPLIT; ++s) a += red[s][rr][b];
      {store}
    }}
  }}"""
    assert 4 * B <= 32 * S_, "epilogue needs 4*B threads"
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define SPLIT {S_}
#define B_ {B}
extern "C" __global__ void __launch_bounds__({32 * S_}, {minb}) k(
    const uint4* __restrict__ W, const void* __restrict__ S, const uint4* __restrict__ X,
    void* Y, const void* AUX, int N, int K)
{{
  __shared__ float red[SPLIT][4][B_];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, sub = threadIdx.x;
  int KW = K / 32;
  int row0 = blockIdx.x * 4, rowEnd = min(N, row0 + 4);
  int chunk = (KW + SPLIT - 1) / SPLIT;
  int kb = warp * chunk, ke = min(KW, kb + chunk);
  float acc[4][B_];
  #pragma unroll
  for (int r = 0; r < 4; ++r)
    for (int b = 0; b < B_; ++b) acc[r][b] = 0.f;
  if (row0 < N) {{
      {body}
  }}
  #pragma unroll
  for (int r = 0; r < 4; ++r) {{
    #pragma unroll
    for (int b = 0; b < B_; ++b) {{
      float a = acc[r][b];
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) red[warp][r][b] = a;
    }}
  }}
  __syncthreads();
  if (row0 < N) {{{fin}
  }}
}}
"""


def gemv_mma_source(S_: int, B: int, epi: str = "store", U: int = 1) -> str:
    """Tensor-core Q4_0 x [B, K] for batched decode (mma.sync m16n8k16, fp16 in, fp32 accumulate).

    A block = S warps on one 16-row tile, splitting K. Per 16-wide k-step each lane needs, for rows
    g and g+8 (g = lane / 4) and t = lane % 4, the weight pairs k = 2t, 2t+1 and 2t+8, 2t+9: nibbles
    2t, 2t+1 of words 2s and 2s+1 of the packed row. They become half2 via the 0x6400 exponent trick
    ((1024 + q) - 1032 = q - 8), times the block's fp16 scale. B operand: x[col = g][k], bf16 -> fp16,
    zero for columns >= B. NT = ceil(B / 8) n-tiles reuse each A fragment."""
    NT = (B + 7) // 8
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define SPLIT {S_}
#define B_ {B}
#define NT {NT}
__device__ __forceinline__ unsigned h2pair(unsigned w, int t, unsigned d2) {{
  // nibbles 2t, 2t+1 of w -> half2 (q0 - 8, q1 - 8) * d
  unsigned lo = (w >> (8 * t)) & 0xFu, hi = (w >> (8 * t + 4)) & 0xFu;
  unsigned h = lo | (hi << 16) | 0x64006400u;
  __half2 v = __hsub2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(1032.f));
  v = __hmul2(v, *reinterpret_cast<__half2*>(&d2));
  return *reinterpret_cast<unsigned*>(&v);
}}
__device__ __forceinline__ unsigned xpair(const __nv_bfloat16* x) {{
  __half2 v = __floats2half2_rn(__bfloat162float(x[0]), __bfloat162float(x[1]));
  return *reinterpret_cast<unsigned*>(&v);
}}
extern "C" __global__ void __launch_bounds__({32 * S_}) k(
    const unsigned* __restrict__ W, const __half* __restrict__ S, const __nv_bfloat16* __restrict__ X,
    void* Y, const void* AUX, int N, int K)
{{
  __shared__ float red[SPLIT][16][NT * 8];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int row0 = blockIdx.x * 16;
  int KW = K / 8;                                  // packed words per row
  int KB = K / 32;                                 // Q4_0 blocks per row
  int chunk = (KB + SPLIT - 1) / SPLIT;
  int bb = warp * chunk, be = min(KB, bb + chunk);
  float c[NT][4];
  #pragma unroll
  for (int n = 0; n < NT; ++n) c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
  int ra = min(row0 + g, N - 1), rb = min(row0 + g + 8, N - 1);
  const unsigned* wa = W + (size_t)ra * KW;
  const unsigned* wb = W + (size_t)rb * KW;
  // U Q4_0 blocks per iteration: their weight loads are all in flight before the first mma
  #pragma unroll 1
  for (int blk0 = bb; blk0 < be; blk0 += {U}) {{
    uint4 A0[{U}], A1[{U}];
    unsigned dA[{U}], dB[{U}];
    #pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      int blk = min(blk0 + u, be - 1);
      A0[u] = *reinterpret_cast<const uint4*>(wa + blk * 4);
      A1[u] = *reinterpret_cast<const uint4*>(wb + blk * 4);
      __half da = S[(size_t)ra * KB + blk], db = S[(size_t)rb * KB + blk];
      __half2 da2 = __halves2half2(da, da), db2 = __halves2half2(db, db);
      dA[u] = *reinterpret_cast<unsigned*>(&da2); dB[u] = *reinterpret_cast<unsigned*>(&db2);
    }}
    #pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      if (blk0 + u >= be) break;
      int blk = blk0 + u;
      #pragma unroll
      for (int s = 0; s < 2; ++s) {{                // two 16-wide k-steps per Q4_0 block
        unsigned w0a = s ? A0[u].z : A0[u].x, w1a = s ? A0[u].w : A0[u].y;
        unsigned w0b = s ? A1[u].z : A1[u].x, w1b = s ? A1[u].w : A1[u].y;
        unsigned a0 = h2pair(w0a, t, dA[u]), a1 = h2pair(w0b, t, dB[u]);
        unsigned a2 = h2pair(w1a, t, dA[u]), a3 = h2pair(w1b, t, dB[u]);
        int kk = blk * 32 + s * 16 + 2 * t;
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          int col = n * 8 + g;
          unsigned b0 = 0u, b1 = 0u;
          if (col < B_) {{
            const __nv_bfloat16* xr = X + (size_t)col * K + kk;
            b0 = xpair(xr); b1 = xpair(xr + 8);
          }}
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
                       : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3])
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }}
      }}
    }}
  }}
  #pragma unroll
  for (int n = 0; n < NT; ++n) {{
    red[warp][g][n * 8 + 2 * t] = c[n][0];      red[warp][g][n * 8 + 2 * t + 1] = c[n][1];
    red[warp][g + 8][n * 8 + 2 * t] = c[n][2];  red[warp][g + 8][n * 8 + 2 * t + 1] = c[n][3];
  }}
  __syncthreads();
  for (int i = threadIdx.x; i < 16 * B_; i += blockDim.x) {{
    int rr = i % 16, b = i / 16, r = row0 + rr;
    float a = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) a += red[s][rr][b];
    {_mma_epi(epi)}
  }}
}}
"""


def _mma_epi(epi: str) -> str:
    if epi == "swiglu":     # rows interleaved g, u: even rr = gate, odd = up; one output per pair
        return """if ((rr & 1) == 0 && r + 1 < N + 1 && r < N) {
      float u = 0.f;
      #pragma unroll
      for (int s = 0; s < SPLIT; ++s) u += red[s][rr + 1][b];
      ((__nv_bfloat16*)Y)[(size_t)b * (N / 2) + r / 2] = __float2bfloat16_rn(a / (1.f + __expf(-a)) * u);
    }"""
    store = {
        "store": "((__nv_bfloat16*)Y)[(size_t)b * N + r] = __float2bfloat16_rn(a);",
        "biasf": "((__nv_bfloat16*)Y)[(size_t)b * N + r] = __float2bfloat16_rn(a + ((const float*)AUX)[r]);",
        "resid": "((float*)Y)[(size_t)b * N + r] += a;",
    }[epi]
    return f"if (r < N) {{ {store} }}"
