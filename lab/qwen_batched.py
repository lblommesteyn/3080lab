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


def gemv_mma_source(S_: int, B: int, epi: str = "store", U: int = 1, MT: int = 1) -> str:
    """Tensor-core Q4_0 x [B, K] for batched decode (mma.sync m16n8k16, fp16 in, fp32 accumulate).

    A block = S warps on one tile of 16*MT rows, splitting K. Per 16-wide k-step each lane needs, for
    rows g and g+8 of every m-tile (g = lane / 4, t = lane % 4), the weight pairs k = 2t, 2t+1 and
    2t+8, 2t+9: nibbles 2t, 2t+1 of words 2s and 2s+1 of the packed row. They become half2 via the
    0x6400 exponent trick ((1024 + q) - 1032 = q - 8), times the block's fp16 scale. B operand:
    x[col = g][k], bf16 -> fp16, zero for columns >= B. NT = ceil(B / 8) n-tiles reuse each A
    fragment; MT m-tiles reuse each B fragment (x traffic per weight byte / MT: at B = 32 a warp
    otherwise reads 2 KB of x per 256 B of weights). U Q4_0 blocks of weights are loaded ahead."""
    NT = (B + 7) // 8
    R = 16 * MT
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define SPLIT {S_}
#define B_ {B}
#define NT {NT}
#define MT {MT}
__device__ __forceinline__ unsigned h2pair(unsigned w, int t, unsigned d2) {{
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
  __shared__ float red[{R}][NT * 8];             // split-K partials, accumulated warp by warp (deterministic)
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int row0 = blockIdx.x * {R};
  int KW = K / 8, KB = K / 32;
  int chunk = (KB + SPLIT - 1) / SPLIT;
  int bb = warp * chunk, be = min(KB, bb + chunk);
  float c[MT][NT][4];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int n = 0; n < NT; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.f;
  const unsigned* wa[MT]; const unsigned* wb[MT]; int ra[MT], rb[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) {{
    ra[m] = min(row0 + 16 * m + g, N - 1); rb[m] = min(row0 + 16 * m + g + 8, N - 1);
    wa[m] = W + (size_t)ra[m] * KW; wb[m] = W + (size_t)rb[m] * KW;
  }}
  #pragma unroll 1
  for (int blk0 = bb; blk0 < be; blk0 += {U}) {{
    uint4 A0[{U}][MT], A1[{U}][MT];
    unsigned dA[{U}][MT], dB[{U}][MT];
    #pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      int blk = min(blk0 + u, be - 1);
      #pragma unroll
      for (int m = 0; m < MT; ++m) {{
        A0[u][m] = *reinterpret_cast<const uint4*>(wa[m] + blk * 4);
        A1[u][m] = *reinterpret_cast<const uint4*>(wb[m] + blk * 4);
        __half da = S[(size_t)ra[m] * KB + blk], db = S[(size_t)rb[m] * KB + blk];
        __half2 da2 = __halves2half2(da, da), db2 = __halves2half2(db, db);
        dA[u][m] = *reinterpret_cast<unsigned*>(&da2); dB[u][m] = *reinterpret_cast<unsigned*>(&db2);
      }}
    }}
    #pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      if (blk0 + u >= be) break;
      int blk = blk0 + u;
      #pragma unroll
      for (int s = 0; s < 2; ++s) {{                // two 16-wide k-steps per Q4_0 block
        unsigned a[MT][4];
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          unsigned w0a = s ? A0[u][m].z : A0[u][m].x, w1a = s ? A0[u][m].w : A0[u][m].y;
          unsigned w0b = s ? A1[u][m].z : A1[u][m].x, w1b = s ? A1[u][m].w : A1[u][m].y;
          a[m][0] = h2pair(w0a, t, dA[u][m]); a[m][1] = h2pair(w0b, t, dB[u][m]);
          a[m][2] = h2pair(w1a, t, dA[u][m]); a[m][3] = h2pair(w1b, t, dB[u][m]);
        }}
        int kk = blk * 32 + s * 16 + 2 * t;
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          int col = n * 8 + g;
          unsigned b0 = 0u, b1 = 0u;
          if (col < B_) {{
            const __nv_bfloat16* xr = X + (size_t)col * K + kk;
            b0 = xpair(xr); b1 = xpair(xr + 8);
          }}
          #pragma unroll
          for (int m = 0; m < MT; ++m)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
                         : "+f"(c[m][n][0]), "+f"(c[m][n][1]), "+f"(c[m][n][2]), "+f"(c[m][n][3])
                         : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(b0), "r"(b1));
        }}
      }}
    }}
  }}
  for (int s = 0; s < SPLIT; ++s) {{
    if (warp == s) {{
      #pragma unroll
      for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          float* r0 = &red[16 * m + g][n * 8 + 2 * t];
          float* r8 = &red[16 * m + g + 8][n * 8 + 2 * t];
          if (s == 0) {{ r0[0] = c[m][n][0]; r0[1] = c[m][n][1]; r8[0] = c[m][n][2]; r8[1] = c[m][n][3]; }}
          else {{ r0[0] += c[m][n][0]; r0[1] += c[m][n][1]; r8[0] += c[m][n][2]; r8[1] += c[m][n][3]; }}
        }}
    }}
    __syncthreads();
  }}
  for (int i = threadIdx.x; i < {R} * B_; i += blockDim.x) {{
    int rr = i % {R}, b = i / {R}, r = row0 + rr;
    float a = red[rr][b];
    {_mma_epi(epi)}
  }}
}}
"""


def _mma_epi(epi: str) -> str:
    if epi == "swiglu":     # rows interleaved g, u: even rr = gate, odd = up; one output per pair
        return """if ((rr & 1) == 0 && r + 1 < N + 1 && r < N) {
      float u = red[rr + 1][b];
      ((__nv_bfloat16*)Y)[(size_t)b * (N / 2) + r / 2] = __float2bfloat16_rn(a / (1.f + __expf(-a)) * u);
    }"""
    if epi == "swiglu16":   # as swiglu, fp16 output (input of a v3 GEMV)
        return """if ((rr & 1) == 0 && r < N) {
      float u = red[rr + 1][b];
      ((__half*)Y)[(size_t)b * (N / 2) + r / 2] = __float2half_rn(a / (1.f + __expf(-a)) * u);
    }"""
    store = {
        "store": "((__nv_bfloat16*)Y)[(size_t)b * N + r] = __float2bfloat16_rn(a);",
        "biasf": "((__nv_bfloat16*)Y)[(size_t)b * N + r] = __float2bfloat16_rn(a + ((const float*)AUX)[r]);",
        "resid": "((float*)Y)[(size_t)b * N + r] += a;",
    }[epi]
    return f"if (r < N) {{ {store} }}"


def head_q6_mma_source(S_: int, B: int, U: int = 1, MT: int = 1) -> str:
    """Tensor-core Q6_K LM head, Y fp32 [B, N] = W x, on qwen_gguf's Q6_K-exact layout:
      L  uint32 [N, K/8]   low 4 bits of q+32 (nibble i = weight 8w+i)
      Hb uint32 [N, K/16]  high 2 bits (field i = weight 16w+i)
      SC int8   [N, K/16]  sub-block scales;  D fp16 [N, K/256] super-block scales
    The A fragment holds the raw q - 32 (exact in fp16 via 0x6400: (1024 + q) - 1056); each 16-wide
    k-step (one Q6_K sub-block) goes into a zeroed accumulator, then c += D * SC * tmp in fp32 per
    row, so no scale is rounded to fp16. Same tiling as gemv_mma_source (S split-K warps, MT
    m-tiles, U 32-weight units in flight, deterministic shared-memory reduction)."""
    NT = (B + 7) // 8
    R = 16 * MT
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define SPLIT {S_}
#define B_ {B}
#define NT {NT}
#define MT {MT}
__device__ __forceinline__ unsigned q6pair(unsigned lw, unsigned hw, int t, int f) {{
  unsigned q0 = ((lw >> (8 * t)) & 0xFu) | (((hw >> (2 * f)) & 3u) << 4);
  unsigned q1 = ((lw >> (8 * t + 4)) & 0xFu) | (((hw >> (2 * f + 2)) & 3u) << 4);
  unsigned h = q0 | (q1 << 16) | 0x64006400u;
  __half2 v = __hsub2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(1056.f));
  return *reinterpret_cast<unsigned*>(&v);
}}
__device__ __forceinline__ unsigned xpair(const __nv_bfloat16* x) {{
  __half2 v = __floats2half2_rn(__bfloat162float(x[0]), __bfloat162float(x[1]));
  return *reinterpret_cast<unsigned*>(&v);
}}
extern "C" __global__ void __launch_bounds__({32 * S_}) k(
    const uint4* __restrict__ L, const uint2* __restrict__ Hb, const unsigned short* __restrict__ SC,
    const __half* __restrict__ D, const __nv_bfloat16* __restrict__ X, float* __restrict__ Y, int N, int K)
{{
  __shared__ float red[{R}][NT * 8];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int row0 = blockIdx.x * {R};
  int KU = K / 32, KD = K / 256;
  int chunk = (KU + SPLIT - 1) / SPLIT;
  int jb = warp * chunk, je = min(KU, jb + chunk);
  float c[MT][NT][4];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int n = 0; n < NT; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.f;
  int ra[MT], rb[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) {{ ra[m] = min(row0 + 16 * m + g, N - 1); rb[m] = min(row0 + 16 * m + g + 8, N - 1); }}
  #pragma unroll 1
  for (int j0 = jb; j0 < je; j0 += {U}) {{
    uint4 LA[{U}][MT], LB[{U}][MT];
    uint2 HA[{U}][MT], HB[{U}][MT];
    float sA[{U}][MT][2], sB[{U}][MT][2];
    #pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      int j = min(j0 + u, je - 1);
      #pragma unroll
      for (int m = 0; m < MT; ++m) {{
        LA[u][m] = L[(size_t)ra[m] * KU + j]; LB[u][m] = L[(size_t)rb[m] * KU + j];
        HA[u][m] = Hb[(size_t)ra[m] * KU + j]; HB[u][m] = Hb[(size_t)rb[m] * KU + j];
        unsigned short ca = SC[(size_t)ra[m] * KU + j], cb = SC[(size_t)rb[m] * KU + j];
        float da = __half2float(D[(size_t)ra[m] * KD + (j >> 3)]), db = __half2float(D[(size_t)rb[m] * KD + (j >> 3)]);
        sA[u][m][0] = da * (float)(signed char)(ca & 0xff); sA[u][m][1] = da * (float)(signed char)(ca >> 8);
        sB[u][m][0] = db * (float)(signed char)(cb & 0xff); sB[u][m][1] = db * (float)(signed char)(cb >> 8);
      }}
    }}
    #pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      if (j0 + u >= je) break;
      int j = j0 + u;
      #pragma unroll
      for (int s = 0; s < 2; ++s) {{                // two sub-blocks of 16 per unit
        unsigned a[MT][4];
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          unsigned l0a = s ? LA[u][m].z : LA[u][m].x, l1a = s ? LA[u][m].w : LA[u][m].y;
          unsigned l0b = s ? LB[u][m].z : LB[u][m].x, l1b = s ? LB[u][m].w : LB[u][m].y;
          unsigned ha = s ? HA[u][m].y : HA[u][m].x, hb = s ? HB[u][m].y : HB[u][m].x;
          a[m][0] = q6pair(l0a, ha, t, 2 * t); a[m][1] = q6pair(l0b, hb, t, 2 * t);
          a[m][2] = q6pair(l1a, ha, t, 2 * t + 8); a[m][3] = q6pair(l1b, hb, t, 2 * t + 8);
        }}
        int kk = j * 32 + s * 16 + 2 * t;
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          int col = n * 8 + g;
          unsigned b0 = 0u, b1 = 0u;
          if (col < B_) {{
            const __nv_bfloat16* xr = X + (size_t)col * K + kk;
            b0 = xpair(xr); b1 = xpair(xr + 8);
          }}
          #pragma unroll
          for (int m = 0; m < MT; ++m) {{
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
                         : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                         : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(b0), "r"(b1));
            c[m][n][0] = fmaf(sA[u][m][s], d0, c[m][n][0]); c[m][n][1] = fmaf(sA[u][m][s], d1, c[m][n][1]);
            c[m][n][2] = fmaf(sB[u][m][s], d2, c[m][n][2]); c[m][n][3] = fmaf(sB[u][m][s], d3, c[m][n][3]);
          }}
        }}
      }}
    }}
  }}
  for (int s = 0; s < SPLIT; ++s) {{
    if (warp == s) {{
      #pragma unroll
      for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          float* r0 = &red[16 * m + g][n * 8 + 2 * t];
          float* r8 = &red[16 * m + g + 8][n * 8 + 2 * t];
          if (s == 0) {{ r0[0] = c[m][n][0]; r0[1] = c[m][n][1]; r8[0] = c[m][n][2]; r8[1] = c[m][n][3]; }}
          else {{ r0[0] += c[m][n][0]; r0[1] += c[m][n][1]; r8[0] += c[m][n][2]; r8[1] += c[m][n][3]; }}
        }}
    }}
    __syncthreads();
  }}
  for (int i = threadIdx.x; i < {R} * B_; i += blockDim.x) {{
    int rr = i % {R}, b = i / {R}, r = row0 + rr;
    if (r < N) Y[(size_t)b * N + r] = red[rr][b];
  }}
}}
"""


def pack_q4_mma(W, S):
    """Repack Q4_0 (W int32 [N, K/8], nibble i of word w = weight 8w+i; S fp16 [N, K/32]) into
    m16n8k16 fragment order for gemv_v2_source. Tile = 16 rows x 64 k (two Q4_0 blocks, four
    k-steps); lane (g, t) gets one uint4 whose word s holds, for k0 = 16s + 2t,
      byte 0: row g k0 | row g+8 k0 << 4,     byte 1: same at k0 + 8,
      byte 2: same at k0 + 1,                  byte 3: same at k0 + 9,
    so (w >> 4j) & 0x000F000F is A register j as a nibble half2 pair. Scales: [tile][chunk][g][4]
    fp16 = (row g blk0, row g+8 blk0, row g blk1, row g+8 blk1); the 4 lanes of a row group read
    the same 8 bytes (broadcast), so no scale bytes are duplicated in memory.
    Returns (Wp int32 [N/16, K/64, 32, 4], Sp fp16 [N/16, K/64, 8, 4]). Same bytes as Q4_0."""
    import torch
    N, KW = W.shape
    K = KW * 8
    T, C = N // 16, K // 64
    w = W.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(w >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K)
    Wp = _i32(_frag_words(q))
    Sv = S.reshape(T, 16, C, 2)
    Sp = torch.stack([Sv[:, :8, :, 0], Sv[:, 8:, :, 0], Sv[:, :8, :, 1], Sv[:, 8:, :, 1]], -1)  # [T, 8, C, 4]
    Sp = Sp.permute(0, 2, 1, 3).contiguous()
    return Wp, Sp


def _frag_words(q):
    """q int64 [N, K] of 4-bit values -> int64 [N/16, K/64, 32, 4] fragment-order words (see pack_q4_mma)."""
    import torch
    N, K = q.shape
    T, C = N // 16, K // 64
    Q = q.reshape(T, 16, C, 64)
    top, bot = Q[:, :8], Q[:, 8:]                                   # [T, 8(g), C, 64]
    s = torch.arange(4, device=q.device)[:, None]
    t = torch.arange(4, device=q.device)[None, :]
    k0 = (16 * s + 2 * t).reshape(-1)                               # (s, t) flattened
    word = torch.zeros(T, 8, C, 16, dtype=torch.int64, device=q.device)
    for byte, off in enumerate((0, 8, 1, 9)):
        word |= (top[..., k0 + off] | (bot[..., k0 + off] << 4)) << (8 * byte)
    word = word.reshape(T, 8, C, 4, 4).permute(0, 2, 1, 4, 3)       # [T, C, g, t, s]
    return word.reshape(T, C, 32, 4)


def _i32(word):
    import torch
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).contiguous()


def pack_q6_mma(L, Hb, SC, D):
    """Repack the Q6_K-exact head from qwen_gguf (L, Hb, SC, D; see head_q6_mma_source) into fragment
    order for head_v2_source, same 6.5625 bits per weight:
      Lp  int32 [T, C, 32, 4]  low nibbles, exactly the pack_q4_mma layout
      Hp  int32 [T, C, 32, 2]  high 2-bit fields in the same nibble slots; word 0 holds k-steps 0
                               (bits 0-1 of each slot) and 1 (bits 2-3), word 1 k-steps 2 and 3
      SCp int8  [T, C, 8, 8]   sub-block scales: row g k-steps 0-3, then row g+8
      Dp  fp16  [T, K/256, 8, 2] super-block scales of rows g, g+8"""
    import torch
    N = L.shape[0]
    K = L.shape[1] * 8
    T, C = N // 16, K // 64
    lw = L.to(torch.int64) & 0xFFFFFFFF
    hw = Hb.to(torch.int64) & 0xFFFFFFFF
    lo = torch.stack([(lw >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K)
    hi = torch.stack([(hw >> (2 * i)) & 3 for i in range(16)], -1).reshape(N, K)
    Lp = _i32(_frag_words(lo))
    hf = _frag_words(hi)
    Hp = _i32(torch.stack([hf[..., 0] | (hf[..., 1] << 2), hf[..., 2] | (hf[..., 3] << 2)], -1))
    sc = SC.reshape(T, 16, C, 4)
    SCp = torch.cat([sc[:, :8], sc[:, 8:]], -1).permute(0, 2, 1, 3).contiguous()    # [T, C, 8, 8]
    d = D.reshape(T, 16, K // 256)
    Dp = torch.stack([d[:, :8], d[:, 8:]], -1).permute(0, 2, 1, 3).contiguous()     # [T, K/256, 8, 2]
    return Lp, Hp, SCp, Dp


def gemv_v2_source(WM: int, WK: int, MT: int, B: int, epi: str = "store", PF: int = 1, KS: int = 1,
                   ACC16: bool = False) -> str:
    """Tensor-core Q4_0 x [B, K] on pack_q4_mma weights, for batched decode.

    Block = WM x WK warps; R = 16 * MT * WM rows. Each iteration the block stages
    X[:, k0 : k0 + 64 * WK] (bf16 -> fp16, once per block) in shared memory; warp (wm, wk) takes the
    64-k chunk wk for its MT m-tiles and reads B fragments with ldmatrix.x4, so the input is read
    from L2 once per R rows instead of once per 16. Weights stream straight to registers, one
    coalesced uint4 per lane per 16 x 64 tile, PF iterations ahead. Split-K partials (WK) are
    reduced in a fixed order (deterministic). KS > 1 also splits K across gridDim.y = KS blocks
    (for short-N shapes like down, 96 row tiles on 68 SMs): each writes its partial to WS
    [KS][B][N] fp32, and the last block of a row tile (per-tile counter in CNT, self-resetting)
    sums the KS partials in a fixed order and applies the epilogue. ACC16: the MMAs accumulate one
    64-k chunk in fp16 (twice the fp32-accumulate rate on GA102), added to the fp32 sums per chunk."""
    NT = (B + 7) // 8
    R = 16 * MT * WM
    KC = 64 * WK
    XS = KC + 8                                                     # row stride (halves): 4-bank skew
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define B_ {B}
#define NT {NT}
#define MT {MT}
#define WM {WM}
#define WK {WK}
#define PF {PF}
#define ACC16 {int(ACC16)}
__device__ __forceinline__ unsigned dq(unsigned w, int j, unsigned d2) {{
  unsigned h = ((w >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
  __half2 v = __hsub2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(1032.f));
  v = __hmul2(v, *reinterpret_cast<__half2*>(&d2));
  return *reinterpret_cast<unsigned*>(&v);
}}
extern "C" __global__ void __launch_bounds__({32 * WM * WK}) k(
    const uint4* __restrict__ W, const uint2* __restrict__ S, const __nv_bfloat16* __restrict__ X,
    void* Y, const void* AUX, int N, int K, float* __restrict__ WS, int* __restrict__ CNT)
{{
  __shared__ __align__(16) __half xs[NT * 8][{XS}];
  __shared__ float red[{R}][NT * 8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp % WM, wk = warp / WM;
  const int nthr = 32 * WM * WK;
  const int row0 = blockIdx.x * {R};
  const int C = K / 64, T = N / 16;
  const int Cy = (C + {KS} - 1) / {KS}, cb = blockIdx.y * Cy, ce = min(C, cb + Cy);
  const int iters = (ce - cb + WK - 1) / WK;
  for (int i = threadIdx.x; i < (NT * 8 - B_) * {XS}; i += nthr) xs[B_ + i / {XS}][i % {XS}] = __float2half(0.f);
  int tile[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) tile[m] = min(row0 / 16 + wm * MT + m, T - 1);
  float c[MT][NT][4];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int n = 0; n < NT; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.f;
  uint4 wq[PF + 1][MT]; uint2 sq[PF + 1][MT];
  #pragma unroll
  for (int p = 0; p < PF; ++p) {{
    int ch = min(cb + p * WK + wk, ce - 1);
    #pragma unroll
    for (int m = 0; m < MT; ++m) {{
      wq[p][m] = W[((size_t)tile[m] * C + ch) * 32 + lane];
      sq[p][m] = S[((size_t)tile[m] * C + ch) * 8 + g];
    }}
  }}
  #pragma unroll 1
  for (int it = 0; it < iters; it += PF + 1) {{
    #pragma unroll
    for (int p = 0; p <= PF; ++p) {{
      const int itp = it + p;
      if (itp >= iters) break;
      {{
        int ch = min(cb + (itp + PF) * WK + wk, ce - 1);
        const int slot = (p + PF) % (PF + 1);
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          wq[slot][m] = W[((size_t)tile[m] * C + ch) * 32 + lane];
          sq[slot][m] = S[((size_t)tile[m] * C + ch) * 8 + g];
        }}
      }}
      const int k0 = (cb + itp * WK) * 64;
      __syncthreads();
      for (int i = threadIdx.x; i < B_ * {KC // 8}; i += nthr) {{
        int b = i / {KC // 8}, kk = (i % {KC // 8}) * 8;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (k0 + kk < K) v = *reinterpret_cast<const uint4*>(X + (size_t)b * K + k0 + kk);
        unsigned in[4] = {{v.x, v.y, v.z, v.w}}, out[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) {{
          __half2 h = __floats2half2_rn(__uint_as_float(in[e] << 16), __uint_as_float(in[e] & 0xffff0000u));
          out[e] = *reinterpret_cast<unsigned*>(&h);
        }}
        *reinterpret_cast<uint4*>(&xs[b][kk]) = make_uint4(out[0], out[1], out[2], out[3]);
      }}
      __syncthreads();
      if (cb + itp * WK + wk < ce) {{
        unsigned dA[MT][2], dB[MT][2];
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          unsigned lo = sq[p][m].x, hi = sq[p][m].y;
          dA[m][0] = __byte_perm(lo, 0, 0x1010); dB[m][0] = __byte_perm(lo, 0, 0x3232);
          dA[m][1] = __byte_perm(hi, 0, 0x1010); dB[m][1] = __byte_perm(hi, 0, 0x3232);
        }}
#if ACC16
        unsigned hc[MT][NT][2];
        #pragma unroll
        for (int m = 0; m < MT; ++m)
          #pragma unroll
          for (int n = 0; n < NT; ++n) hc[m][n][0] = hc[m][n][1] = 0u;
#endif
        #pragma unroll
        for (int h = 0; h < 2; ++h) {{
          unsigned bf[NT][4];
          #pragma unroll
          for (int n = 0; n < NT; ++n) {{
            unsigned addr = (unsigned)__cvta_generic_to_shared(&xs[n * 8 + (lane & 7)][wk * 64 + h * 32 + 8 * (lane >> 3)]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];"
                         : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(addr));
          }}
          #pragma unroll
          for (int ss = 0; ss < 2; ++ss) {{
            const int s = 2 * h + ss;
            unsigned a[MT][4];
            #pragma unroll
            for (int m = 0; m < MT; ++m) {{
              unsigned w = s == 0 ? wq[p][m].x : s == 1 ? wq[p][m].y : s == 2 ? wq[p][m].z : wq[p][m].w;
              a[m][0] = dq(w, 0, dA[m][h]); a[m][1] = dq(w, 1, dB[m][h]);
              a[m][2] = dq(w, 2, dA[m][h]); a[m][3] = dq(w, 3, dB[m][h]);
            }}
            #pragma unroll
            for (int n = 0; n < NT; ++n)
              #pragma unroll
              for (int m = 0; m < MT; ++m)
#if ACC16
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{%0,%1}}, {{%2,%3,%4,%5}}, {{%6,%7}}, {{%0,%1}};"
                             : "+r"(hc[m][n][0]), "+r"(hc[m][n][1])
                             : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(bf[n][2 * ss]), "r"(bf[n][2 * ss + 1]));
#else
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
                             : "+f"(c[m][n][0]), "+f"(c[m][n][1]), "+f"(c[m][n][2]), "+f"(c[m][n][3])
                             : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(bf[n][2 * ss]), "r"(bf[n][2 * ss + 1]));
#endif
          }}
        }}
#if ACC16
        #pragma unroll
        for (int m = 0; m < MT; ++m)
          #pragma unroll
          for (int n = 0; n < NT; ++n) {{
            float2 lo = __half22float2(*reinterpret_cast<__half2*>(&hc[m][n][0]));
            float2 hi = __half22float2(*reinterpret_cast<__half2*>(&hc[m][n][1]));
            c[m][n][0] += lo.x; c[m][n][1] += lo.y; c[m][n][2] += hi.x; c[m][n][3] += hi.y;
          }}
#endif
      }}
    }}
  }}
  for (int s = 0; s < WK; ++s) {{
    if (wk == s) {{
      #pragma unroll
      for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          float* r0 = &red[16 * (wm * MT + m) + g][n * 8 + 2 * t];
          float* r8 = &red[16 * (wm * MT + m) + g + 8][n * 8 + 2 * t];
          if (s == 0) {{ r0[0] = c[m][n][0]; r0[1] = c[m][n][1]; r8[0] = c[m][n][2]; r8[1] = c[m][n][3]; }}
          else {{ r0[0] += c[m][n][0]; r0[1] += c[m][n][1]; r8[0] += c[m][n][2]; r8[1] += c[m][n][3]; }}
        }}
    }}
    __syncthreads();
  }}
{_v2_tail(R, KS, epi)}
}}
"""


def _v2_tail(R: int, KS: int, epi: str) -> str:
    if KS == 1:
        return f"""  for (int i = threadIdx.x; i < {R} * B_; i += nthr) {{
    int rr = i % {R}, b = i / {R}, r = row0 + rr;
    float a = red[rr][b];
    {_mma_epi(epi)}
  }}"""
    # swiglu reads red[rr + 1][b] in the epilogue: the last block rebuilds red from the KS partials
    return f"""  for (int i = threadIdx.x; i < {R} * B_; i += nthr) {{
    int rr = i % {R}, b = i / {R}, r = row0 + rr;
    if (r < N) WS[((size_t)blockIdx.y * B_ + b) * N + r] = red[rr][b];
  }}
  __shared__ int last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {{
    last = atomicAdd(&CNT[blockIdx.x], 1) == {KS} - 1;
    if (last) CNT[blockIdx.x] = 0;
  }}
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int i = threadIdx.x; i < {R} * B_; i += nthr) {{
    int rr = i % {R}, b = i / {R}, r = min(row0 + rr, N - 1);
    float a = 0.f;
    #pragma unroll
    for (int y = 0; y < {KS}; ++y) a += __ldcg(&WS[((size_t)y * B_ + b) * N + r]);
    red[rr][b] = a;
  }}
  __syncthreads();
  for (int i = threadIdx.x; i < {R} * B_; i += nthr) {{
    int rr = i % {R}, b = i / {R}, r = row0 + rr;
    float a = red[rr][b];
    {_mma_epi(epi)}
  }}"""


def head_v2_source(WM: int, WK: int, MT: int, B: int, PF: int = 1) -> str:
    """Q6_K LM head on pack_q6_mma weights, with the gemv_v2_source structure (shared-memory fp16
    inputs via ldmatrix, fragment-order weights streamed PF iterations ahead, WK split-K warps
    reduced in a fixed order). A = raw q - 32 (exact); each k-step is one Q6_K sub-block, MMA'd
    into a zeroed accumulator and added with D * SC in fp32 (as head_q6_mma_source). Y fp32 [B, N]."""
    NT = (B + 7) // 8
    R = 16 * MT * WM
    KC = 64 * WK
    XS = KC + 8
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define B_ {B}
#define NT {NT}
#define MT {MT}
#define WM {WM}
#define WK {WK}
#define PF {PF}
__device__ __forceinline__ unsigned dq6(unsigned lw, unsigned hw, int j) {{
  unsigned h = ((lw >> (4 * j)) & 0x000F000Fu) | (((hw >> (4 * j)) & 0x00030003u) << 4) | 0x64006400u;
  __half2 v = __hsub2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(1056.f));
  return *reinterpret_cast<unsigned*>(&v);
}}
extern "C" __global__ void __launch_bounds__({32 * WM * WK}) k(
    const uint4* __restrict__ L, const uint2* __restrict__ H, const uint2* __restrict__ SC,
    const __half2* __restrict__ D, const __nv_bfloat16* __restrict__ X, float* __restrict__ Y, int N, int K)
{{
  __shared__ __align__(16) __half xs[NT * 8][{XS}];
  __shared__ float red[{R}][NT * 8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp % WM, wk = warp / WM;
  const int nthr = 32 * WM * WK;
  const int row0 = blockIdx.x * {R};
  const int C = K / 64, T = N / 16, KD = K / 256;
  const int iters = (C + WK - 1) / WK;
  for (int i = threadIdx.x; i < (NT * 8 - B_) * {XS}; i += nthr) xs[B_ + i / {XS}][i % {XS}] = __float2half(0.f);
  int tile[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) tile[m] = min(row0 / 16 + wm * MT + m, T - 1);
  float c[MT][NT][4];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int n = 0; n < NT; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.f;
  uint4 lq[PF + 1][MT]; uint2 hq[PF + 1][MT], sq[PF + 1][MT]; __half2 dd[PF + 1][MT];
  #pragma unroll
  for (int p = 0; p < PF; ++p) {{
    int ch = min(p * WK + wk, C - 1);
    #pragma unroll
    for (int m = 0; m < MT; ++m) {{
      size_t o = (size_t)tile[m] * C + ch;
      lq[p][m] = L[o * 32 + lane]; hq[p][m] = H[o * 32 + lane]; sq[p][m] = SC[o * 8 + g];
      dd[p][m] = D[((size_t)tile[m] * KD + (ch >> 2)) * 8 + g];
    }}
  }}
  #pragma unroll 1
  for (int it = 0; it < iters; it += PF + 1) {{
    #pragma unroll
    for (int p = 0; p <= PF; ++p) {{
      const int itp = it + p;
      if (itp >= iters) break;
      {{
        int ch = min((itp + PF) * WK + wk, C - 1);
        const int slot = (p + PF) % (PF + 1);
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          size_t o = (size_t)tile[m] * C + ch;
          lq[slot][m] = L[o * 32 + lane]; hq[slot][m] = H[o * 32 + lane]; sq[slot][m] = SC[o * 8 + g];
          dd[slot][m] = D[((size_t)tile[m] * KD + (ch >> 2)) * 8 + g];
        }}
      }}
      const int k0 = itp * {KC};
      __syncthreads();
      for (int i = threadIdx.x; i < B_ * {KC // 8}; i += nthr) {{
        int b = i / {KC // 8}, kk = (i % {KC // 8}) * 8;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (k0 + kk < K) v = *reinterpret_cast<const uint4*>(X + (size_t)b * K + k0 + kk);
        unsigned in[4] = {{v.x, v.y, v.z, v.w}}, out[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) {{
          __half2 h = __floats2half2_rn(__uint_as_float(in[e] << 16), __uint_as_float(in[e] & 0xffff0000u));
          out[e] = *reinterpret_cast<unsigned*>(&h);
        }}
        *reinterpret_cast<uint4*>(&xs[b][kk]) = make_uint4(out[0], out[1], out[2], out[3]);
      }}
      __syncthreads();
      if (itp * WK + wk < C) {{
        float sA[MT][4], sB[MT][4];
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          float da = __low2float(dd[p][m]), db = __high2float(dd[p][m]);
          unsigned x0 = sq[p][m].x, x1 = sq[p][m].y;
          #pragma unroll
          for (int s = 0; s < 4; ++s) {{
            sA[m][s] = da * (float)(signed char)(x0 >> (8 * s));
            sB[m][s] = db * (float)(signed char)(x1 >> (8 * s));
          }}
        }}
        #pragma unroll
        for (int h = 0; h < 2; ++h) {{
          unsigned bf[NT][4];
          #pragma unroll
          for (int n = 0; n < NT; ++n) {{
            unsigned addr = (unsigned)__cvta_generic_to_shared(&xs[n * 8 + (lane & 7)][wk * 64 + h * 32 + 8 * (lane >> 3)]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];"
                         : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(addr));
          }}
          #pragma unroll
          for (int ss = 0; ss < 2; ++ss) {{
            const int s = 2 * h + ss;
            unsigned a[MT][4];
            #pragma unroll
            for (int m = 0; m < MT; ++m) {{
              unsigned lw = s == 0 ? lq[p][m].x : s == 1 ? lq[p][m].y : s == 2 ? lq[p][m].z : lq[p][m].w;
              unsigned hw = (h ? hq[p][m].y : hq[p][m].x) >> (2 * ss);
              a[m][0] = dq6(lw, hw, 0); a[m][1] = dq6(lw, hw, 1); a[m][2] = dq6(lw, hw, 2); a[m][3] = dq6(lw, hw, 3);
            }}
            #pragma unroll
            for (int n = 0; n < NT; ++n)
              #pragma unroll
              for (int m = 0; m < MT; ++m) {{
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
                             : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                             : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(bf[n][2 * ss]), "r"(bf[n][2 * ss + 1]));
                c[m][n][0] = fmaf(sA[m][s], d0, c[m][n][0]); c[m][n][1] = fmaf(sA[m][s], d1, c[m][n][1]);
                c[m][n][2] = fmaf(sB[m][s], d2, c[m][n][2]); c[m][n][3] = fmaf(sB[m][s], d3, c[m][n][3]);
              }}
          }}
        }}
      }}
    }}
  }}
  for (int s = 0; s < WK; ++s) {{
    if (wk == s) {{
      #pragma unroll
      for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          float* r0 = &red[16 * (wm * MT + m) + g][n * 8 + 2 * t];
          float* r8 = &red[16 * (wm * MT + m) + g + 8][n * 8 + 2 * t];
          if (s == 0) {{ r0[0] = c[m][n][0]; r0[1] = c[m][n][1]; r8[0] = c[m][n][2]; r8[1] = c[m][n][3]; }}
          else {{ r0[0] += c[m][n][0]; r0[1] += c[m][n][1]; r8[0] += c[m][n][2]; r8[1] += c[m][n][3]; }}
        }}
    }}
    __syncthreads();
  }}
  for (int i = threadIdx.x; i < {R} * B_; i += nthr) {{
    int rr = i % {R}, b = i / {R}, r = row0 + rr;
    if (r < N) Y[(size_t)b * N + r] = red[rr][b];
  }}
}}
"""


def gemv_v3_source(WM: int, WK: int, MT: int, B: int, epi: str = "store", NS: int = 3, KS: int = 1) -> str:
    """gemv_v2_source with an asynchronous pipeline: the input X is fp16 already (the producers
    write fp16), and both X chunks and the fragment-packed weights go global -> shared memory with
    cp.async through an NS-stage ring, NS - 1 iterations ahead, one barrier per iteration. v2
    ablations: staging (load + bf16 -> fp16 + store) was a third of the B = 32 time, and the
    synchronous weight loads most of the B = 8 time, while deeper register prefetch made it worse.
    Same arguments as v2 (X is const __half*)."""
    NT = (B + 7) // 8
    R = 16 * MT * WM
    KC = 64 * WK
    XS = KC + 8
    nw = WM * WK
    smem = NS * (NT * 8 * XS * 2 + nw * MT * (512 + 64)) + R * NT * 8 * 4
    assert smem <= 48 * 1024, f"static shared memory {smem} B > 48 KB"
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define B_ {B}
#define NT {NT}
#define MT {MT}
#define WM {WM}
#define WK {WK}
#define NS {NS}
__device__ __forceinline__ unsigned dq(unsigned w, int j, unsigned d2) {{
  unsigned h = ((w >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
  __half2 v = __hsub2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(1032.f));
  v = __hmul2(v, *reinterpret_cast<__half2*>(&d2));
  return *reinterpret_cast<unsigned*>(&v);
}}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {{
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}}
__device__ __forceinline__ void cp8(void* dst, const void* src) {{
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(d), "l"(src));
}}
extern "C" __global__ void __launch_bounds__({32 * nw}) k(
    const uint4* __restrict__ W, const uint2* __restrict__ S, const __half* __restrict__ X,
    void* Y, const void* AUX, int N, int K, float* __restrict__ WS, int* __restrict__ CNT)
{{
  __shared__ __align__(16) __half xs[NS][NT * 8][{XS}];
  __shared__ __align__(16) uint4 wsm[NS][{nw}][MT][32];
  __shared__ __align__(16) uint2 ssm[NS][{nw}][MT][8];
  __shared__ float red[{R}][NT * 8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp % WM, wk = warp / WM;
  const int nthr = 32 * WM * WK;
  const int row0 = blockIdx.x * {R};
  const int C = K / 64, T = N / 16;
  const int Cy = (C + {KS} - 1) / {KS}, cb = blockIdx.y * Cy, ce = min(C, cb + Cy);
  const int iters = (ce - cb + WK - 1) / WK;
  for (int i = threadIdx.x; i < NS * (NT * 8 - B_) * {XS}; i += nthr) {{
    int st = i / ((NT * 8 - B_) * {XS}), r = i % ((NT * 8 - B_) * {XS});
    xs[st][B_ + r / {XS}][r % {XS}] = __float2half(0.f);
  }}
  int tile[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m) tile[m] = min(row0 / 16 + wm * MT + m, T - 1);
  float c[MT][NT][4];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int n = 0; n < NT; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.f;
  auto issue = [&](int itq) {{
    if (itq < iters) {{
      const int st = itq % NS, k0 = (cb + itq * WK) * 64;
      for (int i = threadIdx.x; i < B_ * {KC // 8}; i += nthr) {{
        int b = i / {KC // 8}, kk = (i % {KC // 8}) * 8;
        bool ok = k0 + kk < K;
        cp16(&xs[st][b][kk], X + (size_t)b * K + (ok ? k0 + kk : 0), ok);
      }}
      const int ch = cb + itq * WK + wk;
      if (ch < ce) {{
        #pragma unroll
        for (int m = 0; m < MT; ++m) {{
          cp16(&wsm[st][warp][m][lane], W + ((size_t)tile[m] * C + ch) * 32 + lane, true);
          if (lane < 8) cp8(&ssm[st][warp][m][lane], S + ((size_t)tile[m] * C + ch) * 8 + lane);
        }}
      }}
    }}
    asm volatile("cp.async.commit_group;");
  }};
  #pragma unroll
  for (int p = 0; p < NS - 1; ++p) issue(p);
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {{
    asm volatile("cp.async.wait_group %0;" :: "n"(NS - 2));
    __syncthreads();
    issue(it + NS - 1);
    const int st = it % NS;
    if (cb + it * WK + wk < ce) {{
      unsigned dA[MT][2], dB[MT][2];
      uint4 wq[MT];
      #pragma unroll
      for (int m = 0; m < MT; ++m) {{
        wq[m] = wsm[st][warp][m][lane];
        uint2 sv = ssm[st][warp][m][g];
        dA[m][0] = __byte_perm(sv.x, 0, 0x1010); dB[m][0] = __byte_perm(sv.x, 0, 0x3232);
        dA[m][1] = __byte_perm(sv.y, 0, 0x1010); dB[m][1] = __byte_perm(sv.y, 0, 0x3232);
      }}
      #pragma unroll
      for (int h = 0; h < 2; ++h) {{
        unsigned bf[NT][4];
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          unsigned addr = (unsigned)__cvta_generic_to_shared(&xs[st][n * 8 + (lane & 7)][wk * 64 + h * 32 + 8 * (lane >> 3)]);
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];"
                       : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(addr));
        }}
        #pragma unroll
        for (int ss = 0; ss < 2; ++ss) {{
          const int s = 2 * h + ss;
          unsigned a[MT][4];
          #pragma unroll
          for (int m = 0; m < MT; ++m) {{
            unsigned w = s == 0 ? wq[m].x : s == 1 ? wq[m].y : s == 2 ? wq[m].z : wq[m].w;
            a[m][0] = dq(w, 0, dA[m][h]); a[m][1] = dq(w, 1, dB[m][h]);
            a[m][2] = dq(w, 2, dA[m][h]); a[m][3] = dq(w, 3, dB[m][h]);
          }}
          #pragma unroll
          for (int n = 0; n < NT; ++n)
            #pragma unroll
            for (int m = 0; m < MT; ++m)
              asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
                           : "+f"(c[m][n][0]), "+f"(c[m][n][1]), "+f"(c[m][n][2]), "+f"(c[m][n][3])
                           : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(bf[n][2 * ss]), "r"(bf[n][2 * ss + 1]));
        }}
      }}
    }}
  }}
  asm volatile("cp.async.wait_group 0;");
  for (int s = 0; s < WK; ++s) {{
    if (wk == s) {{
      #pragma unroll
      for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {{
          float* r0 = &red[16 * (wm * MT + m) + g][n * 8 + 2 * t];
          float* r8 = &red[16 * (wm * MT + m) + g + 8][n * 8 + 2 * t];
          if (s == 0) {{ r0[0] = c[m][n][0]; r0[1] = c[m][n][1]; r8[0] = c[m][n][2]; r8[1] = c[m][n][3]; }}
          else {{ r0[0] += c[m][n][0]; r0[1] += c[m][n][1]; r8[0] += c[m][n][2]; r8[1] += c[m][n][3]; }}
        }}
    }}
    __syncthreads();
  }}
{_v2_tail(R, KS, epi)}
}}
"""
