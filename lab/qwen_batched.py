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
