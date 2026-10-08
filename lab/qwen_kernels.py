"""Fused decode kernels for Qwen2-family models (batch 1), compiled with NVRTC.

Per layer: rmsnorm -> qkv GEMV(+bias) -> attention(rope + kv write + GQA decode)
-> o GEMV(+residual) -> rmsnorm -> gate/up GEMV(+SiLU*mul) -> down GEMV(+residual).
The residual stream h is fp32; GEMV inputs are bf16; accumulation is fp32.
"""
from __future__ import annotations

from .experiments.gemv4 import _inner

EPILOGUES = ("store", "bias", "biasf", "resid", "swiglu", "logits")


def gemv_source(S_: int, epi: str, quant: str = "g128f32", G: int = 1, timed: bool = False) -> str:
    """Split-K int4 GEMV. A block holds G row-groups of 4 rows; each row-group is
    reduced by S warps splitting K (block = 32*S*G threads). Fewer, fatter blocks cut
    the ~49 ns/block/SM dispatch cost measured in scripts/graph_overhead.py.
    epi: store (bf16 Y), bias/biasf (bf16 Y = acc + B), resid (fp32 Y += acc),
         swiglu (rows interleaved g,u: bf16 Y[row0/2 + t] = silu(g) * u), logits (fp32 Y)."""
    body = _inner(4, "kb + lane", "ke", "32", "bf16", quant)
    if epi == "swiglu":
        fin = """
  if (sub < 2) {
    float g = 0.f, u = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) { g += red[grp][s][2 * sub]; u += red[grp][s][2 * sub + 1]; }
    ((__nv_bfloat16*)Y)[row0 / 2 + sub] = __float2bfloat16_rn(g / (1.f + __expf(-g)) * u);
  }"""
    else:
        store = {
            "store": "((__nv_bfloat16*)Y)[r] = __float2bfloat16_rn(a);",
            "bias": "((__nv_bfloat16*)Y)[r] = __float2bfloat16_rn(a + __bfloat162float(((const __nv_bfloat16*)AUX)[r]));",
            "biasf": "((__nv_bfloat16*)Y)[r] = __float2bfloat16_rn(a + ((const float*)AUX)[r]);",
            "resid": "((float*)Y)[r] += a;",
            "resid_norm": "((float*)Y)[r] += a;",
            "logits": "((float*)Y)[r] = a;",
        }[epi]
        fin = f"""
  if (sub < 4 && row0 + sub < rowEnd) {{
    float a = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) a += red[grp][s][sub];
    int r = row0 + sub;
    {store}
  }}"""
    tparam = ", unsigned long long* T" if timed else ""
    if epi == "resid_norm":
        tparam += ", unsigned* cnt, const float* __restrict__ nw, __nv_bfloat16* __restrict__ xout, float eps"
    tstart = 'unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));' if timed else ""
    tend = ("""
  __syncthreads();
  if (threadIdx.x == 0) { unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    T[2 * blockIdx.x] = g0; T[2 * blockIdx.x + 1] = g1; }""" if timed else "")
    normtail = ""
    if epi == "resid_norm":
        # last block to finish computes RMSNorm(h) * nw -> xout for the next GEMV (removes a launch)
        normtail = """
  __shared__ unsigned last;
  __shared__ float part[32];
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = (atomicAdd(cnt, 1u) == gridDim.x - 1);
  __syncthreads();
  if (last) {
    __threadfence();
    const float* hf = (const float*)Y;
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) { float v = __ldcg(hf + i); ss += v * v; }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
      float s = 0.f;
      for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += part[w];
      part[0] = rsqrtf(s / N + eps);
      *cnt = 0u;
    }
    __syncthreads();
    float rr = part[0];
    for (int i = threadIdx.x; i < N; i += blockDim.x) xout[i] = __float2bfloat16_rn(__ldcg(hf + i) * rr * nw[i]);
  }"""
    return f"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define SPLIT {S_}
#define GROUPS {G}
extern "C" __global__ void __launch_bounds__({32 * S_ * G}, 1) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const uint4* __restrict__ X,
    void* Y, const void* AUX, int N, int K{tparam})
{{
  {tstart}
  __shared__ float red[GROUPS][SPLIT][4];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int grp = warp / SPLIT, ws = warp % SPLIT, sub = threadIdx.x - grp * SPLIT * 32;
  int KW = K / 32;
  int row0 = (blockIdx.x * GROUPS + grp) * 4, rowEnd = min(N, row0 + 4);
  int chunk = (KW + SPLIT - 1) / SPLIT;
  int kb = ws * chunk, ke = min(KW, kb + chunk);
  float acc[4] = {{0, 0, 0, 0}};
  if (row0 < N) {{
      {body}
  }}
  #pragma unroll
  for (int r = 0; r < 4; ++r) {{
    float a = acc[r];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[grp][ws][r] = a;
  }}
  __syncthreads();
  if (row0 < N) {{{fin}
  }}{tend}{normtail}
}}
"""


RMSNORM = r"""
#include <cuda_bf16.h>
extern "C" __global__ void __launch_bounds__(512) k(const float* __restrict__ h, const __nv_bfloat16* __restrict__ w,
                                               __nv_bfloat16* __restrict__ out, int H, float eps)
{
  __shared__ float part[16];
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += 512) { float v = h[i]; ss += v * v; }
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 16 ? part[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (threadIdx.x == 0) part[0] = rsqrtf(t / H + eps);
  }
  __syncthreads();
  float r = part[0];
  for (int i = threadIdx.x; i < H; i += 512)
    out[i] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(h[i] * r)) * __bfloat162float(w[i]));
}
"""

EMBED = r"""
#include <cuda_bf16.h>
extern "C" __global__ void k(const __nv_bfloat16* __restrict__ E, const long long* tok, float* __restrict__ h, int H)
{
  const __nv_bfloat16* row = E + (size_t)tok[0] * H;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x) h[i] = __bfloat162float(row[i]);
}
"""

# One block per query head; thread d owns head dimension d (head_dim = 128).
ATTN = r"""
#include <cuda_bf16.h>
#define HD 128
#define MAXLEN %(maxlen)d
extern "C" __global__ void __launch_bounds__(128) k(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cosT, const float* __restrict__ sinT,
    const long long* posp, __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ out, int NH, int NKV, float scale)
{
  __shared__ float q[HD], kn[HD], sc[MAXLEN], red[4];
  int h = blockIdx.x, d = threadIdx.x, grp = NH / NKV, kvh = h / grp;
  int pos = (int)posp[0];
  int half = HD / 2, dd = d %% half;
  float c = cosT[pos * half + dd], s = sinT[pos * half + dd];
  // rope(q) and rope(k_new): pairs (d, d + 64)
  float qa = __bfloat162float(qkv[h * HD + d]);
  float qb = __bfloat162float(qkv[h * HD + (d < half ? d + half : d - half)]);
  q[d] = d < half ? qa * c - qb * s : qa * c + qb * s;
  const __nv_bfloat16* kp = qkv + NH * HD + kvh * HD;
  float ka = __bfloat162float(kp[d]), kb = __bfloat162float(kp[d < half ? d + half : d - half]);
  float knew = d < half ? ka * c - kb * s : ka * c + kb * s;
  // match the reference: rotated k is stored (and used) at bf16 precision
  knew = __bfloat162float(__float2bfloat16_rn(knew));
  float vnew = __bfloat162float(qkv[(NH + NKV) * HD + kvh * HD + d]);
  kn[d] = knew;
  q[d] = __bfloat162float(__float2bfloat16_rn(q[d]));
  __nv_bfloat16* kbase = kc + (size_t)kvh * MAXLEN * HD;
  __nv_bfloat16* vbase = vc + (size_t)kvh * MAXLEN * HD;
  if (h %% grp == 0) {  // one writer per kv head; readers use kn/vnew for j == pos (no race)
    kbase[(size_t)pos * HD + d] = __float2bfloat16_rn(knew);
    vbase[(size_t)pos * HD + d] = __float2bfloat16_rn(vnew);
  }
  __syncthreads();
  // scores: thread t handles positions t, t+128, ...
  float mx = -1e30f;
  for (int j = d; j <= pos; j += 128) {
    float acc = 0.f;
    if (j == pos) {
      for (int i = 0; i < HD; ++i) acc += q[i] * kn[i];
    } else {
      const uint4* kr = reinterpret_cast<const uint4*>(kbase + (size_t)j * HD);
      #pragma unroll 4
      for (int i = 0; i < HD / 8; ++i) {
        uint4 t = kr[i];
        unsigned ws[4] = {t.x, t.y, t.z, t.w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
          acc += q[8 * i + 2 * u] * __uint_as_float(ws[u] << 16) + q[8 * i + 2 * u + 1] * __uint_as_float(ws[u] & 0xffff0000u);
        }
      }
    }
    acc *= scale;
    sc[j] = acc;
    mx = fmaxf(mx, acc);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((d & 31) == 0) red[d >> 5] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();
  float sum = 0.f;
  for (int j = d; j <= pos; j += 128) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if ((d & 31) == 0) red[d >> 5] = sum;
  __syncthreads();
  sum = red[0] + red[1] + red[2] + red[3];
  float o_ = 0.f;
  for (int j = 0; j < pos; ++j) o_ += sc[j] * __bfloat162float(vbase[(size_t)j * HD + d]);
  o_ += sc[pos] * __bfloat162float(__float2bfloat16_rn(vnew));
  out[h * HD + d] = __float2bfloat16_rn(o_ / sum);
}
"""

FINISH = r"""
extern "C" __global__ void __launch_bounds__(1024) k(const float* __restrict__ logits, int V, long long* tok, long long* pos)
{
  __shared__ float bv[32]; __shared__ int bi[32];
  float v = -1e30f; int idx = 0;
  for (int i = threadIdx.x; i < V; i += 1024) { float x = logits[i]; if (x > v) { v = x; idx = i; } }
  for (int o = 16; o; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, v, o); int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
  __syncthreads();
  if (threadIdx.x < 32) {
    v = bv[threadIdx.x]; idx = bi[threadIdx.x];
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, v, o); int oi = __shfl_xor_sync(0xffffffffu, idx, o);
      if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (threadIdx.x == 0) { tok[0] = idx; pos[0] += 1; }
  }
}
"""


EMBED_F32 = r"""
extern "C" __global__ void k(const float* __restrict__ E, const long long* tok, float* __restrict__ h, int H)
{
  const float* row = E + (size_t)tok[0] * H;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x) h[i] = row[i];
}
"""


def gemv_i8_source(S_: int) -> str:
    """Exact int8 GEMV for Q6_K-derived weights: W int8 [N, K] (16 per uint4), fp32 scale
    per 16 weights [N, K/16], bf16 X, fp32 logits out. Split-K over S warps, 4 rows/block.
    int8 -> float exactly via the magic constant: bits 0x4B000000 | (b ^ 0x80) = 2^23 + 128 + q."""
    rows = []
    for r in range(4):
        rows.append(f"""
      {{ uint4 w = (row0 + {r} < rowEnd) ? W[(size_t)(row0 + {r}) * KW + j] : make_uint4(0x80808080u,0x80808080u,0x80808080u,0x80808080u);
        unsigned ws[4] = {{w.x, w.y, w.z, w.w}}; float part = 0.f;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {{
          unsigned b = ws[u] ^ 0x80808080u;
          part = fmaf(__int_as_float(((b      ) & 255) | 0x4B000000) - 8388736.0f, xf[4 * u + 0], part);
          part = fmaf(__int_as_float(((b >>  8) & 255) | 0x4B000000) - 8388736.0f, xf[4 * u + 1], part);
          part = fmaf(__int_as_float(((b >> 16) & 255) | 0x4B000000) - 8388736.0f, xf[4 * u + 2], part);
          part = fmaf(__int_as_float(((b >> 24)      ) | 0x4B000000) - 8388736.0f, xf[4 * u + 3], part);
        }}
        if (row0 + {r} < rowEnd) acc[{r}] = fmaf(S[(size_t)(row0 + {r}) * KW + j], part, acc[{r}]); }}""")
    return f"""
extern "C" __global__ void __launch_bounds__({32 * S_}, 1) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const uint4* __restrict__ X,
    float* Y, const void* AUX, int N, int K)
{{
  __shared__ float red[{S_}][4];
  int ws_ = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int KW = K / 16;
  int row0 = blockIdx.x * 4, rowEnd = min(N, row0 + 4);
  int chunk = (KW + {S_} - 1) / {S_};
  int kb = ws_ * chunk, ke = min(KW, kb + chunk);
  float acc[4] = {{0, 0, 0, 0}};
  for (int j = kb + lane; j < ke; j += 32) {{
    uint4 xa = X[(size_t)j * 2], xb = X[(size_t)j * 2 + 1];
    unsigned xw[8] = {{xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w}};
    float xf[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {{ xf[2 * i] = __uint_as_float(xw[i] << 16); xf[2 * i + 1] = __uint_as_float(xw[i] & 0xffff0000u); }}
    {"".join(rows)}
  }}
  #pragma unroll
  for (int r = 0; r < 4; ++r) {{
    float a = acc[r];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[ws_][r] = a;
  }}
  __syncthreads();
  if (threadIdx.x < 4 && row0 + threadIdx.x < rowEnd) {{
    float a = 0.f;
    #pragma unroll
    for (int s = 0; s < {S_}; ++s) a += red[s][threadIdx.x];
    Y[row0 + threadIdx.x] = a;
  }}
}}
"""

# RMSNorm with fp32 weights (GGUF stores norms as F32, with folded channel scales);
# fp32 math like llama.cpp, single rounding to bf16 at the end.
RMSNORM_F32W = RMSNORM.replace("const __nv_bfloat16* __restrict__ w", "const float* __restrict__ w").replace(
    "out[i] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(h[i] * r)) * __bfloat162float(w[i]));",
    "out[i] = __float2bfloat16_rn(h[i] * r * w[i]);")
assert "w[i]);" in RMSNORM_F32W and "__bfloat162float(w[i])" not in RMSNORM_F32W

# Flash-decoding attention in one launch: grid = NH x MAXCH blocks (static for graphs).
# Block (h, c) handles positions [64c, 64c+64) of head h. Partials (max, sum, out[128])
# go to scratch; the last block to arrive for head h (atomic counter) combines them,
# writes the output and resets the counter.
ATTN2 = r"""
#include <cuda_bf16.h>
#define HD 128
#define MAXLEN %(maxlen)d
#define CH 64
#define MAXCH (MAXLEN / CH)
extern "C" __global__ void __launch_bounds__(128) k(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cosT, const float* __restrict__ sinT,
    const long long* posp, __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ out, int NH, int NKV, float scale,
    float* __restrict__ part, unsigned* __restrict__ counter)
{
  __shared__ float q[HD], kn[HD], vn[HD], sc[CH], red[8];
  __shared__ int amLast;
  int h = blockIdx.x / MAXCH, c = blockIdx.x %% MAXCH;
  int d = threadIdx.x, warp = d >> 5, lane = d & 31;
  int grp = NH / NKV, kvh = h / grp;
  int pos = (int)posp[0];
  int j0 = c * CH, j1 = min(pos + 1, j0 + CH);
  float* P = part + ((size_t)h * MAXCH + c) * (HD + 2);
  if (j0 <= pos) {
    int half = HD / 2, dd = d %% half;
    float cs = cosT[pos * half + dd], sn = sinT[pos * half + dd];
    float qa = __bfloat162float(qkv[h * HD + d]);
    float qb = __bfloat162float(qkv[h * HD + (d < half ? d + half : d - half)]);
    q[d] = __bfloat162float(__float2bfloat16_rn(d < half ? qa * cs - qb * sn : qa * cs + qb * sn));
    const __nv_bfloat16* kp = qkv + NH * HD + kvh * HD;
    float ka = __bfloat162float(kp[d]), kb2 = __bfloat162float(kp[d < half ? d + half : d - half]);
    kn[d] = __bfloat162float(__float2bfloat16_rn(d < half ? ka * cs - kb2 * sn : ka * cs + kb2 * sn));
    vn[d] = __bfloat162float(qkv[(NH + NKV) * HD + kvh * HD + d]);
    __nv_bfloat16* kbase = kc + (size_t)kvh * MAXLEN * HD;
    __nv_bfloat16* vbase = vc + (size_t)kvh * MAXLEN * HD;
    if (h %% grp == 0 && c == pos / CH) {  // single writer of the new kv row
      kbase[(size_t)pos * HD + d] = __float2bfloat16_rn(kn[d]);
      vbase[(size_t)pos * HD + d] = __float2bfloat16_rn(vn[d]);
    }
    __syncthreads();
    // scores: warp w handles positions j0 + w, j0 + w + 4, ...; lane covers dims 4*lane .. 4*lane+3
    float q0 = q[4 * lane], q1 = q[4 * lane + 1], q2 = q[4 * lane + 2], q3 = q[4 * lane + 3];
    for (int j = j0 + warp; j < j1; j += 4) {
      float s;
      if (j == pos) {
        s = q0 * kn[4 * lane] + q1 * kn[4 * lane + 1] + q2 * kn[4 * lane + 2] + q3 * kn[4 * lane + 3];
      } else {
        uint2 t = *reinterpret_cast<const uint2*>(kbase + (size_t)j * HD + 4 * lane);
        s = q0 * __uint_as_float(t.x << 16) + q1 * __uint_as_float(t.x & 0xffff0000u)
          + q2 * __uint_as_float(t.y << 16) + q3 * __uint_as_float(t.y & 0xffff0000u);
      }
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0) sc[j - j0] = s * scale;
    }
    __syncthreads();
    float mx = -1e30f;
    for (int j = j0; j < j1; ++j) mx = fmaxf(mx, sc[j - j0]);
    float l = 0.f, o_ = 0.f;
    for (int j = j0; j < j1; ++j) {
      float p = __expf(sc[j - j0] - mx);
      l += p;
      float vv = (j == pos) ? __bfloat162float(__float2bfloat16_rn(vn[d])) : __bfloat162float(vbase[(size_t)j * HD + d]);
      o_ += p * vv;
    }
    P[d] = o_;
    if (d == 0) { P[HD] = mx; P[HD + 1] = l; }
  } else if (d == 0) {
    P[HD] = -1e30f; P[HD + 1] = 0.f;
  }
  __threadfence();
  __syncthreads();
  if (d == 0) amLast = (atomicAdd(&counter[h], 1u) == MAXCH - 1);
  __syncthreads();
  if (!amLast) return;
  __threadfence();
  float M = -1e30f;
  for (int cc = 0; cc < MAXCH; ++cc) M = fmaxf(M, part[((size_t)h * MAXCH + cc) * (HD + 2) + HD]);
  float L = 0.f, O = 0.f;
  for (int cc = 0; cc < MAXCH; ++cc) {
    const float* Q = part + ((size_t)h * MAXCH + cc) * (HD + 2);
    float w = __expf(Q[HD] - M);
    L += w * Q[HD + 1];
    O += w * (Q[HD + 1] > 0.f ? Q[d] : 0.f);
  }
  out[h * HD + d] = __float2bfloat16_rn(O / L);
  if (d == 0) counter[h] = 0;
}
"""

# Attention v3: one block (512 threads) per query head. 16 warps score positions with
# lanes splitting the head dim (coalesced K rows); 4 position-groups of 128 threads
# accumulate P*V in parallel (4x shorter serial L2-latency chain than v1), combined in smem.
ATTN3 = r"""
#include <cuda_bf16.h>
#define HD 128
#define MAXLEN %(maxlen)d
extern "C" __global__ void __launch_bounds__(512) k(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cosT, const float* __restrict__ sinT,
    const long long* posp, __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ out, int NH, int NKV, float scale)
{
  __shared__ float q[HD], kn[HD], vn[HD], sc[MAXLEN], red[16], acc4[4][HD];
  int h = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  int grp = NH / NKV, kvh = h / grp;
  int pos = (int)posp[0];
  __nv_bfloat16* kbase = kc + (size_t)kvh * MAXLEN * HD;
  __nv_bfloat16* vbase = vc + (size_t)kvh * MAXLEN * HD;
  if (t < HD) {
    int d = t, half = HD / 2, dd = d %% half;
    float cs = cosT[pos * half + dd], sn = sinT[pos * half + dd];
    float qa = __bfloat162float(qkv[h * HD + d]);
    float qb = __bfloat162float(qkv[h * HD + (d < half ? d + half : d - half)]);
    q[d] = __bfloat162float(__float2bfloat16_rn(d < half ? qa * cs - qb * sn : qa * cs + qb * sn));
    const __nv_bfloat16* kp = qkv + NH * HD + kvh * HD;
    float ka = __bfloat162float(kp[d]), kb = __bfloat162float(kp[d < half ? d + half : d - half]);
    kn[d] = __bfloat162float(__float2bfloat16_rn(d < half ? ka * cs - kb * sn : ka * cs + kb * sn));
    vn[d] = __bfloat162float(__float2bfloat16_rn(__bfloat162float(qkv[(NH + NKV) * HD + kvh * HD + d])));
    if (h %% grp == 0) {
      kbase[(size_t)pos * HD + d] = __float2bfloat16_rn(kn[d]);
      vbase[(size_t)pos * HD + d] = __float2bfloat16_rn(vn[d]);
    }
  }
  __syncthreads();
  float q0 = q[4 * lane], q1 = q[4 * lane + 1], q2 = q[4 * lane + 2], q3 = q[4 * lane + 3];
  for (int j = warp; j <= pos; j += 16) {
    float s;
    if (j == pos) {
      s = q0 * kn[4 * lane] + q1 * kn[4 * lane + 1] + q2 * kn[4 * lane + 2] + q3 * kn[4 * lane + 3];
    } else {
      uint2 w = *reinterpret_cast<const uint2*>(kbase + (size_t)j * HD + 4 * lane);
      s = q0 * __uint_as_float(w.x << 16) + q1 * __uint_as_float(w.x & 0xffff0000u)
        + q2 * __uint_as_float(w.y << 16) + q3 * __uint_as_float(w.y & 0xffff0000u);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) sc[j] = s * scale;
  }
  __syncthreads();
  float mx = -1e30f;
  for (int j = t; j <= pos; j += 512) mx = fmaxf(mx, sc[j]);
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int i = 1; i < 16; ++i) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int j = t; j <= pos; j += 512) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  sum = 0.f;
  for (int i = 0; i < 16; ++i) sum += red[i];
  int pg = t >> 7, d = t & 127;
  float o_ = 0.f;
  for (int j = pg; j < pos; j += 4) o_ += sc[j] * __bfloat162float(vbase[(size_t)j * HD + d]);
  if (pos %% 4 == pg) o_ += sc[pos] * vn[d];
  acc4[pg][d] = o_;
  __syncthreads();
  if (t < HD) out[h * HD + t] = __float2bfloat16_rn((acc4[0][t] + acc4[1][t] + acc4[2][t] + acc4[3][t]) / sum);
}
"""

ATTN4 = r"""
#include <cuda_bf16.h>
#define HD 128
#define MAXLEN %(maxlen)d
extern "C" __global__ void __launch_bounds__(512) k(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cosT, const float* __restrict__ sinT,
    const long long* posp, __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ out, int NH, int NKV, float scale)
{
  __shared__ float q[HD], kn[HD], vn[HD], sc[MAXLEN], red[16], acc4[4][HD];
  int h = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  int grp = NH / NKV, kvh = h / grp;
  int pos = (int)posp[0];
  __nv_bfloat16* kbase = kc + (size_t)kvh * MAXLEN * HD;
  __nv_bfloat16* vbase = vc + (size_t)kvh * MAXLEN * HD;
  if (t < HD) {
    int d = t, half = HD / 2, dd = d %% half;
    float cs = cosT[pos * half + dd], sn = sinT[pos * half + dd];
    float qa = __bfloat162float(qkv[h * HD + d]);
    float qb = __bfloat162float(qkv[h * HD + (d < half ? d + half : d - half)]);
    q[d] = __bfloat162float(__float2bfloat16_rn(d < half ? qa * cs - qb * sn : qa * cs + qb * sn));
    const __nv_bfloat16* kp = qkv + NH * HD + kvh * HD;
    float ka = __bfloat162float(kp[d]), kb = __bfloat162float(kp[d < half ? d + half : d - half]);
    kn[d] = __bfloat162float(__float2bfloat16_rn(d < half ? ka * cs - kb * sn : ka * cs + kb * sn));
    vn[d] = __bfloat162float(__float2bfloat16_rn(__bfloat162float(qkv[(NH + NKV) * HD + kvh * HD + d])));
    if (h %% grp == 0) {
      kbase[(size_t)pos * HD + d] = __float2bfloat16_rn(kn[d]);
      vbase[(size_t)pos * HD + d] = __float2bfloat16_rn(vn[d]);
    }
  }
  __syncthreads();
  float q0 = q[4 * lane], q1 = q[4 * lane + 1], q2 = q[4 * lane + 2], q3 = q[4 * lane + 3];
  // scores: 4 positions per warp per step, loads issued before the shuffle reductions (ILP)
  for (int j0 = warp; j0 <= pos; j0 += 64) {
    uint2 w[4]; float s[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      int j = j0 + 16 * u;
      w[u] = (j < pos) ? *reinterpret_cast<const uint2*>(kbase + (size_t)j * HD + 4 * lane) : make_uint2(0u, 0u);
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      int j = j0 + 16 * u;
      s[u] = (j == pos) ? q0 * kn[4 * lane] + q1 * kn[4 * lane + 1] + q2 * kn[4 * lane + 2] + q3 * kn[4 * lane + 3]
           : q0 * __uint_as_float(w[u].x << 16) + q1 * __uint_as_float(w[u].x & 0xffff0000u)
           + q2 * __uint_as_float(w[u].y << 16) + q3 * __uint_as_float(w[u].y & 0xffff0000u);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
      #pragma unroll
      for (int u = 0; u < 4; ++u) s[u] += __shfl_xor_sync(0xffffffffu, s[u], o);
    }
    if (lane == 0) {
      #pragma unroll
      for (int u = 0; u < 4; ++u) if (j0 + 16 * u <= pos) sc[j0 + 16 * u] = s[u] * scale;
    }
  }
  __syncthreads();
  float mx = -1e30f;
  for (int j = t; j <= pos; j += 512) mx = fmaxf(mx, sc[j]);
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int i = 1; i < 16; ++i) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int j = t; j <= pos; j += 512) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  sum = 0.f;
  for (int i = 0; i < 16; ++i) sum += red[i];
  int pg = t >> 7, d = t & 127;
  float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;   // 4 independent accumulators: loads overlap
  int j = pg;
  for (; j + 12 < pos; j += 16) {
    float v0 = __bfloat162float(vbase[(size_t)j * HD + d]), v1 = __bfloat162float(vbase[(size_t)(j + 4) * HD + d]);
    float v2 = __bfloat162float(vbase[(size_t)(j + 8) * HD + d]), v3 = __bfloat162float(vbase[(size_t)(j + 12) * HD + d]);
    o0 += sc[j] * v0; o1 += sc[j + 4] * v1; o2 += sc[j + 8] * v2; o3 += sc[j + 12] * v3;
  }
  for (; j < pos; j += 4) o0 += sc[j] * __bfloat162float(vbase[(size_t)j * HD + d]);
  float o_ = (o0 + o1) + (o2 + o3);
  if (pos %% 4 == pg) o_ += sc[pos] * vn[d];
  acc4[pg][d] = o_;
  __syncthreads();
  if (t < HD) out[h * HD + t] = __float2bfloat16_rn((acc4[0][t] + acc4[1][t] + acc4[2][t] + acc4[3][t]) / sum);
}
"""


def gemv_q6_source(S_: int) -> str:
    """Q6_K-exact GEMV (6.5625 bits/weight, same bytes as llama.cpp's Q6_K) in a lane-friendly layout:
      L  uint32 [N, K/8]   low 4 bits of q+32, nibble i of word = weight 8w+i (as the int4 path)
      Hb uint32 [N, K/16]  high 2 bits of q+32, field i of word = weight 16w+i
      SC int8   [N, K/16]  Q6_K sub-block scales;   D fp16 [N, K/256] super-block scales
    weight = D * SC * (q - 32); q - 32 comes out of the magic constant exactly.
    Split-K over S warps, 4 rows per block, bf16 X, fp32 logits out."""
    rows = []
    for r in range(4):
        rows.append(f"""
      {{ int row = row0 + {r};
        if (row < rowEnd) {{
        uint4 lo = L[(size_t)row * (K / 32) + j];
        uint2 hi = Hb[(size_t)row * (K / 32) + j];          // K/16 words per row = K/32 uint2
        unsigned short sc2 = SC[(size_t)row * (K / 32) + j];
        float d = __half2float(D[(size_t)row * (K / 256) + (j >> 3)]);
        float s0 = d * (float)(signed char)(sc2 & 0xff), s1 = d * (float)(signed char)(sc2 >> 8);
        unsigned lw[4] = {{lo.x, lo.y, lo.z, lo.w}};
        unsigned hw[2] = {{hi.x, hi.y}};
        float p0 = 0.f, p1 = 0.f;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {{
          #pragma unroll
          for (int i = 0; i < 8; ++i) {{
            int w = 8 * u + i;                                  // weight index within the 32
            unsigned q = ((lw[u] >> (4 * i)) & 15u) | (((hw[w >> 4] >> (2 * (w & 15))) & 3u) << 4);
            float v = __int_as_float(q | 0x4B000000) - 8388640.0f;    // (2^23 + q) - (2^23 + 32)
            if (w < 16) p0 = fmaf(v, xf[w], p0); else p1 = fmaf(v, xf[w], p1);
          }}
        }}
        acc[{r}] = fmaf(s0, p0, fmaf(s1, p1, acc[{r}]));
        }} }}""")
    return f"""
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__({32 * S_}, 1) k(
    const uint4* __restrict__ L, const uint2* __restrict__ Hb, const unsigned short* __restrict__ SC,
    const __half* __restrict__ D, const uint4* __restrict__ X, float* Y, int N, int K)
{{
  __shared__ float red[{S_}][4];
  int ws_ = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int KW = K / 32;
  int row0 = blockIdx.x * 4, rowEnd = min(N, row0 + 4);
  int chunk = (KW + {S_} - 1) / {S_};
  int kb = ws_ * chunk, ke = min(KW, kb + chunk);
  float acc[4] = {{0, 0, 0, 0}};
  for (int j = kb + lane; j < ke; j += 32) {{
    float xf[32];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {{
      uint4 xv = X[(size_t)j * 4 + t];
      unsigned xw[4] = {{xv.x, xv.y, xv.z, xv.w}};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {{ xf[8 * t + 2 * i] = __uint_as_float(xw[i] << 16); xf[8 * t + 2 * i + 1] = __uint_as_float(xw[i] & 0xffff0000u); }}
    }}
    {"".join(rows)}
  }}
  #pragma unroll
  for (int r = 0; r < 4; ++r) {{
    float a = acc[r];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[ws_][r] = a;
  }}
  __syncthreads();
  if (threadIdx.x < 4 && row0 + threadIdx.x < rowEnd) {{
    float a = 0.f;
    #pragma unroll
    for (int s = 0; s < {S_}; ++s) a += red[s][threadIdx.x];
    Y[row0 + threadIdx.x] = a;
  }}
}}
"""
