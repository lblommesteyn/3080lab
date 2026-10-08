"""Fused decode kernels for Qwen2-family models (batch 1), compiled with NVRTC.

Per layer: rmsnorm -> qkv GEMV(+bias) -> attention(rope + kv write + GQA decode)
-> o GEMV(+residual) -> rmsnorm -> gate/up GEMV(+SiLU*mul) -> down GEMV(+residual).
The residual stream h is fp32; GEMV inputs are bf16; accumulation is fp32.
"""
from __future__ import annotations

from .experiments.gemv4 import _inner

EPILOGUES = ("store", "bias", "resid", "swiglu", "logits")


def gemv_source(S_: int, epi: str) -> str:
    """Split-K int4 GEMV, block = S warps on one 4-row group, bf16 X.
    epi: store (bf16 Y), bias (bf16 Y = acc + B), resid (fp32 Y += acc),
         swiglu (rows interleaved g,u: bf16 Y[row0/2 + t] = silu(g) * u), logits (fp32 Y)."""
    body = _inner(4, "kb + lane", "ke", "32", "bf16")
    if epi == "swiglu":
        fin = """
  if (threadIdx.x < 2 && row0 + 2 * threadIdx.x + 1 < rowEnd + 1) {
    float g = 0.f, u = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) { g += red[s][2 * threadIdx.x]; u += red[s][2 * threadIdx.x + 1]; }
    ((__nv_bfloat16*)Y)[row0 / 2 + threadIdx.x] = __float2bfloat16_rn(g / (1.f + __expf(-g)) * u);
  }"""
    else:
        store = {
            "store": "((__nv_bfloat16*)Y)[r] = __float2bfloat16_rn(a);",
            "bias": "((__nv_bfloat16*)Y)[r] = __float2bfloat16_rn(a + __bfloat162float(((const __nv_bfloat16*)AUX)[r]));",
            "resid": "((float*)Y)[r] += a;",
            "logits": "((float*)Y)[r] = a;",
        }[epi]
        fin = f"""
  if (threadIdx.x < 4 && row0 + threadIdx.x < rowEnd) {{
    float a = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT; ++s) a += red[s][threadIdx.x];
    int r = row0 + threadIdx.x;
    {store}
  }}"""
    return f"""
#include <cuda_bf16.h>
#define SPLIT {S_}
extern "C" __global__ void __launch_bounds__({32 * S_}, 1) k(
    const uint4* __restrict__ W, const float* __restrict__ S, const uint4* __restrict__ X,
    void* Y, const void* AUX, int N, int K)
{{
  __shared__ float red[SPLIT][4];
  int ws = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int KW = K / 32;
  int row0 = blockIdx.x * 4, rowEnd = min(N, row0 + 4);
  int chunk = (KW + SPLIT - 1) / SPLIT;
  int kb = ws * chunk, ke = min(KW, kb + chunk);
  float acc[4] = {{0, 0, 0, 0}};
      {body}
  #pragma unroll
  for (int r = 0; r < 4; ++r) {{
    float a = acc[r];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[ws][r] = a;
  }}
  __syncthreads();{fin}
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
