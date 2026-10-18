"""Decode megakernel for Qwen2.5-1.5B (batch 1): all 28 layers in one persistent launch.

One block per SM. Warp NW of each block is a producer: it streams that block's share of every
layer's weights, in consumption order, from a per-block contiguous buffer into a shared-memory
ring (cp.async, one mbarrier per 4 KB chunk), never waiting on activations. Warps 0..NW-1 are
consumers: they wait at software grid barriers between ops (qkv -> attention -> o -> gate/up ->
down), stage the op's input vector in shared memory, and compute 4-row int4 units out of the ring.
So the DRAM stream keeps running through barriers, attention and every op's tail, which is what a
graph of ~200 kernels cannot do (each kernel boundary drains and refills the memory pipeline).

Weights are llama.cpp's Q4_0 (repacked as in qwen_gguf.py): a unit is 4 rows x K, stored as
sub-units of 4 rows x KWC uint4 (32 weights each) followed by their 4 x KWC fp16 scales
(72 * KWC bytes; KWC = 48 for K = 1536, 56 for down's K = 8960, 5 sub-units).
"""
from __future__ import annotations

from .experiments.gemv3 import _deq


def _sub_body() -> str:
    """Accumulate one sub-unit (4 rows x KWC uint4 at ring offset rp, x from uint4 group jx) into acc."""
    L = ["for (int jl = lane; jl < KWC; jl += 32) {"]
    for r in range(4):
        L.append(f"  unsigned p{r} = rp + ({r} * KWC + jl) * 16; if (p{r} >= RING) p{r} -= RING;")
        L.append(f"  unsigned s{r} = rp + 64 * KWC + ({r} * KWC + jl) * 2; if (s{r} >= RING) s{r} -= RING;")
        L.append(f"  uint4 w{r} = *reinterpret_cast<const uint4*>(ring + p{r});")
        L.append(f"  float sc{r} = __half2float(*reinterpret_cast<const __half*>(ring + s{r}));")
    L.append("  int j = jx + jl, sw = (j >> 1) & 3;")
    for k in range(4):
        L.append(f"  uint4 t{k} = xs[4 * j + ({k} ^ sw)];"
                 f" float4 xa0_{k} = make_float4(__uint_as_float(t{k}.x << 16), __uint_as_float(t{k}.x & 0xffff0000u),"
                 f" __uint_as_float(t{k}.y << 16), __uint_as_float(t{k}.y & 0xffff0000u));"
                 f" float4 xb0_{k} = make_float4(__uint_as_float(t{k}.z << 16), __uint_as_float(t{k}.z & 0xffff0000u),"
                 f" __uint_as_float(t{k}.w << 16), __uint_as_float(t{k}.w & 0xffff0000u));")
    for r in range(4):
        L.append("  { float part = 0.f;")
        L += ["    " + s for s in _deq("magic", f"w{r}", 0)]
        L.append(f"    acc[{r}] = fmaf(sc{r}, part, acc[{r}]); }}")
    L.append("}")
    return "\n    ".join(L)


def mega_source(nl: int, maxlen: int, nw: int = 8, ch: int = 4096, nch: int = 20, cap: int = 1 << 26) -> str:
    xs_bytes = 17920                     # largest op input: down's 8960 bf16
    return r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define H 1536
#define NH 12
#define NKV 2
#define HD 128
#define INTER 8960
#define NQKV 2048
#define NL %(nl)d
#define MAXLEN %(maxlen)d
#define NW %(nw)d
#define NT (32 * NW)
#define CH %(ch)d
#define NCH %(nch)d
#define RING (CH * NCH)
#define XS_BYTES %(xs)d
#define CAP %(cap)d

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(unsigned long long* b, unsigned n) {
  asm volatile("mbarrier.init.shared.b64 [%%0], %%1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared.b64 st, [%%0]; }" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ bool mb_test(unsigned long long* b, unsigned par) {
  unsigned r;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%%1], %%2; selp.u32 %%0, 1, 0, p; }"
               : "=r"(r) : "r"(sa(b)), "r"(par) : "memory");
  return r != 0; }

// every spin is capped (a hung kernel is not reset on this machine); a timeout sets *err and
// makes all later waits fall through, so a bug yields garbage, not a frozen GPU
__device__ __forceinline__ void mb_wait(unsigned long long* b, unsigned par, volatile int* bad, unsigned* err, int code) {
  int n = 0;
  while (!mb_test(b, par)) {
    if (*bad) return;
    if (++n > CAP) { *bad = 1; atomicExch(err, code); return; }
  }
}

extern "C" __global__ void __launch_bounds__(NT + 32, 1) k(
    const uint4* __restrict__ stream, const long long* __restrict__ soff, const int* __restrict__ nchunk,
    const int2* __restrict__ units, const int* __restrict__ ubase, const int2* __restrict__ ptab,
    const float* __restrict__ embed, const long long* __restrict__ tokp, const long long* __restrict__ posp,
    const float* __restrict__ ln1, const float* __restrict__ ln2, const float* __restrict__ bias,
    const float* __restrict__ fnorm, const float* __restrict__ cosT, const float* __restrict__ sinT,
    __nv_bfloat16* kc, __nv_bfloat16* vc, float* h, __nv_bfloat16* qkv, __nv_bfloat16* att,
    __nv_bfloat16* xm, __nv_bfloat16* xout, unsigned* sync, unsigned* err, float eps, float scale)
{
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* ring = smem;
  uint4* xs = reinterpret_cast<uint4*>(smem + RING);
  unsigned long long* full = reinterpret_cast<unsigned long long*>(smem + RING + XS_BYTES);
  unsigned long long* empty = full + NCH;
  float* red = reinterpret_cast<float*>(empty + NCH);          // 16 floats
  volatile int* bad = reinterpret_cast<volatile int*>(red + 16);
  volatile int* last = bad + 1;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, b = blockIdx.x, G = gridDim.x;
  if (t == 0) {
    for (int i = 0; i < NCH; ++i) { mb_init(full + i, 32); mb_init(empty + i, NW); }
    *bad = 0; *last = 0;
  }
  __syncthreads();
  const uint4* sb = stream + soff[b] / 16;
  const int nc = nchunk[b];

  if (warp == NW) {                                            // ---- producer ----
    for (int g = 0; g < nc; ++g) {
      int slot = g %% NCH;
      if (g >= NCH) mb_wait(empty + slot, ((g / NCH) - 1) & 1, bad, err, 2);
      unsigned dst = sa(ring + slot * CH) + lane * 16;
      const uint4* src = sb + (size_t)g * (CH / 16) + lane;
      #pragma unroll
      for (int i = 0; i < CH / 512; ++i)
        asm volatile("cp.async.cg.shared.global [%%0], [%%1], 16;" :: "r"(dst + i * 512), "l"(src + i * 32) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%%0];" :: "r"(sa(full + slot)) : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }

  // ---- consumers ----
  int rel = 0, ready = 0;          // this warp: chunks < rel released, chunks < ready known full
  auto waitc = [&](int c1) {       // chunks [ready, c1] full
    for (; ready <= c1; ++ready) mb_wait(full + (ready %% NCH), (ready / NCH) & 1, bad, err, 3);
  };
  // release chunks < c. A chunk is released only after this warp has seen it full: a warp that
  // skips over other warps' sub-units could otherwise arrive on a slot's empty barrier while the
  // slot's previous phase is still open, completing that phase early (the producer then
  // overwrites data a slower warp is still reading). Wait and release one chunk at a time: a
  // skip longer than the ring would otherwise wait on a chunk that needs this warp's own release
  auto advance = [&](int c) {
    __syncwarp();
    for (; rel < c; ++rel) {
      waitc(rel);
      __syncwarp();
      if (lane == 0) mb_arrive(empty + (rel %% NCH));
    }
  };
  auto csync = [&]() { asm volatile("bar.sync 1, %%0;" :: "r"(NT) : "memory"); };
  unsigned nsp = 0;                // grid sync points passed
  auto arrive = [&]() -> unsigned {
    csync();
    unsigned old = 0;
    if (t == 0) { __threadfence(); old = atomicAdd(sync, 1u); }
    ++nsp;
    return old;
  };
  auto gwait = [&]() {             // all blocks passed sync point nsp - 1
    if (t == 0) {
      unsigned target = (unsigned)G * nsp, v; int n = 0;
      for (;;) {
        asm volatile("ld.acquire.gpu.u32 %%0, [%%1];" : "=r"(v) : "l"(sync) : "memory");
        if (v >= target || *bad) break;
        if (++n > CAP || *(volatile unsigned*)err) { *bad = 1; atomicExch(err, 4); break; }
      }
    }
    csync();
  };
  auto stage_rms = [&](const float* w) {     // xs = bf16(rmsnorm(h) * w), swizzled
    float v[8]; float ss = 0.f;
    if (t < H / 8) {
      float4 a = __ldcg(reinterpret_cast<const float4*>(h) + 2 * t), c = __ldcg(reinterpret_cast<const float4*>(h) + 2 * t + 1);
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
      #pragma unroll
      for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[warp] = ss;
    csync();
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < NW; ++i) tot += red[i];
    float r = rsqrtf(tot / H + eps);
    if (t < H / 8) {
      unsigned u[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat16 lo = __float2bfloat16_rn(v[2 * i] * r * w[8 * t + 2 * i]);
        __nv_bfloat16 hi = __float2bfloat16_rn(v[2 * i + 1] * r * w[8 * t + 2 * i + 1]);
        u[i] = (unsigned)__bfloat16_as_ushort(lo) | ((unsigned)__bfloat16_as_ushort(hi) << 16);
      }
      int j = t >> 2;
      xs[4 * j + ((t & 3) ^ ((j >> 1) & 3))] = make_uint4(u[0], u[1], u[2], u[3]);
    }
    csync();
  };
  auto stage_vec = [&](const __nv_bfloat16* src, int n16) {   // xs = src (n16 uint4), swizzled
    for (int i = t; i < n16; i += NT) {
      int j = i >> 2;
      xs[4 * j + ((i & 3) ^ ((j >> 1) & 3))] = __ldcg(reinterpret_cast<const uint4*>(src) + i);
    }
    csync();
  };
  const int2* myu = units + ubase[b];
  const int pos = (int)posp[0];

  // embed: h = E[tok]
  {
    const float* row = embed + (size_t)tokp[0] * H;
    for (int i = b * NT + t; i < H; i += G * NT) __stcg(h + i, row[i]);
  }
  arrive();

  for (int l = 0; l < NL; ++l) {
    #pragma unroll 1
    for (int op = 0; op < 5; ++op) {
      gwait();
      if (op == 1) {                                           // ---- attention: block = head ----
        if (b < NH) {
          float* q = reinterpret_cast<float*>(xs);
          float* kn = q + HD; float* vn = kn + HD; float* sc = vn + HD; float* acc2 = sc + MAXLEN;
          const int hh = b, grp = NH / NKV, kvh = hh / grp;
          __nv_bfloat16* kbase = kc + ((size_t)l * NKV + kvh) * MAXLEN * HD;
          __nv_bfloat16* vbase = vc + ((size_t)l * NKV + kvh) * MAXLEN * HD;
          if (t < HD) {
            int d = t, half = HD / 2, dd = d %% half, dp = d < half ? d + half : d - half;
            float cs = cosT[pos * half + dd], sn = sinT[pos * half + dd];
            float qa = __bfloat162float(__ldcg(qkv + hh * HD + d)), qb = __bfloat162float(__ldcg(qkv + hh * HD + dp));
            q[d] = __bfloat162float(__float2bfloat16_rn(d < half ? qa * cs - qb * sn : qa * cs + qb * sn));
            const __nv_bfloat16* kp = qkv + NH * HD + kvh * HD;
            float ka = __bfloat162float(__ldcg(kp + d)), kb = __bfloat162float(__ldcg(kp + dp));
            kn[d] = __bfloat162float(__float2bfloat16_rn(d < half ? ka * cs - kb * sn : ka * cs + kb * sn));
            vn[d] = __bfloat162float(__ldcg(qkv + (NH + NKV) * HD + kvh * HD + d));
            if (hh %% grp == 0) {
              kbase[(size_t)pos * HD + d] = __float2bfloat16_rn(kn[d]);
              vbase[(size_t)pos * HD + d] = __float2bfloat16_rn(vn[d]);
            }
          }
          csync();
          float q0 = q[4 * lane], q1 = q[4 * lane + 1], q2 = q[4 * lane + 2], q3 = q[4 * lane + 3];
          for (int j0 = warp; j0 <= pos; j0 += 4 * NW) {
            uint2 w[4]; float s[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
              int j = j0 + NW * u;
              w[u] = (j < pos) ? *reinterpret_cast<const uint2*>(kbase + (size_t)j * HD + 4 * lane) : make_uint2(0u, 0u);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
              int j = j0 + NW * u;
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
              for (int u = 0; u < 4; ++u) if (j0 + NW * u <= pos) sc[j0 + NW * u] = s[u] * scale;
            }
          }
          csync();
          float mx = -1e30f;
          for (int j = t; j <= pos; j += NT) mx = fmaxf(mx, sc[j]);
          for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
          if (lane == 0) red[warp] = mx;
          csync();
          mx = red[0];
          for (int i = 1; i < NW; ++i) mx = fmaxf(mx, red[i]);
          csync();
          float sum = 0.f;
          for (int j = t; j <= pos; j += NT) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
          for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
          if (lane == 0) red[warp] = sum;
          csync();
          sum = 0.f;
          for (int i = 0; i < NW; ++i) sum += red[i];
          const int NPG = NT / HD;                             // position groups
          int pg = t / HD, d = t %% HD;
          float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
          int j = pg;
          for (; j + 3 * NPG < pos; j += 4 * NPG) {
            float v0 = __bfloat162float(vbase[(size_t)j * HD + d]), v1 = __bfloat162float(vbase[(size_t)(j + NPG) * HD + d]);
            float v2 = __bfloat162float(vbase[(size_t)(j + 2 * NPG) * HD + d]), v3 = __bfloat162float(vbase[(size_t)(j + 3 * NPG) * HD + d]);
            o0 += sc[j] * v0; o1 += sc[j + NPG] * v1; o2 += sc[j + 2 * NPG] * v2; o3 += sc[j + 3 * NPG] * v3;
          }
          for (; j < pos; j += NPG) o0 += sc[j] * __bfloat162float(vbase[(size_t)j * HD + d]);
          float o_ = (o0 + o1) + (o2 + o3);
          if (pos %% NPG == pg) o_ += sc[pos] * vn[d];
          acc2[pg * HD + d] = o_;
          csync();
          if (t < HD) {
            float a = 0.f;
            for (int i = 0; i < NPG; ++i) a += acc2[i * HD + t];
            __stcg(att + hh * HD + t, __float2bfloat16_rn(a / sum));
          }
        }
        arrive();
        continue;
      }
      // ---- GEMV ops ----
      if (op == 0) stage_rms(ln1 + (size_t)l * H);
      else if (op == 2) stage_vec(att, H / 8);
      else if (op == 3) stage_rms(ln2 + (size_t)l * H);
      else stage_vec(xm, INTER / 8);
      const int2 ph = ptab[(size_t)b * NL * 5 + l * 5 + op];
      const int KWC = op == 4 ? 56 : 48, NSUB = op == 4 ? 5 : 1, SUB = 72 * KWC;
      for (int e = warp; e < ph.y; e += NW) {
        const int2 U = myu[ph.x + e];
        const int row0 = U.x;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const int gsz = min(NW, ph.y - (e / NW) * NW);          // units in this warp group (sub-units interleaved)
        for (int s = 0; s < NSUB; ++s) {
          unsigned so = (unsigned)U.y + s * gsz * SUB;
          advance(so / CH);
          waitc((so + SUB - 1) / CH);
          unsigned rp = so %% RING;
          int jx = s * KWC;
          %(body)s
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r)
          for (int o = 16; o; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
        float a = lane == 0 ? acc[0] : lane == 1 ? acc[1] : lane == 2 ? acc[2] : acc[3];
        if (op == 0) {
          if (lane < 4) __stcg(qkv + row0 + lane, __float2bfloat16_rn(a + bias[(size_t)l * NQKV + row0 + lane]));
        } else if (op == 3) {
          if (lane < 2) {
            float g = lane == 0 ? acc[0] : acc[2], u = lane == 0 ? acc[1] : acc[3];
            __stcg(xm + row0 / 2 + lane, __float2bfloat16_rn(g / (1.f + __expf(-g)) * u));
          }
        } else {
          if (lane < 4) __stcg(h + row0 + lane, __ldcg(h + row0 + lane) + a);
        }
      }
      advance(myu[ph.x + ph.y].y / CH);
      unsigned old = arrive();
      if (l == NL - 1 && op == 4 && t == 0 && old == (unsigned)G * nsp - 1) *last = 1;   // last block of the step
      if (l == NL - 1 && op == 4) {
        csync();
        if (*last) {                                       // final RMSNorm -> xout, reset the grid counter
          __threadfence();
          float v[8]; float ss = 0.f;
          if (t < H / 8) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) { v[i] = __ldcg(h + 8 * t + i); ss += v[i] * v[i]; }
          }
          for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
          if (lane == 0) red[warp] = ss;
          csync();
          float tot = 0.f;
          for (int i = 0; i < NW; ++i) tot += red[i];
          float r = rsqrtf(tot / H + eps);
          if (t < H / 8)
            for (int i = 0; i < 8; ++i) xout[8 * t + i] = __float2bfloat16_rn(v[i] * r * fnorm[8 * t + i]);
          if (t == 0) *sync = 0u;
        }
      }
    }
  }
  advance(nc);
}
""" % {"nl": nl, "maxlen": maxlen, "nw": nw, "ch": ch, "nch": nch, "xs": xs_bytes, "cap": cap, "body": _sub_body()}


def smem_bytes(ch: int, nch: int) -> int:
    return ch * nch + 17920 + 16 * nch + 16 * 4 + 16
