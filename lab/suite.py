"""Frozen benchmark suite of ordinary CUDA kernels (textbook code, no inline PTX, no hand scheduling).

Every case is one `extern "C" __global__ void k(...)` with the argument order
    input pointers..., output pointers..., scalars...
`make(rng)` returns (inputs, outputs, scalars): input arrays in kernel order, output (shape, dtype)
pairs in kernel order, and scalars as ("i32" | "f32", value). Outputs are zeroed before a checked
launch (the histogram accumulates into its output). `reference(inputs)` is the numpy answer; `tol` is
(rtol, atol) and `frac` the fraction of elements allowed outside it (only Mandelbrot, whose escape
counts flip on FMA contraction). Every kernel is deterministic (integer atomics only), so two
schedules of the same cubin must agree bitwise. Sizes target 20 us - 2 ms on an RTX 3080.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

F32, I32, U32 = np.float32, np.int32, np.uint32


@dataclass
class Case:
    name: str
    kind: str                     # memory | latency | compute
    src: str
    make: Callable
    reference: Callable
    grid: tuple
    block: tuple
    smem: int = 0
    tol: tuple = (1e-5, 1e-6)
    frac: float = 0.0


def compare(out: np.ndarray, ref: np.ndarray, tol: tuple, frac: float = 0.0) -> dict:
    out64, ref64 = out.astype(np.float64), np.asarray(ref, np.float64)
    err = np.abs(out64 - ref64)
    bad = ~(err <= tol[1] + tol[0] * np.abs(ref64))
    bad_frac = float(bad.mean()) if bad.size else 0.0
    return {"ok": bool(bad_frac <= frac), "max_abs": float(np.nanmax(err)) if err.size else 0.0,
            "bad_frac": bad_frac}


def _u(rng, shape, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, shape).astype(F32)


CASES: list[Case] = []


def case(**kw):
    CASES.append(Case(**kw))


# ---- memory bound ----------------------------------------------------------------------------------
N_SAXPY = 1 << 24
case(name="saxpy", kind="memory", grid=(N_SAXPY // 256,), block=(256,), tol=(1e-6, 1e-6), src=r"""
extern "C" __global__ void k(const float* x, const float* y, float* out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}""", make=lambda r: ([_u(r, N_SAXPY), _u(r, N_SAXPY)], [((N_SAXPY,), F32)], [("f32", 2.5), ("i32", N_SAXPY)]),
     reference=lambda i: [np.float32(2.5) * i[0].astype(np.float64) + i[1]])

N_RED, G_RED = 1 << 24, 1024
case(name="reduce", kind="memory", grid=(G_RED,), block=(256,), tol=(1e-4, 1e-3), src=r"""
extern "C" __global__ void k(const float* x, float* partial, int n) {
    __shared__ float s[256];
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += x[i];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int w = blockDim.x / 2; w >= 32; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        v = s[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) partial[blockIdx.x] = v;
    }
}""", make=lambda r: ([_u(r, N_RED)], [((G_RED,), F32)], [("i32", N_RED)]),
     reference=lambda i: [i[0].astype(np.float64).reshape(-1, G_RED, 256).sum(axis=(0, 2))])

T_N = 4096
case(name="transpose", kind="memory", grid=(T_N // 32, T_N // 32), block=(32, 8), tol=(0, 0), src=r"""
extern "C" __global__ void k(const float* in, float* out, int n) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}""", make=lambda r: ([_u(r, (T_N, T_N))], [((T_N, T_N), F32)], [("i32", T_N)]),
     reference=lambda i: [i[0].T])

N_S1, R_S1 = 1 << 24, 3
W_S1 = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
case(name="stencil1d", kind="memory", grid=(N_S1 // 256,), block=(256,), smem=(256 + 2 * R_S1) * 4, src=r"""
#define R 3
__constant__ float W[2 * R + 1] = {0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
extern "C" __global__ void k(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int g = blockIdx.x * blockDim.x + threadIdx.x, l = threadIdx.x + R;
    s[l] = in[min(g, n - 1)];
    if (threadIdx.x < R) {
        s[l - R] = in[max(g - R, 0)];
        s[l + blockDim.x] = in[min(g + (int)blockDim.x, n - 1)];
    }
    __syncthreads();
    float v = 0.f;
    for (int d = -R; d <= R; d++) v += W[d + R] * s[l + d];
    if (g < n) out[g] = v;
}""", make=lambda r: ([_u(r, N_S1)], [((N_S1,), F32)], [("i32", N_S1)]),
     reference=lambda i: [sum(W_S1[d] * np.pad(i[0].astype(np.float64), R_S1, mode="edge")[d:d + N_S1]
                              for d in range(2 * R_S1 + 1))])

J_N = 4096


def _jacobi(i):
    a = i[0]
    o = a.copy()
    o[1:-1, 1:-1] = np.float32(0.25) * (((a[:-2, 1:-1] + a[2:, 1:-1]) + a[1:-1, :-2]) + a[1:-1, 2:])
    return [o]


case(name="jacobi2d", kind="memory", grid=(J_N // 32, J_N // 8), block=(32, 8), tol=(1e-6, 1e-6), src=r"""
extern "C" __global__ void k(const float* in, float* out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int i = y * nx + x;
    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) out[i] = in[i];
    else out[i] = 0.25f * (in[i - nx] + in[i + nx] + in[i - 1] + in[i + 1]);
}""", make=lambda r: ([_u(r, (J_N, J_N))], [((J_N, J_N), F32)], [("i32", J_N), ("i32", J_N)]), reference=_jacobi)

CR_N, CR_R = 4096, 8


def _conv_row(i):
    a = np.pad(i[0].astype(np.float64), ((0, 0), (CR_R, CR_R)), mode="edge")
    return [sum(float(i[1][t]) * a[:, t:t + CR_N] for t in range(2 * CR_R + 1))]


case(name="conv_row", kind="memory", grid=(CR_N // 256, CR_N), block=(256,), src=r"""
#define R 8
extern "C" __global__ void k(const float* in, const float* w, float* out, int width) {
    __shared__ float s[256 + 2 * R];
    __shared__ float ws[2 * R + 1];
    const float* row = in + (size_t)blockIdx.y * width;
    int x = blockIdx.x * 256 + threadIdx.x;
    s[threadIdx.x + R] = row[x];
    if (threadIdx.x < R) {
        s[threadIdx.x] = row[max(x - R, 0)];
        s[threadIdx.x + 256 + R] = row[min(x + 256, width - 1)];
    }
    if (threadIdx.x < 2 * R + 1) ws[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    float v = 0.f;
    #pragma unroll
    for (int t = 0; t <= 2 * R; t++) v += ws[t] * s[threadIdx.x + t];
    out[(size_t)blockIdx.y * width + x] = v;
}""", make=lambda r: ([_u(r, (CR_N, CR_N)), (np.hanning(2 * CR_R + 3)[1:-1] / np.hanning(2 * CR_R + 3).sum()).astype(F32)],
                      [((CR_N, CR_N), F32)], [("i32", CR_N)]), reference=_conv_row)

BB_N = 4096


def _blur(i):
    a = np.pad(i[0].astype(np.float64), 1, mode="edge")
    return [sum(a[dy:dy + BB_N, dx:dx + BB_N] for dy in range(3) for dx in range(3)) / 9.0]


case(name="box_blur3", kind="memory", grid=(BB_N // 16, BB_N // 16), block=(16, 16), src=r"""
extern "C" __global__ void k(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = 0.f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int xx = min(max(x + dx, 0), w - 1), yy = min(max(y + dy, 0), h - 1);
            s += in[yy * w + xx];
        }
    out[y * w + x] = s * (1.0f / 9.0f);
}""", make=lambda r: ([_u(r, (BB_N, BB_N))], [((BB_N, BB_N), F32)], [("i32", BB_N), ("i32", BB_N)]), reference=_blur)

GE_R, GE_C = 8192, 1024


def _gelu(i):
    v = i[0].astype(np.float64) * i[1] + i[2]
    return [0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v ** 3)))]


case(name="gelu_fused", kind="memory", grid=(GE_R * GE_C // 256,), block=(256,), src=r"""
extern "C" __global__ void k(const float* x, const float* w, const float* b, float* out, int n, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % cols;
    float v = x[i] * w[c] + b[c];
    out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}""", make=lambda r: ([_u(r, (GE_R, GE_C), -3, 3), _u(r, GE_C, 0.5, 1.5), _u(r, GE_C, -0.5, 0.5)],
                      [((GE_R, GE_C), F32)], [("i32", GE_R * GE_C), ("i32", GE_C)]), reference=_gelu)

H_N, H_G = 1 << 24, 1024
case(name="histogram", kind="memory", grid=(H_G,), block=(256,), tol=(0, 0), src=r"""
extern "C" __global__ void k(const int* data, unsigned int* hist, int n) {
    __shared__ unsigned int s[256];
    s[threadIdx.x] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&s[data[i] & 255], 1u);
    __syncthreads();
    if (s[threadIdx.x]) atomicAdd(&hist[threadIdx.x], s[threadIdx.x]);
}""", make=lambda r: ([np.clip(r.normal(128, 40, H_N), 0, 255).astype(I32)], [((256,), U32)], [("i32", H_N)]),
     reference=lambda i: [np.bincount(i[0], minlength=256)])

GV_M, GV_K = 8192, 4096
case(name="gemv", kind="memory", grid=(GV_M // 8,), block=(256,), tol=(1e-4, 1e-4), src=r"""
extern "C" __global__ void k(const float* A, const float* x, float* y, int m, int kdim) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= m) return;
    const float* a = A + (size_t)row * kdim;
    float v = 0.f;
    for (int j = lane; j < kdim; j += 32) v += a[j] * x[j];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) y[row] = v;
}""", make=lambda r: ([_u(r, (GV_M, GV_K)), _u(r, GV_K)], [((GV_M,), F32)], [("i32", GV_M), ("i32", GV_K)]),
     reference=lambda i: [i[0].astype(np.float64) @ i[1].astype(np.float64)])

BS_N, BS_R, BS_V = 1 << 23, 0.02, 0.30


def _cnd(d):
    k = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    c = 0.3989422804014327 * np.exp(-0.5 * d * d) * (k * (0.31938153 + k * (-0.356563782 + k * (
        1.781477937 + k * (-1.821255978 + k * 1.330274429)))))
    return np.where(d > 0, 1 - c, c)


def _bs(i):
    s, x, t = (a.astype(np.float64) for a in i)
    sq = np.sqrt(t)
    d1 = (np.log(s / x) + (BS_R + 0.5 * BS_V * BS_V) * t) / (BS_V * sq)
    d2 = d1 - BS_V * sq
    e = np.exp(-BS_R * t)
    return [s * _cnd(d1) - x * e * _cnd(d2), x * e * (1 - _cnd(d2)) - s * (1 - _cnd(d1))]


case(name="black_scholes", kind="memory", grid=(BS_N // 256,), block=(256,), tol=(1e-4, 2e-4), src=r"""
__device__ float cnd(float d) {
    float k = 1.0f / (1.0f + 0.2316419f * fabsf(d));
    float c = 0.3989422804f * expf(-0.5f * d * d) *
              (k * (0.31938153f + k * (-0.356563782f + k * (1.781477937f + k * (-1.821255978f + k * 1.330274429f)))));
    return d > 0 ? 1.0f - c : c;
}
extern "C" __global__ void k(const float* S, const float* X, const float* T, float* call, float* put,
                             float r, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = S[i], x = X[i], t = T[i], sq = sqrtf(t);
    float d1 = (logf(s / x) + (r + 0.5f * v * v) * t) / (v * sq), d2 = d1 - v * sq;
    float c1 = cnd(d1), c2 = cnd(d2), e = expf(-r * t);
    call[i] = s * c1 - x * e * c2;
    put[i] = x * e * (1.0f - c2) - s * (1.0f - c1);
}""", make=lambda r: ([_u(r, BS_N, 5, 30), _u(r, BS_N, 1, 100), _u(r, BS_N, 0.25, 10)],
                      [((BS_N,), F32), ((BS_N,), F32)], [("f32", BS_R), ("f32", BS_V), ("i32", BS_N)]), reference=_bs)

# ---- latency bound (irregular access, shuffles, barriers, short dependent chains) -------------------
SP_ROWS = 1 << 18


def _spmv_make(r):
    cnt = r.integers(16, 49, SP_ROWS)
    rowptr = np.concatenate([[0], np.cumsum(cnt)]).astype(I32)
    nnz = int(rowptr[-1])
    return ([rowptr, r.integers(0, SP_ROWS, nnz).astype(I32), _u(r, nnz), _u(r, SP_ROWS)],
            [((SP_ROWS,), F32)], [("i32", SP_ROWS)])


case(name="spmv_csr", kind="latency", grid=(SP_ROWS // 8,), block=(256,), tol=(1e-4, 1e-4), src=r"""
extern "C" __global__ void k(const int* rowptr, const int* cols, const float* vals, const float* x, float* y,
                             int rows) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    float v = 0.f;
    for (int j = rowptr[row] + lane; j < rowptr[row + 1]; j += 32) v += vals[j] * x[cols[j]];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) y[row] = v;
}""", make=_spmv_make, reference=lambda i: [np.add.reduceat(i[2].astype(np.float64) * i[3][i[1]], i[0][:-1])])

SM_R, SM_C = 8192, 1024


def _softmax(i):
    a = i[0].astype(np.float64)
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return [e / e.sum(axis=1, keepdims=True)]


case(name="softmax", kind="latency", grid=(SM_R // 4,), block=(128,), tol=(1e-5, 1e-7), src=r"""
extern "C" __global__ void k(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* a = in + (size_t)row * cols;
    float m = -3.402823466e38f;
    for (int j = lane; j < cols; j += 32) m = fmaxf(m, a[j]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    float s = 0.f;
    for (int j = lane; j < cols; j += 32) s += expf(a[j] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    float inv = 1.0f / s;
    for (int j = lane; j < cols; j += 32) out[(size_t)row * cols + j] = expf(a[j] - m) * inv;
}""", make=lambda r: ([_u(r, (SM_R, SM_C), -4, 4)], [((SM_R, SM_C), F32)], [("i32", SM_R), ("i32", SM_C)]),
     reference=_softmax)

LN_R, LN_C = 8192, 1024


def _layernorm(i):
    a = i[0].astype(np.float64)
    mu = a.mean(axis=1, keepdims=True)
    var = ((a - mu) ** 2).mean(axis=1, keepdims=True)
    return [(a - mu) / np.sqrt(var + 1e-5) * i[1] + i[2]]


case(name="layernorm", kind="latency", grid=(LN_R,), block=(256,), tol=(1e-4, 1e-4), src=r"""
__device__ float block_sum(float v, float* s) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x % 32 == 0) s[threadIdx.x / 32] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? s[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    v = s[0];
    __syncthreads();
    return v;
}
extern "C" __global__ void k(const float* in, const float* gamma, const float* beta, float* out, int cols) {
    __shared__ float s[32];
    const float* a = in + (size_t)blockIdx.x * cols;
    float sum = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) sum += a[j];
    float mean = block_sum(sum, s) / cols;
    float sq = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) { float d = a[j] - mean; sq += d * d; }
    float rstd = rsqrtf(block_sum(sq, s) / cols + 1e-5f);
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        out[(size_t)blockIdx.x * cols + j] = (a[j] - mean) * rstd * gamma[j] + beta[j];
}""", make=lambda r: ([_u(r, (LN_R, LN_C), -2, 3), _u(r, LN_C, 0.5, 1.5), _u(r, LN_C, -0.5, 0.5)],
                      [((LN_R, LN_C), F32)], [("i32", LN_C)]), reference=_layernorm)

SC_N = 1 << 24
case(name="scan_block", kind="latency", grid=(SC_N // 1024,), block=(1024,), tol=(0, 0), src=r"""
extern "C" __global__ void k(const int* in, int* out, int n) {
    __shared__ int ws[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int v = i < n ? in[i] : 0;
    for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xffffffff, v, o); if (lane >= o) v += t; }
    if (lane == 31) ws[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int s = ws[lane];
        for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xffffffff, s, o); if (lane >= o) s += t; }
        ws[lane] = s;
    }
    __syncthreads();
    if (warp > 0) v += ws[warp - 1];
    if (i < n) out[i] = v;
}""", make=lambda r: ([r.integers(0, 100, SC_N).astype(I32)], [((SC_N,), I32)], [("i32", SC_N)]),
     reference=lambda i: [np.cumsum(i[0].reshape(-1, 1024), axis=1).reshape(-1)])

BT_N = 1 << 22
case(name="bitonic_block", kind="latency", grid=(BT_N // 1024,), block=(512,), tol=(0, 0), src=r"""
extern "C" __global__ void k(const float* in, float* out) {
    __shared__ float s[1024];
    int t = threadIdx.x, base = blockIdx.x * 1024;
    s[t] = in[base + t];
    s[t + 512] = in[base + t + 512];
    __syncthreads();
    for (int kk = 2; kk <= 1024; kk <<= 1)
        for (int j = kk >> 1; j > 0; j >>= 1) {
            int i = 2 * t - (t & (j - 1)), p = i + j;
            float a = s[i], b = s[p];
            if ((a > b) == ((i & kk) == 0)) { s[i] = b; s[p] = a; }
            __syncthreads();
        }
    out[base + t] = s[t];
    out[base + t + 512] = s[t + 512];
}""", make=lambda r: ([_u(r, BT_N)], [((BT_N,), F32)], []),
     reference=lambda i: [np.sort(i[0].reshape(-1, 1024), axis=1).reshape(-1)])

# ---- compute bound ---------------------------------------------------------------------------------
GM_N = 1024
case(name="sgemm_tiled", kind="compute", grid=(GM_N // 64, GM_N // 64), block=(256,), tol=(1e-4, 1e-3), src=r"""
#define BM 64
#define BN 64
#define BK 16
extern "C" __global__ void k(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) As[i % BK][i / BK] = A[(row0 + i / BK) * K + k0 + i % BK];
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) Bs[i / BN][i % BN] = B[(k0 + i / BN) * N + col0 + i % BN];
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < BK; p++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = As[p][ty * 4 + i]; b[i] = Bs[p][tx * 4 + i]; }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) C[(row0 + ty * 4 + i) * N + col0 + tx * 4 + j] = acc[i][j];
}""", make=lambda r: ([_u(r, (GM_N, GM_N)), _u(r, (GM_N, GM_N))], [((GM_N, GM_N), F32)],
                      [("i32", GM_N)] * 3), reference=lambda i: [i[0].astype(np.float64) @ i[1].astype(np.float64)])

GN_N = 768
case(name="sgemm_naive", kind="compute", grid=(GN_N // 16, GN_N // 16), block=(16, 16), tol=(1e-4, 1e-3), src=r"""
extern "C" __global__ void k(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float v = 0.f;
    for (int p = 0; p < K; p++) v += A[row * K + p] * B[p * N + col];
    C[row * N + col] = v;
}""", make=lambda r: ([_u(r, (GN_N, GN_N)), _u(r, (GN_N, GN_N))], [((GN_N, GN_N), F32)],
                      [("i32", GN_N)] * 3), reference=lambda i: [i[0].astype(np.float64) @ i[1].astype(np.float64)])

NB_N, NB_EPS2 = 16384, 1e-2


def _nbody_make(r):
    p = np.empty((NB_N, 4), F32)
    p[:, :3] = _u(r, (NB_N, 3))
    p[:, 3] = 1.0 / NB_N
    return [p], [((NB_N, 4), F32)], [("i32", NB_N), ("f32", NB_EPS2)]


def _nbody(i):
    p = i[0].astype(np.float64)
    acc = np.zeros((NB_N, 4))
    for s in range(0, NB_N, 512):
        d = p[None, :, :3] - p[s:s + 512, None, :3]
        inv = (np.einsum("ijk,ijk->ij", d, d) + NB_EPS2) ** -1.5 * p[None, :, 3]
        acc[s:s + 512, :3] = np.einsum("ij,ijk->ik", inv, d)
    return [acc]


case(name="nbody", kind="compute", grid=(NB_N // 256,), block=(256,), tol=(1e-3, 1e-4), src=r"""
extern "C" __global__ void k(const float4* pos, float4* acc, int n, float eps2) {
    __shared__ float4 tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 me = pos[i];
    float ax = 0.f, ay = 0.f, az = 0.f;
    for (int t = 0; t < n; t += blockDim.x) {
        tile[threadIdx.x] = pos[t + threadIdx.x];
        __syncthreads();
        #pragma unroll 8
        for (int j = 0; j < blockDim.x; j++) {
            float4 o = tile[j];
            float dx = o.x - me.x, dy = o.y - me.y, dz = o.z - me.z;
            float r2 = dx * dx + dy * dy + dz * dz + eps2;
            float inv = rsqrtf(r2);
            float s = o.w * inv * inv * inv;
            ax += dx * s; ay += dy * s; az += dz * s;
        }
        __syncthreads();
    }
    acc[i] = make_float4(ax, ay, az, 0.f);
}""", make=_nbody_make, reference=_nbody)

MB_W, MB_H, MB_IT = 1024, 1024, 512
MB_X0, MB_Y0, MB_D = -2.0, -1.25, 2.5 / 1024


def _mandel(i):
    cx = (np.float32(MB_X0) + np.arange(MB_W, dtype=F32) * np.float32(MB_D))[None, :].repeat(MB_H, 0).ravel()
    cy = (np.float32(MB_Y0) + np.arange(MB_H, dtype=F32) * np.float32(MB_D))[:, None].repeat(MB_W, 1).ravel()
    it = np.zeros(cx.size, I32)
    idx = np.arange(cx.size)
    zx, zy = np.zeros_like(cx), np.zeros_like(cy)
    for n in range(MB_IT):
        live = zx * zx + zy * zy <= 4
        idx, zx, zy, cx, cy = idx[live], zx[live], zy[live], cx[live], cy[live]
        if not idx.size:
            break
        it[idx] = n + 1
        zx, zy = zx * zx - zy * zy + cx, np.float32(2) * zx * zy + cy
    return [it.reshape(MB_H, MB_W)]


case(name="mandelbrot", kind="compute", grid=(MB_W // 16, MB_H // 16), block=(16, 16), tol=(0, 0), frac=0.01, src=r"""
extern "C" __global__ void k(int* out, int w, int h, float x0, float y0, float d, int max_iter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    float cx = x0 + px * d, cy = y0 + py * d, zx = 0.f, zy = 0.f;
    int n = 0;
    while (n < max_iter && zx * zx + zy * zy <= 4.0f) {
        float t = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = t;
        n++;
    }
    out[py * w + px] = n;
}""", make=lambda r: ([], [((MB_H, MB_W), I32)], [("i32", MB_W), ("i32", MB_H), ("f32", MB_X0), ("f32", MB_Y0),
                                                ("f32", MB_D), ("i32", MB_IT)]), reference=_mandel)
