"""Chunked prefill of Qwen2.5-1.5B (llama.cpp's Q4_0 GGUF weights): P prompt tokens of one sequence per pass.

  pcslurm submit -- python scripts/qwen_prefill.py [P=128] [chunk=32,128] [modes=v2,cublas]

Per layer: RMSNorm, QKV, a K/V-write pass for all chunk tokens, causal attention (one block per head
and token, the decode attention kernel with a token index from blockIdx.y on a shared KV cache),
O + residual, RMSNorm, gate_up + SwiGLU, down + residual. GEMMs:
  v2      lab/qwen_batched.gemv_v2_source with B = chunk (chunk <= 32)
  cublas  each layer's Q4_0 weights dequantized to fp16 into a scratch buffer, then torch.matmul
  i8      lab/qwen_batched.gemm_q4i8_source: int8 tensor cores on Q4_0 x per-32 int8 activations
          (llama.cpp MMQ-style), epilogues (bias, residual, SwiGLU) fused
Correctness: the KV cache and the last token's logits against token-by-token prefill through the
batched decode step (scripts/qwen_batched_decode.py, B = 1). Throughput: prompt tokens / time of
the whole prefill (CUDA graph), as llama-batched-bench's PP.
"""
import ctypes
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_batched_decode as D  # noqa: E402
import qwen_e2e as Q  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402
from lab import qwen_kernels as KS  # noqa: E402

dev = "cuda"
Kern, batchify, check, cu = D.Kern, D.batchify, D.check, D.cu

DEQ = r"""
#include <cuda_fp16.h>
extern "C" __global__ void k(const unsigned* __restrict__ W, const __half* __restrict__ S, __half2* __restrict__ out, int N, int K)
{
  // one thread per 32-bit word (8 weights) of Q4_0: out[n, 8w .. 8w+7] = (q - 8) * d
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * (K / 8)) return;
  size_t n = i / (K / 8), w = i % (K / 8);
  unsigned q = W[i];
  __half2 d = __half2half2(S[n * (K / 32) + w / 4]);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    unsigned h = ((q >> (8 * j)) & 0xFu) | (((q >> (8 * j + 4)) & 0xFu) << 16) | 0x64006400u;
    out[i * 4 + j] = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(1032.f)), d);
  }
}
"""

# v2 configs per chunk bucket (from the decode sweeps; B = chunk)
V2CFG = D.CONFIG2
FA = "fa=0" not in sys.argv
I8CFG = (2, 2, 2, 4, 4, 3)       # gemm_q4i8 (WM, WN, MT, NT, KB, MINB), scripts/prefill_gemm_bench.py sweep


class Prefill:
    def __init__(self, m, P, mode):
        self.m, self.P, self.mode = m, P, mode
        H, nh, nkv, hd = m.H, m.nh, m.nkv, m.hd
        self.nq = (nh + 2 * nkv) * hd
        nL = len(m.L)
        self.h = torch.zeros(P, H, device=dev)
        self.x = torch.zeros(P, H, dtype=torch.bfloat16, device=dev)
        self.qkv = torch.zeros(P, self.nq, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(P, nh * hd, dtype=torch.bfloat16, device=dev)
        self.xm = torch.zeros(P, m.inter, dtype=torch.bfloat16, device=dev)
        self.tok = torch.zeros(P, dtype=torch.long, device=dev)
        self.pos = torch.zeros(P, dtype=torch.long, device=dev)
        self.kc = torch.zeros(nL, nkv, Q.MAX_LEN, hd, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.k_emb = Kern(batchify(KS.EMBED_F32, "tok += b_; h += (size_t)b_ * H;"))
        self.k_rms = Kern(batchify(KS.RMSNORM_F32W_1P, "h += (size_t)b_ * H; out += (size_t)b_ * H;"))
        attn = KS.ATTN4 % {"maxlen": Q.MAX_LEN}
        off = "qkv += (size_t)b_ * (NH + 2 * NKV) * HD; posp += b_; out += (size_t)b_ * NH * HD;"
        kvw = attn.replace("  __syncthreads();\n  float q0 = q[4 * lane]", "  return;\n  __syncthreads();\n  float q0 = q[4 * lane]")
        assert kvw != attn
        self.k_kvw = Kern(batchify(kvw, off))
        self.k_attn = Kern(batchify(attn, off))
        self.fa = FA                                    # tensor-core causal attention instead of ATTN4 per token
        if FA:
            self.k_fa = Kern(QB.flash_prefill_source(4, Q.MAX_LEN))
        if mode == "v2":
            bucket = 8 if P <= 8 else 16 if P <= 16 else 32
            assert P <= 32
            self.cfg = V2CFG[bucket]
            self.g = {name: Kern(QB.gemv_v2_source(*self.cfg[name][:3], P, epi, 1, self.cfg[name][3]))
                      for name, epi in (("qkv", "biasf"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"))}
            if not hasattr(m, "packed"):
                m.packed = [{name: QB.pack_q4_mma(*L[name]) for name in ("qkv", "o", "gu", "down")} for L in m.L]
            self.ws = torch.zeros(4 * P * 2 * m.inter, device=dev)
            self.cnt = torch.zeros(2048, dtype=torch.int32, device=dev)
        elif mode == "i8":
            WM, WN, MT, NT, KB, MINB = I8CFG
            self.BM, self.BN, self.nthr = 16 * MT * WM, 8 * NT * WN, 32 * WM * WN
            self.g = {name: Kern(QB.gemm_q4i8_source(WM, WN, MT, NT, epi, KB, MINB))
                      for name, epi in (("qkv", "biasf"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"))}
            self.k_q = Kern(QB.QUANT_Q8)
            if not hasattr(m, "packed_i8"):
                m.packed_i8 = [{name: QB.pack_q4_i8(*L[name]) for name in ("qkv", "o", "gu", "down")} for L in m.L]
            kmax = max(H, m.inter)
            self.xq = torch.zeros(P, kmax, dtype=torch.int8, device=dev)
            self.xd = torch.zeros(kmax // 32, P, device=dev)
            self.xs = torch.zeros(kmax // 32, P, dtype=torch.int32, device=dev)
        else:
            self.k_deq = Kern(DEQ)
            self.w16 = torch.empty(2 * m.inter * H, dtype=torch.float16, device=dev)   # largest layer matrix

    def deq(self, W, S, N, K):
        n = N * (K // 8)
        self.k_deq.launch((n + 255) // 256, 256, [W.data_ptr(), S.data_ptr(), self.w16.data_ptr(), ctypes.c_int32(N),
                                                  ctypes.c_int32(K)])
        return self.w16[:N * K].view(N, K)

    def gemm(self, name, L, X, N, K):
        """-> fp32 [P, N] = X W^T via dequantized fp16 weights (cublas mode)."""
        w = self.deq(*L[name], N, K)
        return torch.matmul(X.to(torch.float16), w.t()).float()

    def gemv(self, i, name, X, Y, aux, N, K):
        W, S = self.m.packed[i][name]
        WM, WK, MT, KS_ = self.cfg[name]
        R = 16 * MT * WM
        self.g[name].launch(((N + R - 1) // R, KS_), 32 * WM * WK,
                            [W.data_ptr(), S.data_ptr(), X.data_ptr(), Y.data_ptr(), aux, ctypes.c_int32(N),
                             ctypes.c_int32(K), self.ws.data_ptr(), self.cnt.data_ptr()])

    def gemm8(self, i, name, X, Y, aux, N, K):
        P = self.P
        nb = P * K // 32
        self.k_q.launch((nb + 7) // 8, 256, [X.data_ptr(), self.xq.data_ptr(), self.xd.data_ptr(), self.xs.data_ptr(),
                                             ctypes.c_int32(P), ctypes.c_int32(K)])
        W, S = self.m.packed_i8[i][name]
        self.g[name].launch((N // self.BM, (P + self.BN - 1) // self.BN), self.nthr,
                            [W.data_ptr(), S.data_ptr(), self.xq.data_ptr(), self.xd.data_ptr(), self.xs.data_ptr(),
                             Y.data_ptr(), aux, ctypes.c_int32(N), ctypes.c_int32(K), ctypes.c_int32(P)])

    def rms(self, w):
        self.k_rms.launch((1, self.P), 512, [self.h.data_ptr(), w.data_ptr(), self.x.data_ptr(),
                                             ctypes.c_int32(self.m.H), ctypes.c_float(self.m.eps)])

    def chunk(self):
        """One pass over P tokens at positions self.pos (filled by the caller)."""
        m, P, H = self.m, self.P, self.m.H
        self.k_emb.launch((4, P), 512, [m.embed.data_ptr(), self.tok.data_ptr(), self.h.data_ptr(), ctypes.c_int32(H)])
        scale = ctypes.c_float(1.0 / math.sqrt(m.hd))
        for i, L in enumerate(m.L):
            self.rms(L["ln1"])
            if self.mode == "i8":
                self.gemm8(i, "qkv", self.x, self.qkv, L["qkv_b"].data_ptr(), self.nq, H)
            elif self.mode == "v2":
                self.gemv(i, "qkv", self.x, self.qkv, L["qkv_b"].data_ptr(), self.nq, H)
            else:
                self.qkv.copy_(self.gemm("qkv", L, self.x, self.nq, H) + L["qkv_b"])
            args = [self.qkv.data_ptr(), m.cos.data_ptr(), m.sin.data_ptr(), self.pos.data_ptr(), self.kc[i].data_ptr(),
                    self.vc[i].data_ptr(), self.att.data_ptr(), ctypes.c_int32(m.nh), ctypes.c_int32(m.nkv), scale]
            self.k_kvw.launch((m.nh, P), 512, args)
            if self.fa:
                self.k_fa.launch((m.nh, (P + 63) // 64), 128, args[:6] + [self.att.data_ptr(), ctypes.c_int32(m.nh),
                                                                           ctypes.c_int32(m.nkv), scale, ctypes.c_int32(P)])
            else:
                self.k_attn.launch((m.nh, P), 512, args)
            if self.mode == "i8":
                self.gemm8(i, "o", self.att, self.h, 0, H, m.nh * m.hd)
            elif self.mode == "v2":
                self.gemv(i, "o", self.att, self.h, 0, H, m.nh * m.hd)
            else:
                self.h += self.gemm("o", L, self.att, H, m.nh * m.hd)
            self.rms(L["ln2"])
            if self.mode == "i8":
                self.gemm8(i, "gu", self.x, self.xm, 0, 2 * m.inter, H)
                self.gemm8(i, "down", self.xm, self.h, 0, H, m.inter)
            elif self.mode == "v2":
                self.gemv(i, "gu", self.x, self.xm, 0, 2 * m.inter, H)
                self.gemv(i, "down", self.xm, self.h, 0, H, m.inter)
            else:
                gu = self.gemm("gu", L, self.x, 2 * m.inter, H)
                self.xm.copy_(TF.silu(gu[:, 0::2]) * gu[:, 1::2])
                self.h += self.gemm("down", L, self.xm, H, m.inter)


def logits_last(m, h_last):
    """fp32 logits of one hidden state through the final norm and the v2 Q6_K head (B = 1)."""
    x = (h_last * torch.rsqrt(h_last.pow(2).mean() + m.eps) * m.norm).to(torch.bfloat16)[None].contiguous()
    if not hasattr(m, "head_packed"):
        m.head_packed = QB.pack_q6_mma(*m.head)
    k = Kern(QB.head_v2_source(8, 1, 1, 1))
    y = torch.zeros(1, m.V, device=dev)
    k.launch(((m.V + 127) // 128), 256, [*(t.data_ptr() for t in m.head_packed), x.data_ptr(), y.data_ptr(),
                                          ctypes.c_int32(m.V), ctypes.c_int32(m.H)])
    torch.cuda.synchronize()
    return y[0]


def prefill(m, ids, chunk, mode):
    """Prefill ids in chunks; returns (Prefill object holding the KV cache and h of the last chunk, ms)."""
    P = len(ids)
    assert P % chunk == 0
    pf = Prefill(m, chunk, mode)
    ids_t = torch.tensor(ids, device=dev)
    starts = list(range(0, P, chunk))

    def run_all():
        for s in starts:
            pf.tok.copy_(ids_t[s:s + chunk])
            pf.pos.copy_(torch.arange(s, s + chunk, device=dev))
            pf.chunk()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run_all()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return pf, sorted(ts)[len(ts) // 2]


def reference(m, ids):
    """Token-by-token prefill through the batched decode step with B = 1 (validated decode path)."""
    bm = D.Batched(m, 1)
    hs = []
    for t in ids:
        bm.tok.fill_(t)
        bm.step()
    torch.cuda.synchronize()
    return bm


def perplexity(m, P):
    """Mean next-token NLL on real text (the first P tokens of README.md), all positions through the final
    norm and the fp16 head, for each GEMM mode in one chunk: the quality cost of int8 activations."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(next(Q.MODEL.iterdir())))
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    ids = tk(text, add_special_tokens=False)["input_ids"][:P]
    if not hasattr(m, "head16"):
        m.head16 = D.head16(m)
    tgt = torch.tensor(ids[1:], device=dev)
    for mode in ("cublas", "i8"):
        pf, _ = prefill(m, ids, P, mode)
        h = pf.h
        x = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + m.eps) * m.norm).half()
        lp = torch.log_softmax(torch.matmul(x, m.head16.t()).float(), -1)[:-1]
        nll = float(-lp.gather(1, tgt[:, None]).mean())
        print(f"{mode:6s} P={P}: mean NLL {nll:.4f}, perplexity {math.exp(nll):.3f}", flush=True)
        del pf
        torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = dict(a.split("=") for a in sys.argv[1:])
    P = int(args.get("P", 128))
    chunks = [int(x) for x in args.get("chunk", "32,128").split(",")]
    modes = args.get("modes", "v2,cublas,i8").split(",")
    torch.zeros(1, device=dev)
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    m = D.load_model()
    if "ppl" in args:
        perplexity(m, P)
        return
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 150000, (P,), generator=g).tolist()
    ref = reference(m, ids)
    # the decode step's state after the last token: KV cache [nL, 1, nkv, MAXLEN, hd]; logits via its own head
    ref_logits = ref.logits[0].clone() if hasattr(ref, "logits") else None
    for mode in modes:
        for chunk in chunks:
            if mode == "v2" and chunk > 32:
                continue
            pf, ms = prefill(m, ids, chunk, mode)
            kerr = float((pf.kc[:, :, :P].float() - ref.kc[:, 0, :, :P].float()).abs().max() / ref.kc[:, 0, :, :P].float().abs().max())
            verr = float((pf.vc[:, :, :P].float() - ref.vc[:, 0, :, :P].float()).abs().max() / ref.vc[:, 0, :, :P].float().abs().max())
            lg = logits_last(m, pf.h[chunk - 1])
            lerr = float((lg - ref_logits).abs().max() / ref_logits.abs().max()) if ref_logits is not None else float("nan")
            same = int(lg.argmax()) == int(ref_logits.argmax()) if ref_logits is not None else None
            print(f"{mode:6s} chunk={chunk:4d} P={P}: {P / ms * 1e3:8.0f} tok/s ({ms:.2f} ms) | K err {kerr:.1e} V err {verr:.1e} "
                  f"| last logits err {lerr:.1e}, argmax same: {same}", flush=True)
            del pf
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
