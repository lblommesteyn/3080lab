"""Batched decode of Qwen2.5-1.5B (llama.cpp's Q4_0 GGUF weights): B sequences per step.

  pcslurm submit -- python scripts/qwen_batched_decode.py 1,4,8,16,32

GEMVs: tensor-core Q4_0 kernels (lab/qwen_batched.gemv_mma_source). Embedding, RMSNorm, attention
and argmax are the single-sequence kernels with a batch index from blockIdx.y offsetting their
pointers. LM head: tensor-core Q6_K kernel (lab/qwen_batched.head_q6_mma_source, 191 MB per step) up
to B = 16; at B = 32 that kernel is slower than cuBLAS on the head dequantized once to fp16 (467 MB).
Correctness: B copies of one prompt must give identical sequences, compared to the validated
single-sequence decode (scripts/qwen_gguf.py) token by token.
Throughput: tok/s = B x generated tokens / time, CUDA-graph replay, like llama-batched-bench TG.
"""
import ctypes
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_e2e as Q  # noqa: E402
import qwen_fused as F  # noqa: E402
import qwen_gguf as GG  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402
from lab import qwen_kernels as KS  # noqa: E402
from lab import toolchain  # noqa: E402
from lab.gguf import GGUF, q6_k_blocks  # noqa: E402

dev = "cuda"
cu, check = F.cu, F.check
GEN = 128
CONFIG = {
    8: {"qkv": (8, 2, 2), "o": (8, 4, 2), "gu": (8, 2, 2), "down": (16, 2, 2)},
    16: {"qkv": (8, 4, 2), "o": (8, 4, 2), "gu": (8, 2, 2), "down": (16, 2, 2)},
    32: {"qkv": (8, 4, 2), "o": (8, 4, 2), "gu": (4, 2, 4), "down": (8, 4, 2)},
}
# v2 GEMVs (packed fragment-order weights): (WM, WK, MT, KS) per shape, from batched_gemv_bench.py v2 sweeps
CONFIG2 = {
    8: {"qkv": (1, 4, 1, 4), "o": (1, 8, 1, 2), "gu": (4, 2, 2, 1), "down": (1, 4, 1, 4)},
    16: {"qkv": (2, 4, 1, 2), "o": (2, 4, 1, 2), "gu": (2, 4, 1, 1), "down": (1, 4, 1, 4)},
    32: {"qkv": (2, 4, 1, 2), "o": (2, 4, 1, 1), "gu": (2, 4, 1, 1), "down": (2, 2, 1, 4)},
}
V2 = "--v1" not in sys.argv
HEAD = {8: (6, 4, 1), 16: (8, 2, 2), 32: None}      # Q6_K head (S, U, MT), from scripts/head_bench.py; None = cuBLAS fp16


class Kern:
    def __init__(self, src):
        b = toolchain.build(src)
        self.fn = check(cu.cuModuleGetFunction(check(cu.cuModuleLoadData(b.cubin)), b"k"))
        self.keep = []

    def launch(self, grid, block, args):
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        a = [x if isinstance(x, (ctypes.c_uint64, ctypes.c_int32, ctypes.c_float)) else ctypes.c_uint64(x) for x in args]
        p = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
        self.keep.append((a, p))
        st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(self.fn, gx, gy, 1, block, 1, 1, 0, st, ctypes.addressof(p), 0))


def batchify(src: str, lines: str) -> str:
    """Insert pointer offsets (using b = blockIdx.y) at the top of kernel k's body."""
    m = re.search(r"\bk\s*\(", src)
    depth, i = 0, m.end() - 1
    while True:                                     # end of the parameter list
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    brace = src.index("{", i)
    return src[:brace + 1] + "\n  const int b_ = blockIdx.y;\n  " + lines + "\n" + src[brace + 1:]


class Batched:
    def __init__(self, m, B):
        self.m, self.B = m, B
        H, nh, nkv, hd = m.H, m.nh, m.nkv, m.hd
        self.nq = (nh + 2 * nkv) * hd
        nL = len(m.L)
        self.h = torch.zeros(B, H, device=dev)
        self.x = torch.zeros(B, H, dtype=torch.bfloat16, device=dev)
        self.qkv = torch.zeros(B, self.nq, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(B, nh * hd, dtype=torch.bfloat16, device=dev)
        self.xm = torch.zeros(B, m.inter, dtype=torch.bfloat16, device=dev)
        self.tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.pos = torch.zeros(B, dtype=torch.long, device=dev)
        self.kc = torch.zeros(nL, B, nkv, Q.MAX_LEN, hd, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        # (S warps, U blocks in flight, MT m-tiles) per shape and batch bucket, from batched_gemv_bench sweeps
        bucket = 8 if B <= 8 else 16 if B <= 16 else 32
        cfg = CONFIG[bucket]
        self.cfg = cfg
        if V2:
            self.cfg = cfg = CONFIG2[bucket]
            self.g = {name: Kern(QB.gemv_v2_source(cfg[name][0], cfg[name][1], cfg[name][2], B, epi, 1, cfg[name][3]))
                      for name, epi in (("qkv", "biasf"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"))}
            if not hasattr(m, "packed"):
                m.packed = [{name: QB.pack_q4_mma(*L[name]) for name in ("qkv", "o", "gu", "down")} for L in m.L]
            self.ws = torch.zeros(4 * B * 2 * m.inter, device=dev)
            self.cnt = torch.zeros(2048, dtype=torch.int32, device=dev)
        else:
            self.g = {name: Kern(QB.gemv_mma_source(cfg[name][0], B, epi, cfg[name][1], cfg[name][2])) for name, epi in
                      (("qkv", "biasf"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"))}
        self.k_emb = Kern(batchify(KS.EMBED_F32, "tok += b_; h += (size_t)b_ * H;"))
        self.k_rms = Kern(batchify(KS.RMSNORM_F32W_1P, "h += (size_t)b_ * H; out += (size_t)b_ * H;"))
        attn = KS.ATTN4 % {"maxlen": Q.MAX_LEN}
        self.k_attn = Kern(batchify(attn, "qkv += (size_t)b_ * (NH + 2 * NKV) * HD; posp += b_; "
                                          "out += (size_t)b_ * NH * HD; "
                                          "kc += (size_t)b_ * NKV * MAXLEN * HD; vc += (size_t)b_ * NKV * MAXLEN * HD;"))
        self.hcfg = HEAD[bucket]
        if self.hcfg:
            S_, U, MT = self.hcfg
            self.k_head = Kern(QB.head_q6_mma_source(S_, B, U, MT))
            self.logits = torch.zeros(B, m.V, device=dev)
        else:
            if not hasattr(m, "head16"):
                m.head16 = head16(m)
            self.head16 = m.head16

    def gemv(self, name, wsplit, X, Y, aux, N, K):
        if V2:
            W, S = self.m.packed[self.layer][name]
            WM, WK, MT, KS = self.cfg[name]
            R = 16 * MT * WM
            self.g[name].launch(((N + R - 1) // R, KS), 32 * WM * WK,
                                [W.data_ptr(), S.data_ptr(), X.data_ptr(), Y.data_ptr(), aux, ctypes.c_int32(N),
                                 ctypes.c_int32(K), self.ws.data_ptr(), self.cnt.data_ptr()])
            return
        W, S = wsplit
        S_, _, MT = self.cfg[name]
        self.g[name].launch((N + 16 * MT - 1) // (16 * MT), 32 * S_, [W.data_ptr(), S.data_ptr(), X.data_ptr(), Y.data_ptr(),
                                                                       aux, ctypes.c_int32(N), ctypes.c_int32(K)])

    def rms(self, w):
        self.k_rms.launch((1, self.B), 512, [self.h.data_ptr(), w.data_ptr(), self.x.data_ptr(),
                                             ctypes.c_int32(self.m.H), ctypes.c_float(self.m.eps)])

    def step(self):
        m, B, H = self.m, self.B, self.m.H
        self.k_emb.launch((4, B), 512, [m.embed.data_ptr(), self.tok.data_ptr(), self.h.data_ptr(), ctypes.c_int32(H)])
        scale = ctypes.c_float(1.0 / math.sqrt(m.hd))
        for i, L in enumerate(m.L):
            self.layer = i
            self.rms(L["ln1"])
            self.gemv("qkv", L["qkv"], self.x, self.qkv, L["qkv_b"].data_ptr(), self.nq, H)
            self.k_attn.launch((m.nh, B), 512, [self.qkv.data_ptr(), m.cos.data_ptr(), m.sin.data_ptr(),
                                                self.pos.data_ptr(), self.kc[i].data_ptr(), self.vc[i].data_ptr(),
                                                self.att.data_ptr(), ctypes.c_int32(m.nh), ctypes.c_int32(m.nkv), scale])
            self.gemv("o", L["o"], self.att, self.h, 0, H, m.nh * m.hd)
            self.rms(L["ln2"])
            self.gemv("gu", L["gu"], self.x, self.xm, 0, 2 * m.inter, H)
            self.gemv("down", L["down"], self.xm, self.h, 0, H, m.inter)
        self.rms(m.norm)
        if self.hcfg:
            S_, _, MT = self.hcfg
            V = m.V
            self.k_head.launch((V + 16 * MT - 1) // (16 * MT), 32 * S_, [*(t.data_ptr() for t in m.head), self.x.data_ptr(),
                                                                         self.logits.data_ptr(), ctypes.c_int32(V), ctypes.c_int32(H)])
            logits = self.logits
        else:
            logits = torch.matmul(self.x.to(torch.float16), self.head16.t())
        self.tok.copy_(logits.argmax(-1))
        self.pos += 1


def head16(m):
    g = GGUF(GG.GGUF_PATH)
    b, t = g.raw("output.weight")
    K, N = t.dims
    q, s = q6_k_blocks(b)
    w = (torch.from_numpy(q.astype(np.float32)).to(dev) * torch.from_numpy(s.astype(np.float32)).to(dev).repeat_interleave(16, 1))
    return w.reshape(N, K).to(torch.float16).contiguous()


def load_model():
    return GG.GGUFQwen(None)


def run(m, B, ids):
    bm = Batched(m, B)
    for t in ids:                                   # prefill one token at a time (all sequences alike)
        bm.tok.fill_(t)
        bm.step()
    torch.cuda.synchronize()
    pos0, tok0 = bm.pos.clone(), bm.tok.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        bm.step()
    torch.cuda.current_stream().wait_stream(side)
    bm.pos.copy_(pos0); bm.tok.copy_(tok0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bm.step()
    bm.pos.copy_(pos0); bm.tok.copy_(tok0)
    torch.cuda.synchronize()
    seqs = [tok0.clone()]
    for _ in range(GEN - 1):
        graph.replay()
        seqs.append(bm.tok.clone())
    toks = torch.stack(seqs, 1).cpu()               # [B, GEN]
    bm.pos.copy_(pos0); bm.tok.copy_(tok0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(GEN):
        graph.replay()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
    del bm
    torch.cuda.empty_cache()
    return toks, ms


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    Bs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,4,8,16").split(",")]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(next(Q.MODEL.iterdir())))
    msgs = [{"role": "user", "content": "Explain in three sentences why GPUs are good at matrix multiplication."}]
    prompt = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tk(prompt, add_special_tokens=False)["input_ids"]
    torch.zeros(1, device=dev)
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    # reference: the validated single-sequence decode
    F.FusedQwen = GG.GGUFQwen
    ref = F.run_fused(ids)["tokens"][:GEN]
    torch.cuda.empty_cache()
    m = load_model()
    for B in Bs:
        toks, ms = run(m, B, ids)
        same_rows = bool((toks == toks[0:1]).all())
        match = next((i for i, (a, r) in enumerate(zip(toks[0].tolist(), ref)) if a != r), len(ref))
        print(f"B={B:3d}: {B * GEN / ms * 1e3:8.1f} tok/s  ({ms / GEN:.3f} ms/step)  rows identical: {same_rows}  "
              f"matches single-sequence decode for {match}/{len(ref)} tokens", flush=True)
    print("text:", tk.decode(toks[0].tolist())[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
