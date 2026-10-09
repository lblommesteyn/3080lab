"""Batched decode of Qwen2.5-1.5B (llama.cpp's Q4_0 GGUF weights): B sequences per step.

  pcslurm submit -- python scripts/qwen_batched_decode.py 1,4,8,16,32

GEMVs: tensor-core Q4_0 kernels (lab/qwen_batched.gemv_mma_source). Embedding, RMSNorm, attention
and argmax are the single-sequence kernels with a batch index from blockIdx.y offsetting their
pointers. LM head: the Q6_K head dequantized once to fp16, cuBLAS matmul (reads 467 MB per step
instead of 191 MB; amortized over B).
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
        S, U = 8, 4
        self.S = S
        self.g = {name: Kern(QB.gemv_mma_source(S, B, epi, U)) for name, epi in
                  (("qkv", "biasf"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"))}
        self.k_emb = Kern(batchify(KS.EMBED_F32, "tok += b_; h += (size_t)b_ * H;"))
        self.k_rms = Kern(batchify(KS.RMSNORM_F32W_1P, "h += (size_t)b_ * H; out += (size_t)b_ * H;"))
        attn = KS.ATTN4 % {"maxlen": Q.MAX_LEN}
        self.k_attn = Kern(batchify(attn, "qkv += (size_t)b_ * (NH + 2 * NKV) * HD; posp += b_; "
                                          "out += (size_t)b_ * NH * HD; "
                                          "kc += (size_t)b_ * NKV * MAXLEN * HD; vc += (size_t)b_ * NKV * MAXLEN * HD;"))
        self.head16 = m.head16

    def gemv(self, name, wsplit, X, Y, aux, N, K):
        W, S = wsplit
        self.g[name].launch((N + 15) // 16, 32 * self.S, [W.data_ptr(), S.data_ptr(), X.data_ptr(), Y.data_ptr(), aux,
                                                         ctypes.c_int32(N), ctypes.c_int32(K)])

    def rms(self, w):
        self.k_rms.launch((1, self.B), 512, [self.h.data_ptr(), w.data_ptr(), self.x.data_ptr(),
                                             ctypes.c_int32(self.m.H), ctypes.c_float(self.m.eps)])

    def step(self):
        m, B, H = self.m, self.B, self.m.H
        self.k_emb.launch((4, B), 512, [m.embed.data_ptr(), self.tok.data_ptr(), self.h.data_ptr(), ctypes.c_int32(H)])
        scale = ctypes.c_float(1.0 / math.sqrt(m.hd))
        for i, L in enumerate(m.L):
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
        logits = torch.matmul(self.x.to(torch.float16), self.head16.t())
        self.tok.copy_(logits.argmax(-1))
        self.pos += 1


def load_model():
    m = GG.GGUFQwen(None)
    g = GGUF(GG.GGUF_PATH)
    b, t = g.raw("output.weight")
    K, N = t.dims
    q, s = q6_k_blocks(b)
    w = (torch.from_numpy(q.astype(np.float32)).to(dev) * torch.from_numpy(s.astype(np.float32)).to(dev).repeat_interleave(16, 1))
    m.head16 = w.reshape(N, K).to(torch.float16).contiguous()
    return m


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
