"""Apples-to-apples vs llama.cpp: run llama.cpp's own Qwen2.5-1.5B Q4_0 GGUF through our
fused kernels.

Every tensor comes from the GGUF:
  linears  Q4_0 blocks repacked bit-exactly into our layout (32-weight groups, fp16 scales)
  output   Q6_K decoded exactly into int8 q + fp32 scale per 16 weights (more bytes than
           llama.cpp's 6.56 bpw: 292 MB vs 191 MB per token, i.e. conservative for us)
  embed    Q4_0 dequantized exactly to an fp32 table (one row read per token)
  norms / biases  F32 as stored (they carry folded per-channel scales)
Then llama-completion (greedy) on the same templated prompt, for text agreement.

  pcslurm submit -- python scripts/qwen_gguf.py
"""
from __future__ import annotations

import ctypes
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_e2e as Q  # noqa: E402
import qwen_fused as F  # noqa: E402
from qwen_e2e import dev  # noqa: E402

from lab import qwen_kernels as KS  # noqa: E402
from lab.gguf import GGUF, q4_0_blocks, q6_k_blocks, q6_k_parts  # noqa: E402

GGUF_PATH = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/"
                           "91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_0.gguf")
LLAMA = Q.ROOT / "vendor" / "llamacpp"
SPLIT = {"qkv": 2, "o": 2, "gu": 1, "down": 2, "head": 1}


def q4_rows(g: GGUF, name: str):
    """-> q uint8 [N, K] (stored 0..15, value = (q-8)*d), d fp16 [N, K/32]."""
    b, t = g.raw(name)
    K, N = t.dims
    blk = b.reshape(-1, 18)
    d = blk[:, :2].copy().view(np.float16).reshape(N, K // 32)
    _, q = q4_0_blocks(b)
    return q.reshape(N, K), d


def to_device_q4(q: np.ndarray, d: np.ndarray):
    qt = torch.from_numpy(q.astype(np.int64)).to(dev)
    N, K = q.shape
    packed = torch.zeros(N, K // 8, dtype=torch.int64, device=dev)
    for i in range(8):
        packed |= qt[:, i::8] << (4 * i)
    return packed.to(torch.int32).contiguous(), torch.from_numpy(d.copy()).to(dev).contiguous()


class GGUFQwen(F.FusedQwen):
    def __init__(self, path: Path):  # noqa: D401 (path unused: everything comes from the GGUF)
        g = GGUF(GGUF_PATH)
        m = g.meta
        self.H, self.nh, self.nkv = m["qwen2.embedding_length"], m["qwen2.attention.head_count"], m["qwen2.attention.head_count_kv"]
        self.hd, self.inter = self.H // self.nh, m["qwen2.feed_forward_length"]
        self.eps = m["qwen2.attention.layer_norm_rms_epsilon"]
        nL = m["qwen2.block_count"]
        f32 = lambda n: torch.from_numpy(g.f32(n).astype(np.float32).copy()).to(dev).contiguous()  # noqa: E731
        self.embed = f32("token_embd.weight")
        self.V = self.embed.shape[0]
        self.L = []
        for i in range(nL):
            p = f"blk.{i}."
            qs = [q4_rows(g, p + f"attn_{x}.weight") for x in "qkv"]
            qkv = to_device_q4(np.concatenate([a for a, _ in qs]), np.concatenate([b for _, b in qs]))
            gq, gd = q4_rows(g, p + "ffn_gate.weight")
            uq, ud = q4_rows(g, p + "ffn_up.weight")
            gu = to_device_q4(np.stack([gq, uq], 1).reshape(-1, gq.shape[1]), np.stack([gd, ud], 1).reshape(-1, gd.shape[1]))
            bias = torch.cat([f32(p + f"attn_{x}.bias") for x in "qkv"])
            self.L.append({"ln1": f32(p + "attn_norm.weight"), "ln2": f32(p + "ffn_norm.weight"),
                           "qkv": qkv, "qkv_b": bias, "o": to_device_q4(*q4_rows(g, p + "attn_output.weight")),
                           "gu": gu, "down": to_device_q4(*q4_rows(g, p + "ffn_down.weight"))})
        self.norm = f32("output_norm.weight")
        b, t = g.raw("output.weight")
        K, N = t.dims
        self.head_q6 = "--head-i8" not in sys.argv
        if self.head_q6:
            # Q6_K-exact repack, same 6.5625 bits/weight as llama.cpp: low-nibble plane, 2-bit high
            # plane, raw int8 sub-block scales and fp16 super-block scales
            q, sc, d = q6_k_parts(b)
            qq = torch.from_numpy((q.astype(np.int16) + 32).reshape(N, K).astype(np.int64)).to(dev)
            lo = torch.zeros(N, K // 8, dtype=torch.int64, device=dev)
            for i in range(8):
                lo |= (qq[:, i::8] & 15) << (4 * i)
            hi = torch.zeros(N, K // 16, dtype=torch.int64, device=dev)
            for i in range(16):
                hi |= (qq[:, i::16] >> 4) << (2 * i)
            self.head = (lo.to(torch.int32).contiguous(), hi.to(torch.int32).contiguous(),
                         torch.from_numpy(sc.reshape(N, K // 16).copy()).to(dev).contiguous(),
                         torch.from_numpy(d.reshape(N, K // 256).copy()).to(dev).contiguous())
        else:
            q6, s6 = q6_k_blocks(b)
            self.head = (torch.from_numpy(q6.reshape(N, K).copy()).to(dev).contiguous(),
                         torch.from_numpy(s6.reshape(N, K // 16).astype(np.float32).copy()).to(dev).contiguous())
        rope_theta = m["qwen2.rope.freq_base"]
        self.kc = torch.zeros(nL, self.nkv, Q.MAX_LEN, self.hd, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        inv = 1.0 / (rope_theta ** (torch.arange(0, self.hd, 2, device=dev).double() / self.hd))
        ang = torch.arange(Q.MAX_LEN, device=dev).double()[:, None] * inv[None]
        self.cos, self.sin = torch.cos(ang).float().contiguous(), torch.sin(ang).float().contiguous()
        self.h = torch.zeros(self.H, device=dev)
        self.x = torch.zeros(self.H, dtype=torch.bfloat16, device=dev)
        self.xm = torch.zeros(max(self.H, self.inter), dtype=torch.bfloat16, device=dev)
        self.qkv = torch.zeros((self.nh + 2 * self.nkv) * self.hd, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(self.nh * self.hd, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(self.V, device=dev)
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        F.SPLIT.update(SPLIT)
        self.k = {name: F.Kern(KS.gemv_source(SPLIT[name], epi, "g32f16")) for name, epi in
                  (("qkv", "biasf"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"))}
        self.k["head"] = F.Kern(KS.gemv_q6_source(SPLIT["head"]) if self.head_q6 else KS.gemv_i8_source(SPLIT["head"]))
        self.k_rms, self.k_emb = F.Kern(KS.RMSNORM_F32W), F.Kern(KS.EMBED_F32)
        attn = "ATTN4" if "--attn-v1" not in sys.argv else "ATTN"
        self.k_attn, self.k_fin = F.Kern(getattr(KS, attn) % {"maxlen": Q.MAX_LEN}), F.Kern(KS.FINISH)
        self.attn_threads = 512 if attn == "ATTN4" else 128


_BASE_GEMV = F.FusedQwen.gemv   # captured before main() rebinds F.FusedQwen


def _gemv(self, name, wsplit, X, Y, aux, N, K):
    if name == "head" and getattr(self, "head_q6", False):
        Lw, Hw, SCw, Dw = wsplit
        self.k[name].launch((N + 3) // 4, 32 * F.SPLIT[name],
                            [Lw.data_ptr(), Hw.data_ptr(), SCw.data_ptr(), Dw.data_ptr(), X.data_ptr(), Y.data_ptr(),
                             F.ctypes.c_int32(N), F.ctypes.c_int32(K)])
    else:
        _BASE_GEMV(self, name, wsplit, X, Y, aux, N, K)


GGUFQwen.gemv = _gemv


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(next(Q.MODEL.iterdir())))
    msgs = [{"role": "user", "content": "Explain in three sentences why GPUs are good at matrix multiplication."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    torch.zeros(1, device=dev)
    F.check(F.cu.cuInit(0))
    F.check(F.cu.cuCtxSetCurrent(F.check(F.cu.cuDevicePrimaryCtxRetain(F.check(F.cu.cuDeviceGet(0))))))
    F.FusedQwen = GGUFQwen  # run_fused instantiates FusedQwen
    res = []
    for _ in range(3):
        r = F.run_fused(ids)
        res.append(r)
        print(f"ours(gguf) {r['tok_per_s']:7.1f} tok/s  {r['ms_per_token']:.3f} ms/token", flush=True)
        torch.cuda.empty_cache()
    ours_text = tok.decode(res[-1]["tokens"][:64])
    # llama.cpp greedy on the identical prompt string (CPU-side tokenization by llama.cpp)
    pf = Q.ROOT / "results" / "qwen_prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    cmd = [str(LLAMA / "llama-completion.exe"), "-m", str(GGUF_PATH), "-f", str(pf), "-n", "64", "--temp", "0",
           "-ngl", "99", "-no-cnv", "--no-display-prompt", "--seed", "0"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    llama_text = out.stdout.strip()
    print("OURS :", ours_text.replace("\n", " ")[:400])
    print("LLAMA:", llama_text.replace("\n", " ")[:400])
    a, b = ours_text.split(), llama_text.split()
    same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"identical words before first divergence: {same} (of {min(len(a), len(b))})")
    (Q.ROOT / "results" / "qwen_gguf.json").write_text(json.dumps({"ours": res, "llama_text": llama_text}, indent=1))


if __name__ == "__main__":
    main()
