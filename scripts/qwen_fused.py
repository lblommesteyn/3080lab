"""Fused Qwen2.5-1.5B int4 decode: ~200 kernels/token, all ours, one CUDA graph.

Same int4 group-128 weights as scripts/qwen_e2e.py (identical quantize()).
Compares against the PyTorch+tinygemm decode from qwen_e2e in the same job.

  pcslurm submit -- python scripts/qwen_fused.py
"""
from __future__ import annotations

import ctypes
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_e2e as Q  # noqa: E402
from qwen_e2e import check, cu, dev  # noqa: E402

from lab import qwen_kernels as KS  # noqa: E402
from lab import toolchain  # noqa: E402

# split-K per GEMV, from the gemv_int4_v4_q15 sweep
SPLIT = {"qkv": 2, "o": 2, "gu": 1, "down": 2, "head": 1}


class Kern:
    def __init__(self, src: str):
        b = toolchain.build(src)
        self.mod = check(cu.cuModuleLoadData(b.cubin))
        self.fn = check(cu.cuModuleGetFunction(self.mod, b"k"))
        self.keep = []

    def launch(self, grid, block, args):
        a = [x if isinstance(x, (ctypes.c_uint64, ctypes.c_int32, ctypes.c_float)) else ctypes.c_uint64(x) for x in args]
        p = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
        self.keep.append((a, p))  # argument blocks must outlive graph capture
        st = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(self.fn, grid, 1, 1, block, 1, 1, 0, st, ctypes.addressof(p), 0))


def pack(w: torch.Tensor):
    q, s = Q.quantize(w)
    N, K = q.shape
    qd = q.to(dev)
    packed = torch.zeros(N, K // 8, dtype=torch.int64, device=dev)
    for i in range(8):
        packed |= qd[:, i::8].long() << (4 * i)
    return packed.to(torch.int32).contiguous(), s.float().contiguous().to(dev)


class FusedQwen:
    def __init__(self, path: Path):
        from safetensors.torch import load_file
        cfg = json.loads((path / "config.json").read_text())
        sd = {}
        for f in sorted(path.glob("*.safetensors")):
            sd.update(load_file(str(f)))
        self.H, self.nh, self.nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd, self.inter, self.eps = self.H // self.nh, cfg["intermediate_size"], cfg["rms_norm_eps"]
        self.V = cfg["vocab_size"]
        bf = lambda t: t.to(torch.bfloat16).contiguous().to(dev)  # noqa: E731
        self.embed = bf(sd["model.embed_tokens.weight"])
        self.L = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}."
            qkv_w = torch.cat([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.v_proj.weight"]])
            qkv_b = torch.cat([sd[p + "self_attn.q_proj.bias"], sd[p + "self_attn.k_proj.bias"], sd[p + "self_attn.v_proj.bias"]])
            g, u = sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]
            # interleave rows g0,u0,g1,u1,... so each 4-row group yields two SiLU(g)*u outputs.
            # quantize gate/up exactly as the baseline does (per row, groups along K), then interleave.
            gu = torch.stack([g, u], 1).reshape(-1, g.shape[1])
            self.L.append({"ln1": bf(sd[p + "input_layernorm.weight"]), "ln2": bf(sd[p + "post_attention_layernorm.weight"]),
                           "qkv": pack(qkv_w), "qkv_b": bf(qkv_b), "o": pack(sd[p + "self_attn.o_proj.weight"]),
                           "gu": pack(gu), "down": pack(sd[p + "mlp.down_proj.weight"])})
        self.norm = bf(sd["model.norm.weight"])
        self.head = pack(sd.get("lm_head.weight", sd["model.embed_tokens.weight"]))
        nL = cfg["num_hidden_layers"]
        self.kc = torch.zeros(nL, self.nkv, Q.MAX_LEN, self.hd, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, device=dev).float() / self.hd))
        ang = torch.arange(Q.MAX_LEN, device=dev).float()[:, None] * inv[None]
        # match the reference: tables rounded to bf16 then used in fp32
        self.cos = torch.cos(ang).to(torch.bfloat16).float().contiguous()
        self.sin = torch.sin(ang).to(torch.bfloat16).float().contiguous()
        # buffers
        self.h = torch.zeros(self.H, device=dev)
        self.x = torch.zeros(self.H, dtype=torch.bfloat16, device=dev)
        self.xm = torch.zeros(max(self.H, self.inter), dtype=torch.bfloat16, device=dev)
        self.qkv = torch.zeros((self.nh + 2 * self.nkv) * self.hd, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(self.nh * self.hd, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(self.V, device=dev)
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        # kernels
        self.k = {name: Kern(KS.gemv_source(SPLIT[name], epi)) for name, epi in
                  (("qkv", "bias"), ("o", "resid"), ("gu", "swiglu"), ("down", "resid"), ("head", "logits"))}
        self.k_rms, self.k_emb = Kern(KS.RMSNORM), Kern(KS.EMBED)
        self.k_attn, self.k_fin = Kern(KS.ATTN % {"maxlen": Q.MAX_LEN}), Kern(KS.FINISH)

    def gemv(self, name, wsplit, X, Y, aux, N, K):
        W, S = wsplit
        self.k[name].launch((N + 3) // 4, 32 * SPLIT[name],
                            [W.data_ptr(), S.data_ptr(), X.data_ptr(), Y.data_ptr(), aux, ctypes.c_int32(N), ctypes.c_int32(K)])

    def rms(self, w, out):
        self.k_rms.launch(1, 512, [self.h.data_ptr(), w.data_ptr(), out.data_ptr(), ctypes.c_int32(self.H),
                                   ctypes.c_float(self.eps)])

    def step(self):
        H, nq = self.H, (self.nh + 2 * self.nkv) * self.hd
        self.k_emb.launch(4, 512, [self.embed.data_ptr(), self.tok.data_ptr(), self.h.data_ptr(), ctypes.c_int32(H)])
        scale = ctypes.c_float(1.0 / math.sqrt(self.hd))
        for i, L in enumerate(self.L):
            self.rms(L["ln1"], self.x)
            self.gemv("qkv", L["qkv"], self.x, self.qkv, L["qkv_b"].data_ptr(), nq, H)
            self.k_attn.launch(self.nh, getattr(self, "attn_threads", 128), [self.qkv.data_ptr(), self.cos.data_ptr(), self.sin.data_ptr(),
                                              self.pos.data_ptr(), self.kc[i].data_ptr(), self.vc[i].data_ptr(),
                                              self.att.data_ptr(), ctypes.c_int32(self.nh), ctypes.c_int32(self.nkv), scale])
            self.gemv("o", L["o"], self.att, self.h, 0, H, self.nh * self.hd)
            self.rms(L["ln2"], self.x)
            self.gemv("gu", L["gu"], self.x, self.xm, 0, 2 * self.inter, H)
            self.gemv("down", L["down"], self.xm, self.h, 0, H, self.inter)
        self.rms(self.norm, self.x)
        self.gemv("head", self.head, self.x, self.logits, 0, self.V, H)
        self.k_fin.launch(1, 1024, [self.logits.data_ptr(), ctypes.c_int32(self.V), self.tok.data_ptr(), self.pos.data_ptr()])


def run_fused(ids):
    m = FusedQwen(next(Q.MODEL.iterdir()))
    for t in ids:   # prefill one token at a time; FINISH advances pos and overwrites tok
        m.tok.fill_(t)
        m.step()
    # state now: pos = len(ids), tok = argmax after the prompt (= first generated token)
    torch.cuda.synchronize()
    pos0, tok0 = m.pos.clone(), m.tok.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        m.step()
    torch.cuda.current_stream().wait_stream(side)
    m.pos.copy_(pos0); m.tok.copy_(tok0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m.step()
    m.pos.copy_(pos0); m.tok.copy_(tok0)
    torch.cuda.synchronize()
    toks = [int(tok0.item())]
    for _ in range(Q.GEN - 1):
        g.replay()
        toks.append(int(m.tok.item()))
    m.pos.copy_(pos0); m.tok.copy_(tok0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(Q.GEN):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
    return {"backend": "fused", "tokens": toks, "ms_per_token": ms / Q.GEN, "tok_per_s": Q.GEN / ms * 1e3}


def main():
    from transformers import AutoTokenizer
    path = next(Q.MODEL.iterdir())
    tok = AutoTokenizer.from_pretrained(str(path))
    msgs = [{"role": "user", "content": "Explain in three sentences why GPUs are good at matrix multiplication."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    torch.zeros(1, device=dev)
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    from lab.experiments.gemv4 import splitk_source
    b = toolchain.build(splitk_source(4, "bf16"))
    Q.Lin.mod = check(cu.cuModuleLoadData(b.cubin))
    Q.Lin.fn = check(cu.cuModuleGetFunction(Q.Lin.mod, b"k"))
    res = {}
    for be in ["torch", "fused", "torch", "fused"]:
        r = Q.run("torch", ids) if be == "torch" else run_fused(ids)
        # Q.run's token list starts at the token generated after the first replay; align on text
        res.setdefault(be, []).append(r)
        text = tok.decode(r["tokens"][:70]).replace("\n", " ")
        print(f"{be:<6} {r['tok_per_s']:7.1f} tok/s  {r['ms_per_token']:.3f} ms/token  | {text[:160]}", flush=True)
        torch.cuda.empty_cache()
    (Q.ROOT / "results" / "qwen_fused.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
