"""End-to-end Qwen2.5-1.5B-Instruct greedy decode with int4 weights.

Identical int4 group-128 weights for every backend; only the linear op differs:
  torch : PyTorch in-tree tinygemm (_weight_int4pack_mm), bf16 activations
  ours  : 3080lab split-K kernel (bf16 in/out, fp32 accumulation), driver API
  none  : linears skipped (output left as-is) = cost of everything else
One decode step (incl. greedy argmax and position increment) is captured as a
CUDA graph and replayed, so the timed loop has no host launches or syncs.

  pcslurm submit -- python scripts/qwen_e2e.py
"""
from __future__ import annotations

import ctypes
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv" / "Lib" / "site-packages"))
from cuda.bindings import driver as cu  # noqa: E402

from lab import toolchain  # noqa: E402
from lab.experiments.gemv4 import splitk_source  # noqa: E402

MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
MAX_LEN = 512
GEN = 256
dev = torch.device("cuda")


def check(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return rest[0] if len(rest) == 1 else (rest or None)


# ---------------------------------------------------------------- int4 weights
def quantize(w: torch.Tensor):
    """w [N, K] float -> q int32 [N, K] in 0..15 (value = (q-8)*s), s [N, K/128]."""
    N, K = w.shape
    g = w.float().view(N, K // 128, 128)
    s = g.abs().amax(-1).clamp_min(1e-8) / 7.0
    q = (torch.round(g / s[..., None]) + 8).clamp(0, 15).to(torch.int32).view(N, K)
    return q, s


class Lin:
    """One quantized linear with all backend representations precomputed."""
    mod = fn = None

    def __init__(self, w: torch.Tensor, backend: str):
        self.N, self.K = w.shape
        q, s = quantize(w)
        self.backend = backend
        if backend == "torch":
            qu8 = ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8).to(dev)
            self.wpk = torch._convert_weight_to_int4pack(qu8, 8)
            self.sz = torch.stack([s.t(), torch.zeros_like(s.t())], -1).to(torch.bfloat16).contiguous().to(dev)
        elif backend == "ours":
            packed = torch.zeros(self.N, self.K // 8, dtype=torch.int64, device=dev)
            qd = q.to(dev)
            for i in range(8):
                packed |= qd[:, i::8].long() << (4 * i)
            self.W = packed.to(torch.int32).contiguous()
            self.S = s.float().contiguous().to(dev)
        self.out = torch.zeros(1, self.N, dtype=torch.bfloat16, device=dev)
        self._args = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # x [1, K] bf16 (contiguous)
        if self.backend == "torch":
            return torch._weight_int4pack_mm(x, self.wpk, 128, self.sz)
        if self.backend == "none":
            return self.out
        # ours: argument block bound to this call's input pointer (stable under graphs)
        key = x.data_ptr()
        if self._args is None or self._args[0] != key:
            a = [ctypes.c_uint64(self.W.data_ptr()), ctypes.c_uint64(self.S.data_ptr()), ctypes.c_uint64(key),
                 ctypes.c_uint64(self.out.data_ptr()), ctypes.c_uint64(0), ctypes.c_int32(self.N),
                 ctypes.c_int32(self.K), ctypes.c_int32(0)]
            self._args = (key, a, (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a]))
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(Lin.fn, (self.N + 3) // 4, 1, 1, 128, 1, 1, 0, stream,
                                ctypes.addressof(self._args[2]), 0))
        return self.out


# ---------------------------------------------------------------- model
class Qwen:
    def __init__(self, path: Path, backend: str):
        from safetensors.torch import load_file
        cfg = json.loads((path / "config.json").read_text())
        self.cfg = cfg
        sd = {}
        for f in sorted(path.glob("*.safetensors")):
            sd.update(load_file(str(f)))
        H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = H // nh
        self.nh, self.nkv, self.eps = nh, nkv, cfg["rms_norm_eps"]
        bf = lambda t: t.to(torch.bfloat16).to(dev)  # noqa: E731
        self.embed = bf(sd["model.embed_tokens.weight"])
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}."
            qkv_w = torch.cat([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"],
                               sd[p + "self_attn.v_proj.weight"]])
            qkv_b = torch.cat([sd[p + "self_attn.q_proj.bias"], sd[p + "self_attn.k_proj.bias"],
                               sd[p + "self_attn.v_proj.bias"]])
            gu_w = torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]])
            self.layers.append({
                "ln1": bf(sd[p + "input_layernorm.weight"]), "ln2": bf(sd[p + "post_attention_layernorm.weight"]),
                "qkv": Lin(qkv_w, backend), "qkv_b": bf(qkv_b), "o": Lin(sd[p + "self_attn.o_proj.weight"], backend),
                "gu": Lin(gu_w, backend), "down": Lin(sd[p + "mlp.down_proj.weight"], backend),
            })
        self.norm = bf(sd["model.norm.weight"])
        head = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
        self.head = Lin(head, backend)
        self.inter = cfg["intermediate_size"]
        L = cfg["num_hidden_layers"]
        self.kc = torch.zeros(L, nkv, MAX_LEN, self.hd, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, device=dev).float() / self.hd))
        ang = torch.arange(MAX_LEN, device=dev).float()[:, None] * inv[None]
        self.cos, self.sin = torch.cos(ang).to(torch.bfloat16), torch.sin(ang).to(torch.bfloat16)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.ar = torch.arange(MAX_LEN, device=dev)

    def rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16) * w

    def rope(self, x):  # x [heads, hd]
        c = self.cos.index_select(0, self.pos)
        s = self.sin.index_select(0, self.pos)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def step(self):
        """One greedy decode step for self.tok at self.pos; writes next token, advances pos."""
        h = self.embed.index_select(0, self.tok)                     # [1, H]
        mask = (self.ar <= self.pos)[None, None, None, :]            # [1,1,1,MAX_LEN]
        for i, L in enumerate(self.layers):
            x = self.rms(h, L["ln1"])
            qkv = L["qkv"](x) + L["qkv_b"]
            nq, nk = self.nh * self.hd, self.nkv * self.hd
            q = self.rope(qkv[0, :nq].view(self.nh, self.hd))
            k = self.rope(qkv[0, nq:nq + nk].view(self.nkv, self.hd))
            v = qkv[0, nq + nk:].view(self.nkv, self.hd)
            self.kc[i].index_copy_(1, self.pos, k[:, None])
            self.vc[i].index_copy_(1, self.pos, v[:, None])
            kk = self.kc[i].repeat_interleave(self.nh // self.nkv, 0)[None]
            vv = self.vc[i].repeat_interleave(self.nh // self.nkv, 0)[None]
            a = F.scaled_dot_product_attention(q[None, :, None], kk, vv, attn_mask=mask)  # [1,nh,1,hd]
            h = h + L["o"](a.reshape(1, -1).contiguous())
            x = self.rms(h, L["ln2"])
            gu = L["gu"](x)
            m = (F.silu(gu[:, : self.inter].float()) * gu[:, self.inter:].float()).to(torch.bfloat16)
            h = h + L["down"](m.contiguous())
        logits = self.head(self.rms(h, self.norm))
        self.tok.copy_(logits.float().argmax(-1))
        self.pos.add_(1)
        return logits


def run(backend: str, prompt_ids: list[int]) -> dict:
    t0 = time.time()
    m = Qwen(next(MODEL.iterdir()), backend)
    load_s = time.time() - t0
    torch.cuda.synchronize()
    # prefill one token at a time (eager), then capture the decode step
    for t in prompt_ids:
        m.tok.fill_(t)
        m.step()
    m.pos.sub_(1)
    m.tok.fill_(prompt_ids[-1])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    pos0, tok0 = m.pos.clone(), m.tok.clone()
    with torch.cuda.stream(side):
        for _ in range(2):
            m.pos.copy_(pos0); m.tok.copy_(tok0)
            m.step()
    torch.cuda.current_stream().wait_stream(side)
    m.pos.copy_(pos0); m.tok.copy_(tok0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m.step()
    m.pos.copy_(pos0); m.tok.copy_(tok0)
    out = []
    torch.cuda.synchronize()
    # correctness pass (collect tokens), then timed pass from the same state
    for _ in range(GEN):
        g.replay()
        out.append(m.tok.clone())
    torch.cuda.synchronize()
    toks = [int(t.item()) for t in out]
    m.pos.copy_(pos0); m.tok.copy_(tok0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(GEN):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
    return {"backend": backend, "tokens": toks, "ms_per_token": ms / GEN, "tok_per_s": GEN / ms * 1e3,
            "load_s": load_s}


def main():
    from transformers import AutoTokenizer
    path = next(MODEL.iterdir())
    tok = AutoTokenizer.from_pretrained(str(path))
    msgs = [{"role": "user", "content": "Explain in three sentences why GPUs are good at matrix multiplication."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    torch.zeros(1, device=dev)
    check(cu.cuInit(0))
    ctx = check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))
    check(cu.cuCtxSetCurrent(ctx))
    b = toolchain.build(splitk_source(4, "bf16"))
    Lin.mod = check(cu.cuModuleLoadData(b.cubin))
    Lin.fn = check(cu.cuModuleGetFunction(Lin.mod, b"k"))
    results = {}
    order = sys.argv[1:] or ["none", "torch", "ours", "torch", "ours"]
    for be in order:
        r = run(be, ids)
        results.setdefault(be, []).append(r)
        text = tok.decode(r["tokens"][:60]).replace("\n", " ")
        print(f"{be:<6} {r['tok_per_s']:7.1f} tok/s  {r['ms_per_token']:.3f} ms/token   | {text[:150]}", flush=True)
        torch.cuda.empty_cache()
    if "torch" in results and "ours" in results:
        a, b2 = results["torch"][-1]["tokens"], results["ours"][-1]["tokens"]
        same = next((i for i, (x, y) in enumerate(zip(a, b2)) if x != y), len(a))
        print(f"tokens identical for the first {same} of {len(a)}")
    (ROOT / "results" / "qwen_e2e.json").write_text(json.dumps(
        {k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
