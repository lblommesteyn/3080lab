"""Decode megakernel (lab/qwen_mega.py) on llama.cpp's Qwen2.5-1.5B Q4_0 GGUF, against the current
~200-kernel graph (qwen_gguf.py), in one process.

  1. correctness: same state (random K/V cache, position `pos`, token 1000) through both steps;
     logits max |diff| and the argmax token;
  2. speed: both steps captured in CUDA graphs and replayed round robin in shuffled order;
  3. text: 64 greedy tokens from the chat prompt with both (token-by-token prompt, as run_fused).

  pcslurm submit -- python scripts/qwen_mega.py [pos=128] [rounds=200] [nch=20] [ch=4096] [nw=8] [notext]
"""
from __future__ import annotations

import ctypes
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_e2e as Q  # noqa: E402
import qwen_fused as F  # noqa: E402
import qwen_gguf as G  # noqa: E402

from lab import qwen_mega as MK  # noqa: E402
from lab import toolchain  # noqa: E402

ARGS = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x and not x.startswith("--"))
NW, CH, NCH = int(ARGS.get("nw", 8)), int(ARGS.get("ch", 4096)), int(ARGS.get("nch", 20))


class MKern:
    def __init__(self, src: str, smem: int):
        b = toolchain.build(src)
        print("megakernel:", toolchain.ptxas_resources(b.ptxas_log), f"smem {smem} B", flush=True)
        self.mod = F.check(F.cu.cuModuleLoadData(b.cubin))
        self.fn = F.check(F.cu.cuModuleGetFunction(self.mod, b"k"))
        F.check(F.cu.cuFuncSetAttribute(self.fn, F.cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        self.smem = smem
        self.keep = []

    def launch(self, grid, block, args):
        a = [x if isinstance(x, (ctypes.c_uint64, ctypes.c_int32, ctypes.c_float)) else ctypes.c_uint64(x) for x in args]
        p = (ctypes.c_void_p * len(a))(*[ctypes.addressof(z) for z in a])
        self.keep.append((a, p))
        st = F.cu.CUstream(torch.cuda.current_stream().cuda_stream)
        F.check(F.cu.cuLaunchKernel(self.fn, grid, 1, 1, block, 1, 1, self.smem, st, ctypes.addressof(p), 0))


def unit_bytes(packed: torch.Tensor, d: torch.Tensor, kwc: int) -> torch.Tensor:
    """[N/4, nsub * 72 * kwc] uint8: per 4-row unit, sub-units of (4 x kwc uint4 weights, 4 x kwc fp16 scales)."""
    N, K8 = packed.shape
    nsub = K8 * 8 // (kwc * 32)
    q = packed.view(N // 4, 4, nsub, kwc * 4).permute(0, 2, 1, 3).contiguous().view(torch.uint8).view(N // 4, nsub, 64 * kwc)
    s = d.view(N // 4, 4, nsub, kwc).permute(0, 2, 1, 3).contiguous().view(torch.uint8).view(N // 4, nsub, 8 * kwc)
    return torch.cat([q, s], 2).reshape(N // 4, nsub * 72 * kwc).contiguous()


def build_streams(m, nblk: int):
    """Per-block weight streams in consumption order + unit and phase tables."""
    nL = len(m.L)
    ops = [("qkv", 48), (None, 0), ("o", 48), ("gu", 48), ("down", 56)]
    cum = [0] * nblk
    per_block = [[] for _ in range(nblk)]        # (tensor slice, [(row0, nbytes)])
    ptab = torch.zeros(nblk, nL * 5, 2, dtype=torch.int32)
    ucount = [0] * nblk
    for li, L in enumerate(m.L):
        for oi, (name, kwc) in enumerate(ops):
            if name is None:
                for b in range(nblk):
                    ptab[b, li * 5 + oi] = torch.tensor([ucount[b], 0])
                continue
            ub = unit_bytes(*L[name], kwc)
            U, size = ub.shape
            base, extra = divmod(U, nblk)
            # extra units go to the blocks that have streamed the least so far (balances the whole step)
            order = sorted(range(nblk), key=lambda b: (cum[b], b))
            cnt = [base] * nblk
            for b in order[:extra]:
                cnt[b] += 1
            nsub = size // (72 * kwc)
            sub = size // nsub
            u0 = 0
            for b in range(nblk):
                ptab[b, li * 5 + oi] = torch.tensor([ucount[b], cnt[b]])
                # units go to warps round robin (unit e -> warp e % NW); multi-sub-unit units are stored
                # sub-unit-interleaved within each group of NW units ([sub][unit]), so all warps advance
                # through the ring together instead of the first warps' units gating the window
                for g0 in range(0, cnt[b], NW):
                    gs = min(NW, cnt[b] - g0)
                    blk = ub[u0 + g0:u0 + g0 + gs].view(gs, nsub, sub).transpose(0, 1).reshape(-1)
                    per_block[b].append((blk, [(4 * (u0 + g0 + i), i * sub, gs * size) for i in range(gs)]))
                u0 += cnt[b]
                cum[b] += cnt[b] * size
                ucount[b] += cnt[b]
    soff, nchunk, units, ubase, chunks = [], [], [], [], []
    total = 0
    for b in range(nblk):
        ubase.append(len(units))
        off = 0
        parts = []
        for t, lst in per_block[b]:
            for row0, rel, _ in lst:
                units.append((row0, off + rel))
            off += lst[0][2]
            parts.append(t.reshape(-1))
        pad = (-off) % CH
        units.append((0, off + pad))             # sentinel: end of stream
        if pad:
            parts.append(torch.zeros(pad, dtype=torch.uint8, device=parts[0].device))
        chunks.append(torch.cat(parts))
        soff.append(total)
        nchunk.append((off + pad) // CH)
        total += off + pad
    stream = torch.cat(chunks)
    print(f"streams: {total / 2**20:.1f} MiB over {nblk} blocks, per block {min(cum) / 2**20:.3f}-{max(cum) / 2**20:.3f} MiB "
          f"(max/mean {max(cum) / (sum(cum) / nblk):.4f})", flush=True)
    dev = stream.device
    return (stream, torch.tensor(soff, dtype=torch.int64, device=dev), torch.tensor(nchunk, dtype=torch.int32, device=dev),
            torch.tensor(units, dtype=torch.int32, device=dev), torch.tensor(ubase, dtype=torch.int32, device=dev),
            ptab.to(dev).contiguous())


class Mega:
    """Megakernel step on top of a loaded GGUFQwen: one launch for the layers, then head + argmax as before."""

    def __init__(self, m):
        self.m = m
        self.G = torch.cuda.get_device_properties(0).multi_processor_count
        self.st = build_streams(m, self.G)
        nL = len(m.L)
        self.ln1 = torch.stack([L["ln1"] for L in m.L]).contiguous()
        self.ln2 = torch.stack([L["ln2"] for L in m.L]).contiguous()
        self.bias = torch.stack([L["qkv_b"] for L in m.L]).contiguous()
        self.sync = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.err = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.k = MKern(MK.mega_source(nL, Q.MAX_LEN, NW, CH, NCH), MK.smem_bytes(CH, NCH))

    def step(self):
        m, c = self.m, ctypes
        s = self.st
        self.k.launch(self.G, 32 * NW + 32,
                      [s[0].data_ptr(), s[1].data_ptr(), s[2].data_ptr(), s[3].data_ptr(), s[4].data_ptr(), s[5].data_ptr(),
                       m.embed.data_ptr(), m.tok.data_ptr(), m.pos.data_ptr(), self.ln1.data_ptr(), self.ln2.data_ptr(),
                       self.bias.data_ptr(), m.norm.data_ptr(), m.cos.data_ptr(), m.sin.data_ptr(), m.kc.data_ptr(),
                       m.vc.data_ptr(), m.h.data_ptr(), m.qkv.data_ptr(), m.att.data_ptr(), m.xm.data_ptr(), m.x.data_ptr(),
                       self.sync.data_ptr(), self.err.data_ptr(), c.c_float(m.eps), c.c_float(1.0 / (m.hd ** 0.5))])
        m.gemv("head", m.head, m.x, m.logits, 0, m.V, m.H)
        nb = m.fin_blocks
        m.k_fin.launch(nb, 256, [m.logits.data_ptr(), c.c_int32(m.V), m.tok.data_ptr(), m.pos.data_ptr(),
                                 m.fin_part.data_ptr(), m.fin_part.data_ptr() + 4 * nb, m.fin_cnt.data_ptr()])


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pos0, rounds = int(ARGS.get("pos", 128)), int(ARGS.get("rounds", 200))
    torch.zeros(1, device="cuda")
    F.check(F.cu.cuInit(0))
    F.check(F.cu.cuCtxSetCurrent(F.check(F.cu.cuDevicePrimaryCtxRetain(F.check(F.cu.cuDeviceGet(0))))))
    m = G.GGUFQwen(None)
    mega = Mega(m)
    torch.manual_seed(0)
    m.kc.normal_()
    m.vc.normal_()
    kc0, vc0 = m.kc.clone(), m.vc.clone()

    def reset():
        m.kc.copy_(kc0)
        m.vc.copy_(vc0)
        m.tok.fill_(1000)
        m.pos.fill_(pos0)

    # 1. correctness (eager first, then graphs)
    reset(); m.step(); torch.cuda.synchronize()
    ref, tref, href = m.logits.clone(), int(m.tok.item()), m.h.clone()
    reset(); mega.step(); torch.cuda.synchronize()
    err = mega.err.tolist()
    out, tout = m.logits.clone(), int(m.tok.item())
    print(f"eager: err {err}  tok ref {tref} mega {tout}  max|logits diff| {float((out - ref).abs().max()):.4g} "
          f"(max|logit| {float(ref.abs().max()):.3g})  max|h diff| {float((m.h - href).abs().max()):.4g} "
          f"(max|h| {float(href.abs().max()):.3g})  sync {mega.sync.tolist()}", flush=True)
    if err[0]:
        print("megakernel reported a wait timeout; stopping")
        return
    if not torch.isfinite(m.h).all() or not 0 <= tout < m.V:
        bad = (~torch.isfinite(m.h)).sum().item()
        print(f"megakernel output not finite ({bad} of {m.h.numel()} h values) or token out of range; stopping")
        return
    for _ in range(3):
        reset(); mega.step()
    torch.cuda.synchronize()
    print(f"repeat: tok {int(m.tok.item())}  max|logits diff| {float((m.logits - ref).abs().max()):.4g}  err {mega.err.tolist()}",
          flush=True)

    # 2. speed (state reset outside the timed region; pos fixed)
    runs = []
    for name, fn in (("graph (200 kernels)", m.step), ("megakernel", mega.step)):
        reset()
        g = capture(fn)
        runs.append([name, g, []])
    for _ in range(20):
        for r in runs:
            m.pos.fill_(pos0); m.tok.fill_(1000); r[1].replay()
    torch.cuda.synchronize()
    rng = random.Random(0)
    for _ in range(rounds):
        order = list(runs)
        rng.shuffle(order)
        for r in order:
            m.pos.fill_(pos0)
            m.tok.fill_(1000)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); r[1].replay(); e.record(); e.synchronize()
            r[2].append(s.elapsed_time(e))
    base = None
    for name, _, ts in runs:
        ts.sort()
        mn, p10, med = ts[0], ts[len(ts) // 10], ts[len(ts) // 2]
        base = base or p10
        print(f"{name:22s} min {mn * 1e3:7.1f}  p10 {p10 * 1e3:7.1f}  median {med * 1e3:7.1f} us  {1e3 / p10:6.1f} tok/s (p10)  "
              f"vs graph {100 * (p10 / base - 1):+.1f}%", flush=True)
    print(f"err after timing {mega.err.tolist()}", flush=True)

    # 3. text: greedy 64 tokens with both
    if "notext" in sys.argv:
        return
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(next(Q.MODEL.iterdir())))
    msgs = [{"role": "user", "content": "Explain in three sentences why GPUs are good at matrix multiplication."}]
    ids = tk(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    texts = {}
    for name, fn in (("graph", m.step), ("mega", mega.step)):
        m.kc.zero_(); m.vc.zero_(); m.pos.fill_(0)
        for t in ids:
            m.tok.fill_(t)
            fn()
        toks = [int(m.tok.item())]
        for _ in range(63):
            fn()
            toks.append(int(m.tok.item()))
        texts[name] = toks
        print(f"{name:6s}: {tk.decode(toks)!r}", flush=True)
    a, b = texts["graph"], texts["mega"]
    same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    print(f"identical tokens before first divergence: {same} of {len(a)}; err {mega.err.tolist()}")


if __name__ == "__main__":
    main()
