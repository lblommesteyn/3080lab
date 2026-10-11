"""Prefill attention kernels on one chunk of P tokens at positions pos0 .. pos0 + P - 1 (Qwen2.5-1.5B
heads: 12 query, 2 KV, head dim 128), against an fp32 torch reference and each other.

  pcslurm submit -- python scripts/flash_bench.py [P=512] [pos=0,512,1024,1536] [kern=v1,v2,v3]

Error: max |O - ref| / max |ref| over the chunk, ref = softmax(q k^T / sqrt(128), causal) v in fp32 from
the same bf16 q (RoPE'd and rounded as the kernels do), k and v.
"""
import ctypes
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batched_gemv_bench as BB  # noqa: E402
from qwen_e2e import check, cu  # noqa: E402

from lab import qwen_batched as QB  # noqa: E402

NH, NKV, HD = 12, 2, 128
# name: (source(maxlen), threads per block, tokens per block)
KERNS = {"v1": (lambda ml: QB.flash_prefill_source(4, ml), 128, 64),
         "v1w2": (lambda ml: QB.flash_prefill_source(2, ml), 64, 32),
         "v2": (lambda ml: QB.flash_prefill_v2_source(4, ml), 128, 64),
         "v2w2": (lambda ml: QB.flash_prefill_v2_source(2, ml), 64, 32),
         "v3": (QB.flash_prefill_v3_source, 256, 64)}


def main():
    a = dict(x.split("=") for x in sys.argv[1:])
    P = int(a.get("P", 512))
    poss = [int(x) for x in a.get("pos", "0,512,1024,1536").split(",")]
    names = a.get("kern", "v1,v2,v3").split(",")
    maxlen = max(poss) + P
    torch.zeros(1, device="cuda")
    check(cu.cuInit(0))
    check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))))
    torch.manual_seed(0)
    dev = "cuda"
    kc = torch.randn(NKV, maxlen, HD, device=dev).to(torch.bfloat16)
    vc = torch.randn(NKV, maxlen, HD, device=dev).to(torch.bfloat16)
    inv = 1.0 / (1e6 ** (torch.arange(0, 64, device=dev).float() / 64))
    ang = torch.arange(maxlen, device=dev).float()[:, None] * inv[None]
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    kern = {n: BB.Kern(KERNS[n][0](maxlen)) for n in names}
    scale = 1.0 / math.sqrt(HD)
    print(f"P={P} maxlen={maxlen} regs " + " ".join(f"{n}:{k.regs}" for n, k in kern.items()), flush=True)
    for pos0 in poss:
        qkv = (torch.randn(P, (NH + 2 * NKV) * HD, device=dev) * 2).to(torch.bfloat16)
        pos = torch.arange(pos0, pos0 + P, device=dev, dtype=torch.int64)
        # reference
        q = qkv[:, :NH * HD].float().view(P, NH, HD)
        qa, qb = q[..., :64], q[..., 64:]
        c, s_ = cos[pos][:, None], sin[pos][:, None]
        qr = torch.cat([qa * c - qb * s_, qb * c + qa * s_], -1).to(torch.bfloat16).float()
        n = pos0 + P
        k = kc[:, :n].float().repeat_interleave(NH // NKV, 0)            # [NH, n, HD]
        v = vc[:, :n].float().repeat_interleave(NH // NKV, 0)
        sc = torch.einsum("phd,hjd->hpj", qr, k) * scale
        mask = torch.arange(n, device=dev)[None, :] > pos[:, None]
        sc.masked_fill_(mask[None], float("-inf"))
        ref = torch.einsum("hpj,hjd->phd", sc.softmax(-1), v).reshape(P, NH * HD)
        line = f"pos0={pos0:5d}:"
        for nm, kr in kern.items():
            out = torch.zeros(P, NH * HD, dtype=torch.bfloat16, device=dev)
            args = [qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), pos.data_ptr(), kc.data_ptr(), vc.data_ptr(),
                    out.data_ptr(), ctypes.c_int32(NH), ctypes.c_int32(NKV), ctypes.c_float(scale), ctypes.c_int32(P)]
            _, nthr, rows = KERNS[nm]
            fn = lambda: kr.launch(NH, nthr, args, (P + rows - 1) // rows)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            err = float((out.float() - ref).abs().max() / ref.abs().max())
            us = min(BB.timeit(fn, 20) for _ in range(3))
            fl = 4 * NH * HD * P * (pos0 + P / 2)
            line += f"  {nm} {us:7.1f} us ({fl / us / 1e6:5.1f} TFLOPS, err {err:.1e})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
