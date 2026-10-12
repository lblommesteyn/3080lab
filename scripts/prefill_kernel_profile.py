"""Time of each kernel in one prefill layer (qwen_prefill.py, i8 mode, one chunk of P tokens at
positions 0 .. P - 1), each launch replayed alone in a CUDA graph (L2 warm), times 28 layers.

  pcslurm submit -- python scripts/prefill_kernel_profile.py [P=512] [pos0=0] [+ qwen_prefill.py flags]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batched_gemv_bench as BB  # noqa: E402
import qwen_prefill as QP  # noqa: E402


def main():
    a = dict(x.split("=") for x in sys.argv[1:] if "=" in x)
    P, pos0 = int(a.get("P", 512)), int(a.get("pos0", 0))
    QP.Q.MAX_LEN = max(512, pos0 + P)
    torch.zeros(1, device="cuda")
    QP.check(QP.cu.cuInit(0))
    QP.check(QP.cu.cuCtxSetCurrent(QP.check(QP.cu.cuDevicePrimaryCtxRetain(QP.check(QP.cu.cuDeviceGet(0))))))
    m = QP.D.load_model()
    pf = QP.Prefill(m, P, "i8")
    names = {id(pf.k_emb): "embed", id(pf.k_rms): "rms", id(pf.k_kvw): "kvw", id(pf.k_q): "quant", id(pf.k_rmsq): "rmsq",
             id(getattr(pf, "k_fa", None)): "attn"}
    names.update({id(k): n for n, k in pf.g.items()})
    rec = []
    kerns = [pf.k_emb, pf.k_rms, pf.k_rmsq, pf.k_kvw, pf.k_q, *pf.g.values()] + ([pf.k_fa] if pf.fa else [])
    orig = {id(k): k.launch for k in kerns}
    for k in kerns:
        k.launch = (lambda kk: lambda *ar, **kw: (rec.append((names[id(kk)], kk, ar, kw)), orig[id(kk)](*ar, **kw)))(k)
    pf.tok.copy_(torch.randint(0, 150000, (P,), device="cuda"))
    pf.pos.copy_(torch.arange(pos0, pos0 + P, device="cuda"))
    pf.chunk()
    torch.cuda.synchronize()
    nL = len(m.L)
    per_layer = (len(rec) - 1) // nL
    layer0 = rec[1:1 + per_layer]
    seen = {}
    rows = []
    for name, k, ar, kw in [rec[0]] + layer0:
        seen[name] = seen.get(name, 0) + 1
        lab = name if name not in ("rms", "rmsq", "quant") else f"{name}{seen[name]}"
        us = min(BB.timeit(lambda: orig[id(k)](*ar, **kw), 20) for _ in range(3))
        rows.append((lab, us, us if name == "embed" else us * nL))
    tot = sum(r[2] for r in rows)
    print(f"P={P} pos0={pos0}: {per_layer} launches per layer")
    for lab, us, t in rows:
        print(f"  {lab:8s} {us:8.1f} us  x layers {t / 1e3:7.3f} ms  {100 * t / tot:5.1f}%")
    print(f"  sum {tot / 1e3:.3f} ms (kernels alone, back to back in the graph: compare the e2e min)")


if __name__ == "__main__":
    main()
