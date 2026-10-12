"""In-process A/B of B = 1 decode configs (qwen_gguf.py's GGUFQwen step): the model is loaded once,
each config's kernels are built and its step captured in its own CUDA graph, then the graphs are
replayed round robin in shuffled order from the same state (random K/V cache, position `pos`), so
desktop load lands on all of them alike. Prints min / p10 / median per step and checks that every
config picks the same next token with logits close to the first config's.

  pcslurm submit -- python scripts/decode_ab.py pos=128 rounds=300 cfg="" cfg="split_down=4" cfg="fin=1"

cfg keys: split_<qkv|o|gu|down>=S (split-K warps), fin=0 (one-block FINISH) or fin=N (FINISH_MB, N blocks).
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_fused as F  # noqa: E402
import qwen_gguf as G  # noqa: E402

from lab import qwen_kernels as KS  # noqa: E402

EPI = {"qkv": "biasf", "o": "resid", "gu": "swiglu", "down": "resid"}


def configure(m, cfg, base_split):
    split = dict(base_split)
    fin = G.FIN_BLOCKS
    for kv in cfg.split():
        k, v = kv.split("=")
        if k.startswith("split_"):
            split[k[6:]] = int(v)
        elif k == "fin":
            fin = int(v)
    for name, epi in EPI.items():
        m.k[name] = F.Kern(KS.gemv_source(split[name], epi, "g32f16"))
    F.SPLIT.update(split)
    m.fin_blocks = fin
    if fin:
        m.k_fin = F.Kern(KS.FINISH_MB)
        m.fin_part = torch.zeros(2 * fin, dtype=torch.int32, device="cuda")
        m.fin_cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    else:
        m.k_fin = F.Kern(KS.FINISH)


def main():
    a = dict(x.split("=", 1) for x in sys.argv[1:] if not x.startswith("cfg=") and "=" in x)
    cfgs = [x[4:] for x in sys.argv[1:] if x.startswith("cfg=")]
    pos0, rounds = int(a.get("pos", 128)), int(a.get("rounds", 300))
    torch.zeros(1, device="cuda")
    F.check(F.cu.cuInit(0))
    F.check(F.cu.cuCtxSetCurrent(F.check(F.cu.cuDevicePrimaryCtxRetain(F.check(F.cu.cuDeviceGet(0))))))
    m = G.GGUFQwen(None)
    base_split = dict(F.SPLIT)
    torch.manual_seed(0)
    m.kc.normal_()
    m.vc.normal_()
    kc0, vc0 = m.kc.clone(), m.vc.clone()

    def reset():
        m.tok.fill_(1000)
        m.pos.fill_(pos0)
    runs = []
    for cfg in cfgs:
        configure(m, cfg, base_split)
        reset()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            m.step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            m.step()
        m.kc.copy_(kc0)
        m.vc.copy_(vc0)
        reset()
        graph.replay()
        torch.cuda.synchronize()
        runs.append([cfg or "(default)", graph, [], m.logits.clone(), int(m.tok.item())])
    ref = runs[0][3]
    for r in runs:
        print(f"{r[0]:28s} tok {r[4]}  max |logits - first| {float((r[3] - ref).abs().max()):.3g} "
              f"(max |logit| {float(ref.abs().max()):.3g})", flush=True)
    for _ in range(30):
        for r in runs:
            reset()
            r[1].replay()
    torch.cuda.synchronize()
    rng = random.Random(0)
    for _ in range(rounds):
        order = list(runs)
        rng.shuffle(order)
        for r in order:
            reset()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            r[1].replay()
            e.record()
            e.synchronize()
            r[2].append(s.elapsed_time(e))
    base = None
    for name, _, ts, _, _ in runs:
        ts.sort()
        mn, p10, med = ts[0], ts[len(ts) // 10], ts[len(ts) // 2]
        base = base or p10
        print(f"{name:28s} min {mn * 1e3:7.1f}  p10 {p10 * 1e3:7.1f}  median {med * 1e3:7.1f} us  {1e3 / mn:6.1f} tok/s (min)  "
              f"p10 vs first {100 * (p10 / base - 1):+.1f}%", flush=True)


if __name__ == "__main__":
    main()
