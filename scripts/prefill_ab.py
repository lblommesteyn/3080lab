"""In-process A/B of prefill configs (qwen_prefill.py i8 mode): every config's CUDA graph is captured in
one process on the same prompt, then the graphs are replayed round robin (config order shuffled each
round), so desktop contention lands on all of them alike. Prints min / 10th percentile / median per
config and the max difference of the K/V cache and the last chunk's hidden state against the first
config (0 = bit-identical).

  pcslurm submit -- python scripts/prefill_ab.py P=512 rounds=200 cfg="fuse=0" cfg="fuse=rms" ...

A cfg is a space-separated list of qwen_prefill flags that are read per Prefill (fuse=...,
tile_<gemm>=WM,WN,MT,NT,KB,NSTG,MINB).
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_prefill as QP  # noqa: E402


def parse_fuse(cfg):
    for a in cfg.split():
        if a.startswith("fuse="):
            return set(a[5:].split(",")) - {"0"}
    return set(QP.FUSE)


def main():
    a = dict(x.split("=", 1) for x in sys.argv[1:] if not x.startswith("cfg="))
    cfgs = [x[4:] for x in sys.argv[1:] if x.startswith("cfg=")]
    P, chunk, rounds = int(a.get("P", 512)), int(a.get("chunk", 512)), int(a.get("rounds", 200))
    QP.Q.MAX_LEN = max(512, P)
    torch.zeros(1, device="cuda")
    QP.check(QP.cu.cuInit(0))
    QP.check(QP.cu.cuCtxSetCurrent(QP.check(QP.cu.cuDevicePrimaryCtxRetain(QP.check(QP.cu.cuDeviceGet(0))))))
    m = QP.D.load_model()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 150000, (P,), generator=g).tolist()
    ids_t = torch.tensor(ids, device="cuda")
    runs = []
    i8ms = dict(QP.I8MS)
    for cfg in cfgs:
        QP.FUSE = parse_fuse(cfg)
        QP.I8MS = dict(i8ms)
        for x in cfg.split():
            if x.startswith("tile_"):
                k, v = x[5:].split("=")
                QP.I8MS[k] = tuple(map(int, v.split(",")))
        pf = QP.Prefill(m, chunk, "i8")

        def run_all(pf=pf):
            for s in range(0, P, chunk):
                pf.tok.copy_(ids_t[s:s + chunk])
                pf.pos.copy_(torch.arange(s, s + chunk, device="cuda"))
                pf.chunk()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_all()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_all()
        graph.replay()
        torch.cuda.synchronize()
        runs.append((cfg, pf, graph, []))
    ref = runs[0][1]
    for cfg, pf, _, _ in runs:
        dk = float((pf.kc[:, :, :P].float() - ref.kc[:, :, :P].float()).abs().max())
        dv = float((pf.vc[:, :, :P].float() - ref.vc[:, :, :P].float()).abs().max())
        dh = float((pf.h - ref.h).abs().max())
        print(f"{cfg:30s} vs first: K {dk:.3g} V {dv:.3g} h {dh:.3g}", flush=True)
    for _ in range(20):                                          # warm the clock
        for r in runs:
            r[2].replay()
    torch.cuda.synchronize()
    rng = random.Random(0)
    for _ in range(rounds):
        order = list(runs)
        rng.shuffle(order)
        for cfg, pf, graph, ts in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
    base = None
    for cfg, _, _, ts in runs:
        ts.sort()
        mn, p10, med = ts[0], ts[len(ts) // 10], ts[len(ts) // 2]
        base = base or p10
        print(f"{cfg:30s} min {mn:7.3f}  p10 {p10:7.3f}  median {med:7.3f} ms  {P / mn * 1e3:7.0f} tok/s (min)  "
              f"p10 vs first {100 * (p10 / base - 1):+.1f}%", flush=True)


if __name__ == "__main__":
    main()
