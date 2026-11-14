"""Per-launch cost of a dependent launch sequence, through scripts/xkernel_bench.Runner (the same Python
launch path the fusion benchmark times), stream vs CUDA graph, vs grid size. Microbenchmark constants
for the fusion/graph hardware model in lab/optsel.fusion_pred; no PolyBench timing is used.

  pcslurm submit -- python scripts/launch_gap.py
"""
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch
import xkernel_bench as X
from lab import xkernel

SRC = 'extern "C" __global__ void k(float* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] + 1.0f; }'


def main():
    X._setup.__globals__  # noqa
    from qwen_e2e import check, cu
    torch.zeros(1, device="cuda"); check(cu.cuInit(0))
    out = {}
    for blocks in [1, 64, 512, 4096, 16384]:
        n = blocks * 256
        bufs = {"p": torch.zeros(n, device="cuda")}
        for L in [1, 2, 4, 8, 20, 30]:
            r = X.Runner([xkernel.Launch(SRC, (blocks,), (256,), ["p", n])] * L, bufs)
            s = torch.cuda.Stream(); g = torch.cuda.CUDAGraph(); torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                r.run()
            res = {}
            for mode, fn in [("stream", r.run), ("graph", g.replay)]:
                ts = []
                for i in range(60):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(); e0.record(); fn(); e1.record(); torch.cuda.synchronize()
                    if i >= 10:
                        ts.append(e0.elapsed_time(e1) * 1e3)
                res[mode] = statistics.median(ts)
            out[f"{blocks}x{L}"] = res
            print(blocks, L, {k: round(v, 2) for k, v in res.items()}, flush=True)
    dst = ROOT / "data/optsel/launch_gap.json"; dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1)); print("wrote", dst)


if __name__ == "__main__":
    main()
