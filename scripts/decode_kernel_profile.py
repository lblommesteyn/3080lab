"""Where a B = 1 decode step's time goes (qwen_gguf.py: llama.cpp's Q4_0 GGUF through our kernels).

One step's launches are recorded and grouped by role (embed, rms1, qkv, attn, o, rms2, gu, down,
final rms, head, finish). Each role's launches (one per layer, so the 28 layers' distinct weights
come from DRAM as in a real step, not from a warm L2) are captured alone in a CUDA graph and timed;
the whole step is timed the same way. The step minus the sum of roles is what the kernels cost
together beyond their separate times (launch/drain overlap makes it negative).

  pcslurm submit -- python scripts/decode_kernel_profile.py [pos=128] [reps=5] [qwen_gguf flags]
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_fused as F  # noqa: E402
import qwen_gguf as G  # noqa: E402

ROLES = ["rms1", "qkv", "attn", "o", "rms2", "gu", "down"]


def graph_time(fn, reps, pos, pos0):
    """min ms of one fn() over 9 timings of a graph holding reps calls (pos reset before each)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    t_end = time.perf_counter() + 0.3
    while time.perf_counter() < t_end:
        pos.fill_(pos0)
        g.replay()
        torch.cuda.synchronize()
    out = []
    for _ in range(9):
        pos.fill_(pos0)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / reps)
    return min(out)


def main():
    a = dict(x.split("=") for x in sys.argv[1:] if "=" in x and not x.startswith("--"))
    pos0, reps = int(a.get("pos", 128)), int(a.get("reps", 5))
    torch.zeros(1, device="cuda")
    F.check(F.cu.cuInit(0))
    F.check(F.cu.cuCtxSetCurrent(F.check(F.cu.cuDevicePrimaryCtxRetain(F.check(F.cu.cuDeviceGet(0))))))
    m = G.GGUFQwen(None)
    torch.manual_seed(0)
    m.kc.normal_()
    m.vc.normal_()
    m.tok.fill_(1000)
    m.pos.fill_(pos0)
    rec = []
    orig = F.Kern.launch

    def recording(self, grid, block, args):
        rec.append((self, grid, block, args))
        orig(self, grid, block, args)
    F.Kern.launch = recording
    m.step()
    torch.cuda.synchronize()
    F.Kern.launch = orig
    nL = len(m.L)
    assert len(rec) == 1 + 7 * nL + 3, len(rec)
    groups = {"embed": [rec[0]]}
    for j, r in enumerate(rec[1:1 + 7 * nL]):
        groups.setdefault(ROLES[j % 7], []).append(r)
    groups["rmsf"], groups["head"], groups["fin"] = [rec[-3]], [rec[-2]], [rec[-1]]
    m.pos.fill_(pos0)
    step_ms = graph_time(m.step, reps, m.pos, pos0)
    rows = []
    for role, launches in groups.items():
        ms = graph_time(lambda L=launches: [orig(k, g, b, ar) for k, g, b, ar in L], reps, m.pos, pos0)
        rows.append((role, len(launches), ms))
    tot = sum(r[2] for r in rows)
    print(f"pos={pos0}: step {step_ms * 1e3:.1f} us ({1e3 / step_ms:.1f} tok/s), {len(rec)} launches")
    for role, n, ms in rows:
        print(f"  {role:6s} x{n:2d} {ms * 1e3:8.1f} us  ({ms * 1e3 / n:6.2f} per launch)  {100 * ms / step_ms:5.1f}% of step")
    print(f"  sum of roles {tot * 1e3:.1f} us; step - sum {1e3 * (step_ms - tot):+.1f} us")


if __name__ == "__main__":
    main()
