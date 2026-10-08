"""Per-kernel GPU time of one eager Qwen decode step (linears skipped = 'none').

Shows where the non-GEMV ~3.3 ms/token goes, to decide what to fuse.
  pcslurm submit -- python scripts/qwen_profile.py [backend]
"""
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_e2e as Q  # noqa: E402


def main():
    be = sys.argv[1] if len(sys.argv) > 1 else "none"
    m = Q.Qwen(next(Q.MODEL.iterdir()), be)
    m.tok.fill_(100)
    for _ in range(3):
        m.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            m.step()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    events.sort(key=lambda e: -e.self_device_time_total)
    total = sum(e.self_device_time_total for e in events) / 5
    print(f"GPU kernel time per step (self, no double counting): {total / 1e3:.3f} ms")
    for e in events[:22]:
        print(f"  {e.self_device_time_total / 5 / 1e3:7.3f} ms  {100 * e.self_device_time_total / 5 / total:5.1f}%  {e.count / 5:6.0f} calls  {e.key[:80]}")


if __name__ == "__main__":
    main()
