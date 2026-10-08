"""Per-kernel cost table for the fused GGUF Qwen2.5-1.5B decode step.

Each kernel type is launched 100x inside one CUDA graph (so the ~1 us per-launch
cost in a graph is included, as in the real decode), timed, and compared with
its bandwidth floor (bytes moved / 705 GB/s). Gap x launches per token =
recoverable time; this ranks optimization targets.

  pcslurm submit -- python scripts/qwen_kernel_times.py
"""
import ctypes
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_e2e as Q  # noqa: E402
import qwen_fused as F  # noqa: E402
import qwen_gguf as GG  # noqa: E402

BW = 705.0  # GB/s == bytes/ns, practical read roofline


def time_graph(fn, reps=100):
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        fn()
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    # warm the clocks: replay for >= 300 ms before timing (short bursty graphs otherwise run at idle P-state clocks)
    import time
    t_end = time.perf_counter() + 0.3
    while time.perf_counter() < t_end:
        g.replay()
        torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (20 * reps) * 1e3  # us


def main():
    torch.zeros(1, device="cuda")
    F.check(F.cu.cuInit(0))
    F.check(F.cu.cuCtxSetCurrent(F.check(F.cu.cuDevicePrimaryCtxRetain(F.check(F.cu.cuDeviceGet(0))))))
    m = GG.GGUFQwen(None)
    m.pos.fill_(150)                # mid-generation context length
    m.tok.fill_(100)
    L = m.L[0]
    H, nh, nkv, hd, inter = m.H, m.nh, m.nkv, m.hd, m.inter
    nq = (nh + 2 * nkv) * hd
    q4 = lambda N, K: N * K / 2 + N * K / 32 * 2           # noqa: E731  Q4_0 repacked: nibbles + fp16 scales
    rows = []

    import pynvml as nv
    nv.nvmlInit()
    hnd = nv.nvmlDeviceGetHandleByIndex(0)

    def add(name, per_token, fn, bytes_):
        us = time_graph(fn)
        mhz = nv.nvmlDeviceGetClockInfo(hnd, nv.NVML_CLOCK_SM)
        floor = bytes_ / BW / 1e3
        rows.append({"kernel": name, "per_token": per_token, "us": us, "floor_us": floor,
                     "gap_us_per_token": (us - floor) * per_token, "share_us_per_token": us * per_token})
        print(f"{name:<12} x{per_token:3d}  {us:7.2f} us  floor {floor:6.2f} us  -> {us * per_token:7.1f} us/token "
              f"(gap {(us - floor) * per_token:6.1f})  [SM clock {mhz} MHz]", flush=True)

    scale = ctypes.c_float(1.0 / math.sqrt(hd))
    nL = len(m.L)
    add("embed", 1, lambda: m.k_emb.launch(4, 512, [m.embed.data_ptr(), m.tok.data_ptr(), m.h.data_ptr(), ctypes.c_int32(H)]),
        H * 4 * 2)
    add("rmsnorm", 2 * nL + 1, lambda: m.rms(L["ln1"], m.x), H * 4 * 3)
    add("gemv_qkv", nL, lambda: m.gemv("qkv", L["qkv"], m.x, m.qkv, L["qkv_b"].data_ptr(), nq, H), q4(nq, H))
    add("attention", nL, lambda: m.k_attn.launch(nh, 128, [m.qkv.data_ptr(), m.cos.data_ptr(), m.sin.data_ptr(),
                                                            m.pos.data_ptr(), m.kc[0].data_ptr(), m.vc[0].data_ptr(),
                                                            m.att.data_ptr(), ctypes.c_int32(nh), ctypes.c_int32(nkv), scale]),
        2 * nkv * 150 * hd * 2)
    add("gemv_o", nL, lambda: m.gemv("o", L["o"], m.att, m.h, 0, H, nh * hd), q4(H, nh * hd))
    add("gemv_gu", nL, lambda: m.gemv("gu", L["gu"], m.x, m.xm, 0, 2 * inter, H), q4(2 * inter, H))
    add("gemv_down", nL, lambda: m.gemv("down", L["down"], m.xm, m.h, 0, H, inter), q4(H, inter))
    add("gemv_head", 1, lambda: m.gemv("head", m.head, m.x, m.logits, 0, m.V, H), m.V * H * 1.25)
    add("finish", 1, lambda: m.k_fin.launch(1, 1024, [m.logits.data_ptr(), ctypes.c_int32(m.V), m.tok.data_ptr(),
                                                     m.pos.data_ptr()]), m.V * 4)
    tot = sum(r["share_us_per_token"] for r in rows)
    gap = sum(r["gap_us_per_token"] for r in rows)
    print(f"\nsum of kernel costs: {tot / 1e3:.3f} ms/token (measured end-to-end 2.29); bandwidth floor "
          f"{(tot - gap) / 1e3:.3f} ms; recoverable gap {gap / 1e3:.3f} ms")
    for r in sorted(rows, key=lambda r: -r["gap_us_per_token"]):
        print(f"  {r['kernel']:<12} gap {r['gap_us_per_token']:7.1f} us/token  ({100 * r['gap_us_per_token'] / gap:4.1f}%)")
    (Q.ROOT / "results" / "qwen_kernel_times.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
