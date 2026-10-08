"""Our int4 GEMV vs PyTorch's in-tree tinygemm (_weight_int4pack_mm), M=1.

Both run on torch's CUDA context and stream; each is launched back to back
200 times between two CUDA events (amortized per-call time, includes launch).
Our cubin is loaded with the driver API into torch's primary context, which is
exactly how an end-to-end Qwen run will call it.

Run as a GPU job with the system Python (torch 2.6 lives there, not in .venv):
  pcslurm submit -- python scripts/bench_vs_torch.py
"""
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
site = ROOT / ".venv" / "Lib" / "site-packages"
sys.path.append(str(site))  # cuda-python + nvidia wheels from the lab venv

from cuda.bindings import driver as cu  # noqa: E402

from lab import toolchain  # noqa: E402
from lab.experiments.gemv3 import Gemv3  # noqa: E402
from lab.experiments.base import Variant  # noqa: E402

SHAPES = [("qkv_kv", 512, 3584), ("q_o", 3584, 3584), ("gate_up", 18944, 3584), ("down", 3584, 18944)]
REPS = 200


def check(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return rest[0] if len(rest) == 1 else rest


def timed_graph(fn, reps=REPS):
    """Capture reps back-to-back calls in one CUDA graph; replay removes host launch cost."""
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (5 * reps) * 1e3


def timed(fn, reps=REPS):
    for _ in range(10):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3  # us


def main():
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)  # create torch's primary context
    check(cu.cuInit(0))
    ctx = check(cu.cuDevicePrimaryCtxRetain(check(cu.cuDeviceGet(0))))
    check(cu.cuCtxSetCurrent(ctx))
    g = Gemv3()
    out = {}
    for name, N, K in SHAPES:
        gen = torch.Generator().manual_seed(N + K)
        q = torch.randint(0, 16, (N, K), generator=gen, dtype=torch.int32)
        scale = (torch.rand(N, K // 128, generator=gen) * 0.018 + 0.002)
        x = torch.randn(K, generator=gen)
        ref = (((q.double() - 8) * scale.double().repeat_interleave(128, 1)) @ x.double())

        # ---- torch tinygemm: uint8 [N, K/2] high-nibble-first, scales_and_zeros [K/G, N, 2] bf16
        qu8 = ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8).to(dev)
        wpk = torch._convert_weight_to_int4pack(qu8, 8)
        sz = torch.stack([scale.t(), torch.zeros_like(scale.t())], dim=-1).to(torch.bfloat16).contiguous().to(dev)
        xb = x.to(torch.bfloat16).to(dev).view(1, K)
        f_torch = lambda: torch._weight_int4pack_mm(xb, wpk, 128, sz)  # noqa: E731
        y_t = f_torch().float().view(-1).cpu().double()

        # ---- ours (R4U1T128 magic): packed uint32 [N, K/8], nibble i = weight 8j+i
        packed = torch.zeros(N, K // 8, dtype=torch.int64)
        for i in range(8):
            packed |= (q[:, i::8].long() << (4 * i))
        Wd = packed.to(torch.int32).to(dev)
        Sd = scale.float().contiguous().to(dev)
        Xd = x.float().to(dev)
        Yd = torch.empty(N, device=dev)
        rows_per_block = 4 * 4
        blocks = (N + rows_per_block - 1) // rows_per_block
        Td = torch.empty(2 * blocks, dtype=torch.int64, device=dev)
        v = Variant("x", {"N": N, "K": K, "R": 4, "U": 1, "T": 128, "deq": "magic", "warps": 4, "iters": 1})
        b = toolchain.build(g.source(v))
        mod = check(cu.cuModuleLoadData(b.cubin))
        fn = check(cu.cuModuleGetFunction(mod, b"k"))
        args = [ctypes.c_uint64(t.data_ptr()) for t in (Wd, Sd, Xd, Yd, Td)] + [ctypes.c_int32(N), ctypes.c_int32(K)]
        ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        def f_ours():  # always launch on torch's *current* stream (captured inside graphs)
            stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
            check(cu.cuLaunchKernel(fn, blocks, 1, 1, 128, 1, 1, 0, stream, ctypes.addressof(ptrs), 0))

        f_ours()
        torch.cuda.synchronize()
        y_o = Yd.cpu().double()
        norm = ref.abs().max().item()
        res = {
            "torch_us": timed(f_torch), "ours_us": timed(f_ours),
            "torch_graph_us": timed_graph(f_torch), "ours_graph_us": timed_graph(f_ours),
            "torch_err": (y_t - ref).abs().max().item() / norm, "ours_err": (y_o - ref).abs().max().item() / norm,
            "weight_bytes": N * K // 2,
        }
        res["speedup"] = res["torch_us"] / res["ours_us"]
        res["graph_speedup"] = res["torch_graph_us"] / res["ours_graph_us"]
        out[name] = res
        print(f"{name:<8} torch {res['torch_us']:7.1f} us (err {res['torch_err']:.1e})   ours {res['ours_us']:7.1f} us "
              f"(err {res['ours_err']:.1e})   {res['speedup']:.2f}x  | GRAPH torch {res['torch_graph_us']:6.1f} ours {res['ours_graph_us']:6.1f} us "
              f"{res['graph_speedup']:.2f}x  ours {res['weight_bytes'] / res['ours_graph_us'] / 1e3:.0f} GB/s",
              flush=True)
    (ROOT / "results" / "bench_vs_torch.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
