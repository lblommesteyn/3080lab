"""Thin CUDA driver-API layer: load cubins, launch, read back, query resources."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from cuda.bindings import driver as cu


def check(res):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, []
    if err != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(err)
        raise RuntimeError(f"CUDA error {name.decode() if name else err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


class Device:
    def __init__(self, ordinal: int = 0):
        check(cu.cuInit(0))
        self.dev = check(cu.cuDeviceGet(ordinal))
        self.ctx = check(cu.cuDevicePrimaryCtxRetain(self.dev))
        check(cu.cuCtxSetCurrent(self.ctx))
        self.stream = check(cu.cuStreamCreate(cu.CUstream_flags.CU_STREAM_NON_BLOCKING))
        a = cu.CUdevice_attribute
        q = lambda attr: check(cu.cuDeviceGetAttribute(attr, self.dev))  # noqa: E731
        self.name = check(cu.cuDeviceGetName(64, self.dev)).split(b"\0")[0].decode()
        self.props = {
            "name": self.name,
            "cc": f"{q(a.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)}.{q(a.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)}",
            "sms": q(a.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            "max_threads_per_sm": q(a.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
            "regs_per_sm": q(a.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),
            "smem_per_sm": q(a.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
            "l2_bytes": q(a.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
            "driver_version": check(cu.cuDriverGetVersion()),
        }

    def alloc(self, nbytes: int) -> int:
        return int(check(cu.cuMemAlloc(max(nbytes, 4))))

    def free(self, ptr: int):
        check(cu.cuMemFree(ptr))

    def htod(self, ptr: int, arr: np.ndarray):
        check(cu.cuMemcpyHtoD(ptr, arr.ctypes.data, arr.nbytes))

    def dtoh(self, arr: np.ndarray, ptr: int) -> np.ndarray:
        check(cu.cuMemcpyDtoH(arr.ctypes.data, ptr, arr.nbytes))
        return arr

    def memset(self, ptr: int, nbytes: int):
        check(cu.cuMemsetD8(ptr, 0, nbytes))

    def sync(self):
        check(cu.cuStreamSynchronize(self.stream))


@dataclass
class Kernel:
    dev: Device
    module: object
    func: object
    name: str

    @classmethod
    def load(cls, dev: Device, cubin: bytes, name: str) -> "Kernel":
        mod = check(cu.cuModuleLoadData(cubin))
        fn = check(cu.cuModuleGetFunction(mod, name.encode()))
        return cls(dev, mod, fn, name)

    def attrs(self, block: int, dyn_smem: int = 0) -> dict:
        a = cu.CUfunction_attribute
        g = lambda attr: check(cu.cuFuncGetAttribute(attr, self.func))  # noqa: E731
        out = {
            "registers": g(a.CU_FUNC_ATTRIBUTE_NUM_REGS),
            "static_smem": g(a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            "local_bytes": g(a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            "const_bytes": g(a.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES),
            "max_threads_per_block": g(a.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        }
        blocks = check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.func, block, dyn_smem))
        warps = blocks * ((block + 31) // 32)
        out["occupancy"] = {
            "max_blocks_per_sm": blocks,
            "max_warps_per_sm": warps,
            "theoretical": warps / (self.dev.props["max_threads_per_sm"] // 32),
        }
        return out

    def launch(self, grid, block, args: list, smem: int = 0):
        """args: list of ctypes scalars (pointers as c_uint64)."""
        ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(x) for x in args])
        g = grid if isinstance(grid, tuple) else (grid, 1, 1)
        b = block if isinstance(block, tuple) else (block, 1, 1)
        check(cu.cuLaunchKernel(self.func, *g, *b, smem, self.dev.stream, ctypes.addressof(ptrs), 0))

    def timed_launch(self, grid, block, args, smem: int = 0) -> float:
        """Launch bracketed by CUDA events; returns elapsed milliseconds."""
        e0 = check(cu.cuEventCreate(0))
        e1 = check(cu.cuEventCreate(0))
        check(cu.cuEventRecord(e0, self.dev.stream))
        self.launch(grid, block, args, smem)
        check(cu.cuEventRecord(e1, self.dev.stream))
        check(cu.cuEventSynchronize(e1))
        ms = check(cu.cuEventElapsedTime(e0, e1))
        check(cu.cuEventDestroy(e0))
        check(cu.cuEventDestroy(e1))
        return float(ms)

    def unload(self):
        check(cu.cuModuleUnload(self.module))
