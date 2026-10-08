"""CUDA C -> PTX (NVRTC) -> CUBIN (ptxas) -> SASS (nvdisasm), all from pip wheels.

Kept as separate stages on purpose: later phases patch cubins and feed
hand-written PTX, so every intermediate is a first-class artifact.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nvidia.cu13 as _cu13

ARCH = "sm_86"
BIN = Path(_cu13.__path__[0]) / "bin"
PTXAS = BIN / "ptxas.exe"
NVDISASM = BIN / "nvdisasm.exe"
CUOBJDUMP = BIN / "cuobjdump.exe"

# NVRTC's DLL lives in bin/x86_64; make sure the loader can find it.
os.add_dll_directory(str(BIN / "x86_64"))
from cuda.bindings import nvrtc  # noqa: E402


@dataclass
class Build:
    source: str
    ptx: str
    cubin: bytes
    ptxas_log: str
    sass: str  # nvdisasm text with hex encodings
    ptxas_flags: list[str] = field(default_factory=list)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.cubin).hexdigest()[:16]


def _check(res):
    err, *rest = res
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error {err}")
    return rest[0] if len(rest) == 1 else rest


def cuda_to_ptx(source: str, name: str = "kernel.cu", opts: list[str] | None = None) -> str:
    prog = _check(nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], []))
    args = [f"--gpu-architecture=compute_{ARCH[3:]}", "-default-device", "-std=c++17",
            f"-I{Path(_cu13.__path__[0]) / 'include'}"]
    args += opts or []
    err, = nvrtc.nvrtcCompileProgram(prog, len(args), [a.encode() for a in args])
    size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * size
    _check(nvrtc.nvrtcGetProgramLog(prog, log))
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC compile failed:\n{log.decode(errors='replace')}")
    size = _check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * size
    _check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)
    return ptx.rstrip(b"\0").decode()


def ptx_to_cubin(ptx: str, flags: list[str] | None = None) -> tuple[bytes, str]:
    flags = flags or []
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "k.ptx", Path(d) / "k.cubin"
        src.write_text(ptx)
        r = subprocess.run([str(PTXAS), f"-arch={ARCH}", "-v", *flags, str(src), "-o", str(out)],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ptxas failed:\n{r.stderr}")
        return out.read_bytes(), r.stderr + r.stdout


def disassemble(cubin: bytes) -> str:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([str(NVDISASM), "-hex", "-c", str(p)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"nvdisasm failed:\n{r.stderr}")
        return r.stdout


def build(source: str, ptxas_flags: list[str] | None = None, nvrtc_opts: list[str] | None = None) -> Build:
    ptx = cuda_to_ptx(source, opts=nvrtc_opts)
    cubin, log = ptx_to_cubin(ptx, ptxas_flags)
    return Build(source, ptx, cubin, log, disassemble(cubin), list(ptxas_flags or []))


def ptxas_resources(log: str) -> dict:
    """Parse `ptxas -v` lines like 'Used 12 registers, used 0 barriers, 360 bytes cmem[0]'."""
    out = {}
    m = re.search(r"Used (\d+) registers", log)
    if m:
        out["registers"] = int(m.group(1))
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if m:
        out["stack_bytes"], out["spill_stores"], out["spill_loads"] = map(int, m.groups())
    m = re.search(r"(\d+) bytes smem", log)
    out["static_smem"] = int(m.group(1)) if m else 0
    return out
