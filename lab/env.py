"""Environment capture via NVML: clocks, temperature, power, throttle reasons, other GPU users.

The rule from the SSD work: control the environment before believing the
measurement. We cannot lock clocks without admin (nvidia-smi -lgc), so every
trial records what the clocks and throttle state actually were, and the
in-kernel cycle counter is the primary unit (clock-frequency independent for
core-bound work).
"""
from __future__ import annotations

import subprocess
import time

import pynvml as nv

_THROTTLE = {
    "gpu_idle": 0x1, "app_clocks": 0x2, "sw_power_cap": 0x4, "hw_slowdown": 0x8,
    "sync_boost": 0x10, "sw_thermal": 0x20, "hw_thermal": 0x40, "hw_power_brake": 0x80,
    "display_clocks": 0x100,
}

_handle = None


def handle():
    global _handle
    if _handle is None:
        nv.nvmlInit()
        _handle = nv.nvmlDeviceGetHandleByIndex(0)
    return _handle


def snapshot() -> dict:
    h = handle()
    reasons = nv.nvmlDeviceGetCurrentClocksEventReasons(h)
    util = nv.nvmlDeviceGetUtilizationRates(h)
    return {
        "t": time.time(),
        "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
        "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
        "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
        "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
        "pstate": nv.nvmlDeviceGetPerformanceState(h),
        "util_gpu": util.gpu,
        "throttle": [k for k, v in _THROTTLE.items() if reasons & v],
    }


def static_info() -> dict:
    h = handle()
    info = {
        "driver": nv.nvmlSystemGetDriverVersion(),
        "max_sm_mhz": nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM),
        "max_mem_mhz": nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM),
        "power_limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
        "vbios": nv.nvmlDeviceGetVbiosVersion(h),
    }
    try:
        procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
        info["other_compute_procs"] = len(procs)
    except nv.NVMLError:
        info["other_compute_procs"] = None  # WDDM often does not report these
    try:
        lo, hi = nv.nvmlDeviceGetGpcClkMinMaxVfOffset(h)
        info["gpc_vf_offset_range"] = [lo, hi]
    except (nv.NVMLError, AttributeError):
        pass
    return info


def try_lock_clocks(mhz: int | None) -> str:
    """Attempt nvidia-smi -lgc. Needs admin on Windows; report, never fail."""
    if not mhz:
        return "unlocked (not requested)"
    r = subprocess.run(["nvidia-smi", "-lgc", f"{mhz},{mhz}"], capture_output=True, text=True)
    if r.returncode == 0:
        return f"locked {mhz} MHz"
    return f"unlocked (lock failed: {(r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else r.returncode})"


def unlock_clocks():
    subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, text=True)


def warnings(snaps: list[dict], max_temp: int = 80, inkernel_mhz: list[float] | None = None) -> list[str]:
    w = []
    bad = {"hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake", "sw_power_cap"}
    hits = sorted({r for s in snaps for r in s["throttle"] if r in bad})
    if hits:
        w.append(f"throttle reasons seen during trials: {', '.join(hits)}")
    if snaps and max(s["temp_c"] for s in snaps) > max_temp:
        w.append(f"temperature exceeded {max_temp} C")
    # NVML is sampled after each launch, when the GPU may already be dropping to idle;
    # clock drift is judged from the in-kernel clock (cycles / globaltimer) instead.
    if inkernel_mhz:
        lo, hi = min(inkernel_mhz), max(inkernel_mhz)
        if hi - lo > 60:
            w.append(f"in-kernel SM clock varied {lo:.0f}-{hi:.0f} MHz across trials")
    return w
