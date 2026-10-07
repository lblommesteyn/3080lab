"""Run an experiment under a controlled protocol and write a full record.

Protocol:
  1. Build every variant; verify the SASS loop body contains exactly the
     expected count of the target opcode and no spills.
  2. Spin the GPU out of idle P-state with warmup launches (time-bounded).
  3. Run trials for all variants in a shuffled, interleaved order (seed saved),
     capturing an NVML snapshot after each launch.
  4. Summarize distributions; attach warnings for anything that undermines
     the measurement (throttling, clock drift, high CV, SASS mismatch).
"""
from __future__ import annotations

import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import env, sass, toolchain
from .gpu import Device, Kernel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {}
    med = float(np.median(a))
    return {
        "n": int(a.size), "median": med, "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()), "max": float(a.max()),
        "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
        "cv": float(a.std(ddof=1) / a.mean()) if a.size > 1 and a.mean() else 0.0,
    }


def run(exp, opts: dict) -> dict:
    trials = int(opts.get("trials", 30))
    warmup_ms = float(opts.get("warmup_ms", 500))
    seed = int(opts.get("seed") or random.SystemRandom().randrange(1 << 31))
    variants = exp.variants(opts)

    # ---- build + static validation (no GPU needed) ----
    builds: dict[str, toolchain.Build] = {}
    per_variant: dict[str, dict] = {}
    warnings: list[str] = []
    for v in variants:
        src = exp.source(v)
        if src not in builds:
            builds[src] = toolchain.build(src, ptxas_flags=exp.ptxas_flags)
        b = builds[src]
        ins = sass.parse(b.sass)
        body = sass.loop_body(ins)
        hist = sass.opcode_histogram(body)
        res = toolchain.ptxas_resources(b.ptxas_log)
        expected = exp.expected_body_ops()
        got = sum(n for op, n in hist.items() if op == exp.target_opcode)
        # The loop counter is itself an IADD3, so allow exactly one extra for that opcode.
        slack = 1 if exp.target_opcode == "IADD3" else 0
        if expected is not None and not (expected <= got <= expected + slack):
            warnings.append(f"[{v.label}] SASS loop body has {got} {exp.target_opcode}, expected {expected}")
        if res.get("spill_stores") or res.get("spill_loads"):
            warnings.append(f"[{v.label}] register spills present")
        per_variant[v.label] = {"build": b, "instrs": ins, "body": body, "hist": hist, "resources": res,
                                "variant": v, "results": [], "env": []}

    if warnings and not opts.get("force"):
        lines = "".join(f"\n  {w}" for w in warnings)
        raise SystemExit(f"refusing to measure, static validation failed:{lines}\n(use --force to measure anyway)")

    # ---- device ----
    dev = Device()
    static_env = env.static_info()
    clock_state = env.try_lock_clocks(opts.get("lock_clock"))
    kernels = {k: Kernel.load(dev, b.cubin, exp.kernel_name) for k, b in builds.items()}
    states = {}
    for v in variants:
        pv = per_variant[v.label]
        pv["kernel"] = kernels[exp.source(v)]
        states[v.label] = exp.prepare(dev, v)
        L = states[v.label]["launch"]
        pv["attrs"] = pv["kernel"].attrs(L["block"] if isinstance(L["block"], int) else int(np.prod(L["block"])))

    def launch(v):
        L = states[v.label]["launch"]
        return per_variant[v.label]["kernel"].timed_launch(L["grid"], L["block"], L["args"], L.get("smem", 0))

    # ---- warmup: run until clocks have had time to ramp ----
    env_before = env.snapshot()
    t_end = time.perf_counter() + warmup_ms / 1e3
    n_warm = 0
    while time.perf_counter() < t_end or n_warm < len(variants):
        launch(variants[n_warm % len(variants)])
        n_warm += 1
    env_after_warm = env.snapshot()

    # ---- trials: shuffled interleaving ----
    order = [v for v in variants for _ in range(trials)]
    random.Random(seed).shuffle(order)
    for v in order:
        ms = launch(v)
        r = exp.collect(dev, v, states[v.label])
        r["runtime_ms"] = ms
        per_variant[v.label]["results"].append(r)
        per_variant[v.label]["env"].append(env.snapshot())

    for v in variants:
        exp.release(dev, states[v.label])
    if opts.get("lock_clock") and clock_state.startswith("locked"):
        env.unlock_clocks()

    # ---- summarize ----
    all_env = [e for pv in per_variant.values() for e in pv["env"]]
    warnings += env.warnings(all_env)
    summary = {}
    for label, pv in per_variant.items():
        rs = pv["results"]
        metrics = {k: stats([r[k] for r in rs]) for k in rs[0]
                   if isinstance(rs[0][k], (int, float)) and not isinstance(rs[0][k], bool)}
        checks = [r["correct"] for r in rs if r.get("correct") is not None]
        if checks and not all(checks):
            warnings.append(f"[{label}] WRONG RESULT in {checks.count(False)}/{len(checks)} trials")
        if metrics.get("cycles", {}).get("cv", 0) > 0.02:
            warnings.append(f"[{label}] cycle-count CV {metrics['cycles']['cv']:.1%} > 2%")
        summary[label] = {
            "params": pv["variant"].params,
            "metrics": metrics,
            "resources": {**pv["resources"], **pv["attrs"]},
            "loop_body_histogram": pv["hist"],
            "correctness": (all(checks) if checks else None),
            "loop_body_instructions": len(pv["body"]),
            "dynamic_instructions_per_warp_est": len(pv["body"]) * pv["variant"].params.get("iters", 0),
            "env": {"sm_mhz_nvml": stats([e["sm_mhz"] for e in pv["env"]]),
                    "temp_c": stats([e["temp_c"] for e in pv["env"]])},
        }

    record = {
        "experiment": exp.name, "description": exp.description, "target_opcode": exp.target_opcode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"trials": trials, "warmup_ms": warmup_ms, "warmup_launches": n_warm, "seed": seed,
                   "variants": [v.label for v in variants], "ptxas_flags": exp.ptxas_flags, **{k: v for k, v in opts.items() if k not in ("trials", "seed")}},
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "device": dev.props, "env_static": static_env, "clock_lock": clock_state,
        "env_before": env_before, "env_after_warmup": env_after_warm,
        "counters": "unavailable: Nsight Compute (ncu) not installed",
        "warnings": warnings,
        "summary": summary,
        "trials": {label: [{**r, "env": e} for r, e in zip(pv["results"], pv["env"])] for label, pv in per_variant.items()},
    }
    record["artifact_dir"] = str(_save(exp, record, builds, per_variant))
    return record


def _save(exp, record, builds, per_variant) -> Path:
    d = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}_{exp.name}"
    d.mkdir(parents=True, exist_ok=True)
    for i, (src, b) in enumerate(builds.items()):
        tag = f"k{i}"
        (d / f"{tag}.cu").write_text(src)
        (d / f"{tag}.ptx").write_text(b.ptx)
        (d / f"{tag}.cubin").write_bytes(b.cubin)
        (d / f"{tag}.ptxas.log").write_text(b.ptxas_log)
        (d / f"{tag}.nvdisasm.txt").write_text(b.sass)
        ins = sass.parse(b.sass)
        (d / f"{tag}.sass.txt").write_text(sass.format_listing(ins))
        (d / f"{tag}.sass.json").write_text(json.dumps([x.to_dict() for x in ins], indent=1))
        for pv in per_variant.values():
            if pv["build"] is b:
                pv["artifact"] = tag
    for label, pv in per_variant.items():
        record["summary"][label]["kernel_artifact"] = pv.get("artifact")
    (d / "record.json").write_text(json.dumps(record, indent=1, default=str))
    return d


def report(record: dict, per_variant_sass: dict | None = None, sass_lines: int = 8) -> str:
    out = [f"experiment:      {record['experiment']}  ({record['description']})"]
    dev = record["device"]
    out.append(f"device:          {dev['name']} (sm_{dev['cc'].replace('.', '')}, {dev['sms']} SMs), driver {record['env_static']['driver']}")
    out.append(f"clocks:          {record['clock_lock']}")
    out.append(f"ptxas flags:     {' '.join(record['config']['ptxas_flags']) or '(default -O3)'}")
    out.append(f"trials:          {record['config']['trials']} per variant, shuffled (seed {record['config']['seed']}), "
               f"{record['config']['warmup_launches']} warmup launches")
    for label, s in record["summary"].items():
        m = s["metrics"]
        r = s["resources"]
        out.append("")
        out.append(f"[{label}]")
        out.append(f"  GPU clock:       {m['sm_mhz_inkernel']['median']:.0f} MHz in-kernel "
                   f"(NVML {s['env']['sm_mhz_nvml']['median']:.0f} MHz, {s['env']['temp_c']['median']:.0f} C)")
        out.append(f"  iterations:      {m['ops_per_thread']['median']:,.0f} ops/thread")
        out.append(f"  warps:           {s['params'].get('warps')}")
        out.append(f"  registers:       {r.get('registers')}  (spills {r.get('spill_stores', 0)}/{r.get('spill_loads', 0)}, "
                   f"occupancy {r['occupancy']['theoretical']:.0%} = {r['occupancy']['max_warps_per_sm']} warps/SM)")
        c = m["cycles_per_op"]
        out.append(f"  cycles/op:       {c['median']:.3f}  [p05 {c['p05']:.3f}, p95 {c['p95']:.3f}, CV {c['cv']:.2%}]")
        t = m["warp_ops_per_cycle"]
        out.append(f"  throughput:      {t['median']:.3f} warp-instr/cycle/SM  = {t['median'] * 32:.1f} thread-ops/cycle/SM")
        if s.get("correctness") is not None:
            out.append(f"  result check:    {'PASS' if s['correctness'] else 'FAIL'} (host reference)")
        out.append(f"  runtime:         {m['runtime_ms']['median']:.3f} ms (event-timed)")
        out.append(f"  instructions:    ~{s['dynamic_instructions_per_warp_est']:,} per warp (static loop body {s['loop_body_instructions']} x trips)")
        out.append(f"  loop body:       {', '.join(f'{k}x{v}' for k, v in s['loop_body_histogram'].items())}")
    out.append("")
    out.append("SASS (timed loop body, control: Stall Yield Wbar Rbar wait-mask reuse):")
    k0 = next(iter(record["summary"].values()))["kernel_artifact"]
    lst = (Path(record["artifact_dir"]) / f"{k0}.sass.txt").read_text().splitlines()
    loop = [ln for ln in lst if record["target_opcode"] and record["target_opcode"] in ln]
    out += ["  " + ln.strip() for ln in loop[:sass_lines]]
    if len(loop) > sass_lines:
        out.append(f"  ... ({len(loop)} {record['target_opcode']} total)")
    out.append("")
    out.append(f"counters:        {record['counters']}")
    if record["warnings"]:
        out.append("WARNINGS:")
        out += [f"  ! {w}" for w in record["warnings"]]
    else:
        out.append("warnings:        none")
    out.append(f"artifacts:       {record['artifact_dir']}")
    return "\n".join(out)
