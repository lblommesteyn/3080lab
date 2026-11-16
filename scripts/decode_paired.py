"""Paired decode benchmark: ours vs two llama.cpp builds, rotated rounds, central estimates.

Replaces the best-of rule of the Oct 11/13 night runs (scripts/night_summary.py). Each round runs every
arm once, in an order rotated by one position per round, so slow drift or a background load hits each
arm at every position equally. `ours_dup` is a second, identical run of our B = 1 decode in the same
round: its ratio to `ours` is the A/A noise of the comparison.

Statistic per (workload, pair): the per-round ratio of throughputs; we report the median over rounds,
its min-max, and a bootstrap 95% interval of the mean log ratio. llama.cpp B = 1 is run with flash
attention on and off; the summary uses whichever has the higher median (generous to llama.cpp).

  pcslurm submit -- python scripts/decode_paired.py rounds=6 old=<dir of b11485> new=<dir of b12428>
  python scripts/decode_paired.py summarize [in=data/decode_paired/raw.json]
"""
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/"
                      "91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_0.gguf")
BS = [1, 2, 4, 8, 16, 32]
OUT = ROOT / "data" / "decode_paired"


def run(cmd, timeout=1800):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       cwd=ROOT)
    return p.stdout + "\n" + p.stderr, time.time() - t


def ours_b1(_):
    txt, dt = run([sys.executable, "scripts/qwen_gguf.py"])
    v = [float(x) for x in re.findall(r"ours\(gguf\)\s+([\d.]+)", txt)]
    # qwen_gguf prints 3 runs; the first includes graph capture warm-up effects, take the median of all 3
    return {"tg_b1": statistics.median(v) if v else None, "runs": v, "sec": dt}, txt


def ours_batched(_):
    txt, dt = run([sys.executable, "scripts/qwen_batched_decode.py", ",".join(map(str, BS))])
    r = {"sec": dt}
    for m in re.finditer(r"B=\s*(\d+):\s+([\d.]+) tok/s.*rows identical: (\w+)", txt):
        r[f"tg_b{m[1]}"] = float(m[2])
        r[f"rows_identical_b{m[1]}"] = m[3] == "True"
    return r, txt


def llama_bench(d):
    def f(_):
        txt, dt = run([str(Path(d) / "llama-bench.exe"), "-m", str(GGUF), "-ngl", "99", "-p", "0", "-n", "256",
                       "-fa", "0,1", "-r", "5", "-o", "json"])
        r = {"sec": dt}
        try:
            for x in json.loads(txt[txt.find("["):txt.rfind("]") + 1]):
                r[f"tg_b1_fa{int(bool(x['flash_attn']))}"] = x["avg_ts"]
                r["build"] = x.get("build_number")
        except ValueError:
            pass
        return r, txt
    return f


def llama_batched(d):
    def f(_):
        txt, dt = run([str(Path(d) / "llama-batched-bench.exe"), "-m", str(GGUF), "-ngl", "99", "-fa", "on",
                       "-c", "8192", "-npp", "128", "-ntg", "128", "-npl", ",".join(map(str, BS))])
        r = {"sec": dt}
        for line in txt.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 8 and c[2].isdigit():
                r[f"tg_b{int(c[2])}"] = float(c[7])
        return r, txt
    return f


def collect(kv):
    rounds = int(kv.get("rounds", 6))
    arms = {"ours": ours_b1, "ours_dup": ours_b1, "ours_batched": ours_batched,
            "old_bench": llama_bench(kv["old"]), "old_batched": llama_batched(kv["old"]),
            "new_bench": llama_bench(kv["new"]), "new_batched": llama_batched(kv["new"])}
    names = list(arms)
    random.Random(0).shuffle(names)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {"gguf": str(GGUF.name), "order_seed": 0, "rounds": []}
    smi, _ = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                  "--format=csv,noheader"])
    raw["gpu_start"] = smi.strip()
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        rec = {"round": r, "order": order}
        for a in order:
            res, txt = arms[a](r)
            rec[a] = res
            (OUT / f"r{r}_{a}.txt").write_text(txt, encoding="utf-8")
            print(f"round {r} {a:13s} " + " ".join(f"{k}={v:.1f}" for k, v in res.items()
                                                      if isinstance(v, float) and k.startswith("tg")), flush=True)
        raw["rounds"].append(rec)
        (OUT / "raw.json").write_text(json.dumps(raw, indent=1))
    return raw


def boot_ci(xs, n=4000):
    rng = random.Random(1)
    ms = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def ratios(rounds, num, den):
    """num/den are (arm, key) or a function round -> value."""
    out = []
    for rec in rounds:
        a = num(rec) if callable(num) else rec.get(num[0], {}).get(num[1])
        b = den(rec) if callable(den) else rec.get(den[0], {}).get(den[1])
        if a and b:
            out.append(a / b)
    return out


def summarize(raw):
    R = raw["rounds"]
    lines = [f"{len(R)} rounds, GPU at start: {raw.get('gpu_start')}", "",
             "| workload | ours (median tok/s) | vs | llama.cpp (median tok/s) | ratio median [min, max] | "
             "mean log ratio 95% CI (as x) |", "|---|---:|---|---:|---:|---:|"]
    res = {}
    for build in ("old", "new"):
        fa = max((0, 1), key=lambda f: statistics.median(
            [x[f"{build}_bench"].get(f"tg_b1_fa{f}", 0) for x in R]))
        rows = [("B = 1", ("ours", "tg_b1"), (f"{build}_bench", f"tg_b1_fa{fa}"), f"fa{fa}")]
        rows += [(f"B = {b} (batched)", ("ours_batched", f"tg_b{b}"), (f"{build}_batched", f"tg_b{b}"), "")
                 for b in BS]
        bn = next((x[f"{build}_bench"].get("build") for x in R if x[f"{build}_bench"].get("build")), build)
        for name, num, den, note in rows:
            rs = ratios(R, num, den)
            if not rs:
                continue
            o = statistics.median(x[num[0]][num[1]] for x in R if num[1] in x[num[0]])
            l = statistics.median(x[den[0]][den[1]] for x in R if den[1] in x[den[0]])
            lo, hi = boot_ci([math.log(z) for z in rs])
            res[f"b{bn}:{name}"] = dict(ours=o, llama=l, ratio_median=statistics.median(rs), ratio_min=min(rs),
                                        ratio_max=max(rs), ci=[math.exp(lo), math.exp(hi)], n=len(rs))
            lines.append(f"| {name} | {o:.0f} | b{bn} {note} | {l:.0f} | **{statistics.median(rs):.2f}x** "
                         f"[{min(rs):.2f}, {max(rs):.2f}] | [{math.exp(lo):.2f}, {math.exp(hi):.2f}] |")
    aa = ratios(R, ("ours_dup", "tg_b1"), ("ours", "tg_b1"))
    if aa:
        res["aa_ours_b1"] = dict(median=statistics.median(aa), min=min(aa), max=max(aa))
        lines += ["", f"A/A (ours_dup / ours, B = 1): median {statistics.median(aa):.3f}, "
                      f"range [{min(aa):.3f}, {max(aa):.3f}] over {len(aa)} rounds"]
    bad = [(x["round"], k) for x in R for k, v in x.get("ours_batched", {}).items()
           if k.startswith("rows_identical") and v is False]
    lines.append(f"batched rows identical in every round: {not bad}" + (f" (failures {bad})" if bad else ""))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.md").write_text(txt + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if "summarize" in sys.argv:
        summarize(json.loads((ROOT / kv.get("in", "data/decode_paired/raw.json")).read_text()))
    else:
        summarize(collect(kv))


if __name__ == "__main__":
    main()
