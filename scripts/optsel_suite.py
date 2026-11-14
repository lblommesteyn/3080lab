"""Suite decisions under timing protocol v2 (data/optsel/PROTOCOL.md), with an A/A arm in every case.

Protocol v2 vs suite_run.py:
  * K arms share K output-buffer sets, rotated every round (arm i uses set (i + round) mod K), so a
    placement effect (A/A job 2216: 1.0% on transpose, stable across processes) is spread over all arms.
  * `ptxas_dup` is a second load of ptxas's own cubin: an A/A control timed exactly like a candidate.
  * statistic: 10%-trimmed mean of the per-round paired log ratio log(t_ptxas / t_arm); the CUDA event
    timer has a 1.024 us quantum, which a median cannot resolve below and a mean can.
  * `procs` fresh processes per case (allocation and module placement re-drawn), pooled.
Correctness: every candidate must match ptxas bitwise (suite_run's _check child); ptxas against numpy.

  pcslurm submit -- python scripts/optsel_suite.py [rounds=36] [procs=2] [cases=regex]
"""
import ctypes, json, math, random, re, statistics, subprocess, sys, tempfile, zlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa
from lab import toolchain
import suite_run as S

CANDS = ["identity_late", "crit_late", "sb_late", "sb_wide_late", "split_identity_late", "split_sb_late", "split24_sb_late"]


def tmean(xs, trim=0.1):
    xs = sorted(xs); k = int(len(xs) * trim)
    xs = xs[k:len(xs) - k] or xs
    return sum(xs) / len(xs)


def child(cubin_dir, name, arms, rounds, seed):
    from lab.gpu import Device, Kernel
    c = S.case_by_name(name)
    dev = Device()
    _, outs, in_ptrs, scal = S.setup(dev, c)
    src = {a: ("ptxas" if a == "ptxas_dup" else a) for a in arms}
    kern = {a: Kernel.load(dev, (Path(cubin_dir) / f"{name}.{src[a]}.cubin").read_bytes(), "k") for a in arms}
    K = len(arms)
    bufs = [S.alloc_outs(dev, outs) for _ in range(K)]
    fl = Kernel.load(dev, toolchain.build(S.FLUSH_SRC).cubin, "flush")
    buf, sink = dev.alloc(S.FLUSH_BYTES), dev.alloc(4)
    dev.memset(buf, S.FLUSH_BYTES)
    fargs = [ctypes.c_uint64(buf), ctypes.c_int32(S.FLUSH_BYTES // 16), ctypes.c_uint64(sink)]
    rng = random.Random(seed)
    t = {a: [] for a in arms}
    for r in range(rounds + 2):
        order = list(range(K)); rng.shuffle(order)
        for i in order:
            a = arms[i]
            fl.launch(dev.props["sms"] * 4, 256, fargs)
            x = kern[a].timed_launch(S.dim3(c.grid), S.dim3(c.block), S.args_for(in_ptrs, bufs[(i + r) % K], scal), c.smem)
            if r >= 2:
                t[a].append(x * 1e3)
    print("RESULT " + json.dumps(t))


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:])
    rounds, procs, pat = int(kv.get("rounds", 36)), int(kv.get("procs", 2)), kv.get("cases", "")
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for c in S.CASES:
            if not re.search(pat, c.name):
                continue
            b = toolchain.build(c.src)
            row = {"resources": toolchain.ptxas_resources(b.ptxas_log), "variants": {}}
            ok = []
            for v in ["ptxas"] + CANDS:
                try:
                    cub = b.cubin if v == "ptxas" else S.transform(b.cubin, v)
                except Exception as e:  # noqa: BLE001
                    row["variants"][v] = {"status": "build_failed", "error": str(e)[:300]}; continue
                p = Path(tmp) / f"{c.name}.{v}.cubin"; p.write_bytes(cub)
                (ROOT / "data/optsel/cubins").mkdir(parents=True, exist_ok=True)
                (ROOT / "data/optsel/cubins" / f"{c.name}.{v}.cubin").write_bytes(cub)
                r, err = S.run_child("_check", p, c.name, v)
                info = {"identical_to_ptxas": cub == b.cubin}
                if err:
                    info.update(status="fault", error=err[-300:])
                else:
                    info.update(sha=r["sha"], registers=r["attrs"]["registers"], attrs=r["attrs"])
                    if v == "ptxas":
                        info["status"] = "ok" if r["ref"]["ok"] else "ref_mismatch"
                    else:
                        info["status"] = "ok" if r["sha"] == row["variants"]["ptxas"].get("sha") else "bitwise_mismatch"
                row["variants"][v] = info
                if info["status"] == "ok" or (v == "ptxas" and info.get("sha")):
                    ok.append(v)
            if "ptxas" not in ok:
                res[c.name] = row; continue
            arms = ["ptxas", "ptxas_dup"] + [v for v in ok if v != "ptxas"]
            times = {a: [] for a in arms}
            for p in range(procs):
                r = subprocess.run([sys.executable, __file__, "_child", tmp, c.name, ",".join(arms), str(rounds), str(zlib.crc32(c.name.encode()) + p)],
                                   capture_output=True, text=True, timeout=1800)
                line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
                if not line:
                    row["timing_error"] = r.stderr[-400:]; break
                for a, xs in json.loads(line[7:]).items():
                    times[a].append(xs)
            row["times"] = times
            if "timing_error" not in row:
                for a in arms[1:]:
                    lr = [math.log(x / y) for pb, pa in zip(times["ptxas"], times[a]) for x, y in zip(pb, pa)]
                    row.setdefault("y", {})[a] = tmean(lr)
            res[c.name] = row
            print(c.name, row["variants"]["ptxas"]["status"], " ".join(f"{a}:{v:+.4f}" for a, v in row.get("y", {}).items()),
                  row.get("timing_error", "")[-200:], flush=True)
    dst = ROOT / "data/optsel" / kv.get("out", "suite_v2.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res)); print("wrote", dst)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_child"]:
        child(sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
