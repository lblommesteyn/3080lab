"""Outside-kernel benchmark for scripts/sassopt.py: PolyBench/GPU (University of Delaware), CUDA versions.

  pcslurm submit -- python scripts/ext_polybench.py [policies=sb,sb_wide] [seeds=3] [rounds=20]
  python scripts/ext_polybench.py cpu            compatibility + static stages only, no GPU

The source is cloned at a pinned commit into vendor/polybenchGpu (gitignored, not redistributed). Every
__global__ kernel becomes a standalone extern "C" file under vendor/ext_kernels: the benchmark's STANDARD
dataset sizes, DATA_TYPE float, its own block shape; the grid covers the largest dimension (the kernels
bounds-check), and every buffer holds (max dim + 64)^2 elements, ^3 for the 3-D benchmarks. Scalars: sizes
from the dataset, loop indices 1, alpha 1.5, other floats 0.5. Nothing here was used to develop the
scheduler; it was first run on Oct 18 2026, after the policies were frozen. Results: data/ext_polybench/.
"""
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO, COMMIT = "https://github.com/sgrauerg/polybenchGpu.git", "5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c"
VENDOR = ROOT / "vendor" / "polybenchGpu"
KDIR = ROOT / "vendor" / "ext_kernels"
OUT = ROOT / "data" / "ext_polybench"



def std_defines(cuh: str) -> dict:
    m = re.search(r"#\s*ifdef STANDARD_DATASET(.*?)#\s*endif", cuh, re.S)
    body = m.group(1) if m else ""
    return {k: v.strip() for k, v in re.findall(r"#define\s+(\w+)\s+([^\n]+)", body)}


def simple_defines(text: str) -> dict:
    out = {}
    for k, v in re.findall(r"^\s*#\s*define\s+(\w+)[ \t]+([^\n/]+)", text, re.M):
        v = v.strip()
        if re.fullmatch(r"[\w.+\-*/() ]+", v) and not k.endswith("_H") and not k.endswith("_CUH"):
            out[k] = v
    return out


def functions(text: str, kind: str):
    for m in re.finditer(rf"{kind}\s+[^;{{]*?\b(\w+)\s*\(([^)]*)\)\s*{{", text):
        depth, i = 0, m.end() - 1
        while True:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
            if depth == 0:
                break
        yield m.group(1), m.group(2), text[m.start():i]


def fetch():
    if not (VENDOR / ".git").exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(VENDOR)], check=True)
    subprocess.run(["git", "-C", str(VENDOR), "checkout", "-q", COMMIT], check=True)


def extract():
    SRC, DST = VENDOR / "CUDA", KDIR
    DST.mkdir(parents=True, exist_ok=True)
    manifest = []
    for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
        cu = next(d.glob("*.cu"), None)
        cuh = next(d.glob("*.cuh"), None)
        if not cu or not cuh:
            continue
        t_cu, t_cuh = cu.read_text(errors="replace"), cuh.read_text(errors="replace")
        sizes = std_defines(t_cuh)
        defs = {**simple_defines(t_cu), **sizes}
        defs.pop("RUN_ON_CPU", None)
        devs = "\n\n".join(body for _, _, body in functions(t_cu, r"__device__"))
        for name, params, body in functions(t_cu, r"__global__"):
            hdr = ["#define POLYBENCH_LOOP_BOUND(x, y) x", "#define DATA_TYPE float"]
            hdr += [f"#define {k} {v}" for k, v in defs.items() if k not in ("DATA_TYPE",)]
            hdr += [f"#define _PB_{k} {k}" for k in sizes]
            src = "\n".join(hdr) + "\n\n" + devs + "\n\n" + 'extern "C" ' + body.replace("__global__", "__global__", 1) + "\n"
            try:
                vals = {k: int(eval(v, {}, {})) for k, v in sizes.items()}
            except Exception:
                continue
            if not vals:
                print("no sizes:", d.name, name); continue
            mx = max(vals.values())
            three_d = "blockIdx.z" in body or d.name in ("3DCONV", "DOITGEN")
            bx = int(eval(defs.get("DIM_THREAD_BLOCK_X", "32"), {}, {}))
            by = int(eval(defs.get("DIM_THREAD_BLOCK_Y", "8"), {}, {}))
            two_d = "blockIdx.y" in body or "threadIdx.y" in body
            grid = [math.ceil(mx / bx), math.ceil(mx / by)] if two_d else [math.ceil(mx / bx)]
            block = [bx, by] if two_d else [bx]
            n = (mx + 64) ** 3 if three_d else (mx + 64) ** 2
            scalars = {}
            for p in filter(None, (x.strip() for x in params.split(","))):
                if "*" in p:
                    continue
                pname = p.split()[-1]
                if p.startswith("int"):
                    scalars[pname] = vals.get(pname.upper(), 1)
                else:
                    scalars[pname] = 1.5 if "alpha" in pname else 0.5
            fname = f"{d.name.lower()}__{name}.cu"
            (DST / fname).write_text(src)
            manifest.append({"file": fname, "bench": d.name, "kernel": name, "grid": grid, "block": block, "n": n,
                             "scalars": scalars})
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def run(policies, seeds, rounds, cpu):
    fetch()
    man = extract()
    OUT.mkdir(parents=True, exist_ok=True)
    for pol in policies:
        rows = []
        for x in man:
            out = KDIR / f"{x['file'][:-3]}.{pol}.cubin"
            a = [sys.executable, str(ROOT / "scripts" / "sassopt.py"), str(KDIR / x["file"]), "--kernel", x["kernel"],
                 "--grid", ",".join(map(str, x["grid"])), "--block", ",".join(map(str, x["block"])), "--n", str(x["n"]),
                 "--policy", pol, "--seeds", str(seeds), "--rounds", str(rounds), "--out", str(out)]
            a += ["--cpu-only"] if cpu else []
            for k, v in x["scalars"].items():
                a += ["--scalar", f"{k}={v}"]
            r = subprocess.run(a, capture_output=True, text=True, timeout=3600)
            rep = json.loads(out.with_suffix(".json").read_text()) if out.with_suffix(".json").exists() else {}
            st = rep.get("stages", {})
            row = {"file": x["file"], "decision": rep.get("decision"), "returncode": r.returncode,
                   "compat": st.get("compat", {}).get("ok"), "static": st.get("static", {}).get("ok"),
                   "differential": st.get("differential", {}).get("ok"),
                   "speedup": st.get("timing", {}).get("speedup"),
                   "changed": not st.get("compat", {}).get("identical", True)}
            rows.append(row)
            print(f"{pol:8s} {x['file']:40s} {row['decision']}", flush=True)
        timed = [r["speedup"] for r in rows if r["speedup"]]
        geo = math.exp(statistics.fmean(map(math.log, timed))) if timed else None
        kept = [r["speedup"] if r["decision"] and r["decision"].startswith("kept rewrite") else 1.0 for r in rows]
        summ = {"policy": pol, "kernels": len(rows), "changed": sum(r["changed"] for r in rows),
                "static_fail": sum(r["static"] is False for r in rows),
                "differential_fail": sum(r["differential"] is False for r in rows),
                "timed": len(timed), "geomean_raw_rewrite": geo,
                "geomean_with_decision": math.exp(statistics.fmean(map(math.log, kept))), "rows": rows}
        (OUT / f"{pol}{'_cpu' if cpu else ''}.json").write_text(json.dumps(summ, indent=1))
        print(f"{pol}: {summ['changed']}/{len(rows)} changed, static fails {summ['static_fail']}, differential fails "
              f"{summ['differential_fail']}, raw rewrite geomean over {len(timed)} timed: {geo}, "
              f"with keep/revert decision over all {len(rows)}: {summ['geomean_with_decision']:.4f}", flush=True)


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    run(kv.get("policies", "sb,sb_wide").split(","), int(kv.get("seeds", 3)), int(kv.get("rounds", 20)),
        "cpu" in sys.argv[1:])
