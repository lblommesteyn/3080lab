"""Summarize data/night_oct11: ours vs llama.cpp b11485, best of all rounds and repetitions on each side.

  python scripts/night_summary.py [dir=data/night_oct11]
"""
import glob
import json
import re
import sys
from pathlib import Path


def main():
    d = Path(dict(a.split("=") for a in sys.argv[1:]).get("dir", "data/night_oct11"))
    best = lambda xs: max(xs) if xs else float("nan")  # noqa: E731
    # llama-bench: per-sample tok/s; best sample, and best per-round mean
    lb = {}
    for f in sorted(glob.glob(str(d / "lb_*_r*.json"))):
        txt = Path(f).read_text(encoding="utf-8", errors="replace")
        txt = txt[txt.find("["):]
        for r in json.loads(txt):
            key = (f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}") + f" fa{int(bool(r['flash_attn']))}"
            lb.setdefault(key, {"samples": [], "means": []})
            lb[key]["samples"] += r["samples_ts"]
            lb[key]["means"].append(r["avg_ts"])
    ours_pp, ours_tg, ours_btg, lbb = {}, [], {}, {}
    for f in glob.glob(str(d / "ours_pp_r*.txt")):
        for m in re.finditer(r"P=(\d+):\s+(\d+) tok/s", Path(f).read_text()):
            ours_pp.setdefault(int(m[1]), []).append(int(m[2]))
    for f in glob.glob(str(d / "ours_tg_r*.txt")):
        ours_tg += [float(x) for x in re.findall(r"ours\(gguf\)\s+([\d.]+)", Path(f).read_text())]
    for f in glob.glob(str(d / "ours_btg_r*.txt")):
        for m in re.finditer(r"B=\s*(\d+):\s+([\d.]+) tok/s", Path(f).read_text()):
            ours_btg.setdefault(int(m[1]), []).append(float(m[2]))
    for f in glob.glob(str(d / "lbb_r*.txt")):
        for line in Path(f).read_text().splitlines():
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) >= 8 and c[2].isdigit():
                lbb.setdefault(int(c[2]), []).append(float(c[7]))
    print("prefill (tok/s, best)            ours   llama.cpp fa1 (best sample / best round mean)   ratio")
    for P in sorted(ours_pp):
        k = f"pp{P} fa1"
        if k in lb:
            print(f"  P={P:5d}  {best(ours_pp[P]):8.0f}   {best(lb[k]['samples']):8.0f} / {best(lb[k]['means']):8.0f}"
                  f"   {best(ours_pp[P]) / best(lb[k]['samples']):.2f}x (runs: ours {sorted(ours_pp[P])})")
    print("decode B=1 (tok/s, best)")
    for k in ("tg256 fa0", "tg256 fa1"):
        if k in lb:
            print(f"  llama.cpp {k}: {best(lb[k]['samples']):.1f} / mean {best(lb[k]['means']):.1f}")
    print(f"  ours (qwen_gguf.py): {best(ours_tg):.1f}  (runs {sorted(ours_tg)})")
    print("batched decode (tok/s, best of rounds)  ours   llama-batched-bench S_TG   ratio")
    for B in sorted(ours_btg):
        o, l_ = best(ours_btg[B]), best(lbb.get(B, []))
        print(f"  B={B:3d}  {o:8.1f}   {l_:8.1f}   {o / l_:.2f}x")


if __name__ == "__main__":
    main()
