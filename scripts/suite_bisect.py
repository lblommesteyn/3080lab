"""Find which rescheduled basic block breaks a suite case (bitwise mismatch vs ptxas).

  pcslurm submit -- python scripts/suite_bisect.py cases=layernorm,spmv_csr policies=sb,sb_wide [port=old|new] [out=name]

For each (case, policy): the full rewrite, a padding-only rewrite (no block scheduled), then one cubin
per scheduled block with only that block rewritten. Each runs once in its own process on the suite's
seeded inputs (scripts/suite_run.py child) and is compared with ptxas's output hash. port=old
reproduces the Oct 18 suite run (memory ops at 1-cycle issue cost); port=new is the current scheduler.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401

import suite_run as SR  # noqa: E402
from lab import resched, sass, toolchain, verify  # noqa: E402
from lab.suite import CASES  # noqa: E402


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    cases = kv.get("cases", "layernorm,spmv_csr,transpose,bitonic_block").split(",")
    policies = kv.get("policies", "sb,sb_wide").split(",")
    if kv.get("port", "old") == "old":
        resched._port = lambda i: resched.port_of_v1(i.opcode)[:2]
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in cases:
            c = next(c for c in CASES if c.name == name)
            b = toolchain.build(c.src)
            p = Path(tmp) / f"{name}.ptxas.cubin"
            p.write_bytes(b.cubin)
            base, err = SR.run_child("_check", p, name, "ptxas")
            if err:
                print(name, "ptxas fault", err[-200:], flush=True)
                continue
            ins = sass.parse(b.sass)
            for pol in policies:
                kw = SR.VARIANTS.get(pol, dict(policy=pol))          # suite_run variant names too
                _, log = resched.reschedule(b.cubin, "k", **kw)
                starts = sorted({s for s, m in log if m.startswith("scheduled")})
                variants = [("all", None), ("pad-only", set())] + [(f"blk@{s}", {s}) for s in starts]
                rows = []
                for tag, only in variants:
                    cub, _ = resched.reschedule(b.cubin, "k", **kw, only_blocks=only)
                    if cub == b.cubin:
                        rows.append((tag, "identical"))
                        continue
                    q = Path(tmp) / f"{name}.{pol}.{tag.replace('@', '_')}.cubin"
                    q.write_bytes(cub)
                    r, err = SR.run_child("_check", q, name, pol)
                    st = "fault" if err else ("ok" if r["sha"] == base["sha"] else "MISMATCH")
                    hz = verify.new_hazards(cub, b.cubin)
                    first = ins[int(tag[4:])].text.strip()[:40] if tag.startswith("blk@") else ""
                    rows.append((tag, st, len(hz), first))
                    print(f"{name:14s} {pol:8s} {tag:10s} {st:9s} hazards={len(hz):3d} {first}", flush=True)
                out[f"{name}/{pol}"] = rows
    dst = ROOT / "data" / "suite" / f"bisect_{kv.get('out', kv.get('port', 'old'))}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
