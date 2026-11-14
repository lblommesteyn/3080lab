"""Static headroom for software pipelining (cross-iteration motion), CPU only.

For each hot loop (randk dev + held-out + suite):
  cp_iter   critical path of one iteration in program-order DAG terms: RAW at resched latencies,
            WAR/WAW 1 cycle, memory order kept (no alias facts). What a block-local scheduler
            is bound by, plus the loop's own dependences.
  rec       recurrence bound (RecMII) with distance-1 loop-carried RAW edges: for a value defined at u
            and read at v <= u (it feeds the next trip), the true path v -> u plus u's latency.
  mem       chain of memory-order edges within one trip (must also hold across trips without
            alias facts, so it is a recurrence too).
  issue     instruction count (one issue slot each).
  bound     max(rec, mem, issue): a modulo schedule's best initiation interval in this model.
  headroom  cp_iter / bound: the best single-warp gain from overlapping trips. It only turns into
            time where the loop is latency bound (low occupancy); the report weights by that.

usage: python scripts/swp_headroom.py [out=data/ra/swp_headroom.json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(r"C:\Users\LukeB\dev\3080lab\.venv\Lib\site-packages")
import torch  # noqa: E402,F401
from lab import resched, sass, suite, toolchain  # noqa: E402
from lab import operands as opnd  # noqa: E402
from lab.experiments import randk  # noqa: E402


def lat(body, table, p, v):
    ip = body[p]
    return resched.var_latency(ip) if ip.wbar != 7 else resched.latency(ip, table, body[v])


def analyze(body, cache) -> dict:
    table = resched.fixed_latency_table(body, cache)
    res = [resched._resources(x, cache) for x in body]
    n = len(body)
    if any(r is None for r in res):
        return {"skip": "unmodelled instruction in loop"}
    # one-trip DAG (all deps + memory order) -> cp_iter; RAW-only forward paths for recurrences
    full = [0] * n
    raw_pred = [[] for _ in range(n)]
    last_def, readers = {}, {}
    mem_t = [0] * n
    for v in range(n):
        d, u = res[v]
        f = 0
        if resched._memory(body[v]):
            for p in range(v):
                if resched._memory(body[p]) and (resched._writes_mem(body[p]) or resched._writes_mem(body[v])):
                    f = max(f, full[p] + 1)
                    mem_t[v] = max(mem_t[v], mem_t[p] + 1)
        for x in u:
            if x in last_def:
                p = last_def[x]
                L = lat(body, table, p, v)
                f = max(f, full[p] + L)
                raw_pred[v].append((p, L))
        for x in d:
            if x in last_def:
                f = max(f, full[last_def[x]] + 1)
            for p in readers.get(x, ()):
                f = max(f, full[p] + 1)
        full[v] = f
        for x in u:
            readers.setdefault(x, []).append(v)
        for x in d:
            last_def[x], readers[x] = v, []
    cp_iter = max(full) + 1
    # loop-carried RAW: value defined at u (last def in the trip) and read at v before any def in the trip
    first_read, defined = {}, set()
    for v in range(n):
        for x in res[v][1]:
            if x not in defined:
                first_read.setdefault(x, []).append(v)
        defined |= res[v][0]
    rec = 0
    for x, u in last_def.items():
        for v in first_read.get(x, []):
            if v > u:
                continue
            # longest RAW path v -> u inside the trip
            dist = {v: 0}
            for k in range(v + 1, u + 1):
                best = max((dist[p] + L for p, L in raw_pred[k] if p in dist), default=None)
                if best is not None:
                    dist[k] = best
            if u in dist:
                rec = max(rec, dist[u] + lat(body, table, u, v))
    # memory order across trips (no alias facts): a memory op v of trip i+1 follows every earlier-trip
    # memory op u when either writes memory, so path(v -> u) through RAW + memory-order edges, plus 1,
    # is a recurrence too (a load feeding a store recurs through the next trip's load)
    mem = 0
    memops = [k for k in range(n) if resched._memory(body[k])]
    for v in memops:
        dist = {v: 0}
        for k in range(v + 1, n):
            c = [dist[p] + L for p, L in raw_pred[k] if p in dist]
            if resched._memory(body[k]):
                c += [dist[p] + 1 for p in memops if p in dist and p < k
                      and (resched._writes_mem(body[p]) or resched._writes_mem(body[k]))]
            if c:
                dist[k] = max(c)
        for u in memops:
            if u >= v and u in dist and (resched._writes_mem(body[u]) or resched._writes_mem(body[v])):
                mem = max(mem, dist[u] + 1)
    bound = max(rec, mem, n)
    return {"n": n, "cp_iter": cp_iter, "rec": rec, "mem": mem, "bound": bound,
            "headroom": cp_iter / bound, "stall_sum": sum(x.stall or 32 for x in body)}


def ipc_from_log(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r"^\[(seed\d+)/ptxas\]\n((?:  .*\n)*)", txt, re.M):
        t = re.search(r"throughput:\s*([\d.]+)", m.group(2))
        w = re.search(r"= (\d+) warps/SM", m.group(2))
        out[m.group(1)] = {"ipc": float(t[1]) if t else None, "warps_sm": int(w[1]) if w else None}
    return out


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    cache = opnd._load_cache()
    rows = []
    logs = {"resched_rand": Path(kv.get("dev_log", ROOT / "data/ra/ra-rand-2211.log")),
            "resched_rand_b": Path(kv.get("heldout_log", ROOT / "data/ra/ra-rand-b-2212.log"))}
    for key, log in logs.items():
        meas = ipc_from_log(log)
        e = randk.registry()[key]
        for v in e.variants({"n": 40, "policies": "ptxas"}):
            s = v.label.split("/")[0]
            body = sass.loop_body(sass.parse(toolchain.disassemble(toolchain.build(e.source(v)).cubin)))
            r = {"name": f"{key}:{s}", **meas.get(s, {})}
            r.update(analyze(body, cache) if body else {"skip": "no loop"})
            rows.append(r)
            print(json.dumps(r), flush=True)
    for c in suite.CASES:
        body = sass.loop_body(sass.parse(toolchain.disassemble(toolchain.build(c.src).cubin)))
        r = {"name": "suite:" + c.name}
        r.update(analyze(body, cache) if body else {"skip": "no loop"})
        rows.append(r)
        print(json.dumps(r), flush=True)
    out = ROOT / kv.get("out", "data/ra/swp_headroom.json")
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
