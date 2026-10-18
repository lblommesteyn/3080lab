"""Cross-kernel optimizer: optimize a SEQUENCE of CUDA kernels together instead of one cubin at a time.

A Program is an ordered list of launches of extern "C" kernels over named buffers. Passes:

  vertical fusion    launches i, i+1 with the same grid and block, where every buffer one of them writes
                     and the other touches is accessed by both through ONE identical subscript built
                     from identically defined index variables: each thread only consumes what it
                     produced, so the two bodies can run back to back in one thread with no grid-wide
                     barrier. Saves a launch, the drain/fill between kernels, and lets the consumer read
                     the producer's value while it is still in L1/L2.
  horizontal fusion  consecutive launches that share no written buffer (no RAW, WAR or WAW between
                     them) with the same block shape: one launch whose grid is the concatenation of
                     theirs; each block runs the body it belongs to with blockIdx/gridDim remapped.
                     Saves launches and fills the tail of one kernel's last wave with the other's blocks.
  graph              whatever sequence remains is captured into one CUDA graph (launch overhead).

Rejected outright (never fused): __syncthreads, shared memory, atomics, warp intrinsics, inline asm,
pointer arithmetic on a buffer parameter or a buffer passed to another function, so dependence can be
read from subscripts. Fused bodies become __device__ __forceinline__ functions; every buffer shared by
fused bodies loses const and __restrict__ (a read-only/no-alias promise would let the compiler hoist the
consumer's load above the producer's store, or use the non-coherent load path on data written in the
same kernel). Correctness is not left to this analysis alone: scripts/xkernel_bench.py checks every
buffer of the optimized program bitwise against the original launch sequence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SIG = re.compile(r'extern\s+"C"\s+__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(\w+)\s*\(([^)]*)\)\s*\{', re.S)
_FORBID = re.compile(r"__syncthreads|__shared__|\batomic\w*\s*\(|__shfl|__ballot|__any_sync|__all_sync|\basm\b|"
                     r"__syncwarp|cooperative_groups|\bthis_grid\b")


@dataclass
class Launch:
    src: str                       # translation unit holding exactly one extern "C" __global__ kernel
    grid: tuple
    block: tuple
    args: list                     # per parameter: buffer name (pointer params) or a scalar value


@dataclass
class KernelInfo:
    name: str
    params: list                   # [(name, decl, is_ptr)]
    body: str                      # text inside the outer braces
    prelude: str                   # everything before the kernel (macros, device functions)
    fusable: bool
    why: str = ""
    access: dict = field(default_factory=dict)   # param -> {"r": set(subscripts), "w": set(subscripts)}


def _match_brace(s: str, i: int) -> int:
    depth = 0
    while True:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
        if depth == 0:
            return i


def _subscripts(body: str, p: str):
    """(subscript text, is_write) for every p[...] in body; None if p is used any other way."""
    out = []
    for m in re.finditer(rf"(?<![.\w]){p}\b", body):          # not a member access like threadIdx.x
        j = m.end()
        while j < len(body) and body[j] in " \t":
            j += 1
        if j >= len(body) or body[j] != "[":
            return None
        depth, k = 0, j
        while True:
            depth += {"[": 1, "]": -1}.get(body[k], 0)
            k += 1
            if depth == 0:
                break
        sub = " ".join(body[j + 1:k - 1].split())
        rest = body[k:k + 3].lstrip()
        write = bool(re.match(r"(=(?!=)|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\+\+|--)", rest))
        before = body[max(0, m.start() - 2):m.start()]
        if "++" in before or "--" in before:
            write = True
        out.append((sub, write))
        if write and not rest.startswith("="):       # compound assignment also reads
            out.append((sub, False))
    return out


def analyze(src: str) -> KernelInfo:
    m = _SIG.search(src)
    if not m or len(_SIG.findall(src)) != 1:
        raise ValueError("need exactly one extern \"C\" __global__ kernel")
    end = _match_brace(src, m.end() - 1)
    body = src[m.end():end - 1]
    params = []
    for p in filter(None, (x.strip() for x in m.group(2).split(","))):
        pname = re.findall(r"\w+", p)[-1]
        params.append((pname, p, "*" in p))
    info = KernelInfo(m.group(1), params, body, src[:m.start()], True)
    bad = _FORBID.search(body)
    if bad:
        info.fusable, info.why = False, f"uses {bad.group(0).strip()}"
    for pname, _, is_ptr in params:
        if not is_ptr:
            continue
        subs = _subscripts(body, pname)
        if subs is None:
            info.fusable, info.why = False, f"buffer {pname} used other than p[...]"
            continue
        info.access[pname] = {"r": {s for s, w in subs if not w}, "w": {s for s, w in subs if w}}
    return info


def _index_defs(body: str, names: set) -> dict:
    """Declaration text of each local variable among names (e.g. 'int i = blockIdx.x * ...')."""
    out = {}
    for v in names:
        ds = re.findall(rf"(?:\bint|\bunsigned|\blong|\bsize_t|\bconst int)\s+[^;]*?\b{v}\s*=\s*([^;,]+)", body)
        out[v] = tuple(" ".join(d.split()) for d in ds)
    return out


def _buffers(launch: Launch, info: KernelInfo) -> dict:
    """buffer name -> {"r": subs, "w": subs} in this launch (params renamed to the buffer they get)."""
    out = {}
    for (pname, _, is_ptr), a in zip(info.params, launch.args):
        if is_ptr:
            acc = info.access.get(pname, {"r": {"?"}, "w": {"?"}})
            d = out.setdefault(a, {"r": set(), "w": set(), "params": []})
            d["r"] |= acc["r"]
            d["w"] |= acc["w"]
            d["params"].append(pname)
    return out


def _scalar_env(launch: Launch, info: KernelInfo) -> dict:
    return {p: a for (p, _, is_ptr), a in zip(info.params, launch.args) if not is_ptr}


def vertical_ok(a: Launch, ia: KernelInfo, b: Launch, ib: KernelInfo) -> tuple[bool, str]:
    if not (ia.fusable and ib.fusable):
        return False, ia.why or ib.why
    if tuple(a.grid) != tuple(b.grid) or tuple(a.block) != tuple(b.block):
        return False, "different grid or block"
    ba, bb = _buffers(a, ia), _buffers(b, ib)
    shared = [n for n in ba if n in bb and (ba[n]["w"] or bb[n]["w"])]
    if not shared:
        return False, "no dependence (horizontal candidate)"
    for n in shared:
        subs = ba[n]["r"] | ba[n]["w"] | bb[n]["r"] | bb[n]["w"]
        if len(subs) != 1:
            return False, f"buffer {n} accessed through {len(subs)} different subscripts"
        sub = next(iter(subs))
        names = set(re.findall(r"[A-Za-z_]\w*", sub))
        da, db = _index_defs(ia.body, names), _index_defs(ib.body, names)
        sa, sb = _scalar_env(a, ia), _scalar_env(b, ib)
        for v in names:
            if v in sa or v in sb:                          # a scalar parameter: same value in both launches
                if sa.get(v) != sb.get(v):
                    return False, f"scalar {v} differs"
            elif v in ("threadIdx", "blockIdx", "blockDim", "gridDim", "x", "y", "z") or v.isupper():
                continue                                    # builtins and macros: identical by construction
            elif len(da[v]) != 1 or da[v] != db[v]:
                return False, f"index {v} not defined identically ({da[v]} vs {db[v]})"
            elif re.search(rf"\b{v}\s*(\+\+|--|[+\-*/]?=(?!=))", ia.body.split(da[v][0], 1)[-1]) or \
                    re.search(rf"\b{v}\s*(\+\+|--|[+\-*/]?=(?!=))", ib.body.split(db[v][0], 1)[-1]):
                return False, f"index {v} reassigned"
    return True, f"thread-local dependence on {shared}"


def horizontal_ok(a: Launch, ia: KernelInfo, b: Launch, ib: KernelInfo) -> tuple[bool, str]:
    if not (ia.fusable and ib.fusable):
        return False, ia.why or ib.why
    if tuple(a.block) != tuple(b.block):
        return False, "different block shape"
    ba, bb = _buffers(a, ia), _buffers(b, ib)
    for n in set(ba) & set(bb):
        if ba[n]["w"] or bb[n]["w"]:
            return False, f"buffer {n} written by one, used by the other"
    return True, "independent"


# ---- code generation --------------------------------------------------------------------------------

def _device_fn(info: KernelInfo, fname: str, shared_bufs: set, launch: Launch, remap: bool) -> str:
    ps = []
    for (pname, decl, is_ptr), a in zip(info.params, launch.args):
        if is_ptr and a in shared_bufs:
            decl = re.sub(r"\b(const|__restrict__|restrict)\b", " ", decl)
            decl = " ".join(decl.split())
        ps.append(decl)
    if remap:
        ps += ["uint3 blockIdx", "dim3 gridDim"]                # shadow the builtins inside this body
    return f"__device__ __forceinline__ void {fname}({', '.join(ps)}) {{{info.body}}}\n"


def fuse(group: list[tuple[Launch, KernelInfo]], kind: str, name: str = "fused") -> Launch:
    """One launch running every member: kind 'v' = bodies back to back per thread, 'h' = grid concat."""
    counts = {}
    for l, _ in group:
        for a in l.args:
            if isinstance(a, str):
                counts[a] = counts.get(a, 0) + 1
    shared = {n for n, c in counts.items() if c > 1}
    # every buffer used by >1 member is shared; also strip const if any member writes it
    preludes, seen = [], set()
    for _, inf in group:
        for line in inf.prelude.splitlines():
            if line not in seen:
                seen.add(line)
                preludes.append(line)
    bufs, scalars, fns, calls = [], [], [], []
    ptr_decl, scal_decl = {}, {}
    for i, (l, inf) in enumerate(group):
        fns.append(_device_fn(inf, f"_m{i}", shared, l, kind == "h"))
        argv = []
        for (pname, decl, is_ptr), a in zip(inf.params, l.args):
            if is_ptr:
                if a not in ptr_decl:
                    base = re.sub(r"\b(const|__restrict__|restrict)\b", " ", decl)
                    typ = " ".join(base.split()).rsplit(pname, 1)[0].strip()
                    ptr_decl[a] = f"{typ} b_{a}"
                    bufs.append(a)
                argv.append(f"b_{a}")
            else:
                key = f"s{i}_{pname}"
                scal_decl[key] = " ".join(decl.split()).rsplit(pname, 1)[0].strip() + f" {key}"
                scalars.append((key, a))
                argv.append(key)
        calls.append(argv)
    params = [ptr_decl[b] for b in bufs] + [scal_decl[k] for k, _ in scalars]
    if kind == "v":
        body = "\n".join(f"    _m{i}({', '.join(c)});" for i, c in enumerate(calls))
        grid = group[0][0].grid
    else:
        # concatenate along a linear block index; each member keeps its own (x, y, z) grid shape
        tot, lines = 0, ["    unsigned _lin = blockIdx.x;"]
        for i, (l, _) in enumerate(group):
            g = tuple(l.grid) + (1,) * (3 - len(l.grid))
            n = g[0] * g[1] * g[2]
            lines.append(f"    if (_lin < {tot + n}u) {{ unsigned _r = _lin - {tot}u;"
                         f" uint3 _b = make_uint3(_r % {g[0]}u, (_r / {g[0]}u) % {g[1]}u, _r / ({g[0]}u * {g[1]}u));"
                         f" _m{i}({', '.join(calls[i] + ['_b', f'dim3({g[0]}, {g[1]}, {g[2]})'])}); return; }}")
            tot += n
        body = "\n".join(lines)
        grid = (tot,)
    src = "\n".join(preludes) + "\n\n" + "\n".join(fns) + \
        f'\nextern "C" __global__ void {name}({", ".join(params)}) {{\n{body}\n}}\n'
    return Launch(src, tuple(grid), tuple(group[0][0].block), bufs + [v for _, v in scalars])


def plan(prog: list[Launch], vertical=True, horizontal=True) -> tuple[list[Launch], list[str]]:
    """Greedy left-to-right grouping. Returns (optimized launches, log)."""
    infos = [analyze(l.src) for l in prog]
    out, log, i = [], [], 0
    while i < len(prog):
        grp, kind = [(prog[i], infos[i])], None
        j = i + 1
        while j < len(prog):
            cand = (prog[j], infos[j])
            if vertical and kind in (None, "v"):
                ok, why = vertical_ok(*grp[-1], *cand)
                if ok and all(vertical_ok(*g, *cand)[0] or horizontal_ok(*g, *cand)[0] for g in grp[:-1]):
                    grp.append(cand)
                    kind = "v"
                    log.append(f"vertical {infos[i].name}..{cand[1].name}: {why}")
                    j += 1
                    continue
            if horizontal and kind in (None, "h"):
                ok = all(horizontal_ok(*g, *cand)[0] for g in grp)
                if ok:
                    grp.append(cand)
                    kind = "h"
                    log.append(f"horizontal {infos[i].name}..{cand[1].name}: independent")
                    j += 1
                    continue
            log.append(f"boundary {grp[-1][1].name} | {cand[1].name}: v: {vertical_ok(*grp[-1], *cand)[1]}; "
                       f"h: {horizontal_ok(*grp[-1], *cand)[1]}")
            break
        out.append(grp[0][0] if len(grp) == 1 else fuse(grp, kind, f"fused{len(out)}"))
        i = j
    return out, log
