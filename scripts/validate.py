"""Static validation of every experiment variant (compile + transform + SASS check). No GPU."""
import sys

from lab import sass, toolchain
from lab.experiments import registry

names = sys.argv[1:] or list(registry())
bad = 0
for n in names:
    e = registry()[n]
    for v in e.variants({}):
        b = toolchain.build(e.source(v), ptxas_flags=e.ptxas_flags)
        c = e.transform(b.cubin, v)
        body = sass.loop_body(sass.parse(toolchain.disassemble(c)))
        h = sass.opcode_histogram(body)
        miss = {k: (h.get(k, 0), x) for k, x in e.expected(v).items()
                if not x <= h.get(k, 0) <= x + (k == "IADD3")}
        stalls = sorted({i.stall for i in body if i.opcode == e.target_opcode})
        regs = toolchain.ptxas_resources(b.ptxas_log).get("registers")
        flag = "OK " if not miss else f"BAD {miss}"
        bad += bool(miss)
        print(f"{n:<20} {v.label:<12} {flag} stalls={stalls} regs={regs}")
print(f"{bad} bad variants")
sys.exit(1 if bad else 0)
