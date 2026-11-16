import subprocess, sys
sys.path.insert(0, "scripts")
import decode_paired as d
p = subprocess.run(d.py("scripts/qwen_batched_decode.py", "1,32"), capture_output=True, text=True, cwd=d.ROOT)
print("exit", p.returncode, hex(p.returncode & 0xFFFFFFFF))
print(p.stdout[-3000:]); print(p.stderr[-3000:])
