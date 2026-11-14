"""Run a script with environment variables set: python scripts/run_env.py K=V ... -- script.py args (pcslurm has no env)."""
import os, runpy, sys
i = sys.argv.index("--")
for kv in sys.argv[1:i]:
    k, v = kv.split("=", 1); os.environ[k] = v
sys.argv = sys.argv[i + 1:]
runpy.run_path(sys.argv[0], run_name="__main__")
