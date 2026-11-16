"""Make the project venv's packages importable when a script runs under a bare `python` (pcslurm jobs do).

Lookup order: $LAB_VENV, <repo>/.venv, then the main checkout's .venv when <repo> is a git worktree
(its .git file points at <main>/.git/worktrees/<name>). No match is a no-op, so a script run with the
venv's own interpreter, or on another machine, is unaffected.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _candidates():
    if os.environ.get("LAB_VENV"):
        yield Path(os.environ["LAB_VENV"])
    yield ROOT / ".venv"
    g = ROOT / ".git"
    if g.is_file():
        line = g.read_text().strip()
        if line.startswith("gitdir:"):
            gd = Path(line[7:].strip())
            if gd.parent.name == "worktrees":
                yield gd.parent.parent.parent / ".venv"


def site_packages(venv):
    if os.name == "nt":
        return [venv / "Lib" / "site-packages"]
    return sorted((venv / "lib").glob("python3*/site-packages"))


for _v in _candidates():
    _sp = [p for p in site_packages(_v) if p.is_dir()]
    if _sp:
        for p in _sp:
            if str(p) not in sys.path:
                sys.path.append(str(p))
        break
