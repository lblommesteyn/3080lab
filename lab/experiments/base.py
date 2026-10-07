from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Variant:
    label: str
    params: dict = field(default_factory=dict)


@dataclass
class Experiment:
    """One microbenchmark family. Subclasses generate source per variant,
    allocate inputs, and turn device buffers into measurements."""
    name: str = ""
    description: str = ""
    target_opcode: str = ""
    kernel_name: str = "k"
    ptxas_flags: list[str] = field(default_factory=list)

    def source(self, v: Variant) -> str: ...
    def variants(self, opts: dict) -> list[Variant]: ...
    def expected_body_ops(self) -> int | None: return None
    def prepare(self, dev, v: Variant) -> dict: ...
    def collect(self, dev, v: Variant, state: dict) -> dict: ...
    def release(self, dev, state: dict): ...
