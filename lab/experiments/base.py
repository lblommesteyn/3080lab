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

    def expected(self, v: Variant) -> dict[str, int]:
        """Opcode -> exact count required in the timed loop body."""
        n = self.expected_body_ops()
        return {self.target_opcode: n} if n is not None else {}

    def build_key(self, v: Variant):
        """Variants with different keys get separately transformed cubins."""
        return None

    def kernel_attrs(self, v: Variant) -> dict[str, int]:
        """CUfunction attributes to set after loading (e.g. carveout)."""
        return {}

    def finalize(self, results: dict[str, list[dict]]):
        """Cross-variant post-processing (e.g. compare outputs to a golden variant)."""

    def transform(self, cubin: bytes, v: Variant) -> bytes:
        """Post-ptxas cubin edit (control-bit patching etc.). Identity by default."""
        return cubin
    def prepare(self, dev, v: Variant) -> dict: ...
    def collect(self, dev, v: Variant, state: dict) -> dict: ...
    def release(self, dev, state: dict): ...
