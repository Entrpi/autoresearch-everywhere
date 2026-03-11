from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LabCapabilities:
    supports_workspace_init: bool
    supports_fixed_bench: bool
    supports_profile_extract: bool


@dataclass(frozen=True)
class LabTarget:
    key: str
    description: str
    metric: str
    status: str
    notes: str | None = None


@dataclass(frozen=True)
class LabBenchResult:
    target: str
    status: str
    metric_name: str
    metric_value: float | None
    wall_seconds: float
    details: dict[str, Any]


class KernelLab(Protocol):
    """Shared boundary for backend-specific kernel labs."""

    name: str
    backend_family: str
    capabilities: LabCapabilities

    def target_catalog(self) -> dict[str, LabTarget]:
        ...

    def init_workspace(self, *, target: str, workspace: Path) -> Path:
        ...

    def bench_workspace(self, *, workspace: Path, quick: bool = False) -> LabBenchResult:
        ...


def available_labs() -> tuple[str, ...]:
    return ("mlx",)


def get_lab(name: str) -> KernelLab:
    if name == "mlx":
        from autoresearch_mlx.lab_workspace import MLXKernelLab

        return MLXKernelLab()
    raise ValueError(f"Unknown lab engine: {name}")

