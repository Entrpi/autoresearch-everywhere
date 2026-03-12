from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LabCapabilities:
    supports_workspace_init: bool
    supports_fixed_bench: bool
    supports_profile: bool
    supports_extract: bool
    supports_orchestrate: bool
    supports_verify: bool
    supports_capture: bool
    supports_trace_profile: bool
    supports_auto_trace_review: bool
    supports_deep_trace_profile: bool = False


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


@dataclass(frozen=True)
class LabProfileCandidate:
    target: str
    rank: int
    priority_score: float
    category: str
    status: str
    rationale: str
    details: dict[str, Any]


@dataclass(frozen=True)
class LabProfileResult:
    engine: str
    backend_family: str
    preset: str
    status: str
    wall_seconds: float
    candidates: tuple[LabProfileCandidate, ...]
    details: dict[str, Any]


@dataclass(frozen=True)
class LabExtractResult:
    engine: str
    target: str
    workspace: str
    status: str
    wall_seconds: float
    details: dict[str, Any]


@dataclass(frozen=True)
class LabEvidenceResult:
    engine: str
    backend_family: str
    target: str
    preset: str | None
    status: str
    details: dict[str, Any]


@dataclass(frozen=True)
class LabOrchestrationPlan:
    engine: str
    target: str
    workspace: str
    status: str
    commands: tuple[str, ...]
    details: dict[str, Any]


@dataclass(frozen=True)
class LabPromotionCheck:
    engine: str
    backend_family: str
    target: str
    preset: str | None
    workspace: str | None
    status: str
    commands: tuple[str, ...]
    details: dict[str, Any]


@dataclass(frozen=True)
class LabTraceResult:
    engine: str
    backend_family: str
    target: str
    workspace: str
    trace_path: str
    metadata_path: str | None
    status: str
    wall_seconds: float
    details: dict[str, Any]


@dataclass(frozen=True)
class LabTraceProfileCandidate:
    target: str
    rank: int
    priority_score: float
    time_share_pct: float | None
    category: str
    status: str
    rationale: str
    details: dict[str, Any]


@dataclass(frozen=True)
class LabTraceProfileResult:
    engine: str
    backend_family: str
    preset: str
    status: str
    trace_path: str | None
    wall_seconds: float
    dominant_issue: str | None
    candidates: tuple[LabTraceProfileCandidate, ...]
    details: dict[str, Any]


@dataclass(frozen=True)
class LabAutoTraceReviewResult:
    engine: str
    backend_family: str
    preset: str
    status: str
    dominant_issue: str | None
    confidence: float | None
    details: dict[str, Any]


@dataclass(frozen=True)
class LabDeepProfileResult:
    engine: str
    backend_family: str
    preset: str
    target: str
    status: str
    wall_seconds: float
    diagnosis: str | None
    confidence: float | None
    details: dict[str, Any]


@dataclass(frozen=True)
class LabIntegrationABResult:
    engine: str
    backend_family: str
    target: str
    preset: str
    workspace: str
    status: str
    wall_seconds: float
    details: dict[str, Any]


@dataclass(frozen=True)
class LabIntegrationSuiteResult:
    engine: str
    backend_family: str
    target: str
    presets: tuple[str, ...]
    workspace: str
    status: str
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

    def verify_workspace(self, *, workspace: Path, quick: bool = False) -> LabBenchResult:
        ...

    def profile_targets(self, *, preset: str, top_k: int = 10) -> LabProfileResult:
        ...

    def extract_from_profile(self, *, profile_path: Path, workspace: Path, rank: int = 1) -> LabExtractResult:
        ...

    def orchestrate_from_profile(
        self,
        *,
        profile_path: Path,
        workspace_root: Path,
        rank: int = 1,
        trace_metadata_path: Path | None = None,
    ) -> LabOrchestrationPlan:
        ...

    def capture_workspace(self, *, workspace: Path, output: Path, quick: bool = False) -> LabTraceResult:
        ...

    def trace_profile(self, *, metadata_path: Path) -> LabTraceProfileResult:
        ...

    def auto_review_trace(self, *, trace_profile_path: Path) -> LabAutoTraceReviewResult:
        ...

    def summarize_evidence(self, *, target: str, preset: str | None = None) -> LabEvidenceResult:
        ...

    def promotion_check(self, *, target: str, preset: str | None = None, workspace: Path | None = None) -> LabPromotionCheck:
        ...

    def run_integration_ab(
        self,
        *,
        workspace: Path,
        time_budget: float,
        preset: str | None = None,
        repeats: int = 2,
        benchmark_skip_eval: bool = True,
        no_checkpoint: bool = True,
    ) -> LabIntegrationABResult:
        ...

    def run_integration_suite(
        self,
        *,
        workspace: Path,
        time_budget: float,
        preset: str | None = None,
        presets: tuple[str, ...] | None = None,
        repeats: int = 2,
        benchmark_skip_eval: bool = True,
        no_checkpoint: bool = True,
    ) -> LabIntegrationSuiteResult:
        ...


def available_labs() -> tuple[str, ...]:
    return ("mlx", "cuda")


def get_lab(name: str) -> KernelLab:
    if name == "mlx":
        from autoresearch_mlx.lab_workspace import MLXKernelLab

        return MLXKernelLab()
    if name == "cuda":
        from autoresearch_cuda.lab_trace import CudaKernelLab

        return CudaKernelLab()
    raise ValueError(f"Unknown lab engine: {name}")
