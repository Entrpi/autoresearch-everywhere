from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .lr_profile import LrMultipliers, LrProfile


DEFAULT_ENGINE_NAME = "mlx"


@dataclass(frozen=True)
class EngineCapabilities:
    supports_platform_bringup: bool
    supports_local_search: bool
    supports_eval_calibration: bool
    supports_checkpoint_mint: bool
    supports_runtime_eval_policy: bool
    mutable_axes: tuple[str, ...]


@dataclass(frozen=True)
class EnginePreset:
    key: str
    description: str
    seq_len: int
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int
    lr_profile: LrProfile | None = None
    canonical_eval_seq_len: int | None = None
    canonical_eval_tokens: int | None = None
    canonical_eval_batch_size: int | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwareFingerprint:
    engine: str
    backend_family: str
    hardware_key: str
    platform: str
    machine: str
    processor: str
    python_version: str
    os_version: str | None = None
    runtime_version: str | None = None
    driver_version: str | None = None
    accelerator_vendor: str | None = None
    accelerator_model: str | None = None
    accelerator_architecture: str | None = None
    accelerator_compute_capability: str | None = None
    accelerator_cores: int | None = None
    memory_bytes: int | None = None
    memory_gb: float | None = None
    attention_backend: str | None = None
    flash_attention_generation: int | None = None


@dataclass(frozen=True)
class ProbeResult:
    preset: str
    stage: str
    seq_len: int
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int
    grad_accum_steps: int | None
    status: str
    returncode: int
    wall_seconds: float
    stdout_path: str
    stderr_path: str
    curve_output_path: str | None = None
    curve_eval_points: int | None = None
    val_bpb: float | None = None
    proxy_val_bpb: float | None = None
    steady_state_tok_per_sec: float | None = None
    peak_vram_mb: float | None = None
    training_seconds: float | None = None
    total_seconds: float | None = None
    eval_percent: float | None = None
    optimizer_percent: float | None = None
    accum_percent: float | None = None
    loader_percent: float | None = None
    input_pipeline_percent: float | None = None
    grad_percent: float | None = None
    forward_backward_percent: float | None = None
    other_step_percent: float | None = None
    compute_share_percent: float | None = None
    train_tflops: float | None = None
    peak_flop_utilization_percent: float | None = None
    util_window_steps: int | None = None
    util_window: str | None = None
    control_overhead_percent: float | None = None
    canonical_rung: str | None = None
    canonical_seq_len: int | None = None
    canonical_tokens: int | None = None
    canonical_batch: int | None = None
    canonical_slices: int | None = None
    streaming_eval_points: int | None = None
    streaming_eval_cycles_completed: int | None = None
    streaming_eval_cycle_batches: int | None = None
    streaming_eval_total_seconds: float | None = None
    streaming_val_auc: float | None = None
    honest_val_bpb: float | None = None
    subref_one_sixth_eval_points: int | None = None
    subref_one_sixth_eval_cycles_completed: int | None = None
    honest_subref_one_sixth_bpb: float | None = None
    reference_streaming_eval_cycles_completed: int | None = None
    honest_reference_bpb: float | None = None
    eval_calibration_status: str | None = None
    eval_calibration_effective_confidence: str | None = None
    eval_calibration_freshness: str | None = None
    eval_calibration_limited_by: str | None = None
    error_tail: str | None = None


class TrainingEngine(Protocol):
    """Shared backend contract for the important training-stack features.

    Platform bring-up is the first major consumer, but the boundary is
    intentionally broader than calibration alone: new engines should plug into
    train probes, local search, checkpoint minting, eval calibration, and
    runtime capability reporting through this contract instead of growing
    separate orchestration trees.
    """

    name: str
    backend_family: str
    capabilities: EngineCapabilities
    reference_preset: str | None

    def preset_catalog(self) -> dict[str, EnginePreset]:
        ...

    def preset_order(self) -> tuple[str, ...]:
        ...

    def default_platform_presets(self) -> tuple[str, ...]:
        ...

    def detect_hardware_fingerprint(self) -> HardwareFingerprint:
        ...

    def run_train_probe(
        self,
        *,
        preset: str,
        time_budget: float,
        logs_dir: Path,
        stage: str,
        benchmark_skip_eval: bool,
        checkpoint_path: Path | None = None,
        seq_len: int | None = None,
        window_pattern: str | None = None,
        device_batch_size: int | None = None,
        total_batch_size: int | None = None,
        curve_eval_seconds: tuple[float, ...] | None = None,
        eval_seq_len: int | None = None,
        eval_tokens: int | None = None,
        eval_batch_size: int | None = None,
        no_checkpoint: bool = True,
        lr_multiplier: float | None = None,
        lr_multipliers: LrMultipliers | None = None,
        streaming_eval_interval_steps: int | None = None,
        streaming_eval_mode: str | None = None,
        streaming_eval_tokens: int | None = None,
        streaming_eval_seq_len: int | None = None,
        streaming_eval_batch_size: int | None = None,
        streaming_eval_history_output: Path | None = None,
        complete_streaming_eval_cycle: bool = False,
    ) -> ProbeResult:
        ...

    def default_local_seq_lens(self, preset: str, *, mode: str) -> list[int]:
        ...

    def default_local_window_patterns(self, preset: str, *, mode: str) -> list[str]:
        ...

    def local_batch_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        ...

    def batch_profile_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        ...

    def batch_profile_refinement_candidates(
        self,
        preset: str,
        *,
        seq_len: int,
        coarse_winner: tuple[int, int],
    ) -> list[tuple[int, int]]:
        ...

    def calibration_signatures(self) -> dict[str, str | None]:
        ...

    def run_eval_calibration(
        self,
        *,
        preset: str,
        checkpoint_dir: Path,
        hardware_key: str,
        rungs: list[str],
        budget_seconds: list[float],
        markdown_path: Path | None = None,
    ) -> dict:
        ...


def available_engines() -> tuple[str, ...]:
    return ("mlx", "cuda")


def get_engine(name: str) -> TrainingEngine:
    if name == "mlx":
        from .mlx_engine import MLXEngine

        return MLXEngine()
    if name == "cuda":
        from .cuda_engine import CUDAEngine

        return CUDAEngine()
    raise ValueError(f"Unknown engine: {name}")
