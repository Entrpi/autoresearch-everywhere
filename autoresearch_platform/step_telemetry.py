from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


PHASE_LOADER = "loader"
PHASE_GRAD = "grad"
PHASE_ACCUMULATE = "accumulate"
PHASE_OPTIMIZER = "optimizer"
PHASE_FORWARD_BACKWARD = "forward_backward"
OBS_INPUT_PIPELINE = "input_pipeline"


def percent(part: float, whole: float) -> float:
    if whole <= 0.0:
        return 0.0
    return 100.0 * part / whole


def estimate_step_tflops(num_flops_per_token: int, total_batch_size: int, step_seconds: float) -> float:
    if step_seconds <= 0.0:
        return 0.0
    return (num_flops_per_token * total_batch_size) / step_seconds / 1e12


@dataclass
class StepTiming:
    total_seconds: float = 0.0
    phase_seconds: dict[str, float] = field(default_factory=dict)
    observed_seconds: dict[str, float] = field(default_factory=dict)

    def add_phase_seconds(self, phase: str, seconds: float) -> None:
        if seconds <= 0.0:
            return
        self.phase_seconds[phase] = self.phase_seconds.get(phase, 0.0) + seconds

    def phase_seconds_for(self, phase: str) -> float:
        return float(self.phase_seconds.get(phase, 0.0))

    def add_observed_seconds(self, name: str, seconds: float) -> None:
        if seconds <= 0.0:
            return
        self.observed_seconds[name] = self.observed_seconds.get(name, 0.0) + seconds

    def observed_seconds_for(self, name: str) -> float:
        return float(self.observed_seconds.get(name, 0.0))

    def sum_phase_seconds(self, phases: tuple[str, ...] | list[str]) -> float:
        return sum(self.phase_seconds_for(phase) for phase in phases)

    @property
    def accounted_seconds(self) -> float:
        return sum(self.phase_seconds.values())

    @property
    def other_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.accounted_seconds)


@dataclass
class StepTelemetry:
    total_steps: int = 0
    total_step_seconds: float = 0.0
    total_phase_seconds: dict[str, float] = field(default_factory=dict)
    total_observed_seconds: dict[str, float] = field(default_factory=dict)
    steady_steps: int = 0
    steady_step_seconds: float = 0.0
    steady_phase_seconds: dict[str, float] = field(default_factory=dict)
    steady_observed_seconds: dict[str, float] = field(default_factory=dict)

    def record_step(self, timing: StepTiming, *, include_in_steady: bool) -> None:
        self.total_steps += 1
        self.total_step_seconds += timing.total_seconds
        for phase, seconds in timing.phase_seconds.items():
            self.total_phase_seconds[phase] = self.total_phase_seconds.get(phase, 0.0) + seconds
        for name, seconds in timing.observed_seconds.items():
            self.total_observed_seconds[name] = self.total_observed_seconds.get(name, 0.0) + seconds
        if include_in_steady:
            self.steady_steps += 1
            self.steady_step_seconds += timing.total_seconds
            for phase, seconds in timing.phase_seconds.items():
                self.steady_phase_seconds[phase] = self.steady_phase_seconds.get(phase, 0.0) + seconds
            for name, seconds in timing.observed_seconds.items():
                self.steady_observed_seconds[name] = self.steady_observed_seconds.get(name, 0.0) + seconds

    def to_dict(self) -> dict[str, object]:
        return {
            "total_steps": self.total_steps,
            "total_step_seconds": self.total_step_seconds,
            "total_phase_seconds": dict(self.total_phase_seconds),
            "total_observed_seconds": dict(self.total_observed_seconds),
            "steady_steps": self.steady_steps,
            "steady_step_seconds": self.steady_step_seconds,
            "steady_phase_seconds": dict(self.steady_phase_seconds),
            "steady_observed_seconds": dict(self.steady_observed_seconds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "StepTelemetry":
        if not payload:
            return cls()
        if (
            "total_phase_seconds" in payload
            or "steady_phase_seconds" in payload
            or "total_observed_seconds" in payload
            or "steady_observed_seconds" in payload
        ):
            return cls(
                total_steps=int(payload.get("total_steps", 0) or 0),
                total_step_seconds=float(payload.get("total_step_seconds", 0.0) or 0.0),
                total_phase_seconds=_coerce_phase_seconds(payload.get("total_phase_seconds")),
                total_observed_seconds=_coerce_phase_seconds(payload.get("total_observed_seconds")),
                steady_steps=int(payload.get("steady_steps", 0) or 0),
                steady_step_seconds=float(payload.get("steady_step_seconds", 0.0) or 0.0),
                steady_phase_seconds=_coerce_phase_seconds(payload.get("steady_phase_seconds")),
                steady_observed_seconds=_coerce_phase_seconds(payload.get("steady_observed_seconds")),
            )
        return cls(
            total_steps=int(payload.get("total_steps", 0) or 0),
            total_step_seconds=float(payload.get("total_step_seconds", 0.0) or 0.0),
            total_phase_seconds=_legacy_phase_seconds(payload, prefix="total"),
            steady_steps=int(payload.get("steady_steps", 0) or 0),
            steady_step_seconds=float(payload.get("steady_step_seconds", 0.0) or 0.0),
            steady_phase_seconds=_legacy_phase_seconds(payload, prefix="steady"),
        )


@dataclass(frozen=True)
class StepTelemetrySummary:
    window_label: str
    window_steps: int
    train_tflops: float
    compute_share_percent: float
    other_step_percent: float
    phase_percents: dict[str, float]
    observed_percents: dict[str, float]
    peak_flop_utilization_percent: float | None = None

    def phase_percent(self, phase: str) -> float:
        return float(self.phase_percents.get(phase, 0.0))

    def observed_percent(self, name: str) -> float:
        return float(self.observed_percents.get(name, 0.0))


def summarize_step_telemetry(
    step_telemetry: StepTelemetry,
    *,
    num_flops_per_token: int,
    total_batch_size: int,
    compute_phases: tuple[str, ...],
    peak_flop_utilization_percent: float | None = None,
) -> StepTelemetrySummary:
    if step_telemetry.steady_steps > 0:
        label = "steady-state"
        steps = step_telemetry.steady_steps
        step_seconds = step_telemetry.steady_step_seconds
        phase_seconds = step_telemetry.steady_phase_seconds
        observed_seconds = step_telemetry.steady_observed_seconds
    else:
        label = "all-steps"
        steps = step_telemetry.total_steps
        step_seconds = step_telemetry.total_step_seconds
        phase_seconds = step_telemetry.total_phase_seconds
        observed_seconds = step_telemetry.total_observed_seconds
    phase_percents = {
        phase: percent(seconds, step_seconds)
        for phase, seconds in phase_seconds.items()
    }
    observed_percents = {
        name: percent(seconds, step_seconds)
        for name, seconds in observed_seconds.items()
    }
    compute_seconds = sum(float(phase_seconds.get(phase, 0.0)) for phase in compute_phases)
    other_seconds = max(0.0, step_seconds - sum(phase_seconds.values()))
    return StepTelemetrySummary(
        window_label=label,
        window_steps=steps,
        train_tflops=estimate_step_tflops(
            num_flops_per_token,
            total_batch_size,
            step_seconds / max(steps, 1),
        ),
        compute_share_percent=percent(compute_seconds, step_seconds),
        other_step_percent=percent(other_seconds, step_seconds),
        phase_percents=phase_percents,
        observed_percents=observed_percents,
        peak_flop_utilization_percent=peak_flop_utilization_percent,
    )


def _coerce_phase_seconds(payload: object) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(phase): float(seconds or 0.0)
        for phase, seconds in payload.items()
    }


def _legacy_phase_seconds(payload: Mapping[str, object], *, prefix: str) -> dict[str, float]:
    mapping = {
        PHASE_LOADER: f"{prefix}_loader_seconds",
        PHASE_GRAD: f"{prefix}_grad_seconds",
        PHASE_ACCUMULATE: f"{prefix}_accumulate_seconds",
        PHASE_OPTIMIZER: f"{prefix}_optimizer_seconds",
    }
    return {
        phase: float(payload.get(key, 0.0) or 0.0)
        for phase, key in mapping.items()
        if float(payload.get(key, 0.0) or 0.0) > 0.0
    }
