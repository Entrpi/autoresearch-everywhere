from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import CHECKPOINT_DIR


AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC = 300.0


@dataclass(frozen=True)
class CheckpointCalibration:
    key: str
    label: str
    max_params_m: float
    checkpoint_cost_sec: float
    source: str
    notes: str = ""


@dataclass(frozen=True)
class HumanIntervalPolicy:
    label: str
    anchor_interval_sec: float
    save_only_overhead_cap_fraction: float
    friendly_intervals_sec: tuple[float, ...]


@dataclass(frozen=True)
class CheckpointIntervalRecommendation:
    interval_sec: float
    interval_label: str
    save_only_overhead_fraction: float
    failed_shorter_interval_label: str | None


@dataclass(frozen=True)
class AutoCheckpointDecision:
    calibration: CheckpointCalibration
    recommendation: CheckpointIntervalRecommendation


DEFAULT_CHECKPOINT_CALIBRATIONS = (
    CheckpointCalibration(
        key="exact_full_state_m5_large",
        label="Exact full-state resume (m5-large calibrated)",
        max_params_m=30.0,
        checkpoint_cost_sec=0.07,
        source="Measured from 20s matched m5-large runs: +0.7s across 10 saves.",
        notes="Conservative calibration used for presets up to roughly the 26.3M-parameter m5-large shape.",
    ),
    CheckpointCalibration(
        key="exact_full_state_m5_xlarge",
        label="Exact full-state resume (m5-xlarge calibrated)",
        max_params_m=float("inf"),
        checkpoint_cost_sec=0.11,
        source="Measured from 60s matched m5-xlarge runs: +3.3s across 29 saves.",
        notes="Conservative calibration used for the 50.3M-parameter xlarge/upstream model shape.",
    ),
)


DEFAULT_HUMAN_INTERVAL_POLICY = HumanIntervalPolicy(
    label="Generalized human-friendly interval scan anchored at hourly <= 0.1% save-only overhead",
    anchor_interval_sec=3600.0,
    save_only_overhead_cap_fraction=0.001,
    friendly_intervals_sec=(
        60.0,
        120.0,
        300.0,
        600.0,
        900.0,
        1800.0,
        3600.0,
        7200.0,
        14400.0,
        28800.0,
        43200.0,
        86400.0,
    ),
)


def save_only_overhead_fraction(*, checkpoint_cost_sec: float, interval_sec: float) -> float:
    return checkpoint_cost_sec / interval_sec


def format_interval_label(interval_sec: float) -> str:
    if interval_sec < 3600.0:
        return f"{interval_sec / 60.0:.0f}m"
    if interval_sec < 86400.0:
        return f"{interval_sec / 3600.0:.0f}h"
    return f"{interval_sec / 86400.0:.0f}d"


def select_checkpoint_calibration(
    num_params_m: float,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> CheckpointCalibration:
    for calibration in calibrations:
        if num_params_m <= calibration.max_params_m:
            return calibration
    return calibrations[-1]


def recommend_interval_for_cost(
    checkpoint_cost_sec: float,
    policy: HumanIntervalPolicy = DEFAULT_HUMAN_INTERVAL_POLICY,
) -> CheckpointIntervalRecommendation:
    last_fail_label = None
    for interval_sec in policy.friendly_intervals_sec:
        overhead_fraction = save_only_overhead_fraction(
            checkpoint_cost_sec=checkpoint_cost_sec,
            interval_sec=interval_sec,
        )
        if overhead_fraction <= policy.save_only_overhead_cap_fraction:
            return CheckpointIntervalRecommendation(
                interval_sec=interval_sec,
                interval_label=format_interval_label(interval_sec),
                save_only_overhead_fraction=overhead_fraction,
                failed_shorter_interval_label=last_fail_label,
            )
        last_fail_label = format_interval_label(interval_sec)
    minimum_interval_sec = checkpoint_cost_sec / policy.save_only_overhead_cap_fraction
    return CheckpointIntervalRecommendation(
        interval_sec=minimum_interval_sec,
        interval_label=f">= {minimum_interval_sec / 60.0:.2f} min",
        save_only_overhead_fraction=policy.save_only_overhead_cap_fraction,
        failed_shorter_interval_label=last_fail_label,
    )


def choose_auto_checkpoint_decision(
    num_params_m: float,
    policy: HumanIntervalPolicy = DEFAULT_HUMAN_INTERVAL_POLICY,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> AutoCheckpointDecision:
    calibration = select_checkpoint_calibration(num_params_m, calibrations)
    recommendation = recommend_interval_for_cost(calibration.checkpoint_cost_sec, policy)
    return AutoCheckpointDecision(calibration=calibration, recommendation=recommendation)


def default_auto_checkpoint_path(
    preset: str,
    *,
    seq_len: int,
    depth: int,
    total_batch_size: int,
    window_pattern: str,
) -> Path:
    slug = f"{preset}-seq{seq_len}-d{depth}-tb{total_batch_size}-w{window_pattern.lower()}"
    return CHECKPOINT_DIR / "auto" / slug
