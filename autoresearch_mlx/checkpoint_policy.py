from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import CHECKPOINT_DIR
from .checkpoints import CHECKPOINT_MODE_EXACT, CHECKPOINT_MODE_WEIGHTS_ONLY


AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC = 300.0


@dataclass(frozen=True)
class CheckpointCalibration:
    key: str
    label: str
    checkpoint_mode: str
    max_params_m: float
    checkpoint_cost_sec: float
    resume_ready_penalty_sec: float
    source: str
    resume_ready_source: str = ""
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
        checkpoint_mode=CHECKPOINT_MODE_EXACT,
        max_params_m=30.0,
        checkpoint_cost_sec=0.07,
        resume_ready_penalty_sec=0.641,
        source="Measured from 20s matched m5-large runs: +0.7s across 10 saves.",
        resume_ready_source="Measured from repeated resume-ready trials: median 0.641s to first completed resumed optimizer step.",
        notes="Conservative calibration used for presets up to roughly the 26.3M-parameter m5-large shape.",
    ),
    CheckpointCalibration(
        key="weights_only_m5_large",
        label="Weights-only approximate resume (m5-large calibrated)",
        checkpoint_mode=CHECKPOINT_MODE_WEIGHTS_ONLY,
        max_params_m=30.0,
        checkpoint_cost_sec=0.02,
        resume_ready_penalty_sec=1.176,
        source="Measured from matched 20s m5-large runs: about +0.2s across 10 weights-only saves.",
        resume_ready_source="Measured from repeated weights-only resume-ready trials: median 1.176s to first completed resumed optimizer step.",
        notes="Approximate resume restores model weights only and restarts from a fresh optimizer and train-loader state.",
    ),
    CheckpointCalibration(
        key="exact_full_state_m5_xlarge",
        label="Exact full-state resume (m5-xlarge calibrated)",
        checkpoint_mode=CHECKPOINT_MODE_EXACT,
        max_params_m=float("inf"),
        checkpoint_cost_sec=0.11,
        resume_ready_penalty_sec=0.683,
        source="Measured from 60s matched m5-xlarge runs: +3.3s across 29 saves.",
        resume_ready_source="Measured from repeated resume-ready trials: median 0.683s to first completed resumed optimizer step.",
        notes="Conservative calibration used for the 50.3M-parameter xlarge/upstream model shape.",
    ),
    CheckpointCalibration(
        key="weights_only_m5_xlarge",
        label="Weights-only approximate resume (m5-xlarge calibrated)",
        checkpoint_mode=CHECKPOINT_MODE_WEIGHTS_ONLY,
        max_params_m=float("inf"),
        checkpoint_cost_sec=0.03,
        resume_ready_penalty_sec=1.236,
        source="Measured from matched 60s m5-xlarge runs: about +0.8s across 28 weights-only saves.",
        resume_ready_source="Measured from repeated weights-only resume-ready trials: median 1.236s to first completed resumed optimizer step.",
        notes="Approximate resume restores model weights only and restarts from a fresh optimizer and train-loader state.",
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
    checkpoint_mode: str,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> CheckpointCalibration:
    matching = tuple(
        calibration
        for calibration in calibrations
        if calibration.checkpoint_mode == checkpoint_mode
    )
    if not matching:
        raise ValueError(f"No checkpoint calibration registered for checkpoint_mode={checkpoint_mode!r}.")
    for calibration in matching:
        if num_params_m <= calibration.max_params_m:
            return calibration
    return matching[-1]


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
    checkpoint_mode: str,
    policy: HumanIntervalPolicy = DEFAULT_HUMAN_INTERVAL_POLICY,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> AutoCheckpointDecision:
    calibration = select_checkpoint_calibration(
        num_params_m,
        checkpoint_mode,
        calibrations,
    )
    recommendation = recommend_interval_for_cost(calibration.checkpoint_cost_sec, policy)
    return AutoCheckpointDecision(calibration=calibration, recommendation=recommendation)


def default_auto_checkpoint_path(
    preset: str,
    *,
    seq_len: int,
    depth: int,
    total_batch_size: int,
    window_pattern: str,
    checkpoint_mode: str,
    checkpoint_save_mode: str,
) -> Path:
    slug = (
        f"{preset}-seq{seq_len}-d{depth}-tb{total_batch_size}-w{window_pattern.lower()}"
        f"-ckpt{checkpoint_mode}-save{checkpoint_save_mode}"
    )
    return CHECKPOINT_DIR / "auto" / slug
