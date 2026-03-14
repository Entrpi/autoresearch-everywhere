from __future__ import annotations

from pathlib import Path

from autoresearch_platform.checkpoint_policy import (
    AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC,
    DEFAULT_HUMAN_INTERVAL_POLICY,
    AutoCheckpointDecision,
    CheckpointCalibration,
    CheckpointIntervalRecommendation,
    HumanIntervalPolicy,
    choose_auto_checkpoint_decision as choose_shared_auto_checkpoint_decision,
    recommend_interval_for_cost,
    select_checkpoint_calibration as select_shared_checkpoint_calibration,
)

from .constants import CHECKPOINT_DIR
from .checkpoints import CHECKPOINT_MODE_EXACT, CHECKPOINT_MODE_WEIGHTS_ONLY


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


def select_checkpoint_calibration(
    num_params_m: float,
    checkpoint_mode: str,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> CheckpointCalibration:
    return select_shared_checkpoint_calibration(
        num_params_m,
        checkpoint_mode,
        calibrations=calibrations,
    )


def choose_auto_checkpoint_decision(
    num_params_m: float,
    checkpoint_mode: str,
    policy: HumanIntervalPolicy = DEFAULT_HUMAN_INTERVAL_POLICY,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> AutoCheckpointDecision:
    return choose_shared_auto_checkpoint_decision(
        num_params_m,
        checkpoint_mode,
        policy=policy,
        calibrations=calibrations,
    )


def default_auto_checkpoint_path(
    preset: str,
    *,
    seq_len: int,
    depth: int,
    total_batch_size: int,
    window_pattern: str,
    time_budget_mode: str,
    checkpoint_mode: str,
    checkpoint_save_mode: str,
) -> Path:
    slug = (
        f"{preset}-seq{seq_len}-d{depth}-tb{total_batch_size}-w{window_pattern.lower()}"
        f"-budget{time_budget_mode}-ckpt{checkpoint_mode}-save{checkpoint_save_mode}"
    )
    return CHECKPOINT_DIR / "auto" / slug
