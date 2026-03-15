from __future__ import annotations

from pathlib import Path

from autoresearch_platform.checkpoint_policy import (
    AUTO_CHECKPOINT_INTERVAL_SEC,
    AUTO_CHECKPOINT_INTERVAL_TOKENS,
    AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC,
    AUTO_CHECKPOINT_MIN_TOKEN_BUDGET,
    CHECKPOINT_SAVE_MODE_ASYNC,
    CHECKPOINT_SAVE_MODE_SYNC,
    AutoCheckpointDecision,
    AutoCheckpointPlan,
    CheckpointCalibration,
    CheckpointInterval,
    auto_checkpoint_due,
    checkpoint_interval_due,
    choose_auto_checkpoint_decision as choose_shared_auto_checkpoint_decision,
    choose_auto_checkpoint_plan as choose_shared_auto_checkpoint_plan,
    default_time_budget_checkpoint_interval,
    default_token_budget_checkpoint_interval,
    format_interval_label,
    format_interval_spec,
    format_token_count,
    parse_checkpoint_interval_spec,
)


DEFAULT_CHECKPOINT_CALIBRATIONS: tuple[CheckpointCalibration, ...] = (
    CheckpointCalibration(
        key="cuda_exact_sync_gb10_balanced_band",
        label="Exact sync checkpointing (GB10 balanced-band calibrated)",
        checkpoint_mode="exact",
        max_params_m=30.0,
        checkpoint_cost_sec=0.5215833333333333,
        resume_ready_penalty_sec=0.0,
        source=(
            "Measured on GB10 under FA4 with repeated 60s matched m5-balanced "
            "runs at seq=1024, db=32, tb=32768 and 5s explicit checkpoint cadence."
        ),
        checkpoint_save_mode=CHECKPOINT_SAVE_MODE_SYNC,
        notes=(
            "Uses blocking exact-save cost for presets up to roughly the 26.3M-parameter "
            "balanced band. Resume-ready latency has not been reprofiled on CUDA yet."
        ),
    ),
    CheckpointCalibration(
        key="cuda_exact_async_gb10_balanced_band",
        label="Exact async checkpointing (GB10 balanced-band calibrated)",
        checkpoint_mode="exact",
        max_params_m=30.0,
        checkpoint_cost_sec=0.29700000000000004,
        resume_ready_penalty_sec=0.0,
        source=(
            "Measured on GB10 under FA4 with repeated 60s matched m5-balanced "
            "runs at seq=1024, db=32, tb=32768 and 5s explicit checkpoint cadence."
        ),
        checkpoint_save_mode=CHECKPOINT_SAVE_MODE_ASYNC,
        notes=(
            "Uses blocking snapshot-capture cost for interval choice and leaves the larger "
            "background write share out of the save-only overhead cap."
        ),
    ),
    CheckpointCalibration(
        key="cuda_exact_sync_gb10_xlarge_band",
        label="Exact sync checkpointing (GB10 xlarge-band calibrated)",
        checkpoint_mode="exact",
        max_params_m=float("inf"),
        checkpoint_cost_sec=0.3804166666666667,
        resume_ready_penalty_sec=0.0,
        source=(
            "Measured on GB10 under FA4 with repeated 60s matched m5-xlarge "
            "runs at seq=2048, db=16, tb=32768 and 5s explicit checkpoint cadence."
        ),
        checkpoint_save_mode=CHECKPOINT_SAVE_MODE_SYNC,
        notes=(
            "Uses blocking exact-save cost for the 50.3M-parameter xlarge band. "
            "Resume-ready latency has not been reprofiled on CUDA yet."
        ),
    ),
    CheckpointCalibration(
        key="cuda_exact_async_gb10_xlarge_band",
        label="Exact async checkpointing (GB10 xlarge-band calibrated)",
        checkpoint_mode="exact",
        max_params_m=float("inf"),
        checkpoint_cost_sec=0.6935833333333333,
        resume_ready_penalty_sec=0.0,
        source=(
            "Measured on GB10 under FA4 with repeated 60s matched m5-xlarge "
            "runs at seq=2048, db=16, tb=32768 and 5s explicit checkpoint cadence."
        ),
        checkpoint_save_mode=CHECKPOINT_SAVE_MODE_ASYNC,
        notes=(
            "Uses blocking snapshot-capture cost for interval choice. On the xlarge band "
            "the exact async path is currently more blocking than exact sync, so the "
            "recommended interval is correspondingly longer."
        ),
    ),
)


def choose_auto_checkpoint_decision(
    num_params_m: float,
    checkpoint_mode: str,
    checkpoint_save_mode: str = CHECKPOINT_SAVE_MODE_SYNC,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> AutoCheckpointDecision | None:
    if not calibrations:
        return None
    return choose_shared_auto_checkpoint_decision(
        num_params_m,
        checkpoint_mode,
        checkpoint_save_mode,
        calibrations=calibrations,
    )


def choose_auto_checkpoint_plan(
    *,
    num_params_m: float,
    checkpoint_mode: str,
    checkpoint_save_mode: str = CHECKPOINT_SAVE_MODE_SYNC,
    time_budget: float | None,
    token_budget: int | None,
    calibrations: tuple[CheckpointCalibration, ...] = DEFAULT_CHECKPOINT_CALIBRATIONS,
) -> AutoCheckpointPlan | None:
    return choose_shared_auto_checkpoint_plan(
        num_params_m=num_params_m,
        checkpoint_mode=checkpoint_mode,
        checkpoint_save_mode=checkpoint_save_mode,
        time_budget=time_budget,
        token_budget=token_budget,
        calibrations=calibrations,
    )


def default_auto_checkpoint_path(
    preset: str,
    *,
    seq_len: int,
    depth: int,
    total_batch_size: int,
    window_pattern: str,
    checkpoint_save_mode: str,
) -> Path:
    slug = (
        f"{preset}-seq{seq_len}-d{depth}-tb{total_batch_size}-w{window_pattern.lower()}"
        f"-save{checkpoint_save_mode}"
    )
    return Path.home() / ".cache" / "autoresearch" / "checkpoints" / "auto" / slug
