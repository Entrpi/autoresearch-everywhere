"""
Autoresearch pretraining script for MLX on Apple Silicon.
Use via the generic top-level entrypoint:
    uv run train.py
or directly as:
    python -m autoresearch_mlx.train
"""

import argparse
import gc
import os
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from autoresearch_mlx.constants import (
    CANONICAL_EVAL_SEQ_LEN,
    EVAL_SLICE_CAP,
    EVAL_TOKENS,
    CANONICAL_EVAL_STEP_TOKENS,
    CANONICAL_EVAL_TOKENS,
    MAX_SEQ_LEN,
    PROXY_EVAL_TOKENS,
    TIME_BUDGET,
)
from autoresearch_mlx.checkpoint_policy import (
    AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC,
    choose_auto_checkpoint_decision,
    default_auto_checkpoint_path,
)
from autoresearch_mlx.checkpoints import (
    CHECKPOINT_MODE_EXACT,
    CHECKPOINT_MODES,
    CHECKPOINT_SAVE_MODE_ASYNC,
    CHECKPOINT_SAVE_MODE_SYNC,
    CHECKPOINT_SAVE_MODES,
    AsyncCheckpointWriter,
    capture_async_checkpoint_snapshot,
    load_checkpoint_metadata,
    restore_checkpoint,
    save_checkpoint,
)
from autoresearch_mlx.calibration_signature import (
    current_eval_semantics_signature,
    current_runtime_shape_signature,
)
from autoresearch_mlx.data import Tokenizer, evaluate_bpb, make_dataloader
from autoresearch_mlx.eval_policy import (
    EVAL_POLICY_VERSION,
    choose_auto_eval_decision,
    detect_current_hardware_key,
    find_eval_calibration,
)
from autoresearch_mlx.eval_telemetry import EvalTelemetryRecord, append_eval_telemetry, now_iso
from autoresearch_mlx.model import GPT, GPTConfig
from autoresearch_mlx.optim import MuonAdamW


@dataclass(frozen=True)
class RunPreset:
    description: str
    seq_len: int
    eval_tokens: int
    canonical_eval_seq_len: int
    canonical_eval_tokens: int
    canonical_eval_batch_size: int
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int


@dataclass(frozen=True)
class RunConfig:
    preset: str
    time_budget: float
    time_budget_mode: str
    seq_len: int
    eval_tokens: int
    canonical_eval_seq_len: int
    canonical_eval_tokens: int
    canonical_eval_batch_size: int
    canonical_eval_rung: str | None
    canonical_eval_slices: int
    canonical_eval_reference_tokens: int | None
    eval_hardware_key: str
    eval_calibration_status: str
    eval_calibration_key: str | None
    eval_calibration_confidence: str | None
    eval_calibration_effective_confidence: str | None
    eval_calibration_freshness: str | None
    eval_calibration_repeat_count: int | None
    eval_calibration_measured_train_seconds: float | None
    eval_calibration_measured_on: str | None
    eval_calibration_eval_semantics_signature: str | None
    eval_calibration_runtime_shape_signature: str | None
    eval_current_eval_semantics_signature: str | None
    eval_current_runtime_shape_signature: str | None
    eval_calibration_telemetry_count: int | None
    eval_calibration_commit_count: int | None
    eval_calibration_day_count: int | None
    eval_calibration_observed_rungs: str | None
    eval_calibration_stable_rungs: str | None
    eval_calibration_last_seen_on: str | None
    eval_calibration_last_seen_age_days: int | None
    eval_calibration_limited_by: str | None
    eval_policy_version: int | None
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int
    seed: int
    smoke: bool
    benchmark_warmup_steps: int | None
    benchmark_skip_eval: bool
    prefer_prepacked_cache: bool
    no_checkpoint: bool
    checkpoint_mode: str
    checkpoint_save_mode: str
    checkpoint_path: str | None
    checkpoint_interval: float | None
    resume_from: str | None


@dataclass
class StepTiming:
    total_seconds: float = 0.0
    loader_seconds: float = 0.0
    grad_seconds: float = 0.0
    accumulate_seconds: float = 0.0
    optimizer_seconds: float = 0.0

    @property
    def compute_seconds(self) -> float:
        return self.grad_seconds + self.accumulate_seconds + self.optimizer_seconds

    @property
    def other_seconds(self) -> float:
        accounted = self.loader_seconds + self.compute_seconds
        return max(0.0, self.total_seconds - accounted)


@dataclass
class StepTelemetry:
    total_steps: int = 0
    total_step_seconds: float = 0.0
    total_loader_seconds: float = 0.0
    total_grad_seconds: float = 0.0
    total_accumulate_seconds: float = 0.0
    total_optimizer_seconds: float = 0.0
    steady_steps: int = 0
    steady_step_seconds: float = 0.0
    steady_loader_seconds: float = 0.0
    steady_grad_seconds: float = 0.0
    steady_accumulate_seconds: float = 0.0
    steady_optimizer_seconds: float = 0.0

    def record_step(self, timing: StepTiming, *, include_in_steady: bool) -> None:
        self.total_steps += 1
        self.total_step_seconds += timing.total_seconds
        self.total_loader_seconds += timing.loader_seconds
        self.total_grad_seconds += timing.grad_seconds
        self.total_accumulate_seconds += timing.accumulate_seconds
        self.total_optimizer_seconds += timing.optimizer_seconds
        if include_in_steady:
            self.steady_steps += 1
            self.steady_step_seconds += timing.total_seconds
            self.steady_loader_seconds += timing.loader_seconds
            self.steady_grad_seconds += timing.grad_seconds
            self.steady_accumulate_seconds += timing.accumulate_seconds
            self.steady_optimizer_seconds += timing.optimizer_seconds

    @classmethod
    def from_dict(cls, payload: dict | None) -> "StepTelemetry":
        if not payload:
            return cls()
        valid = {field: payload.get(field, 0) for field in cls.__dataclass_fields__}
        return cls(**valid)


TIME_BUDGET_MODE_TRAIN = "train"
TIME_BUDGET_MODE_WALL = "wall"
TIME_BUDGET_MODES = (TIME_BUDGET_MODE_TRAIN, TIME_BUDGET_MODE_WALL)


def default_canonical_eval_batch_size(seq_len: int) -> int:
    if seq_len <= 0:
        raise ValueError("canonical eval sequence length must be positive")
    return max(1, CANONICAL_EVAL_STEP_TOKENS // seq_len)


def current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def uses_default_preset_shape(config: RunConfig) -> bool:
    preset = PRESETS[config.preset]
    return (
        config.seq_len == preset.seq_len
        and config.depth == preset.depth
        and config.window_pattern == preset.window_pattern
        and config.device_batch_size == preset.device_batch_size
        and config.total_batch_size == preset.total_batch_size
    )


def resolve_eval_settings(
    config: RunConfig,
    *,
    explicit_canonical_overrides: bool,
) -> tuple[RunConfig, str | None]:
    if config.smoke:
        return replace(
            config,
            canonical_eval_rung="smoke",
            canonical_eval_slices=1,
            canonical_eval_reference_tokens=None,
            eval_calibration_status="smoke",
            eval_calibration_key=None,
            eval_calibration_confidence=None,
            eval_calibration_effective_confidence=None,
            eval_calibration_freshness=None,
            eval_calibration_repeat_count=None,
            eval_calibration_measured_train_seconds=None,
            eval_calibration_measured_on=None,
            eval_calibration_eval_semantics_signature=None,
            eval_calibration_runtime_shape_signature=None,
            eval_current_eval_semantics_signature=current_eval_semantics_signature(),
            eval_current_runtime_shape_signature=current_runtime_shape_signature(),
            eval_calibration_telemetry_count=None,
            eval_calibration_commit_count=None,
            eval_calibration_day_count=None,
            eval_calibration_observed_rungs=None,
            eval_calibration_stable_rungs=None,
            eval_calibration_last_seen_on=None,
            eval_calibration_last_seen_age_days=None,
            eval_calibration_limited_by=None,
            eval_policy_version=EVAL_POLICY_VERSION,
        ), None
    if explicit_canonical_overrides:
        return replace(
            config,
            canonical_eval_rung="manual",
            canonical_eval_slices=1,
            canonical_eval_reference_tokens=None,
            eval_calibration_status="manual",
            eval_calibration_key=None,
            eval_calibration_confidence=None,
            eval_calibration_effective_confidence=None,
            eval_calibration_freshness=None,
            eval_calibration_repeat_count=None,
            eval_calibration_measured_train_seconds=None,
            eval_calibration_measured_on=None,
            eval_calibration_eval_semantics_signature=None,
            eval_calibration_runtime_shape_signature=None,
            eval_current_eval_semantics_signature=current_eval_semantics_signature(),
            eval_current_runtime_shape_signature=current_runtime_shape_signature(),
            eval_calibration_telemetry_count=None,
            eval_calibration_commit_count=None,
            eval_calibration_day_count=None,
            eval_calibration_observed_rungs=None,
            eval_calibration_stable_rungs=None,
            eval_calibration_last_seen_on=None,
            eval_calibration_last_seen_age_days=None,
            eval_calibration_limited_by=None,
            eval_policy_version=EVAL_POLICY_VERSION,
        ), "manual canonical eval override"
    if not uses_default_preset_shape(config):
        return replace(
            config,
            canonical_eval_rung="default",
            canonical_eval_slices=EVAL_SLICE_CAP,
            canonical_eval_reference_tokens=EVAL_TOKENS,
            canonical_eval_batch_size=default_canonical_eval_batch_size(config.canonical_eval_seq_len),
            eval_calibration_status="shape-fallback",
            eval_calibration_key=None,
            eval_calibration_confidence=None,
            eval_calibration_effective_confidence=None,
            eval_calibration_freshness=None,
            eval_calibration_repeat_count=None,
            eval_calibration_measured_train_seconds=None,
            eval_calibration_measured_on=None,
            eval_calibration_eval_semantics_signature=None,
            eval_calibration_runtime_shape_signature=None,
            eval_current_eval_semantics_signature=current_eval_semantics_signature(),
            eval_current_runtime_shape_signature=current_runtime_shape_signature(),
            eval_calibration_telemetry_count=None,
            eval_calibration_commit_count=None,
            eval_calibration_day_count=None,
            eval_calibration_observed_rungs=None,
            eval_calibration_stable_rungs=None,
            eval_calibration_last_seen_on=None,
            eval_calibration_last_seen_age_days=None,
            eval_calibration_limited_by=None,
            eval_policy_version=EVAL_POLICY_VERSION,
        ), "preset shape mutated; keeping default canonical eval settings until calibrated"

    calibration = find_eval_calibration(
        config.preset,
        hardware_key=config.eval_hardware_key,
    )
    if calibration is None:
        return replace(
            config,
            canonical_eval_rung="default",
            canonical_eval_slices=EVAL_SLICE_CAP,
            canonical_eval_reference_tokens=EVAL_TOKENS,
            canonical_eval_batch_size=default_canonical_eval_batch_size(config.canonical_eval_seq_len),
            eval_calibration_status="hardware-unmatched",
            eval_calibration_key=None,
            eval_calibration_confidence=None,
            eval_calibration_effective_confidence=None,
            eval_calibration_freshness=None,
            eval_calibration_repeat_count=None,
            eval_calibration_measured_train_seconds=None,
            eval_calibration_measured_on=None,
            eval_calibration_eval_semantics_signature=None,
            eval_calibration_runtime_shape_signature=None,
            eval_current_eval_semantics_signature=current_eval_semantics_signature(),
            eval_current_runtime_shape_signature=current_runtime_shape_signature(),
            eval_calibration_telemetry_count=None,
            eval_calibration_commit_count=None,
            eval_calibration_day_count=None,
            eval_calibration_observed_rungs=None,
            eval_calibration_stable_rungs=None,
            eval_calibration_last_seen_on=None,
            eval_calibration_last_seen_age_days=None,
            eval_calibration_limited_by=None,
            eval_policy_version=EVAL_POLICY_VERSION,
        ), (
            f"no eval calibration row for preset={config.preset} on hardware={config.eval_hardware_key}; "
            "keeping default canonical eval settings"
        )
    if calibration.policy_version != EVAL_POLICY_VERSION:
        return replace(
            config,
            canonical_eval_rung="default",
            canonical_eval_slices=EVAL_SLICE_CAP,
            canonical_eval_reference_tokens=EVAL_TOKENS,
            canonical_eval_batch_size=default_canonical_eval_batch_size(config.canonical_eval_seq_len),
            eval_calibration_status="stale-policy",
            eval_calibration_key=calibration.key,
            eval_calibration_confidence=calibration.confidence,
            eval_calibration_effective_confidence=calibration.confidence,
            eval_calibration_freshness="unknown",
            eval_calibration_repeat_count=calibration.repeat_count,
            eval_calibration_measured_train_seconds=calibration.measured_train_seconds,
            eval_calibration_measured_on=calibration.measured_on,
            eval_calibration_eval_semantics_signature=calibration.eval_semantics_signature,
            eval_calibration_runtime_shape_signature=calibration.runtime_shape_signature,
            eval_current_eval_semantics_signature=current_eval_semantics_signature(),
            eval_current_runtime_shape_signature=current_runtime_shape_signature(),
            eval_calibration_telemetry_count=0,
            eval_calibration_commit_count=0,
            eval_calibration_day_count=0,
            eval_calibration_observed_rungs=None,
            eval_calibration_stable_rungs=None,
            eval_calibration_last_seen_on=None,
            eval_calibration_last_seen_age_days=None,
            eval_calibration_limited_by="policy-version",
            eval_policy_version=EVAL_POLICY_VERSION,
        ), (
            f"calibration row {calibration.key} is policy_version={calibration.policy_version}, "
            f"expected {EVAL_POLICY_VERSION}; keeping default canonical eval settings"
        )
    current_eval_signature = current_eval_semantics_signature()
    current_runtime_signature = current_runtime_shape_signature()
    if (
        calibration.eval_semantics_signature != current_eval_signature
        or calibration.runtime_shape_signature != current_runtime_signature
    ):
        limited_by = []
        if calibration.eval_semantics_signature != current_eval_signature:
            limited_by.append("eval-signature")
        if calibration.runtime_shape_signature != current_runtime_signature:
            limited_by.append("runtime-signature")
        limited_by_text = ",".join(limited_by)
        return replace(
            config,
            canonical_eval_rung="default",
            canonical_eval_slices=EVAL_SLICE_CAP,
            canonical_eval_reference_tokens=EVAL_TOKENS,
            canonical_eval_batch_size=default_canonical_eval_batch_size(config.canonical_eval_seq_len),
            eval_calibration_status="signature-mismatch",
            eval_calibration_key=calibration.key,
            eval_calibration_confidence=calibration.confidence,
            eval_calibration_effective_confidence=calibration.confidence,
            eval_calibration_freshness="unknown",
            eval_calibration_repeat_count=calibration.repeat_count,
            eval_calibration_measured_train_seconds=calibration.measured_train_seconds,
            eval_calibration_measured_on=calibration.measured_on,
            eval_calibration_eval_semantics_signature=calibration.eval_semantics_signature,
            eval_calibration_runtime_shape_signature=calibration.runtime_shape_signature,
            eval_current_eval_semantics_signature=current_eval_signature,
            eval_current_runtime_shape_signature=current_runtime_signature,
            eval_calibration_telemetry_count=0,
            eval_calibration_commit_count=0,
            eval_calibration_day_count=0,
            eval_calibration_observed_rungs=None,
            eval_calibration_stable_rungs=None,
            eval_calibration_last_seen_on=None,
            eval_calibration_last_seen_age_days=None,
            eval_calibration_limited_by=limited_by_text,
            eval_policy_version=EVAL_POLICY_VERSION,
        ), (
            f"calibration row {calibration.key} does not match the current code signatures "
            f"({limited_by_text}); keeping default canonical eval settings"
        )

    decision = choose_auto_eval_decision(
        config.preset,
        time_budget_sec=config.time_budget,
        hardware_key=config.eval_hardware_key,
    )
    if decision.freshness == "stale-age":
        return replace(
            config,
            canonical_eval_rung="default",
            canonical_eval_slices=EVAL_SLICE_CAP,
            canonical_eval_reference_tokens=EVAL_TOKENS,
            canonical_eval_batch_size=default_canonical_eval_batch_size(config.canonical_eval_seq_len),
            eval_calibration_status="stale-age",
            eval_calibration_key=decision.calibration.key,
            eval_calibration_confidence=decision.calibration.confidence,
            eval_calibration_effective_confidence=decision.effective_confidence,
            eval_calibration_freshness=decision.freshness,
            eval_calibration_repeat_count=decision.calibration.repeat_count,
            eval_calibration_measured_train_seconds=decision.calibration.measured_train_seconds,
            eval_calibration_measured_on=decision.calibration.measured_on,
            eval_calibration_eval_semantics_signature=decision.calibration.eval_semantics_signature,
            eval_calibration_runtime_shape_signature=decision.calibration.runtime_shape_signature,
            eval_current_eval_semantics_signature=current_eval_semantics_signature(),
            eval_current_runtime_shape_signature=current_runtime_shape_signature(),
            eval_calibration_telemetry_count=decision.telemetry_count,
            eval_calibration_commit_count=decision.telemetry_commit_count,
            eval_calibration_day_count=decision.telemetry_day_count,
            eval_calibration_observed_rungs=",".join(decision.observed_rungs) or None,
            eval_calibration_stable_rungs=",".join(decision.stable_rungs) or None,
            eval_calibration_last_seen_on=decision.last_seen_on,
            eval_calibration_last_seen_age_days=decision.last_seen_age_days,
            eval_calibration_limited_by=decision.limited_by,
            eval_policy_version=decision.calibration.policy_version,
        ), (
            f"calibration row {decision.calibration.key} is stale ({decision.last_seen_age_days}d old); "
            "keeping default canonical eval settings until refreshed"
        )
    rung = decision.recommendation.rung
    updated = replace(
        config,
        canonical_eval_seq_len=decision.calibration.seq_len,
        canonical_eval_tokens=rung.spec.eval_tokens,
        canonical_eval_batch_size=decision.calibration.batch_size,
        canonical_eval_rung=rung.spec.key,
        canonical_eval_slices=rung.spec.eval_slices,
        canonical_eval_reference_tokens=rung.spec.reference_eval_tokens,
        eval_calibration_status="calibrated" if decision.limited_by is None else "calibrated-limited",
        eval_calibration_key=decision.calibration.key,
        eval_calibration_confidence=decision.calibration.confidence,
        eval_calibration_effective_confidence=decision.effective_confidence,
        eval_calibration_freshness=decision.freshness,
        eval_calibration_repeat_count=decision.calibration.repeat_count,
        eval_calibration_measured_train_seconds=decision.calibration.measured_train_seconds,
        eval_calibration_measured_on=decision.calibration.measured_on,
        eval_calibration_eval_semantics_signature=decision.calibration.eval_semantics_signature,
        eval_calibration_runtime_shape_signature=decision.calibration.runtime_shape_signature,
        eval_current_eval_semantics_signature=current_eval_semantics_signature(),
        eval_current_runtime_shape_signature=current_runtime_shape_signature(),
        eval_calibration_telemetry_count=decision.telemetry_count,
        eval_calibration_commit_count=decision.telemetry_commit_count,
        eval_calibration_day_count=decision.telemetry_day_count,
        eval_calibration_observed_rungs=",".join(decision.observed_rungs) or None,
        eval_calibration_stable_rungs=",".join(decision.stable_rungs) or None,
        eval_calibration_last_seen_on=decision.last_seen_on,
        eval_calibration_last_seen_age_days=decision.last_seen_age_days,
        eval_calibration_limited_by=decision.limited_by,
        eval_policy_version=decision.calibration.policy_version,
    )
    selected_label = "auto-selected" if decision.limited_by is None else "auto-selected with safety cap"
    reason = (
        f"{selected_label} {rung.spec.key} eval rung from {decision.calibration.label}; "
        f"projected overhead {decision.recommendation.projected_overhead_fraction * 100.0:.2f}% "
        f"for time_budget={config.time_budget:.1f}s; effective_confidence={decision.effective_confidence}; "
        f"freshness={decision.freshness}; telemetry_count={decision.telemetry_count}; "
        f"commit_count={decision.telemetry_commit_count}; day_count={decision.telemetry_day_count}; "
        f"stable_rungs={','.join(decision.stable_rungs) or 'none'}"
    )
    if decision.limited_by is not None:
        reason += f"; limited_by={decision.limited_by}"
    return updated, reason


def verify_mlx_env() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(f"autoresearch_mlx.train requires macOS. Detected platform: {sys.platform}")
    if not mx.is_available(mx.gpu) or not mx.metal.is_available():
        raise RuntimeError("MLX GPU/Metal is not available. This script expects Apple Silicon with Metal.")
    print("Environment verified: macOS detected with MLX Metal GPU support available.")
    print()


def build_model_config(depth: int, vocab_size: int, *, sequence_len: int, window_pattern: str) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    if WARMDOWN_RATIO == 0:
        return FINAL_LR_FRAC
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float) -> float:
    return WEIGHT_DECAY * (1.0 - progress)


def make_grad_step_fn(model):
    def loss_fn(model, inputs, targets):
        return model(inputs, targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def grad_step(inputs, targets):
        return loss_and_grad(model, inputs, targets)

    return grad_step


def make_apply_grads_fn(model, optimizer):
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def apply_grads(grads):
        optimizer.update(model, grads)
        return optimizer.state["step"]

    return apply_grads


def percent(part: float, whole: float) -> float:
    if whole <= 0.0:
        return 0.0
    return 100.0 * part / whole


def median_abs_deviation(samples: list[float], center: float) -> float:
    if not samples:
        return 0.0
    return statistics.median([abs(sample - center) for sample in samples])


def detect_benchmark_warmup_steps(step_seconds: list[float]) -> int:
    if len(step_seconds) < BENCHMARK_AUTO_MIN_TOTAL_STEPS:
        return 0

    # Compare early-step windows to a trailing reference window and pick the
    # first prefix after which step times look statistically stable.
    reference_window = min(
        BENCHMARK_AUTO_MAX_REFERENCE_STEPS,
        max(BENCHMARK_AUTO_MIN_REFERENCE_STEPS, len(step_seconds) // 5),
    )
    stability_window = min(
        BENCHMARK_AUTO_MAX_STABILITY_STEPS,
        max(BENCHMARK_AUTO_MIN_STABILITY_STEPS, reference_window // 2),
    )
    reference = step_seconds[-reference_window:]
    reference_median = statistics.median(reference)
    reference_mad = median_abs_deviation(reference, reference_median)
    tolerance = max(
        BENCHMARK_AUTO_MIN_REL_TOL * reference_median,
        BENCHMARK_AUTO_MAD_MULTIPLIER * reference_mad,
    )

    max_warmup_steps = max(0, len(step_seconds) - stability_window)
    for warmup_steps in range(max_warmup_steps + 1):
        window = step_seconds[warmup_steps:warmup_steps + stability_window]
        if len(window) < stability_window:
            break
        if abs(statistics.median(window) - reference_median) > tolerance:
            continue
        if max(abs(sample - reference_median) for sample in window) > tolerance:
            continue
        return warmup_steps
    return 0


def estimate_step_tflops(num_flops_per_token: int, total_batch_size: int, step_seconds: float) -> float:
    if step_seconds <= 0.0:
        return 0.0
    return (num_flops_per_token * total_batch_size) / step_seconds / 1e12


def summarize_step_telemetry(step_telemetry: StepTelemetry, *, num_flops_per_token: int, total_batch_size: int) -> dict[str, float | int]:
    if step_telemetry.steady_steps > 0:
        label = "steady-state"
        steps = step_telemetry.steady_steps
        step_seconds = step_telemetry.steady_step_seconds
        loader_seconds = step_telemetry.steady_loader_seconds
        grad_seconds = step_telemetry.steady_grad_seconds
        accumulate_seconds = step_telemetry.steady_accumulate_seconds
        optimizer_seconds = step_telemetry.steady_optimizer_seconds
    else:
        label = "all-steps"
        steps = step_telemetry.total_steps
        step_seconds = step_telemetry.total_step_seconds
        loader_seconds = step_telemetry.total_loader_seconds
        grad_seconds = step_telemetry.total_grad_seconds
        accumulate_seconds = step_telemetry.total_accumulate_seconds
        optimizer_seconds = step_telemetry.total_optimizer_seconds

    other_seconds = max(0.0, step_seconds - loader_seconds - grad_seconds - accumulate_seconds - optimizer_seconds)
    compute_seconds = grad_seconds + accumulate_seconds + optimizer_seconds
    return {
        "window_label": label,
        "window_steps": steps,
        "train_tflops": estimate_step_tflops(num_flops_per_token, total_batch_size, step_seconds / max(steps, 1)),
        "mfu_percent": percent(compute_seconds, step_seconds),
        "loader_percent": percent(loader_seconds, step_seconds),
        "grad_percent": percent(grad_seconds, step_seconds),
        "accumulate_percent": percent(accumulate_seconds, step_seconds),
        "optimizer_percent": percent(optimizer_seconds, step_seconds),
        "other_step_percent": percent(other_seconds, step_seconds),
    }


def run_train_step(loader, grad_step, apply_grads, grad_accum_steps: int, model, optimizer):
    total_loss = None
    total_grads = None
    epoch = 1
    step_timing = StepTiming()
    t_step_start = time.perf_counter()

    for _ in range(grad_accum_steps):
        t_loader_start = time.perf_counter()
        batch_inputs, batch_targets, epoch = next(loader)
        step_timing.loader_seconds += time.perf_counter() - t_loader_start

        t_grad_start = time.perf_counter()
        loss, grads = grad_step(batch_inputs, batch_targets)
        mx.eval(loss, grads)
        step_timing.grad_seconds += time.perf_counter() - t_grad_start

        t_accumulate_start = time.perf_counter()
        scaled_loss = loss / grad_accum_steps
        scaled_grads = tree_map(lambda grad: grad / grad_accum_steps, grads)
        if total_grads is None:
            total_loss = scaled_loss
            total_grads = scaled_grads
        else:
            total_loss = total_loss + scaled_loss
            total_grads = tree_map(lambda left, right: left + right, total_grads, scaled_grads)
        mx.eval(total_loss, total_grads)
        step_timing.accumulate_seconds += time.perf_counter() - t_accumulate_start

    t_optimizer_start = time.perf_counter()
    optimizer_step = apply_grads(total_grads)
    mx.eval(optimizer_step, model.state, optimizer.state)
    step_timing.optimizer_seconds += time.perf_counter() - t_optimizer_start
    step_timing.total_seconds = time.perf_counter() - t_step_start
    return total_loss, epoch, step_timing


# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128

# Optimization
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.02
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
UTILIZATION_WARMUP_STEPS = 1
BENCHMARK_AUTO_MIN_TOTAL_STEPS = 8
BENCHMARK_AUTO_MIN_REFERENCE_STEPS = 5
BENCHMARK_AUTO_MAX_REFERENCE_STEPS = 12
BENCHMARK_AUTO_MIN_STABILITY_STEPS = 3
BENCHMARK_AUTO_MAX_STABILITY_STEPS = 6
BENCHMARK_AUTO_MIN_REL_TOL = 0.03
BENCHMARK_AUTO_MAD_MULTIPLIER = 3.0

PRESETS = {
    "m5-fast": RunPreset(
        description="Fast local iteration on Apple Silicon.",
        seq_len=256,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=default_canonical_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        depth=2,
        window_pattern="L",
        device_batch_size=2,
        total_batch_size=512,
    ),
    "m5-balanced": RunPreset(
        description="Default M5 baseline with materially better throughput than the upstream shape.",
        seq_len=512,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=default_canonical_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        depth=4,
        window_pattern="L",
        device_batch_size=4,
        total_batch_size=12288,
    ),
    "m5-large": RunPreset(
        description="Larger M5 run when you want more model capacity and can accept slower updates.",
        seq_len=1024,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=default_canonical_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        depth=6,
        window_pattern="L",
        device_batch_size=2,
        total_batch_size=4096,
    ),
    "m5-xlarge": RunPreset(
        description="Upstream-scale model shape with an M5-sized batch and dense attention.",
        seq_len=2048,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=default_canonical_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        depth=8,
        window_pattern="L",
        device_batch_size=2,
        total_batch_size=4096,
    ),
    "upstream": RunPreset(
        description="Original upstream-shaped MLX port for reference, closest to the H100-oriented defaults.",
        seq_len=MAX_SEQ_LEN,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=default_canonical_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        depth=8,
        window_pattern="SSSL",
        device_batch_size=8,
        total_batch_size=2**16,
    ),
}
DEFAULT_PRESET = "m5-balanced"


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    if args.benchmark_warmup_steps is not None and args.benchmark_warmup_steps < 0:
        raise ValueError("--benchmark-warmup-steps must be non-negative.")
    preset_name = args.preset or DEFAULT_PRESET
    preset = PRESETS[preset_name]
    config = RunConfig(
        preset=preset_name,
        time_budget=TIME_BUDGET,
        time_budget_mode=TIME_BUDGET_MODE_TRAIN if args.time_budget_mode is None else args.time_budget_mode,
        seq_len=preset.seq_len,
        eval_tokens=preset.eval_tokens,
        canonical_eval_seq_len=preset.canonical_eval_seq_len,
        canonical_eval_tokens=preset.canonical_eval_tokens,
        canonical_eval_batch_size=preset.canonical_eval_batch_size,
        canonical_eval_rung=None,
        canonical_eval_slices=EVAL_SLICE_CAP,
        canonical_eval_reference_tokens=EVAL_TOKENS,
        eval_hardware_key=detect_current_hardware_key(),
        eval_calibration_status="unresolved",
        eval_calibration_key=None,
        eval_calibration_confidence=None,
        eval_calibration_effective_confidence=None,
        eval_calibration_freshness=None,
        eval_calibration_repeat_count=None,
        eval_calibration_measured_train_seconds=None,
        eval_calibration_measured_on=None,
        eval_calibration_eval_semantics_signature=None,
        eval_calibration_runtime_shape_signature=None,
        eval_current_eval_semantics_signature=current_eval_semantics_signature(),
        eval_current_runtime_shape_signature=current_runtime_shape_signature(),
        eval_calibration_telemetry_count=None,
        eval_calibration_commit_count=None,
        eval_calibration_day_count=None,
        eval_calibration_observed_rungs=None,
        eval_calibration_stable_rungs=None,
        eval_calibration_last_seen_on=None,
        eval_calibration_last_seen_age_days=None,
        eval_calibration_limited_by=None,
        eval_policy_version=EVAL_POLICY_VERSION,
        depth=preset.depth,
        window_pattern=preset.window_pattern,
        device_batch_size=preset.device_batch_size,
        total_batch_size=preset.total_batch_size,
        seed=42 if args.seed is None else args.seed,
        smoke=args.smoke,
        benchmark_warmup_steps=args.benchmark_warmup_steps,
        benchmark_skip_eval=args.benchmark_skip_eval,
        prefer_prepacked_cache=not args.no_prepacked_cache,
        no_checkpoint=args.no_checkpoint,
        checkpoint_mode=CHECKPOINT_MODE_EXACT if args.checkpoint_mode is None else args.checkpoint_mode,
        checkpoint_save_mode=(
            CHECKPOINT_SAVE_MODE_SYNC if args.checkpoint_save_mode is None else args.checkpoint_save_mode
        ),
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )

    if args.smoke:
        smoke_seq_len = min(config.seq_len, 256)
        smoke_depth = min(config.depth, 2)
        smoke_device_batch_size = min(config.device_batch_size, 2)
        config = replace(
            config,
            time_budget=1.0,
            seq_len=smoke_seq_len,
            eval_tokens=smoke_seq_len * smoke_device_batch_size,
            canonical_eval_seq_len=smoke_seq_len,
            canonical_eval_tokens=smoke_seq_len * smoke_device_batch_size,
            canonical_eval_batch_size=smoke_device_batch_size,
            depth=smoke_depth,
            window_pattern="L",
            device_batch_size=smoke_device_batch_size,
            total_batch_size=smoke_seq_len * smoke_device_batch_size,
        )

    overrides = {}
    for field in (
        "time_budget",
        "seq_len",
        "eval_tokens",
        "canonical_eval_seq_len",
        "canonical_eval_tokens",
        "canonical_eval_batch_size",
        "depth",
        "window_pattern",
        "device_batch_size",
        "total_batch_size",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if overrides:
        config = replace(config, **overrides)
    explicit_canonical_overrides = any(
        getattr(args, field) is not None
        for field in ("canonical_eval_seq_len", "canonical_eval_tokens", "canonical_eval_batch_size")
    )
    if explicit_canonical_overrides and args.canonical_eval_batch_size is None:
        config = replace(
            config,
            canonical_eval_batch_size=default_canonical_eval_batch_size(config.canonical_eval_seq_len),
        )

    return resolve_eval_settings(
        config,
        explicit_canonical_overrides=explicit_canonical_overrides,
    )[0]


def resolve_resume_config(args: argparse.Namespace) -> RunConfig:
    disallowed = []
    for field in (
        "preset",
        "seq_len",
        "eval_tokens",
        "canonical_eval_seq_len",
        "canonical_eval_tokens",
        "canonical_eval_batch_size",
        "depth",
        "window_pattern",
        "device_batch_size",
        "total_batch_size",
        "seed",
    ):
        if getattr(args, field) is not None:
            disallowed.append(field)
    if args.smoke:
        disallowed.append("smoke")
    if args.benchmark_warmup_steps is not None:
        disallowed.append("benchmark_warmup_steps")
    if args.benchmark_skip_eval:
        disallowed.append("benchmark_skip_eval")
    if args.no_prepacked_cache:
        disallowed.append("no_prepacked_cache")
    if disallowed:
        raise ValueError(
            "--resume-from restores the saved run configuration. Only "
            "--time-budget, --time-budget-mode, --checkpoint-path, --checkpoint-interval, and --no-checkpoint may be overridden. "
            f"Got overrides for: {', '.join(disallowed)}"
        )

    metadata = load_checkpoint_metadata(args.resume_from)
    run_config = dict(metadata["run_config"])
    run_config.setdefault("benchmark_warmup_steps", None)
    run_config.setdefault("benchmark_skip_eval", False)
    run_config.setdefault("no_checkpoint", False)
    run_config.setdefault("canonical_eval_rung", None)
    run_config.setdefault("canonical_eval_slices", EVAL_SLICE_CAP)
    run_config.setdefault("canonical_eval_reference_tokens", EVAL_TOKENS)
    run_config.setdefault("eval_hardware_key", detect_current_hardware_key())
    run_config.setdefault("eval_calibration_status", "unknown")
    run_config.setdefault("eval_calibration_key", None)
    run_config.setdefault("eval_calibration_confidence", None)
    run_config.setdefault("eval_calibration_effective_confidence", None)
    run_config.setdefault("eval_calibration_freshness", None)
    run_config.setdefault("eval_calibration_repeat_count", None)
    run_config.setdefault("eval_calibration_measured_train_seconds", None)
    run_config.setdefault("eval_calibration_measured_on", None)
    run_config.setdefault("eval_calibration_eval_semantics_signature", None)
    run_config.setdefault("eval_calibration_runtime_shape_signature", None)
    run_config.setdefault("eval_current_eval_semantics_signature", current_eval_semantics_signature())
    run_config.setdefault("eval_current_runtime_shape_signature", current_runtime_shape_signature())
    run_config.setdefault("eval_calibration_telemetry_count", None)
    run_config.setdefault("eval_calibration_commit_count", None)
    run_config.setdefault("eval_calibration_day_count", None)
    run_config.setdefault("eval_calibration_observed_rungs", None)
    run_config.setdefault("eval_calibration_stable_rungs", None)
    run_config.setdefault("eval_calibration_last_seen_on", None)
    run_config.setdefault("eval_calibration_last_seen_age_days", None)
    run_config.setdefault("eval_calibration_limited_by", None)
    run_config.setdefault("eval_policy_version", EVAL_POLICY_VERSION)
    run_config.setdefault("time_budget_mode", TIME_BUDGET_MODE_TRAIN)
    run_config.setdefault("checkpoint_mode", CHECKPOINT_MODE_EXACT)
    run_config.setdefault("checkpoint_save_mode", CHECKPOINT_SAVE_MODE_SYNC)
    run_config.setdefault("checkpoint_path", None)
    run_config.setdefault("checkpoint_interval", None)
    run_config["time_budget"] = args.time_budget if args.time_budget is not None else run_config["time_budget"]
    run_config["time_budget_mode"] = (
        args.time_budget_mode
        if args.time_budget_mode is not None
        else run_config["time_budget_mode"]
    )
    run_config["no_checkpoint"] = args.no_checkpoint
    run_config["checkpoint_mode"] = (
        args.checkpoint_mode
        if args.checkpoint_mode is not None
        else run_config["checkpoint_mode"]
    )
    run_config["checkpoint_save_mode"] = (
        args.checkpoint_save_mode
        if args.checkpoint_save_mode is not None
        else run_config["checkpoint_save_mode"]
    )
    if args.no_checkpoint:
        run_config["checkpoint_path"] = None
        run_config["checkpoint_interval"] = None
    else:
        run_config["checkpoint_path"] = (
            args.checkpoint_path
            if args.checkpoint_path is not None
            else run_config["checkpoint_path"] or args.resume_from
        )
        run_config["checkpoint_interval"] = (
            args.checkpoint_interval
            if args.checkpoint_interval is not None
            else run_config["checkpoint_interval"]
        )
    run_config["resume_from"] = args.resume_from
    return RunConfig(**run_config)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        prog=os.environ.get("AUTORESEARCH_ENTRYPOINT_PROG"),
        description="Run autoresearch pretraining with MLX on Apple Silicon.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        help="Named runtime preset. Defaults to the M5-friendly balanced preset.",
    )
    parser.add_argument("--time-budget", type=float, help="Training budget in seconds.")
    parser.add_argument(
        "--time-budget-mode",
        choices=TIME_BUDGET_MODES,
        help="Budget accounting mode. 'train' stops on accumulated optimizer-step time; 'wall' stops on elapsed training-loop wall time.",
    )
    parser.add_argument("--seq-len", type=int, help="Sequence length for training and proxy evaluation.")
    parser.add_argument("--eval-tokens", type=int, help="Proxy validation token budget.")
    parser.add_argument(
        "--canonical-eval-seq-len",
        type=int,
        help="Fixed evaluation sequence length used for cross-preset comparisons.",
    )
    parser.add_argument(
        "--canonical-eval-tokens",
        type=int,
        help="Fixed evaluation token budget used for cross-preset comparisons.",
    )
    parser.add_argument(
        "--canonical-eval-batch-size",
        type=int,
        help="Batch size for fixed canonical evaluation. Defaults to a constant 4096 tokens per eval step.",
    )
    parser.add_argument("--depth", type=int, help="Number of transformer blocks.")
    parser.add_argument(
        "--window-pattern",
        help="Sliding-window pattern (S/L) repeated across layers.",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        help="Per-step device batch size.",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        help="Total batch size in tokens across gradient accumulation.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short end-to-end sanity check with smaller defaults.",
    )
    parser.add_argument(
        "--benchmark-warmup-steps",
        type=int,
        help="Override the warmup cutoff used for steady-state benchmark metrics. If omitted, the cutoff is auto-detected from step-time stabilization.",
    )
    parser.add_argument(
        "--benchmark-skip-eval",
        action="store_true",
        help="Skip proxy and canonical evaluation to reduce benchmark noise.",
    )
    parser.add_argument(
        "--no-prepacked-cache",
        action="store_true",
        help="Disable optional prepacked row caches and use the live packing path instead.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable resumable checkpoints, including the automatic defaults for longer runs.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=CHECKPOINT_MODES,
        help="Checkpoint semantics. 'exact' restores optimizer and loader state; 'weights_only' is retained only for failed-experiment comparison and resumes approximately from model weights alone.",
    )
    parser.add_argument(
        "--checkpoint-save-mode",
        choices=CHECKPOINT_SAVE_MODES,
        help="Checkpoint write path. 'sync' writes on the training thread; 'async' captures an exact host snapshot and writes it in the background. Async currently supports only exact checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Directory to save a resumable training checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        help="Save a checkpoint every N training seconds. If no path is provided, an automatic checkpoint directory is used.",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume training from a checkpoint directory.",
    )
    args = parser.parse_args()
    if args.resume_from:
        return resolve_resume_config(args)
    return resolve_run_config(args)


def resolve_checkpoint_settings(args: RunConfig, num_params: int) -> tuple[RunConfig, str | None]:
    if args.no_checkpoint:
        return replace(args, checkpoint_path=None, checkpoint_interval=None), "disabled via --no-checkpoint"

    auto_path = default_auto_checkpoint_path(
        args.preset,
        seq_len=args.seq_len,
        depth=args.depth,
        total_batch_size=args.total_batch_size,
        window_pattern=args.window_pattern,
        time_budget_mode=args.time_budget_mode,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_save_mode=args.checkpoint_save_mode,
    )

    if args.checkpoint_interval is not None:
        if args.checkpoint_path is None:
            return replace(args, checkpoint_path=str(auto_path)), "auto-selected checkpoint path for explicit interval"
        return args, None

    if args.time_budget <= AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC:
        return args, None

    decision = choose_auto_checkpoint_decision(
        num_params / 1e6,
        checkpoint_mode=args.checkpoint_mode,
    )
    checkpoint_path = args.checkpoint_path or str(auto_path)
    resolved = replace(
        args,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=decision.recommendation.interval_sec,
    )
    path_source = "existing path" if args.checkpoint_path is not None else "auto path"
    reason = (
        f"auto-enabled for time_budget>{AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC:.0f}s using "
        f"{decision.calibration.label}; selected {decision.recommendation.interval_label} "
        f"at {decision.recommendation.save_only_overhead_fraction * 100.0:.4f}% save-only overhead "
        f"with {path_source}; time_budget_mode={args.time_budget_mode}; checkpoint_mode={args.checkpoint_mode}; "
        f"checkpoint_save_mode={args.checkpoint_save_mode}; measured resume-ready penalty "
        f"{decision.calibration.resume_ready_penalty_sec:.3f}s"
    )
    return resolved, reason


def describe_eval_policy(args: RunConfig) -> str:
    if args.canonical_eval_rung == "smoke":
        return "smoke canonical eval"
    if args.canonical_eval_rung == "manual":
        return "manual canonical eval override"
    if args.canonical_eval_rung == "default":
        if args.eval_calibration_status == "shape-fallback":
            return "default canonical eval settings (mutated preset shape is not yet calibrated)"
        if args.eval_calibration_status == "hardware-unmatched":
            return (
                f"default canonical eval settings (no exact calibration row for hardware={args.eval_hardware_key})"
            )
        if args.eval_calibration_status == "stale-age":
            return (
                f"default canonical eval settings (calibration row {args.eval_calibration_key} is stale-age; "
                f"last_seen_on={args.eval_calibration_last_seen_on}, age_days={args.eval_calibration_last_seen_age_days})"
            )
        if args.eval_calibration_status == "stale-policy":
            return (
                f"default canonical eval settings (calibration row {args.eval_calibration_key} is stale for "
                f"policy_version={args.eval_policy_version})"
            )
        if args.eval_calibration_status == "signature-mismatch":
            return (
                f"default canonical eval settings (calibration row {args.eval_calibration_key} does not match "
                f"current code signatures; limited_by={args.eval_calibration_limited_by})"
            )
        return "default canonical eval settings"
    if args.canonical_eval_rung in {"cheap", "reference", "full"}:
        reference = (
            f", reference_eval_tokens={args.canonical_eval_reference_tokens}"
            if args.canonical_eval_reference_tokens is not None
            else ""
        )
        return (
            f"auto-selected {args.canonical_eval_rung} rung for {args.preset} on {args.eval_hardware_key}; "
            f"slices={args.canonical_eval_slices}{reference}; "
            f"calibration={args.eval_calibration_key}; confidence={args.eval_calibration_confidence}; "
            f"effective_confidence={args.eval_calibration_effective_confidence}; "
            f"freshness={args.eval_calibration_freshness}; telemetry_count={args.eval_calibration_telemetry_count}; "
            f"commit_count={args.eval_calibration_commit_count}; day_count={args.eval_calibration_day_count}; "
            f"observed_rungs={args.eval_calibration_observed_rungs}; stable_rungs={args.eval_calibration_stable_rungs}; "
            f"last_seen_on={args.eval_calibration_last_seen_on}; "
            f"repeats={args.eval_calibration_repeat_count}; measured_on={args.eval_calibration_measured_on}; "
            f"limited_by={args.eval_calibration_limited_by}"
        )
    return "canonical eval policy unresolved"


def main() -> None:
    args = parse_args()
    if args.checkpoint_save_mode == CHECKPOINT_SAVE_MODE_ASYNC and args.checkpoint_mode != CHECKPOINT_MODE_EXACT:
        raise ValueError("Async checkpoint writes currently support only --checkpoint-mode exact.")
    verify_mlx_env()
    t_start = time.perf_counter()
    mx.random.seed(args.seed)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = build_model_config(
        args.depth,
        vocab_size,
        sequence_len=max(args.seq_len, args.canonical_eval_seq_len),
        window_pattern=args.window_pattern,
    )

    model = GPT(config)
    model.init_weights()
    model.ensure_runtime_caches(max(args.seq_len, args.canonical_eval_seq_len))
    mx.eval(model.state)

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts["total"]
    args, checkpoint_resolution = resolve_checkpoint_settings(args, num_params)
    print(f"Model config: {asdict(config)}")
    print(f"Run preset: {args.preset} ({PRESETS[args.preset].description})")
    print(
        "Run config: "
        f"time_budget={args.time_budget}s, time_budget_mode={args.time_budget_mode}, "
        f"seq_len={args.seq_len}, eval_tokens={args.eval_tokens}, "
        f"canonical_eval_seq_len={args.canonical_eval_seq_len}, "
        f"canonical_eval_tokens={args.canonical_eval_tokens}, "
        f"canonical_eval_batch_size={args.canonical_eval_batch_size}, "
        f"canonical_eval_rung={args.canonical_eval_rung}, "
        f"canonical_eval_slices={args.canonical_eval_slices}, "
        f"canonical_eval_reference_tokens={args.canonical_eval_reference_tokens}, "
        f"eval_hardware_key={args.eval_hardware_key}, "
        f"eval_calibration_status={args.eval_calibration_status}, "
        f"eval_calibration_key={args.eval_calibration_key}, "
        f"eval_calibration_confidence={args.eval_calibration_confidence}, "
        f"eval_calibration_effective_confidence={args.eval_calibration_effective_confidence}, "
        f"eval_calibration_freshness={args.eval_calibration_freshness}, "
        f"eval_calibration_repeat_count={args.eval_calibration_repeat_count}, "
        f"eval_calibration_measured_train_seconds={args.eval_calibration_measured_train_seconds}, "
        f"eval_calibration_measured_on={args.eval_calibration_measured_on}, "
        f"eval_calibration_eval_semantics_signature={args.eval_calibration_eval_semantics_signature}, "
        f"eval_calibration_runtime_shape_signature={args.eval_calibration_runtime_shape_signature}, "
        f"eval_current_eval_semantics_signature={args.eval_current_eval_semantics_signature}, "
        f"eval_current_runtime_shape_signature={args.eval_current_runtime_shape_signature}, "
        f"eval_calibration_telemetry_count={args.eval_calibration_telemetry_count}, "
        f"eval_calibration_commit_count={args.eval_calibration_commit_count}, "
        f"eval_calibration_day_count={args.eval_calibration_day_count}, "
        f"eval_calibration_observed_rungs={args.eval_calibration_observed_rungs}, "
        f"eval_calibration_stable_rungs={args.eval_calibration_stable_rungs}, "
        f"eval_calibration_last_seen_on={args.eval_calibration_last_seen_on}, "
        f"eval_calibration_last_seen_age_days={args.eval_calibration_last_seen_age_days}, "
        f"eval_calibration_limited_by={args.eval_calibration_limited_by}, "
        f"eval_policy_version={args.eval_policy_version}, "
        f"device_batch_size={args.device_batch_size}, total_batch_size={args.total_batch_size}, "
        f"smoke={args.smoke}, benchmark_warmup_steps={args.benchmark_warmup_steps if args.benchmark_warmup_steps is not None else 'auto'}, "
        f"benchmark_skip_eval={args.benchmark_skip_eval}, prefer_prepacked_cache={args.prefer_prepacked_cache}, "
        f"no_checkpoint={args.no_checkpoint}, checkpoint_mode={args.checkpoint_mode}, checkpoint_save_mode={args.checkpoint_save_mode}, checkpoint_path={args.checkpoint_path}, "
        f"checkpoint_interval={args.checkpoint_interval}, resume_from={args.resume_from}"
    )
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
    if checkpoint_resolution is not None:
        print(f"Checkpoint policy: {checkpoint_resolution}")
    print(f"Eval policy: {describe_eval_policy(args)}")

    tokens_per_fwdbwd = args.device_batch_size * args.seq_len
    if args.total_batch_size % tokens_per_fwdbwd != 0:
        raise ValueError("total_batch_size must be divisible by device_batch_size * seq_len")
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

    optimizer = MuonAdamW(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_dataloader(
        tokenizer,
        args.device_batch_size,
        args.seq_len,
        "train",
        prefer_prepacked_cache=args.prefer_prepacked_cache,
    )
    if args.resume_from:
        restored = restore_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        resume_mode = restored["loaded_checkpoint_mode"]
        resume_semantics = "exact step-boundary state restored"
        if not restored["restored_optimizer_state"] or not restored["restored_loader_state"]:
            resume_semantics = "approximate resume: optimizer and/or loader state reset"
        print(
            f"Resumed from {args.resume_from}: step={restored['step']}, "
            f"cumulative_training_seconds={restored['total_training_time']:.1f}, "
            f"checkpoint_mode={resume_mode}, checkpoint_save_mode={restored['loaded_checkpoint_save_mode']} "
            f"({resume_semantics})"
        )
    else:
        restored = {
            "step": 0,
            "total_training_time": 0.0,
            "smooth_train_loss": 0.0,
            "step_telemetry": None,
            "total_checkpoint_time": 0.0,
            "checkpoint_count": 0,
        }
    grad_step = make_grad_step_fn(model)
    apply_grads = make_apply_grads_fn(model, optimizer)

    print(f"Time budget: {args.time_budget}s ({args.time_budget_mode})")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    if args.checkpoint_interval is not None and args.checkpoint_path is None:
        raise ValueError("--checkpoint-interval requires --checkpoint-path")

    mx.reset_peak_memory()
    smooth_train_loss = float(restored["smooth_train_loss"])
    total_training_time = float(restored["total_training_time"])
    resumed_training_time = total_training_time
    step = int(restored["step"])
    resumed_step = step
    local_step_index = 0
    step_telemetry = StepTelemetry.from_dict(restored.get("step_telemetry"))
    resumed_checkpoint_seconds = float(restored.get("total_checkpoint_time", 0.0))
    resumed_checkpoint_write_seconds = float(restored.get("total_checkpoint_write_time", resumed_checkpoint_seconds))
    resumed_checkpoint_count = int(restored.get("checkpoint_count", 0))
    checkpoint_seconds = 0.0
    checkpoint_count = 0
    checkpoint_write_seconds = 0.0
    last_checkpoint_time = total_training_time
    checkpoint_run_config = asdict(replace(args, resume_from=None))
    session_step_seconds: list[float] = []
    session_post_step_wall_seconds: list[float] = []
    t_budget_start = time.perf_counter()
    async_checkpoint_writer = (
        AsyncCheckpointWriter()
        if args.checkpoint_path is not None and args.checkpoint_save_mode == CHECKPOINT_SAVE_MODE_ASYNC
        else None
    )

    def budget_elapsed_seconds() -> float:
        if args.time_budget_mode == TIME_BUDGET_MODE_TRAIN:
            return total_training_time
        return time.perf_counter() - t_budget_start

    def maybe_save_checkpoint(*, force: bool = False) -> None:
        nonlocal last_checkpoint_time, checkpoint_seconds, checkpoint_count, checkpoint_write_seconds
        if args.checkpoint_path is None:
            return
        if async_checkpoint_writer is not None:
            async_checkpoint_writer.check_health()
        if force and total_training_time == last_checkpoint_time:
            return
        if not force:
            if args.checkpoint_interval is None:
                return
            if (total_training_time - last_checkpoint_time) < args.checkpoint_interval:
                return
        if async_checkpoint_writer is None:
            elapsed = save_checkpoint(
                args.checkpoint_path,
                checkpoint_mode=args.checkpoint_mode,
                checkpoint_save_mode=args.checkpoint_save_mode,
                run_config=checkpoint_run_config,
                model_config=asdict(config),
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                step=step,
                total_training_time=total_training_time,
                smooth_train_loss=smooth_train_loss,
                step_telemetry=asdict(step_telemetry),
                total_checkpoint_time=resumed_checkpoint_seconds + checkpoint_seconds,
                total_checkpoint_write_time=resumed_checkpoint_write_seconds + checkpoint_write_seconds,
                checkpoint_count=resumed_checkpoint_count + checkpoint_count,
            )
            checkpoint_seconds += elapsed
            checkpoint_write_seconds += elapsed
            checkpoint_count += 1
            print(f"\nCheckpoint saved to {args.checkpoint_path} at step {step}.")
        else:
            snapshot, capture_seconds = capture_async_checkpoint_snapshot(
                args.checkpoint_path,
                checkpoint_mode=args.checkpoint_mode,
                run_config=checkpoint_run_config,
                model_config=asdict(config),
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                step=step,
                total_training_time=total_training_time,
                smooth_train_loss=smooth_train_loss,
                step_telemetry=asdict(step_telemetry),
                total_checkpoint_time=resumed_checkpoint_seconds + checkpoint_seconds,
                total_checkpoint_write_time=resumed_checkpoint_write_seconds + checkpoint_write_seconds,
                checkpoint_count=resumed_checkpoint_count + checkpoint_count,
            )
            checkpoint_seconds += capture_seconds
            async_checkpoint_writer.submit(snapshot)
            checkpoint_write_seconds = async_checkpoint_writer.completed_write_seconds
            checkpoint_count = async_checkpoint_writer.completed_count
            print(f"\nCheckpoint queued to {args.checkpoint_path} at step {step}.")
        last_checkpoint_time = total_training_time

    while budget_elapsed_seconds() < args.time_budget:
        progress = min(budget_elapsed_seconds() / args.time_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        optimizer.set_schedule(
            lr_multiplier=lrm,
            muon_momentum=muon_momentum,
            muon_weight_decay=muon_weight_decay,
        )

        loss, epoch, step_timing = run_train_step(
            train_loader,
            grad_step,
            apply_grads,
            grad_accum_steps,
            model,
            optimizer,
        )
        dt = step_timing.total_seconds

        train_loss = loss.item()
        if train_loss > 100:
            print("FAIL")
            raise SystemExit(1)

        step_telemetry.record_step(
            step_timing,
            include_in_steady=local_step_index >= UTILIZATION_WARMUP_STEPS,
        )
        total_training_time += dt
        session_step_seconds.append(dt)

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(args.total_batch_size / dt)
        step_tflops = estimate_step_tflops(num_flops_per_token, args.total_batch_size, dt)
        remaining = max(0.0, args.time_budget - budget_elapsed_seconds())
        step_util = percent(step_timing.compute_seconds, dt)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"util: {step_util:.1f}% | tflops: {step_tflops:.2f} | "
            f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        local_step_index += 1
        maybe_save_checkpoint()
        session_post_step_wall_seconds.append(time.perf_counter() - t_budget_start)

    budget_elapsed_at_cutoff = budget_elapsed_seconds()
    print()
    maybe_save_checkpoint(force=True)
    if async_checkpoint_writer is not None:
        t_flush_start = time.perf_counter()
        async_checkpoint_writer.wait_until_idle()
        checkpoint_seconds += time.perf_counter() - t_flush_start
        checkpoint_write_seconds = async_checkpoint_writer.completed_write_seconds
        checkpoint_count = async_checkpoint_writer.completed_count
        async_checkpoint_writer.close()

    total_tokens = step * args.total_batch_size
    session_steps = step - resumed_step
    session_tokens = session_steps * args.total_batch_size
    if args.benchmark_warmup_steps is None:
        benchmark_warmup_mode = "auto"
        benchmark_warmup_steps_done = detect_benchmark_warmup_steps(session_step_seconds)
    else:
        benchmark_warmup_mode = "fixed"
        benchmark_warmup_steps_done = min(args.benchmark_warmup_steps, session_steps)
    benchmark_warmup_step_seconds = sum(session_step_seconds[:benchmark_warmup_steps_done])
    benchmark_warmup_wall_seconds = (
        session_post_step_wall_seconds[benchmark_warmup_steps_done - 1] if benchmark_warmup_steps_done > 0 else 0.0
    )
    warmup_done_step = resumed_step + benchmark_warmup_steps_done
    steady_state_steps = max(0, session_steps - benchmark_warmup_steps_done)
    steady_state_tokens = max(0, session_tokens - benchmark_warmup_steps_done * args.total_batch_size)
    if args.benchmark_skip_eval:
        proxy_val_bpb = None
        val_bpb = None
        proxy_eval_seconds = 0.0
        canonical_eval_seconds = 0.0
    else:
        model.eval()
        t_proxy_eval_start = time.perf_counter()
        proxy_val_bpb = evaluate_bpb(
            model,
            tokenizer,
            args.device_batch_size,
            seq_len=args.seq_len,
            eval_tokens=args.eval_tokens,
            prefer_prepacked_cache=args.prefer_prepacked_cache,
        )
        proxy_eval_seconds = time.perf_counter() - t_proxy_eval_start
        t_canonical_eval_start = time.perf_counter()
        val_bpb = evaluate_bpb(
            model,
            tokenizer,
            args.canonical_eval_batch_size,
            seq_len=args.canonical_eval_seq_len,
            eval_tokens=args.canonical_eval_tokens,
            prefer_prepacked_cache=args.prefer_prepacked_cache,
            eval_slices=args.canonical_eval_slices,
            reference_eval_tokens=args.canonical_eval_reference_tokens,
        )
        canonical_eval_seconds = time.perf_counter() - t_canonical_eval_start
    total_wall_seconds = time.perf_counter() - t_start
    telemetry_summary = summarize_step_telemetry(
        step_telemetry,
        num_flops_per_token=num_flops_per_token,
        total_batch_size=args.total_batch_size,
    )
    session_training_seconds = total_training_time - resumed_training_time
    cumulative_training_seconds = total_training_time
    session_total_seconds = total_wall_seconds
    session_eval_seconds = proxy_eval_seconds + canonical_eval_seconds
    session_checkpoint_count = checkpoint_count
    cumulative_checkpoint_seconds = resumed_checkpoint_seconds + checkpoint_seconds
    cumulative_checkpoint_count = resumed_checkpoint_count + checkpoint_count
    cumulative_checkpoint_write_seconds = resumed_checkpoint_write_seconds + checkpoint_write_seconds
    steady_state_training_seconds = max(0.0, session_training_seconds - benchmark_warmup_step_seconds)
    steady_state_tok_per_sec = (
        steady_state_tokens / steady_state_training_seconds if steady_state_training_seconds > 0.0 else 0.0
    )
    eval_percent = percent(session_eval_seconds, session_total_seconds)
    checkpoint_percent = percent(checkpoint_seconds, session_total_seconds)
    checkpoint_write_percent = percent(checkpoint_write_seconds, session_total_seconds)
    peak_vram_mb = mx.get_peak_memory() / 1024 / 1024

    if val_bpb is not None:
        recorded_at, recorded_on = now_iso()
        telemetry_record = EvalTelemetryRecord(
            recorded_at=recorded_at,
            recorded_on=recorded_on,
            commit=current_git_commit(),
            preset=args.preset,
            hardware_key=args.eval_hardware_key,
            policy_version=args.eval_policy_version,
            default_shape=uses_default_preset_shape(args),
            smoke=args.smoke,
            benchmark_skip_eval=args.benchmark_skip_eval,
            calibration_status=args.eval_calibration_status,
            canonical_rung=args.canonical_eval_rung,
            canonical_eval_seq_len=args.canonical_eval_seq_len,
            canonical_eval_tokens=args.canonical_eval_tokens,
            canonical_eval_batch_size=args.canonical_eval_batch_size,
            canonical_eval_slices=args.canonical_eval_slices,
            canonical_eval_reference_tokens=args.canonical_eval_reference_tokens,
            time_budget=args.time_budget,
            time_budget_mode=args.time_budget_mode,
            training_seconds=session_training_seconds,
            total_seconds=session_total_seconds,
            canonical_eval_seconds=canonical_eval_seconds,
            val_bpb=float(val_bpb),
        )
        try:
            append_eval_telemetry(telemetry_record)
        except Exception as exc:
            print(f"Eval telemetry: failed to append ({exc})")

    print("---")
    if val_bpb is None:
        print("val_bpb:          skipped")
        print("proxy_val_bpb:    skipped")
    else:
        print(f"val_bpb:          {val_bpb:.6f}")
        print(f"proxy_val_bpb:    {proxy_val_bpb:.6f}")
    print(f"training_seconds: {session_training_seconds:.1f}")
    print(f"total_seconds:    {session_total_seconds:.1f}")
    print(f"time_budget_mode: {args.time_budget_mode}")
    print(f"budget_elapsed_seconds: {budget_elapsed_at_cutoff:.1f}")
    print(f"canonical_rung:   {args.canonical_eval_rung}")
    print(f"eval_hardware_key: {args.eval_hardware_key}")
    print(f"eval_calibration_status: {args.eval_calibration_status}")
    print(f"eval_calibration_key: {args.eval_calibration_key}")
    print(f"eval_calibration_confidence: {args.eval_calibration_confidence}")
    print(f"eval_calibration_effective_confidence: {args.eval_calibration_effective_confidence}")
    print(f"eval_calibration_freshness: {args.eval_calibration_freshness}")
    print(f"eval_calibration_repeat_count: {args.eval_calibration_repeat_count}")
    print(f"eval_calibration_measured_train_seconds: {args.eval_calibration_measured_train_seconds}")
    print(f"eval_calibration_measured_on: {args.eval_calibration_measured_on}")
    print(f"eval_calibration_eval_semantics_signature: {args.eval_calibration_eval_semantics_signature}")
    print(f"eval_calibration_runtime_shape_signature: {args.eval_calibration_runtime_shape_signature}")
    print(f"eval_current_eval_semantics_signature: {args.eval_current_eval_semantics_signature}")
    print(f"eval_current_runtime_shape_signature: {args.eval_current_runtime_shape_signature}")
    print(f"eval_calibration_telemetry_count: {args.eval_calibration_telemetry_count}")
    print(f"eval_calibration_commit_count: {args.eval_calibration_commit_count}")
    print(f"eval_calibration_day_count: {args.eval_calibration_day_count}")
    print(f"eval_calibration_observed_rungs: {args.eval_calibration_observed_rungs}")
    print(f"eval_calibration_stable_rungs: {args.eval_calibration_stable_rungs}")
    print(f"eval_calibration_last_seen_on: {args.eval_calibration_last_seen_on}")
    print(f"eval_calibration_last_seen_age_days: {args.eval_calibration_last_seen_age_days}")
    print(f"eval_calibration_limited_by: {args.eval_calibration_limited_by}")
    print(f"eval_policy_version: {args.eval_policy_version}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {telemetry_summary['mfu_percent']:.2f}")
    print(f"train_tflops:     {telemetry_summary['train_tflops']:.3f}")
    print(f"loader_percent:   {telemetry_summary['loader_percent']:.2f}")
    print(f"grad_percent:     {telemetry_summary['grad_percent']:.2f}")
    print(f"accum_percent:    {telemetry_summary['accumulate_percent']:.2f}")
    print(f"optimizer_percent: {telemetry_summary['optimizer_percent']:.2f}")
    print(f"other_step_percent: {telemetry_summary['other_step_percent']:.2f}")
    print(f"checkpoint_percent: {checkpoint_percent:.2f}")
    print(f"checkpoint_write_percent: {checkpoint_write_percent:.2f}")
    print(f"eval_percent:     {eval_percent:.2f}")
    print(f"util_window_steps: {telemetry_summary['window_steps']}")
    print(f"util_window:      cumulative {telemetry_summary['window_label']}")
    print(f"checkpoint_count: {session_checkpoint_count}")
    print(f"session_tokens_M: {session_tokens / 1e6:.3f}")
    print(f"session_steps:    {session_steps}")
    print(f"benchmark_warmup_mode: {benchmark_warmup_mode}")
    print(f"benchmark_warmup_steps: {benchmark_warmup_steps_done}")
    print(f"warmup_done_step: {warmup_done_step}")
    print(f"benchmark_warmup_step_seconds: {benchmark_warmup_step_seconds:.3f}")
    print(f"benchmark_warmup_wall_seconds: {benchmark_warmup_wall_seconds:.3f}")
    print(f"steady_state_training_seconds: {steady_state_training_seconds:.3f}")
    print(f"steady_state_tokens_M: {steady_state_tokens / 1e6:.3f}")
    print(f"steady_state_steps: {steady_state_steps}")
    print(f"steady_state_tok_per_sec: {steady_state_tok_per_sec:.1f}")
    print(f"cumulative_training_seconds: {cumulative_training_seconds:.1f}")
    print(f"cumulative_checkpoint_seconds: {cumulative_checkpoint_seconds:.3f}")
    print(f"cumulative_checkpoint_write_seconds: {cumulative_checkpoint_write_seconds:.3f}")
    print(f"cumulative_checkpoint_count: {cumulative_checkpoint_count}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.3f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {args.depth}")
    if args.benchmark_skip_eval:
        print("proxy_eval_tokens: skipped")
        print("canonical_seq_len: skipped")
        print("canonical_tokens: skipped")
        print("canonical_batch:  skipped")
    else:
        print(f"proxy_eval_tokens: {args.eval_tokens}")
        print(f"canonical_seq_len: {args.canonical_eval_seq_len}")
        print(f"canonical_tokens: {args.canonical_eval_tokens}")
        print(f"canonical_batch:  {args.canonical_eval_batch_size}")
        print(f"canonical_slices: {args.canonical_eval_slices}")


if __name__ == "__main__":
    main()
