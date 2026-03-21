from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json

from .config import CUDA_PRESETS


EVAL_POLICY_VERSION = 1
EVAL_CALIBRATION_AGING_DAYS = 30
EVAL_CALIBRATION_STALE_DAYS = 90


@dataclass(frozen=True)
class EvalRungSpec:
    key: str
    label: str
    seq_len: int
    eval_tokens: int
    batch_size: int


@dataclass(frozen=True)
class EvalRungMeasurement:
    spec: EvalRungSpec
    eval_seconds: float
    val_bpb: float
    abs_error_vs_full: float | None = None

    def overhead_fraction(self, time_budget_sec: float) -> float:
        if time_budget_sec <= 0:
            raise ValueError("time_budget_sec must be positive")
        return self.eval_seconds / time_budget_sec


@dataclass(frozen=True)
class EvalCalibration:
    key: str
    label: str
    hardware_key: str
    preset: str
    seq_len: int
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int
    source: str
    policy_version: int
    confidence: str
    measured_train_seconds: float
    repeat_count: int
    measured_on: str
    eval_semantics_signature: str
    runtime_shape_signature: str
    notes: str = ""
    cheap: EvalRungMeasurement | None = None
    reference: EvalRungMeasurement | None = None
    full: EvalRungMeasurement | None = None

    def rung(self, key: str) -> EvalRungMeasurement:
        value = getattr(self, key, None)
        if value is None:
            raise KeyError(f"No rung {key!r} for calibration {self.key}.")
        return value

    def available_rungs(self) -> tuple[EvalRungMeasurement, ...]:
        return tuple(r for r in (self.cheap, self.reference, self.full) if r is not None)


@dataclass(frozen=True)
class AutoEvalDecision:
    calibration: EvalCalibration
    rung_key: str
    rung: EvalRungMeasurement
    projected_overhead_fraction: float
    freshness: str
    effective_confidence: str
    limited_by: str | None


def _short_hash(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def current_eval_semantics_signature() -> str:
    payload = {
        "policy_version": EVAL_POLICY_VERSION,
        "rungs": {
            "cheap": {"seq_len": 2048, "eval_tokens": 262144, "batch_size": 32},
            "reference": {"seq_len": 2048, "eval_tokens": 3 * 524288, "batch_size": 32},
            "full": {"seq_len": 2048, "eval_tokens": 40 * 524288, "batch_size": 32},
        },
        "budget_caps": {"full": 0.01, "reference": 0.06},
    }
    return _short_hash(payload)


def current_runtime_shape_signature() -> str:
    payload = {
        key: {
            "seq_len": value.seq_len,
            "depth": value.depth,
            "window_pattern": value.window_pattern,
        }
        for key, value in CUDA_PRESETS.items()
    }
    return _short_hash(payload)


CUDA_CHEAP_RUNG = EvalRungSpec(
    key="cheap",
    label="Cheap canonical",
    seq_len=2048,
    eval_tokens=262144,
    batch_size=32,
)
CUDA_REFERENCE_RUNG = EvalRungSpec(
    key="reference",
    label="Reference canonical",
    seq_len=2048,
    eval_tokens=3 * 524288,
    batch_size=32,
)
CUDA_SUBREF_ONE_SIXTH_RUNG = EvalRungSpec(
    key="subref-one-sixth",
    label="Subreference one-sixth",
    seq_len=2048,
    eval_tokens=262144,
    batch_size=32,
)
CUDA_FULL_RUNG = EvalRungSpec(
    key="full",
    label="Full upstream audit",
    seq_len=2048,
    eval_tokens=40 * 524288,
    batch_size=32,
)


CURRENT_EVAL_SIGNATURE = current_eval_semantics_signature()
CURRENT_RUNTIME_SIGNATURE = current_runtime_shape_signature()


CUDA_EVAL_CALIBRATIONS: tuple[EvalCalibration, ...] = (
    EvalCalibration(
        key="gb10-m5-small-fa4-fast-2026-03-13",
        label="GB10 m5-small FA4 fast bring-up",
        hardware_key="nvidia-blackwell-gb10-120gb",
        preset="m5-small",
        seq_len=512,
        depth=4,
        window_pattern="L",
        device_batch_size=32,
        total_batch_size=32768,
        source="real-fast-bringup",
        policy_version=EVAL_POLICY_VERSION,
        confidence="seed-single-checkpoint",
        measured_train_seconds=120.0,
        repeat_count=1,
        measured_on="2026-03-13",
        eval_semantics_signature=CURRENT_EVAL_SIGNATURE,
        runtime_shape_signature=CURRENT_RUNTIME_SIGNATURE,
        notes="FA4-backed GB10 fast bring-up candidate default.",
        cheap=EvalRungMeasurement(
            spec=CUDA_CHEAP_RUNG,
            eval_seconds=0.8,
            val_bpb=1.282243,
        ),
        reference=EvalRungMeasurement(
            spec=CUDA_REFERENCE_RUNG,
            eval_seconds=4.5,
            val_bpb=1.253053,
        ),
    ),
    EvalCalibration(
        key="gb10-upstream-sdpa-2026-03-13",
        label="GB10 upstream checkpoint-backed ladder",
        hardware_key="nvidia-blackwell-gb10-120gb",
        preset="upstream",
        seq_len=2048,
        depth=8,
        window_pattern="SSSSL",
        device_batch_size=16,
        total_batch_size=65536,
        source="shared-engine-eval-calibration",
        policy_version=EVAL_POLICY_VERSION,
        confidence="seed-single-checkpoint",
        measured_train_seconds=300.0,
        repeat_count=1,
        measured_on="2026-03-13",
        eval_semantics_signature=CURRENT_EVAL_SIGNATURE,
        runtime_shape_signature=CURRENT_RUNTIME_SIGNATURE,
        notes="Minimal-container GB10 checkpoint-backed upstream ladder.",
        cheap=EvalRungMeasurement(
            spec=CUDA_CHEAP_RUNG,
            eval_seconds=1.0,
            val_bpb=2.212186,
            abs_error_vs_full=0.038056,
        ),
        reference=EvalRungMeasurement(
            spec=CUDA_REFERENCE_RUNG,
            eval_seconds=5.5,
            val_bpb=2.161972,
            abs_error_vs_full=0.012158,
        ),
        full=EvalRungMeasurement(
            spec=CUDA_FULL_RUNG,
            eval_seconds=72.2,
            val_bpb=2.174130,
            abs_error_vs_full=0.0,
        ),
    ),
)


def find_calibration(*, preset: str, hardware_key: str) -> EvalCalibration | None:
    for calibration in CUDA_EVAL_CALIBRATIONS:
        if calibration.preset == preset and calibration.hardware_key == hardware_key:
            return calibration
    return None


def has_calibration_for_preset(*, preset: str) -> bool:
    return any(calibration.preset == preset for calibration in CUDA_EVAL_CALIBRATIONS)


def _freshness(calibration: EvalCalibration) -> str:
    measured = date.fromisoformat(calibration.measured_on)
    age_days = (date.today() - measured).days
    if age_days >= EVAL_CALIBRATION_STALE_DAYS:
        return "stale-age"
    if age_days >= EVAL_CALIBRATION_AGING_DAYS:
        return "aging"
    return "fresh"


def choose_runtime_eval(
    *,
    calibration: EvalCalibration,
    time_budget: float,
) -> AutoEvalDecision:
    freshness = _freshness(calibration)
    effective_confidence = calibration.confidence

    if freshness == "stale-age":
        raise ValueError("stale-age calibrations should not be auto-selected")

    # Seed calibrations can auto-pick cheap or reference, but not full.
    allowed_max_rung = "reference"
    limited_by = "confidence"
    if freshness == "aging":
        allowed_max_rung = "reference"
        limited_by = "aging"

    candidates = list(calibration.available_rungs())
    if allowed_max_rung == "reference":
        candidates = [r for r in candidates if r.spec.key in {"cheap", "reference"}]

    full = next((r for r in candidates if r.spec.key == "full"), None)
    reference = next((r for r in candidates if r.spec.key == "reference"), None)
    cheap = next((r for r in candidates if r.spec.key == "cheap"), None)

    if full is not None and full.overhead_fraction(time_budget) <= 0.01:
        return AutoEvalDecision(
            calibration=calibration,
            rung_key="full",
            rung=full,
            projected_overhead_fraction=full.overhead_fraction(time_budget),
            freshness=freshness,
            effective_confidence=effective_confidence,
            limited_by=None,
        )
    if reference is not None and reference.overhead_fraction(time_budget) <= 0.06:
        return AutoEvalDecision(
            calibration=calibration,
            rung_key="reference",
            rung=reference,
            projected_overhead_fraction=reference.overhead_fraction(time_budget),
            freshness=freshness,
            effective_confidence=effective_confidence,
            limited_by=limited_by,
        )
    if cheap is None:
        raise ValueError(f"Calibration {calibration.key} has no cheap rung")
    return AutoEvalDecision(
        calibration=calibration,
        rung_key="cheap",
        rung=cheap,
        projected_overhead_fraction=cheap.overhead_fraction(time_budget),
        freshness=freshness,
        effective_confidence=effective_confidence,
        limited_by=limited_by,
    )
