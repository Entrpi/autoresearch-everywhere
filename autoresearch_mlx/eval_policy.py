from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    CANONICAL_EVAL_SEQ_LEN,
    CANONICAL_EVAL_STEP_TOKENS,
    CANONICAL_EVAL_TOKENS,
    EVAL_SLICE_CAP,
    EVAL_TOKENS,
)


DEFAULT_EVAL_HARDWARE_KEY = "apple-m5-32gb-10gpu"
REFERENCE_EVAL_TOKENS = 3 * 524288


@dataclass(frozen=True)
class EvalRungSpec:
    key: str
    label: str
    eval_tokens: int
    eval_slices: int
    reference_eval_tokens: int | None


@dataclass(frozen=True)
class EvalRungMeasurement:
    spec: EvalRungSpec
    eval_seconds: float
    val_bpb: float
    abs_error_vs_full: float

    def overhead_fraction(self, time_budget_sec: float) -> float:
        if time_budget_sec <= 0.0:
            raise ValueError("time_budget_sec must be positive")
        return self.eval_seconds / time_budget_sec


@dataclass(frozen=True)
class EvalCalibration:
    key: str
    label: str
    hardware_key: str
    preset: str
    seq_len: int
    batch_size: int
    source: str
    notes: str = ""
    cheap: EvalRungMeasurement | None = None
    reference: EvalRungMeasurement | None = None
    full: EvalRungMeasurement | None = None

    def rung(self, rung_key: str) -> EvalRungMeasurement:
        value = getattr(self, rung_key, None)
        if value is None:
            raise KeyError(f"No rung {rung_key!r} registered for {self.key}.")
        return value

    def available_rungs(self) -> tuple[EvalRungMeasurement, ...]:
        return tuple(
            rung
            for rung in (self.cheap, self.reference, self.full)
            if rung is not None
        )


@dataclass(frozen=True)
class EvalBudgetPolicy:
    label: str
    full_overhead_cap_fraction: float
    reference_overhead_cap_fraction: float


@dataclass(frozen=True)
class EvalRecommendation:
    rung_key: str
    rung: EvalRungMeasurement
    projected_overhead_fraction: float
    failed_more_expensive_rung_key: str | None


@dataclass(frozen=True)
class AutoEvalDecision:
    calibration: EvalCalibration
    recommendation: EvalRecommendation


def default_eval_batch_size(seq_len: int) -> int:
    if seq_len <= 0:
        raise ValueError("eval sequence length must be positive")
    return max(1, CANONICAL_EVAL_STEP_TOKENS // seq_len)


CHEAP_EVAL_RUNG = EvalRungSpec(
    key="cheap",
    label="Cheap canonical",
    eval_tokens=CANONICAL_EVAL_TOKENS,
    eval_slices=EVAL_SLICE_CAP,
    reference_eval_tokens=EVAL_TOKENS,
)
REFERENCE_EVAL_RUNG = EvalRungSpec(
    key="reference",
    label="Reference canonical",
    eval_tokens=REFERENCE_EVAL_TOKENS,
    eval_slices=EVAL_SLICE_CAP,
    reference_eval_tokens=EVAL_TOKENS,
)
FULL_EVAL_RUNG = EvalRungSpec(
    key="full",
    label="Full upstream audit",
    eval_tokens=EVAL_TOKENS,
    eval_slices=1,
    reference_eval_tokens=None,
)
DEFAULT_EVAL_RUNGS = (
    CHEAP_EVAL_RUNG,
    REFERENCE_EVAL_RUNG,
    FULL_EVAL_RUNG,
)


DEFAULT_EVAL_BUDGET_POLICY = EvalBudgetPolicy(
    label="Use full when <=1% of the training budget, otherwise reference when <=6%, otherwise cheap.",
    full_overhead_cap_fraction=0.01,
    reference_overhead_cap_fraction=0.06,
)


def _measurement(
    spec: EvalRungSpec,
    *,
    eval_seconds: float,
    val_bpb: float,
    abs_error_vs_full: float,
) -> EvalRungMeasurement:
    return EvalRungMeasurement(
        spec=spec,
        eval_seconds=eval_seconds,
        val_bpb=val_bpb,
        abs_error_vs_full=abs_error_vs_full,
    )


DEFAULT_EVAL_CALIBRATIONS = (
    EvalCalibration(
        key="m5-fast_apple-m5-32gb-10gpu",
        label="m5-fast eval ladder on Apple M5 32GB / 10 GPU cores",
        hardware_key=DEFAULT_EVAL_HARDWARE_KEY,
        preset="m5-fast",
        seq_len=CANONICAL_EVAL_SEQ_LEN,
        batch_size=default_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        source="Measured from a 120s m5-fast checkpoint on the reference M5 machine.",
        notes="Reference rung is a plausible 5-minute default; full is primarily an audit rung.",
        cheap=_measurement(
            CHEAP_EVAL_RUNG,
            eval_seconds=1.3409303750377148,
            val_bpb=2.002362005413137,
            abs_error_vs_full=0.008803160051600756,
        ),
        reference=_measurement(
            REFERENCE_EVAL_RUNG,
            eval_seconds=7.735471999971196,
            val_bpb=2.0143204051162025,
            abs_error_vs_full=0.003155239651464914,
        ),
        full=_measurement(
            FULL_EVAL_RUNG,
            eval_seconds=109.7370084580034,
            val_bpb=2.0111651654647376,
            abs_error_vs_full=0.0,
        ),
    ),
    EvalCalibration(
        key="m5-balanced_apple-m5-32gb-10gpu",
        label="m5-balanced eval ladder on Apple M5 32GB / 10 GPU cores",
        hardware_key=DEFAULT_EVAL_HARDWARE_KEY,
        preset="m5-balanced",
        seq_len=CANONICAL_EVAL_SEQ_LEN,
        batch_size=default_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        source="Measured from a 120s m5-balanced checkpoint on the reference M5 machine.",
        notes="Reference is the cleanest middle rung; full becomes practical only on much longer runs.",
        cheap=_measurement(
            CHEAP_EVAL_RUNG,
            eval_seconds=2.963485582964495,
            val_bpb=1.6250616868446086,
            abs_error_vs_full=0.0021893173372982133,
        ),
        reference=_measurement(
            REFERENCE_EVAL_RUNG,
            eval_seconds=17.341864292044193,
            val_bpb=1.6277542594613608,
            abs_error_vs_full=0.0005032552794539402,
        ),
        full=_measurement(
            FULL_EVAL_RUNG,
            eval_seconds=183.78859833301976,
            val_bpb=1.6272510041819068,
            abs_error_vs_full=0.0,
        ),
    ),
    EvalCalibration(
        key="m5-large_apple-m5-32gb-10gpu",
        label="m5-large eval ladder on Apple M5 32GB / 10 GPU cores",
        hardware_key=DEFAULT_EVAL_HARDWARE_KEY,
        preset="m5-large",
        seq_len=CANONICAL_EVAL_SEQ_LEN,
        batch_size=default_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        source="Measured from a 120s m5-large checkpoint on the reference M5 machine.",
        notes="Reference reads more like a long-run rung than a 5-minute default.",
        cheap=_measurement(
            CHEAP_EVAL_RUNG,
            eval_seconds=5.866491249995306,
            val_bpb=1.8033380842562028,
            abs_error_vs_full=0.004303172826900958,
        ),
        reference=_measurement(
            REFERENCE_EVAL_RUNG,
            eval_seconds=34.821197957964614,
            val_bpb=1.811423111547698,
            abs_error_vs_full=0.0037818544645942254,
        ),
        full=_measurement(
            FULL_EVAL_RUNG,
            eval_seconds=434.0882288750727,
            val_bpb=1.8076412570831037,
            abs_error_vs_full=0.0,
        ),
    ),
    EvalCalibration(
        key="m5-xlarge_apple-m5-32gb-10gpu",
        label="m5-xlarge eval ladder on Apple M5 32GB / 10 GPU cores",
        hardware_key=DEFAULT_EVAL_HARDWARE_KEY,
        preset="m5-xlarge",
        seq_len=CANONICAL_EVAL_SEQ_LEN,
        batch_size=default_eval_batch_size(CANONICAL_EVAL_SEQ_LEN),
        source="Measured from a 120s m5-xlarge checkpoint on the reference M5 machine.",
        notes="Reference improves on cheap but remains too expensive for a short-run default.",
        cheap=_measurement(
            CHEAP_EVAL_RUNG,
            eval_seconds=6.545518999919295,
            val_bpb=1.9091981270870741,
            abs_error_vs_full=0.008205720233501933,
        ),
        reference=_measurement(
            REFERENCE_EVAL_RUNG,
            eval_seconds=38.03362387488596,
            val_bpb=1.9214453781594023,
            abs_error_vs_full=0.004041530838826211,
        ),
        full=_measurement(
            FULL_EVAL_RUNG,
            eval_seconds=475.16593666700646,
            val_bpb=1.9174038473205761,
            abs_error_vs_full=0.0,
        ),
    ),
)


def select_eval_calibration(
    preset: str,
    *,
    hardware_key: str = DEFAULT_EVAL_HARDWARE_KEY,
    calibrations: tuple[EvalCalibration, ...] = DEFAULT_EVAL_CALIBRATIONS,
) -> EvalCalibration:
    matches = tuple(
        calibration
        for calibration in calibrations
        if calibration.preset == preset and calibration.hardware_key == hardware_key
    )
    if not matches:
        raise ValueError(
            f"No eval calibration registered for preset={preset!r}, hardware_key={hardware_key!r}."
        )
    return matches[0]


def recommend_eval_rung(
    calibration: EvalCalibration,
    *,
    time_budget_sec: float,
    policy: EvalBudgetPolicy = DEFAULT_EVAL_BUDGET_POLICY,
) -> EvalRecommendation:
    if time_budget_sec <= 0.0:
        raise ValueError("time_budget_sec must be positive")
    full = calibration.full
    reference = calibration.reference
    cheap = calibration.cheap
    if full is None or reference is None or cheap is None:
        raise ValueError(f"Calibration {calibration.key} is missing one or more default rungs.")

    full_overhead = full.overhead_fraction(time_budget_sec)
    if full_overhead <= policy.full_overhead_cap_fraction:
        return EvalRecommendation(
            rung_key=full.spec.key,
            rung=full,
            projected_overhead_fraction=full_overhead,
            failed_more_expensive_rung_key=None,
        )

    reference_overhead = reference.overhead_fraction(time_budget_sec)
    if reference_overhead <= policy.reference_overhead_cap_fraction:
        return EvalRecommendation(
            rung_key=reference.spec.key,
            rung=reference,
            projected_overhead_fraction=reference_overhead,
            failed_more_expensive_rung_key=full.spec.key,
        )

    return EvalRecommendation(
        rung_key=cheap.spec.key,
        rung=cheap,
        projected_overhead_fraction=cheap.overhead_fraction(time_budget_sec),
        failed_more_expensive_rung_key=reference.spec.key,
    )


def choose_auto_eval_decision(
    preset: str,
    *,
    time_budget_sec: float,
    hardware_key: str = DEFAULT_EVAL_HARDWARE_KEY,
    calibrations: tuple[EvalCalibration, ...] = DEFAULT_EVAL_CALIBRATIONS,
    policy: EvalBudgetPolicy = DEFAULT_EVAL_BUDGET_POLICY,
) -> AutoEvalDecision:
    calibration = select_eval_calibration(
        preset,
        hardware_key=hardware_key,
        calibrations=calibrations,
    )
    recommendation = recommend_eval_rung(
        calibration,
        time_budget_sec=time_budget_sec,
        policy=policy,
    )
    return AutoEvalDecision(calibration=calibration, recommendation=recommendation)
