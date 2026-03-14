from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECTION_STD_FLOOR = 0.01
MONTE_CARLO_SAMPLES = 20000
MIN_PROJECTION_R2 = 0.85
MARGIN_SNR_THRESHOLD = 1.5
EXTRAPOLATION_BASE_DAMPING = 0.25
MAX_CONFIDENT_EXTRAPOLATION_RATIO = 1.25


def _infer_curve_engine(payload: dict) -> str | None:
    engine = payload.get("engine")
    if isinstance(engine, str) and engine:
        return engine
    hardware_key = payload.get("hardware_key")
    if isinstance(hardware_key, str):
        if hardware_key.startswith("nvidia-") or hardware_key.startswith("amd-"):
            return "cuda"
        if hardware_key.startswith("apple-"):
            return "mlx"
    accelerator_architecture = payload.get("accelerator_architecture")
    if isinstance(accelerator_architecture, str):
        arch = accelerator_architecture.lower()
        if any(token in arch for token in ("hopper", "blackwell", "ampere", "ada", "rubin")):
            return "cuda"
        if any(token in arch for token in ("apple", "m-series", "m1", "m2", "m3", "m4", "m5")):
            return "mlx"
    if payload.get("resolved_attention_backend") is not None:
        return "cuda"
    return None


@dataclass(frozen=True)
class CurvePoint:
    target_training_seconds: float
    actual_training_seconds: float
    step: int
    total_tokens: float
    val_bpb: float
    eval_seconds: float | None = None
    eval_seq_len: int | None = None
    eval_tokens: int | None = None
    eval_batch_size: int | None = None
    canonical_rung: str | None = None


@dataclass(frozen=True)
class CurveArtifact:
    engine: str | None
    preset: str
    hardware_key: str | None
    device_batch_size: int | None
    total_batch_size: int | None
    curve_points: tuple[CurvePoint, ...]
    final_eval: dict | None
    raw_payload: dict


@dataclass(frozen=True)
class CurveProjection:
    target_seconds: float
    projected_seconds: float
    projected_tokens: float
    projected_val_bpb: float
    method: str
    observed_point_count: int
    fit_r2: float | None = None
    fit_sigma: float | None = None


@dataclass(frozen=True)
class CurveSummary:
    engine: str | None
    preset: str
    hardware_key: str | None
    device_batch_size: int | None
    total_batch_size: int | None
    curve_points: int
    target_seconds: float
    observed_seconds: float | None
    observed_tokens: float | None
    projected_val_bpb: float | None
    projected_tokens: float | None
    projection_method: str | None
    projection_fit_r2: float | None
    projection_fit_sigma: float | None
    final_val_bpb: float | None
    final_eval_seconds: float | None
    final_training_seconds: float | None
    projected_margin_to_best: float | None = None
    final_margin_to_best: float | None = None


@dataclass(frozen=True)
class ProjectionCalibrationPoint:
    horizon_seconds: float
    horizon_tokens: float
    sample_count: int
    residual_mean: float
    residual_std: float
    residual_mad: float


@dataclass(frozen=True)
class ProjectionCalibration:
    target_seconds: float
    points: tuple[ProjectionCalibrationPoint, ...]


@dataclass(frozen=True)
class ProjectedCurveEstimate:
    summary: CurveSummary
    corrected_val_bpb: float
    projection_std: float
    fit_r2: float | None
    fit_sigma: float | None
    calibration_horizon_seconds: float | None
    calibration_horizon_tokens: float | None
    calibration_sample_count: int
    correction_mean: float
    projection_source: str = "generic-projection"
    matched_truth_count: int = 0
    winner_probability: float | None = None
    enough_signal: bool | None = None
    confidence_reason: str | None = None
    truth_anchor_seconds: float | None = None
    truth_anchor_tokens: float | None = None
    extrapolation_ratio: float | None = None


@dataclass(frozen=True)
class HorizonProjectionRow:
    target_seconds: float
    preset: str
    engine: str | None
    hardware_key: str | None
    device_batch_size: int | None
    total_batch_size: int | None
    observed_seconds: float | None
    observed_tokens: float | None
    target_tokens: float | None
    corrected_val_bpb: float
    correction_mean: float
    projection_std: float
    fit_r2: float | None
    fit_sigma: float | None
    interval_low: float
    interval_high: float
    winner_probability: float | None
    enough_signal: bool | None
    projection_source: str
    matched_truth_count: int
    truth_anchor_seconds: float | None
    truth_anchor_tokens: float | None
    extrapolation_ratio: float | None
    calibration_sample_count: int
    confidence_label: str
    confidence_reason: str


@dataclass(frozen=True)
class HorizonProjectionDecision:
    target_seconds: float
    top_preset: str | None
    top_projected_tokens: float | None
    top_winner_probability: float | None
    top_margin_to_second: float | None
    top_margin_snr: float | None
    enough_signal: bool
    confidence_reason: str | None = None


@dataclass(frozen=True)
class ProjectionDecision:
    target_seconds: float
    top_preset: str
    top_projected_tokens: float | None
    top_winner_probability: float
    top_margin_to_second: float | None
    top_margin_snr: float | None
    enough_signal: bool
    confidence_reason: str | None = None


@dataclass(frozen=True)
class MultiHorizonProjectionDiagnostics:
    preset: str
    short_observed_seconds: float | None
    long_observed_seconds: float | None
    short_observed_tokens: float | None
    long_observed_tokens: float | None
    short_projected_val_bpb: float | None
    long_projected_val_bpb: float
    short_fit_r2: float | None
    long_fit_r2: float | None
    short_fit_sigma: float | None
    long_fit_sigma: float | None
    fit_quality_min: float | None
    horizon_alpha: float | None
    effective_damping: float | None
    horizon_correction: float | None
    projection_delta: float | None
    projection_sigma: float | None
    projection_snr: float | None
    stability_gap: float | None
    stability_snr: float | None
    stable_projection: bool
    stability_reason: str


@dataclass(frozen=True)
class MultiHorizonProjectionDecision:
    target_seconds: float
    short_horizon_seconds: float | None
    long_horizon_seconds: float | None
    top_preset: str
    top_winner_probability: float
    top_margin_to_second: float | None
    top_margin_snr: float | None
    top_stability_gap: float | None
    top_stability_snr: float | None
    enough_signal: bool
    stability_reason: str | None = None
    confidence_reason: str | None = None


@dataclass(frozen=True)
class NextHorizonSuggestion:
    target_seconds: float
    current_observed_seconds: float | None
    current_observed_tokens: float | None
    suggested_seconds: float | None
    top_preset: str | None
    top_winner_probability: float | None
    target_enough_signal: bool
    target_confidence_reason: str | None
    suggestion_reason: str


@dataclass(frozen=True)
class ScalingCandidate:
    target_seconds: float
    scaling_target_seconds: float
    strict_winner_preset: str
    strict_winner_val_bpb: float
    strict_winner_device_batch_size: int | None
    strict_winner_total_batch_size: int | None
    candidate_preset: str
    candidate_val_bpb: float
    candidate_device_batch_size: int | None
    candidate_total_batch_size: int | None
    candidate_gap_at_target: float
    strict_projected_val_bpb: float | None
    candidate_projected_val_bpb: float | None
    projected_gap_at_scaling_target: float | None
    strict_last_segment_gain: float | None
    candidate_last_segment_gain: float | None
    source: str
    rationale: str


@dataclass(frozen=True)
class LongerHorizonProjection:
    target_seconds: float
    scaling_target_seconds: float
    target_winner_preset: str
    target_winner_val_bpb: float
    scaling_winner_preset: str
    scaling_winner_val_bpb: float
    scaling_winner_device_batch_size: int | None
    scaling_winner_total_batch_size: int | None
    scaling_winner_probability: float | None
    scaling_margin_to_second: float | None
    scaling_margin_snr: float | None
    scaling_enough_signal: bool
    scaling_confidence_label: str
    scaling_confidence_reason: str
    scaling_projection_source: str
    scaling_matched_truth_count: int
    crossover_from_target: bool
    target_gap_at_scaling: float | None
    scaling_gap_at_target: float | None
    target_winner_rank_at_scaling: int | None
    scaling_winner_rank_at_target: int | None


def final_training_seconds(curve: CurveArtifact) -> float | None:
    final_eval = curve.final_eval or {}
    if final_eval.get("training_seconds") is not None:
        return float(final_eval["training_seconds"])
    if curve.curve_points:
        return float(curve.curve_points[-1].actual_training_seconds)
    raw_total = curve.raw_payload.get("curve_eval_total_seconds")
    if raw_total is not None:
        return float(raw_total)
    return None


def final_val_bpb(curve: CurveArtifact) -> float | None:
    final_eval = curve.final_eval or {}
    if final_eval.get("val_bpb") is not None:
        return float(final_eval["val_bpb"])
    if curve.curve_points:
        return float(curve.curve_points[-1].val_bpb)
    return None


def load_curve_artifact(path: Path) -> CurveArtifact:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "curve_points" not in payload:
        raise ValueError(f"{path} is not a valid curve artifact.")
    points = tuple(
        CurvePoint(
            target_training_seconds=float(item["target_training_seconds"]),
            actual_training_seconds=float(item["actual_training_seconds"]),
            step=int(item["step"]),
            total_tokens=float(item["total_tokens"]),
            val_bpb=float(item["val_bpb"]),
            eval_seconds=float(item["eval_seconds"]) if item.get("eval_seconds") is not None else None,
            eval_seq_len=int(item["eval_seq_len"]) if item.get("eval_seq_len") is not None else None,
            eval_tokens=int(item["eval_tokens"]) if item.get("eval_tokens") is not None else None,
            eval_batch_size=int(item["eval_batch_size"]) if item.get("eval_batch_size") is not None else None,
            canonical_rung=item.get("canonical_rung"),
        )
        for item in payload["curve_points"]
    )
    points = tuple(sorted(points, key=lambda item: item.actual_training_seconds))
    return CurveArtifact(
        engine=_infer_curve_engine(payload),
        preset=str(payload.get("preset")),
        hardware_key=payload.get("hardware_key"),
        device_batch_size=int(payload["device_batch_size"]) if payload.get("device_batch_size") is not None else None,
        total_batch_size=int(payload["total_batch_size"]) if payload.get("total_batch_size") is not None else None,
        curve_points=points,
        final_eval=payload.get("final_eval"),
        raw_payload=payload,
    )


def load_curve_artifacts(paths: Iterable[Path]) -> list[CurveArtifact]:
    return [load_curve_artifact(path) for path in paths]


def load_curve_artifacts_from_dir(
    directory: Path,
    *,
    engine: str | None = None,
    hardware_key: str | None = None,
    require_target_seconds: float | None = None,
) -> list[CurveArtifact]:
    curves: list[CurveArtifact] = []
    for path in sorted(directory.glob("*.json")):
        try:
            curve = load_curve_artifact(path)
        except Exception:
            continue
        if engine is not None and curve.engine != engine:
            continue
        if hardware_key is not None and curve.hardware_key != hardware_key:
            continue
        horizon_seconds = final_training_seconds(curve)
        if require_target_seconds is not None and (
            horizon_seconds is None or float(horizon_seconds) < require_target_seconds
        ):
            continue
        curves.append(curve)
    return curves


def _interpolate_seconds(points: tuple[CurvePoint, ...], target_seconds: float) -> CurvePoint | None:
    if not points:
        return None
    if target_seconds <= points[0].actual_training_seconds:
        return points[0]
    if target_seconds >= points[-1].actual_training_seconds:
        return points[-1]
    for left, right in zip(points, points[1:]):
        if left.actual_training_seconds <= target_seconds <= right.actual_training_seconds:
            if math.isclose(left.actual_training_seconds, right.actual_training_seconds):
                return right
            alpha = (target_seconds - left.actual_training_seconds) / (
                right.actual_training_seconds - left.actual_training_seconds
            )
            return CurvePoint(
                target_training_seconds=target_seconds,
                actual_training_seconds=target_seconds,
                step=round(left.step + alpha * (right.step - left.step)),
                total_tokens=left.total_tokens + alpha * (right.total_tokens - left.total_tokens),
                val_bpb=left.val_bpb + alpha * (right.val_bpb - left.val_bpb),
                eval_seconds=(
                    None
                    if left.eval_seconds is None or right.eval_seconds is None
                    else left.eval_seconds + alpha * (right.eval_seconds - left.eval_seconds)
                ),
                eval_seq_len=right.eval_seq_len if right.eval_seq_len is not None else left.eval_seq_len,
                eval_tokens=right.eval_tokens if right.eval_tokens is not None else left.eval_tokens,
                eval_batch_size=right.eval_batch_size if right.eval_batch_size is not None else left.eval_batch_size,
                canonical_rung=right.canonical_rung or left.canonical_rung,
            )
    return points[-1]


def truncate_curve(curve: CurveArtifact, *, horizon_seconds: float) -> CurveArtifact | None:
    points = tuple(point for point in curve.curve_points if point.actual_training_seconds <= horizon_seconds + 1e-9)
    interpolated = _interpolate_seconds(curve.curve_points, horizon_seconds)
    if interpolated is not None:
        if not points or not math.isclose(points[-1].actual_training_seconds, interpolated.actual_training_seconds):
            points = points + (interpolated,)
    if not points:
        return None
    payload = dict(curve.raw_payload)
    payload["curve_points"] = [
        {
            "target_training_seconds": point.target_training_seconds,
            "actual_training_seconds": point.actual_training_seconds,
            "step": point.step,
            "total_tokens": point.total_tokens,
            "val_bpb": point.val_bpb,
            "eval_seconds": point.eval_seconds,
            "eval_seq_len": point.eval_seq_len,
            "eval_tokens": point.eval_tokens,
            "eval_batch_size": point.eval_batch_size,
            "canonical_rung": point.canonical_rung,
        }
        for point in points
    ]
    return CurveArtifact(
        engine=curve.engine,
        preset=curve.preset,
        hardware_key=curve.hardware_key,
        device_batch_size=curve.device_batch_size,
        total_batch_size=curve.total_batch_size,
        curve_points=points,
        final_eval=curve.final_eval,
        raw_payload=payload,
    )


def _estimate_tokens_at_seconds(points: tuple[CurvePoint, ...], target_seconds: float) -> float | None:
    interpolated = _interpolate_seconds(points, target_seconds)
    if interpolated is not None and target_seconds <= points[-1].actual_training_seconds:
        return interpolated.total_tokens
    if not points:
        return None
    if len(points) == 1:
        rate = points[0].total_tokens / max(points[0].actual_training_seconds, 1e-9)
        return points[0].total_tokens + max(0.0, target_seconds - points[0].actual_training_seconds) * rate
    intervals = []
    for left, right in zip(points[1:], points[2:]):
        delta_t = right.actual_training_seconds - left.actual_training_seconds
        if delta_t <= 0:
            continue
        intervals.append((right.total_tokens - left.total_tokens) / delta_t)
    if not intervals:
        left, right = points[-2], points[-1]
        delta_t = max(right.actual_training_seconds - left.actual_training_seconds, 1e-9)
        rate = (right.total_tokens - left.total_tokens) / delta_t
    else:
        rate = statistics.median(intervals)
    return points[-1].total_tokens + max(0.0, target_seconds - points[-1].actual_training_seconds) * rate


def _fit_log_token_projection(points: tuple[CurvePoint, ...], target_tokens: float) -> tuple[float, float | None, float | None]:
    fit_points = list(points[1:] if len(points) > 2 else points)
    xs = [math.log(max(point.total_tokens, 1.0)) for point in fit_points]
    ys = [point.val_bpb for point in fit_points]
    if len(fit_points) == 1 or math.isclose(max(xs), min(xs)):
        return fit_points[-1].val_bpb, None, None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(denominator, 0.0):
        return fit_points[-1].val_bpb, None, None
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    residuals = [y - yhat for y, yhat in zip(ys, fitted)]
    ss_res = sum(residual * residual for residual in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    fit_r2 = 1.0 if math.isclose(ss_tot, 0.0) else max(0.0, 1.0 - (ss_res / ss_tot))
    fit_sigma = max(PROJECTION_STD_FLOOR, math.sqrt(ss_res / max(len(residuals) - 1, 1)))
    projected = intercept + slope * math.log(max(target_tokens, 1.0))
    last_observed = fit_points[-1].val_bpb
    return min(last_observed, projected), fit_r2, fit_sigma


def project_curve(curve: CurveArtifact, *, target_seconds: float) -> CurveProjection | None:
    points = curve.curve_points
    if not points:
        return None
    if target_seconds <= points[-1].actual_training_seconds:
        interpolated = _interpolate_seconds(points, target_seconds)
        if interpolated is None:
            return None
        return CurveProjection(
            target_seconds=target_seconds,
            projected_seconds=target_seconds,
            projected_tokens=interpolated.total_tokens,
            projected_val_bpb=interpolated.val_bpb,
            method="interpolate-seconds",
            observed_point_count=len(points),
        )

    projected_tokens = _estimate_tokens_at_seconds(points, target_seconds)
    if projected_tokens is None:
        return None
    projected_val_bpb, fit_r2, fit_sigma = _fit_log_token_projection(points, projected_tokens)
    return CurveProjection(
        target_seconds=target_seconds,
        projected_seconds=target_seconds,
        projected_tokens=projected_tokens,
        projected_val_bpb=projected_val_bpb,
        method="log-token-fit",
        observed_point_count=len(points),
        fit_r2=fit_r2,
        fit_sigma=fit_sigma,
    )


def summarize_curve(curve: CurveArtifact, *, target_seconds: float) -> dict:
    projection = project_curve(curve, target_seconds=target_seconds)
    final_eval = curve.final_eval or {}
    return {
        "engine": curve.engine,
        "preset": curve.preset,
        "hardware_key": curve.hardware_key,
        "device_batch_size": curve.device_batch_size,
        "total_batch_size": curve.total_batch_size,
        "curve_points": len(curve.curve_points),
        "target_seconds": target_seconds,
        "observed_seconds": curve.curve_points[-1].actual_training_seconds if curve.curve_points else None,
        "observed_tokens": curve.curve_points[-1].total_tokens if curve.curve_points else None,
        "projected_val_bpb": projection.projected_val_bpb if projection else None,
        "projected_tokens": projection.projected_tokens if projection else None,
        "projection_method": projection.method if projection else None,
        "projection_fit_r2": projection.fit_r2 if projection else None,
        "projection_fit_sigma": projection.fit_sigma if projection else None,
        "final_val_bpb": final_eval.get("val_bpb"),
        "final_eval_seconds": final_eval.get("eval_seconds"),
        "final_training_seconds": final_training_seconds(curve),
    }


def summarize_curve_artifact(curve: CurveArtifact, *, target_seconds: float) -> CurveSummary:
    summary = summarize_curve(curve, target_seconds=target_seconds)
    return CurveSummary(**summary)


def summarize_curves(curves: list[CurveArtifact], *, target_seconds: float) -> list[CurveSummary]:
    rows = [summarize_curve_artifact(curve, target_seconds=target_seconds) for curve in curves]
    rows.sort(key=lambda row: (row.projected_val_bpb is None, row.projected_val_bpb))

    valid_projected = [row.projected_val_bpb for row in rows if row.projected_val_bpb is not None]
    best_projected = min(valid_projected) if valid_projected else None
    valid_final = [row.final_val_bpb for row in rows if row.final_val_bpb is not None]
    best_final = min(valid_final) if valid_final else None

    enriched: list[CurveSummary] = []
    for row in rows:
        projected_margin = None
        if best_projected is not None and row.projected_val_bpb is not None:
            projected_margin = row.projected_val_bpb - best_projected
        final_margin = None
        if best_final is not None and row.final_val_bpb is not None:
            final_margin = row.final_val_bpb - best_final
        enriched.append(
            CurveSummary(
                engine=row.engine,
                preset=row.preset,
                hardware_key=row.hardware_key,
                device_batch_size=row.device_batch_size,
                total_batch_size=row.total_batch_size,
                curve_points=row.curve_points,
                target_seconds=row.target_seconds,
                observed_seconds=row.observed_seconds,
                observed_tokens=row.observed_tokens,
                projected_val_bpb=row.projected_val_bpb,
                projected_tokens=row.projected_tokens,
                projection_method=row.projection_method,
                projection_fit_r2=row.projection_fit_r2,
                projection_fit_sigma=row.projection_fit_sigma,
                final_val_bpb=row.final_val_bpb,
                final_eval_seconds=row.final_eval_seconds,
                final_training_seconds=row.final_training_seconds,
                projected_margin_to_best=projected_margin,
                final_margin_to_best=final_margin,
            )
        )
    return enriched


def _robust_std(values: list[float]) -> float:
    if len(values) <= 1:
        return PROJECTION_STD_FLOOR
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        stdev = 0.0
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    mad_std = 1.4826 * mad
    return max(PROJECTION_STD_FLOOR, stdev, mad_std)


def build_projection_calibration(
    truth_curves: list[CurveArtifact],
    *,
    target_seconds: float,
) -> ProjectionCalibration:
    residuals_by_horizon: dict[tuple[float, float], list[float]] = {}
    for curve in truth_curves:
        final_eval = curve.final_eval or {}
        target_horizon = final_training_seconds(curve)
        final_val_bpb = final_eval.get("val_bpb")
        if target_horizon is None or final_val_bpb is None:
            continue
        if float(target_horizon) < target_seconds:
            continue
        horizons = sorted(
            {
                point.actual_training_seconds
                for point in curve.curve_points
                if point.actual_training_seconds < target_seconds - 1e-9
            }
        )
        for horizon_seconds in horizons:
            truncated = truncate_curve(curve, horizon_seconds=horizon_seconds)
            if truncated is None:
                continue
            horizon_tokens = truncated.curve_points[-1].total_tokens if truncated.curve_points else None
            if horizon_tokens is None:
                continue
            projection = project_curve(truncated, target_seconds=target_seconds)
            if projection is None:
                continue
            residual = float(final_val_bpb) - projection.projected_val_bpb
            residuals_by_horizon.setdefault((float(horizon_seconds), float(horizon_tokens)), []).append(residual)

    points: list[ProjectionCalibrationPoint] = []
    for horizon_seconds, horizon_tokens in sorted(residuals_by_horizon):
        residuals = residuals_by_horizon[(horizon_seconds, horizon_tokens)]
        points.append(
            ProjectionCalibrationPoint(
                horizon_seconds=horizon_seconds,
                horizon_tokens=horizon_tokens,
                sample_count=len(residuals),
                residual_mean=statistics.fmean(residuals),
                residual_std=_robust_std(residuals),
                residual_mad=statistics.median(abs(value - statistics.median(residuals)) for value in residuals),
            )
        )
    return ProjectionCalibration(target_seconds=target_seconds, points=tuple(points))


def nearest_calibration_point(
    calibration: ProjectionCalibration,
    *,
    observed_tokens: float,
    observed_seconds: float | None = None,
) -> ProjectionCalibrationPoint | None:
    if not calibration.points:
        return None
    return min(
        calibration.points,
        key=lambda point: (
            abs(point.horizon_tokens - observed_tokens),
            abs(point.horizon_seconds - observed_seconds) if observed_seconds is not None else 0.0,
            -point.sample_count,
        ),
    )


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return statistics.fmean(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_std(values: list[float], weights: list[float], *, center: float) -> float:
    total = sum(weights)
    if total <= 0 or len(values) <= 1:
        return PROJECTION_STD_FLOOR
    variance = sum(weight * ((value - center) ** 2) for value, weight in zip(values, weights)) / total
    return max(PROJECTION_STD_FLOOR, math.sqrt(max(0.0, variance)))


def _curve_distance_to_truth(curve: CurveArtifact, truth_curve: CurveArtifact) -> float:
    if not curve.curve_points or not truth_curve.curve_points:
        return float("inf")
    horizon_seconds = curve.curve_points[-1].actual_training_seconds
    truth_partial = truncate_curve(truth_curve, horizon_seconds=horizon_seconds)
    if truth_partial is None or not truth_partial.curve_points:
        return float("inf")
    errors: list[float] = []
    for point in curve.curve_points:
        truth_point = _interpolate_seconds(truth_partial.curve_points, point.actual_training_seconds)
        if truth_point is None:
            continue
        errors.append(abs(point.val_bpb - truth_point.val_bpb))
    if not errors:
        return float("inf")
    distance = statistics.fmean(errors)
    if (
        curve.device_batch_size is not None
        and truth_curve.device_batch_size is not None
        and curve.device_batch_size != truth_curve.device_batch_size
    ):
        distance += 0.01 * abs(curve.device_batch_size - truth_curve.device_batch_size) / max(curve.device_batch_size, 1)
    if (
        curve.total_batch_size is not None
        and truth_curve.total_batch_size is not None
        and curve.total_batch_size != truth_curve.total_batch_size
    ):
        distance += 0.01 * abs(curve.total_batch_size - truth_curve.total_batch_size) / max(curve.total_batch_size, 1)
    return distance


def _best_truth_curve_by_preset(
    truth_curves: list[CurveArtifact],
    *,
    engine: str | None,
    hardware_key: str | None,
    target_seconds: float,
) -> dict[str, CurveArtifact]:
    best: dict[str, CurveArtifact] = {}
    for curve in truth_curves:
        if engine is not None and curve.engine is not None and curve.engine != engine:
            continue
        if hardware_key is not None and curve.hardware_key is not None and curve.hardware_key != hardware_key:
            continue
        horizon_seconds = final_training_seconds(curve)
        if horizon_seconds is None or horizon_seconds < target_seconds:
            continue
        final_bpb = final_val_bpb(curve)
        if final_bpb is None:
            continue
        current = best.get(curve.preset)
        if current is None:
            best[curve.preset] = curve
            continue
        current_bpb = final_val_bpb(current)
        if current_bpb is None or final_bpb < current_bpb:
            best[curve.preset] = curve
    return best


def _last_segment_gain(curve: CurveArtifact, *, target_seconds: float) -> float | None:
    points = tuple(point for point in curve.curve_points if point.actual_training_seconds <= target_seconds + 1e-9)
    if len(points) < 2:
        return None
    left, right = points[-2], points[-1]
    return left.val_bpb - right.val_bpb


def estimate_from_matching_truth(
    curve: CurveArtifact,
    *,
    truth_curves: list[CurveArtifact],
    target_seconds: float,
) -> tuple[float, float, int] | None:
    finals: list[float] = []
    distances: list[float] = []
    for truth_curve in truth_curves:
        if truth_curve.preset != curve.preset:
            continue
        if curve.engine is not None and truth_curve.engine is not None and truth_curve.engine != curve.engine:
            continue
        if curve.hardware_key is not None and truth_curve.hardware_key is not None and truth_curve.hardware_key != curve.hardware_key:
            continue
        final_eval = truth_curve.final_eval or {}
        final_val_bpb = final_eval.get("val_bpb")
        if final_val_bpb is None:
            continue
        truth_horizon = final_training_seconds(truth_curve)
        if truth_horizon is None or truth_horizon < target_seconds:
            continue
        distance = _curve_distance_to_truth(curve, truth_curve)
        if not math.isfinite(distance):
            continue
        finals.append(float(final_val_bpb))
        distances.append(distance)
    if not finals:
        return None
    weights = [math.exp(-distance / 0.05) for distance in distances]
    center = _weighted_mean(finals, weights)
    std = _weighted_std(finals, weights, center=center)
    if len(finals) == 1:
        std = max(std, 0.01 + distances[0])
    return center, std, len(finals)


def estimate_from_truth_anchor(
    curve: CurveArtifact,
    *,
    truth_curves: list[CurveArtifact],
    target_seconds: float,
) -> tuple[float, float, int, float, float | None, float | None] | None:
    matching: list[CurveArtifact] = []
    for truth_curve in truth_curves:
        if truth_curve.preset != curve.preset:
            continue
        if curve.engine is not None and truth_curve.engine is not None and truth_curve.engine != curve.engine:
            continue
        if curve.hardware_key is not None and truth_curve.hardware_key is not None and truth_curve.hardware_key != curve.hardware_key:
            continue
        truth_horizon = final_training_seconds(truth_curve)
        if truth_horizon is None:
            continue
        matching.append(truth_curve)
    if not matching:
        return None

    anchor_seconds = max(final_training_seconds(item) or 0.0 for item in matching)
    anchor_truths = [item for item in matching if math.isclose(final_training_seconds(item) or 0.0, anchor_seconds)]
    anchor_finals = [final_val_bpb(item) for item in anchor_truths]
    anchor_finals = [float(item) for item in anchor_finals if item is not None]
    if not anchor_finals:
        return None

    observed_anchor_projection = project_curve(curve, target_seconds=anchor_seconds)
    target_projection = project_curve(curve, target_seconds=target_seconds)
    if observed_anchor_projection is None or target_projection is None:
        return None

    anchor_center = statistics.fmean(anchor_finals)
    anchor_std = _robust_std(anchor_finals)
    anchor_tokens = observed_anchor_projection.projected_tokens
    target_tokens = target_projection.projected_tokens
    if anchor_tokens is None or target_tokens is None or anchor_tokens <= 0 or target_tokens <= 0:
        return None

    raw_delta = target_projection.projected_val_bpb - observed_anchor_projection.projected_val_bpb
    extrapolation_ratio = target_tokens / anchor_tokens
    fit_quality = target_projection.fit_r2 if target_projection.fit_r2 is not None else 0.5
    damping = EXTRAPOLATION_BASE_DAMPING * max(0.25, min(1.0, fit_quality))
    damping /= max(1.0, math.sqrt(extrapolation_ratio))
    corrected = anchor_center + damping * raw_delta
    std = max(
        anchor_std,
        target_projection.fit_sigma or PROJECTION_STD_FLOOR,
        PROJECTION_STD_FLOOR * extrapolation_ratio,
    )
    return corrected, std, len(anchor_finals), anchor_seconds, anchor_tokens, extrapolation_ratio


def select_scaling_candidate(
    estimates: list[ProjectedCurveEstimate],
    *,
    truth_curves: list[CurveArtifact],
    preset_order: list[str],
    target_seconds: float,
    scaling_target_seconds: float,
    max_gap_at_target: float = 0.01,
) -> ScalingCandidate | None:
    if not estimates or not preset_order:
        return None

    strict_winner = estimates[0]
    try:
        strict_index = preset_order.index(strict_winner.summary.preset)
    except ValueError:
        return None

    truth_by_preset = _best_truth_curve_by_preset(
        truth_curves,
        engine=strict_winner.summary.engine,
        hardware_key=strict_winner.summary.hardware_key,
        target_seconds=target_seconds,
    )
    strict_truth = truth_by_preset.get(strict_winner.summary.preset)
    if strict_truth is None:
        return None

    strict_long = project_curve(strict_truth, target_seconds=scaling_target_seconds)
    strict_last_gain = _last_segment_gain(strict_truth, target_seconds=target_seconds)
    estimates_by_preset = {item.summary.preset: item for item in estimates}

    best_candidate: ScalingCandidate | None = None
    best_key: tuple[float, float, float] | None = None
    for preset in preset_order[strict_index + 1 :]:
        candidate = estimates_by_preset.get(preset)
        if candidate is None:
            continue
        candidate_gap = candidate.corrected_val_bpb - strict_winner.corrected_val_bpb
        if candidate_gap > max_gap_at_target:
            continue
        candidate_truth = truth_by_preset.get(preset)
        if candidate_truth is None:
            continue
        candidate_long = project_curve(candidate_truth, target_seconds=scaling_target_seconds)
        if candidate_long is None or strict_long is None:
            continue
        projected_gap = candidate_long.projected_val_bpb - strict_long.projected_val_bpb
        candidate_last_gain = _last_segment_gain(candidate_truth, target_seconds=target_seconds)
        if projected_gap > candidate_gap and projected_gap > max_gap_at_target:
            continue
        candidate_record = ScalingCandidate(
            target_seconds=target_seconds,
            scaling_target_seconds=scaling_target_seconds,
            strict_winner_preset=strict_winner.summary.preset,
            strict_winner_val_bpb=strict_winner.corrected_val_bpb,
            strict_winner_device_batch_size=strict_truth.device_batch_size,
            strict_winner_total_batch_size=strict_truth.total_batch_size,
            candidate_preset=preset,
            candidate_val_bpb=candidate.corrected_val_bpb,
            candidate_device_batch_size=candidate_truth.device_batch_size,
            candidate_total_batch_size=candidate_truth.total_batch_size,
            candidate_gap_at_target=candidate_gap,
            strict_projected_val_bpb=strict_long.projected_val_bpb,
            candidate_projected_val_bpb=candidate_long.projected_val_bpb,
            projected_gap_at_scaling_target=projected_gap,
            strict_last_segment_gain=strict_last_gain,
            candidate_last_segment_gain=candidate_last_gain,
            source="truth-curves",
            rationale="larger-near-frontier",
        )
        improvement_advantage = (candidate_last_gain or 0.0) - (strict_last_gain or 0.0)
        ranking_key = (
            projected_gap,
            candidate_gap,
            -improvement_advantage,
        )
        if best_key is None or ranking_key < best_key:
            best_key = ranking_key
            best_candidate = candidate_record

    return best_candidate


def estimate_projected_curve(
    curve: CurveArtifact,
    *,
    target_seconds: float,
    calibration: ProjectionCalibration | None = None,
    truth_curves: list[CurveArtifact] | None = None,
) -> ProjectedCurveEstimate:
    summary = summarize_curve_artifact(curve, target_seconds=target_seconds)
    if summary.projected_val_bpb is None:
        return ProjectedCurveEstimate(
            summary=summary,
            corrected_val_bpb=float("inf"),
            projection_std=float("inf"),
            fit_r2=summary.projection_fit_r2,
            fit_sigma=summary.projection_fit_sigma,
            calibration_horizon_seconds=None,
            calibration_horizon_tokens=None,
            calibration_sample_count=0,
            correction_mean=0.0,
        )
    truth_estimate = estimate_from_matching_truth(
        curve,
        truth_curves=truth_curves or [],
        target_seconds=target_seconds,
    )
    if truth_estimate is not None:
        corrected_val_bpb, projection_std, matched_truth_count = truth_estimate
        return ProjectedCurveEstimate(
            summary=summary,
            corrected_val_bpb=corrected_val_bpb,
            projection_std=projection_std,
            fit_r2=summary.projection_fit_r2,
            fit_sigma=summary.projection_fit_sigma,
            calibration_horizon_seconds=curve.curve_points[-1].actual_training_seconds if curve.curve_points else None,
            calibration_horizon_tokens=curve.curve_points[-1].total_tokens if curve.curve_points else None,
            calibration_sample_count=matched_truth_count,
            correction_mean=corrected_val_bpb - summary.projected_val_bpb,
            projection_source="truth-match",
            matched_truth_count=matched_truth_count,
        )
    truth_anchor_estimate = estimate_from_truth_anchor(
        curve,
        truth_curves=truth_curves or [],
        target_seconds=target_seconds,
    )
    if truth_anchor_estimate is not None:
        (
            corrected_val_bpb,
            projection_std,
            matched_truth_count,
            truth_anchor_seconds,
            truth_anchor_tokens,
            extrapolation_ratio,
        ) = truth_anchor_estimate
        return ProjectedCurveEstimate(
            summary=summary,
            corrected_val_bpb=corrected_val_bpb,
            projection_std=projection_std,
            fit_r2=summary.projection_fit_r2,
            fit_sigma=summary.projection_fit_sigma,
            calibration_horizon_seconds=curve.curve_points[-1].actual_training_seconds if curve.curve_points else None,
            calibration_horizon_tokens=curve.curve_points[-1].total_tokens if curve.curve_points else None,
            calibration_sample_count=matched_truth_count,
            correction_mean=corrected_val_bpb - summary.projected_val_bpb,
            projection_source="calibrated-extrapolation",
            matched_truth_count=matched_truth_count,
            confidence_reason="extrapolative",
            truth_anchor_seconds=truth_anchor_seconds,
            truth_anchor_tokens=truth_anchor_tokens,
            extrapolation_ratio=extrapolation_ratio,
        )
    observed_seconds = curve.curve_points[-1].actual_training_seconds if curve.curve_points else 0.0
    observed_tokens = curve.curve_points[-1].total_tokens if curve.curve_points else 0.0
    calibration_point = (
        nearest_calibration_point(
            calibration,
            observed_tokens=observed_tokens,
            observed_seconds=observed_seconds,
        )
        if calibration is not None
        else None
    )
    if calibration_point is None:
        corrected_val_bpb = summary.projected_val_bpb
        projection_std = PROJECTION_STD_FLOOR
        calibration_horizon_seconds = None
        calibration_horizon_tokens = None
        calibration_sample_count = 0
        correction_mean = 0.0
        projection_source = "generic-projection"
        matched_truth_count = 0
    else:
        corrected_val_bpb = summary.projected_val_bpb + calibration_point.residual_mean
        projection_std = calibration_point.residual_std
        calibration_horizon_seconds = calibration_point.horizon_seconds
        calibration_horizon_tokens = calibration_point.horizon_tokens
        calibration_sample_count = calibration_point.sample_count
        correction_mean = calibration_point.residual_mean
        projection_source = "calibrated-projection"
        matched_truth_count = 0
    return ProjectedCurveEstimate(
        summary=summary,
        corrected_val_bpb=corrected_val_bpb,
        projection_std=projection_std,
        fit_r2=summary.projection_fit_r2,
        fit_sigma=summary.projection_fit_sigma,
        calibration_horizon_seconds=calibration_horizon_seconds,
        calibration_horizon_tokens=calibration_horizon_tokens,
        calibration_sample_count=calibration_sample_count,
        correction_mean=correction_mean,
        projection_source=projection_source,
        matched_truth_count=matched_truth_count,
    )


def compare_projected_curves(
    curves: list[CurveArtifact],
    *,
    target_seconds: float,
    calibration: ProjectionCalibration | None = None,
    truth_curves: list[CurveArtifact] | None = None,
    winner_probability_threshold: float = 0.9,
    projected_margin_threshold: float = 0.01,
    monte_carlo_samples: int = MONTE_CARLO_SAMPLES,
) -> tuple[list[ProjectedCurveEstimate], ProjectionDecision | None]:
    estimates = [
        estimate_projected_curve(
            curve,
            target_seconds=target_seconds,
            calibration=calibration,
            truth_curves=truth_curves,
        )
        for curve in curves
    ]
    estimates.sort(key=lambda item: item.corrected_val_bpb)
    valid = [item for item in estimates if math.isfinite(item.corrected_val_bpb) and math.isfinite(item.projection_std)]
    if not valid:
        return estimates, None

    rng = random.Random(0)
    wins = {item.summary.preset: 0 for item in valid}
    for _ in range(monte_carlo_samples):
        samples = [
            (
                rng.gauss(item.corrected_val_bpb, item.projection_std),
                item.summary.preset,
            )
            for item in valid
        ]
        samples.sort()
        wins[samples[0][1]] += 1
    win_probabilities = {
        preset: count / monte_carlo_samples for preset, count in wins.items()
    }
    enriched: list[ProjectedCurveEstimate] = []
    for item in estimates:
        if item.summary.preset in win_probabilities:
            enriched.append(
                ProjectedCurveEstimate(
                    summary=item.summary,
                    corrected_val_bpb=item.corrected_val_bpb,
                    projection_std=item.projection_std,
                    fit_r2=item.fit_r2,
                    fit_sigma=item.fit_sigma,
                    calibration_horizon_seconds=item.calibration_horizon_seconds,
                    calibration_horizon_tokens=item.calibration_horizon_tokens,
                    calibration_sample_count=item.calibration_sample_count,
                    correction_mean=item.correction_mean,
                    projection_source=item.projection_source,
                    matched_truth_count=item.matched_truth_count,
                    winner_probability=win_probabilities[item.summary.preset],
                    enough_signal=item.enough_signal,
                    confidence_reason=item.confidence_reason,
                    truth_anchor_seconds=item.truth_anchor_seconds,
                    truth_anchor_tokens=item.truth_anchor_tokens,
                    extrapolation_ratio=item.extrapolation_ratio,
                )
            )
        else:
            enriched.append(item)
    estimates = enriched
    estimates.sort(key=lambda item: item.corrected_val_bpb)
    top = estimates[0]
    second = estimates[1] if len(estimates) > 1 else None
    top_margin = None
    top_margin_snr = None
    if second is not None:
        top_margin = second.corrected_val_bpb - top.corrected_val_bpb
        pooled_std = max(
            PROJECTION_STD_FLOOR,
            math.sqrt(top.projection_std**2 + second.projection_std**2),
        )
        top_margin_snr = top_margin / pooled_std
    fit_quality_ok = (
        top.projection_source == "truth-match"
        or top.fit_r2 is None
        or top.fit_r2 >= MIN_PROJECTION_R2
    )
    margin_snr_ok = top_margin_snr is None or top_margin_snr >= MARGIN_SNR_THRESHOLD
    extrapolative = top.projection_source in {"calibrated-extrapolation", "generic-projection"} and (
        top.extrapolation_ratio is not None and top.extrapolation_ratio > MAX_CONFIDENT_EXTRAPOLATION_RATIO
    )
    if top.projection_source == "truth-match":
        confidence_reason = "truth-backed" if top.matched_truth_count >= 2 else "single-truth-match"
    elif top.projection_source == "calibrated-extrapolation":
        confidence_reason = "extrapolative"
    elif not fit_quality_ok:
        confidence_reason = "low_r2"
    elif not margin_snr_ok:
        confidence_reason = "insufficient_snr"
    elif top.projection_source == "calibrated-projection" and top.calibration_sample_count < 5:
        confidence_reason = "limited_calibration"
    else:
        confidence_reason = "projected"
    enough_signal = (
        (top.winner_probability or 0.0) >= winner_probability_threshold
        and (top_margin is None or top_margin >= projected_margin_threshold)
        and fit_quality_ok
        and margin_snr_ok
        and not extrapolative
    )
    decision = ProjectionDecision(
        target_seconds=target_seconds,
        top_preset=top.summary.preset,
        top_projected_tokens=top.summary.projected_tokens,
        top_winner_probability=top.winner_probability or 0.0,
        top_margin_to_second=top_margin,
        top_margin_snr=top_margin_snr,
        enough_signal=enough_signal,
        confidence_reason=confidence_reason,
    )
    enriched = []
    for item in estimates:
        enriched.append(
            ProjectedCurveEstimate(
                summary=item.summary,
                corrected_val_bpb=item.corrected_val_bpb,
                projection_std=item.projection_std,
                fit_r2=item.fit_r2,
                fit_sigma=item.fit_sigma,
                calibration_horizon_seconds=item.calibration_horizon_seconds,
                calibration_horizon_tokens=item.calibration_horizon_tokens,
                calibration_sample_count=item.calibration_sample_count,
                correction_mean=item.correction_mean,
                projection_source=item.projection_source,
                matched_truth_count=item.matched_truth_count,
                winner_probability=item.winner_probability,
                enough_signal=enough_signal if item.summary.preset == top.summary.preset else False,
                confidence_reason=confidence_reason if item.summary.preset == top.summary.preset else None,
                truth_anchor_seconds=item.truth_anchor_seconds,
                truth_anchor_tokens=item.truth_anchor_tokens,
                extrapolation_ratio=item.extrapolation_ratio,
            )
        )
    return enriched, decision


def projection_confidence_label(estimate: ProjectedCurveEstimate) -> str:
    if estimate.confidence_reason == "extrapolative":
        return "low"
    if estimate.confidence_reason == "low_r2":
        return "low"
    if estimate.confidence_reason == "insufficient_snr":
        return "low"
    if estimate.projection_source == "truth-match":
        return "high" if estimate.matched_truth_count >= 2 else "medium"
    if estimate.projection_source == "calibrated-projection":
        if estimate.calibration_sample_count >= 5 and estimate.projection_std <= 0.02:
            return "medium"
        return "low"
    return "low"


def classify_multi_horizon_reason(
    short_item: ProjectedCurveEstimate | None,
    long_item: ProjectedCurveEstimate,
    *,
    horizon_alpha: float | None,
    projection_snr: float | None,
    stability_snr: float | None,
    stable_projection: bool,
) -> str:
    short_fit_r2 = short_item.fit_r2 if short_item is not None else None
    long_fit_r2 = long_item.fit_r2
    fit_values = [value for value in (short_fit_r2, long_fit_r2) if value is not None]
    if fit_values and min(fit_values) < MIN_PROJECTION_R2:
        return "low_r2"
    if horizon_alpha is None:
        return "single_horizon"
    if abs(horizon_alpha) < 0.02:
        return "no_drift"
    if projection_snr is not None and projection_snr < MARGIN_SNR_THRESHOLD:
        return "insufficient_projection_snr"
    if stability_snr is not None and stability_snr < MARGIN_SNR_THRESHOLD:
        return "insufficient_snr"
    if stable_projection:
        return "corrected"
    return "unstable"


def estimate_confidence_interval(
    estimate: ProjectedCurveEstimate,
    *,
    z_score: float = 1.96,
) -> tuple[float, float]:
    width = z_score * estimate.projection_std
    return estimate.corrected_val_bpb - width, estimate.corrected_val_bpb + width


def build_horizon_projection_table(
    curves: list[CurveArtifact],
    *,
    horizons_seconds: Iterable[float],
    calibration: ProjectionCalibration | None = None,
    truth_curves: list[CurveArtifact] | None = None,
    winner_probability_threshold: float = 0.9,
    projected_margin_threshold: float = 0.01,
    monte_carlo_samples: int = MONTE_CARLO_SAMPLES,
) -> tuple[list[HorizonProjectionRow], list[HorizonProjectionDecision]]:
    rows: list[HorizonProjectionRow] = []
    decisions: list[HorizonProjectionDecision] = []
    all_truth_curves = truth_curves
    for horizon in sorted({float(value) for value in horizons_seconds}):
        horizon_truth_curves = all_truth_curves
        horizon_calibration_truth_curves = None
        if all_truth_curves is not None:
            horizon_calibration_truth_curves = [
                curve
                for curve in all_truth_curves
                if (final_training_seconds(curve) or 0.0) >= horizon
            ]
        horizon_calibration = calibration
        if horizon_calibration is not None and not math.isclose(horizon_calibration.target_seconds, horizon):
            horizon_calibration = None
        if horizon_calibration is None and horizon_calibration_truth_curves:
            horizon_calibration = build_projection_calibration(horizon_calibration_truth_curves, target_seconds=horizon)
        estimates, decision = compare_projected_curves(
            curves,
            target_seconds=horizon,
            calibration=horizon_calibration,
            truth_curves=horizon_truth_curves,
            winner_probability_threshold=winner_probability_threshold,
            projected_margin_threshold=projected_margin_threshold,
            monte_carlo_samples=monte_carlo_samples,
        )
        if decision is not None:
            decisions.append(
                HorizonProjectionDecision(
                    target_seconds=horizon,
                    top_preset=decision.top_preset,
                    top_projected_tokens=next(
                        (
                            item.summary.projected_tokens
                            for item in estimates
                            if item.summary.preset == decision.top_preset
                        ),
                        None,
                    ),
                    top_winner_probability=decision.top_winner_probability,
                    top_margin_to_second=decision.top_margin_to_second,
                    top_margin_snr=decision.top_margin_snr,
                    enough_signal=decision.enough_signal,
                    confidence_reason=decision.confidence_reason,
                )
            )
        else:
            decisions.append(
                HorizonProjectionDecision(
                    target_seconds=horizon,
                    top_preset=None,
                    top_projected_tokens=None,
                    top_winner_probability=None,
                    top_margin_to_second=None,
                    top_margin_snr=None,
                    enough_signal=False,
                    confidence_reason=None,
                )
            )
        for estimate in estimates:
            interval_low, interval_high = estimate_confidence_interval(estimate)
            rows.append(
                HorizonProjectionRow(
                    target_seconds=horizon,
                    preset=estimate.summary.preset,
                    engine=estimate.summary.engine,
                    hardware_key=estimate.summary.hardware_key,
                    device_batch_size=estimate.summary.device_batch_size,
                    total_batch_size=estimate.summary.total_batch_size,
                    observed_seconds=estimate.summary.observed_seconds,
                    observed_tokens=estimate.summary.observed_tokens,
                    target_tokens=estimate.summary.projected_tokens,
                    corrected_val_bpb=estimate.corrected_val_bpb,
                    correction_mean=estimate.correction_mean,
                    projection_std=estimate.projection_std,
                    fit_r2=estimate.fit_r2,
                    fit_sigma=estimate.fit_sigma,
                    interval_low=interval_low,
                    interval_high=interval_high,
                    winner_probability=estimate.winner_probability,
                    enough_signal=estimate.enough_signal,
                    projection_source=estimate.projection_source,
                    matched_truth_count=estimate.matched_truth_count,
                    truth_anchor_seconds=estimate.truth_anchor_seconds,
                    truth_anchor_tokens=estimate.truth_anchor_tokens,
                    extrapolation_ratio=estimate.extrapolation_ratio,
                    calibration_sample_count=estimate.calibration_sample_count,
                    confidence_label=projection_confidence_label(estimate),
                    confidence_reason=estimate.confidence_reason or "n/a",
                )
            )
    return rows, decisions


def summarize_longer_horizon_projection(
    rows: list[HorizonProjectionRow],
    decisions: list[HorizonProjectionDecision],
    *,
    target_seconds: float,
    scaling_target_seconds: float,
) -> LongerHorizonProjection | None:
    target_rows = sorted(
        (row for row in rows if math.isclose(row.target_seconds, target_seconds)),
        key=lambda row: row.corrected_val_bpb,
    )
    scaling_rows = sorted(
        (row for row in rows if math.isclose(row.target_seconds, scaling_target_seconds)),
        key=lambda row: row.corrected_val_bpb,
    )
    if not target_rows or not scaling_rows:
        return None
    target_winner = target_rows[0]
    scaling_winner = scaling_rows[0]
    scaling_decision = next(
        (item for item in decisions if math.isclose(item.target_seconds, scaling_target_seconds)),
        None,
    )
    target_gap_at_scaling = None
    target_winner_rank_at_scaling = None
    for idx, row in enumerate(scaling_rows, start=1):
        if row.preset == target_winner.preset:
            target_winner_rank_at_scaling = idx
            target_gap_at_scaling = row.corrected_val_bpb - scaling_winner.corrected_val_bpb
            break
    scaling_gap_at_target = None
    scaling_winner_rank_at_target = None
    for idx, row in enumerate(target_rows, start=1):
        if row.preset == scaling_winner.preset:
            scaling_winner_rank_at_target = idx
            scaling_gap_at_target = row.corrected_val_bpb - target_winner.corrected_val_bpb
            break
    return LongerHorizonProjection(
        target_seconds=target_seconds,
        scaling_target_seconds=scaling_target_seconds,
        target_winner_preset=target_winner.preset,
        target_winner_val_bpb=target_winner.corrected_val_bpb,
        scaling_winner_preset=scaling_winner.preset,
        scaling_winner_val_bpb=scaling_winner.corrected_val_bpb,
        scaling_winner_device_batch_size=scaling_winner.device_batch_size,
        scaling_winner_total_batch_size=scaling_winner.total_batch_size,
        scaling_winner_probability=(
            scaling_decision.top_winner_probability if scaling_decision is not None else scaling_winner.winner_probability
        ),
        scaling_margin_to_second=(
            scaling_decision.top_margin_to_second if scaling_decision is not None else None
        ),
        scaling_margin_snr=(
            scaling_decision.top_margin_snr if scaling_decision is not None else None
        ),
        scaling_enough_signal=(
            scaling_decision.enough_signal if scaling_decision is not None else bool(scaling_winner.enough_signal)
        ),
        scaling_confidence_label=scaling_winner.confidence_label,
        scaling_confidence_reason=(
            scaling_decision.confidence_reason if scaling_decision is not None else scaling_winner.confidence_reason
        ),
        scaling_projection_source=scaling_winner.projection_source,
        scaling_matched_truth_count=scaling_winner.matched_truth_count,
        crossover_from_target=scaling_winner.preset != target_winner.preset,
        target_gap_at_scaling=target_gap_at_scaling,
        scaling_gap_at_target=scaling_gap_at_target,
        target_winner_rank_at_scaling=target_winner_rank_at_scaling,
        scaling_winner_rank_at_target=scaling_winner_rank_at_target,
    )


def suggest_next_horizon(
    rows: list[HorizonProjectionRow],
    decisions: list[HorizonProjectionDecision],
    *,
    target_seconds: float,
    candidate_horizons: Iterable[float] | None = None,
) -> NextHorizonSuggestion | None:
    target_decision = next(
        (item for item in decisions if math.isclose(item.target_seconds, target_seconds)),
        None,
    )
    if target_decision is None:
        return None
    target_rows = [row for row in rows if math.isclose(row.target_seconds, target_seconds)]
    current_observed_seconds = max(
        (row.observed_seconds for row in target_rows if row.observed_seconds is not None),
        default=None,
    )
    current_observed_tokens = max(
        (row.observed_tokens for row in target_rows if row.observed_tokens is not None),
        default=None,
    )
    if target_decision.enough_signal:
        return NextHorizonSuggestion(
            target_seconds=target_seconds,
            current_observed_seconds=current_observed_seconds,
            current_observed_tokens=current_observed_tokens,
            suggested_seconds=None,
            top_preset=target_decision.top_preset,
            top_winner_probability=target_decision.top_winner_probability,
            target_enough_signal=True,
            target_confidence_reason=target_decision.confidence_reason,
            suggestion_reason="enough-signal",
        )

    horizon_candidates = sorted(
        {
            float(value)
            for value in (candidate_horizons or [])
            if float(value) > 0.0
        }
        | {float(target_seconds)}
    )

    if current_observed_seconds is None:
        suggested_seconds = horizon_candidates[0] if horizon_candidates else target_seconds
        return NextHorizonSuggestion(
            target_seconds=target_seconds,
            current_observed_seconds=None,
            current_observed_tokens=None,
            suggested_seconds=suggested_seconds,
            top_preset=target_decision.top_preset,
            top_winner_probability=target_decision.top_winner_probability,
            target_enough_signal=False,
            target_confidence_reason=target_decision.confidence_reason,
            suggestion_reason="no-observed-horizon",
        )

    future_candidates = [value for value in horizon_candidates if value > current_observed_seconds]
    if not future_candidates:
        return NextHorizonSuggestion(
            target_seconds=target_seconds,
            current_observed_seconds=current_observed_seconds,
            current_observed_tokens=current_observed_tokens,
            suggested_seconds=None,
            top_preset=target_decision.top_preset,
            top_winner_probability=target_decision.top_winner_probability,
            target_enough_signal=False,
            target_confidence_reason=target_decision.confidence_reason,
            suggestion_reason="no-larger-horizon-available",
        )

    confidence_reason = target_decision.confidence_reason or "insufficient-signal"
    if current_observed_seconds < target_seconds:
        desired_threshold = min(target_seconds, current_observed_seconds * 2.0)
        suggested_seconds = next((value for value in future_candidates if value >= desired_threshold), future_candidates[-1])
        suggestion_reason = (
            "reach-target-horizon"
            if math.isclose(suggested_seconds, target_seconds)
            else f"extend-horizon:{confidence_reason}"
        )
    else:
        suggested_seconds = future_candidates[0]
        suggestion_reason = f"beyond-target:{confidence_reason}"

    return NextHorizonSuggestion(
        target_seconds=target_seconds,
        current_observed_seconds=current_observed_seconds,
        current_observed_tokens=current_observed_tokens,
        suggested_seconds=suggested_seconds,
        top_preset=target_decision.top_preset,
        top_winner_probability=target_decision.top_winner_probability,
        target_enough_signal=False,
        target_confidence_reason=target_decision.confidence_reason,
        suggestion_reason=suggestion_reason,
    )


def compare_multi_horizon_curves(
    short_curves: list[CurveArtifact],
    long_curves: list[CurveArtifact],
    *,
    target_seconds: float,
    calibration: ProjectionCalibration | None = None,
    truth_curves: list[CurveArtifact] | None = None,
    winner_probability_threshold: float = 0.9,
    projected_margin_threshold: float = 0.01,
    monte_carlo_samples: int = MONTE_CARLO_SAMPLES,
) -> tuple[list[ProjectedCurveEstimate], MultiHorizonProjectionDecision | None, list[MultiHorizonProjectionDiagnostics]]:
    short_estimates, _ = compare_projected_curves(
        short_curves,
        target_seconds=target_seconds,
        calibration=calibration,
        truth_curves=truth_curves,
        winner_probability_threshold=winner_probability_threshold,
        projected_margin_threshold=projected_margin_threshold,
        monte_carlo_samples=monte_carlo_samples,
    )
    long_estimates, long_decision = compare_projected_curves(
        long_curves,
        target_seconds=target_seconds,
        calibration=calibration,
        truth_curves=truth_curves,
        winner_probability_threshold=winner_probability_threshold,
        projected_margin_threshold=projected_margin_threshold,
        monte_carlo_samples=monte_carlo_samples,
    )
    if long_decision is None:
        return long_estimates, None, []

    short_by_preset = {item.summary.preset: item for item in short_estimates}
    short_curve_by_preset = {curve.preset: curve for curve in short_curves}
    long_curve_by_preset = {curve.preset: curve for curve in long_curves}

    diagnostics: list[MultiHorizonProjectionDiagnostics] = []
    diagnostics_by_preset: dict[str, MultiHorizonProjectionDiagnostics] = {}
    for item in long_estimates:
        short_item = short_by_preset.get(item.summary.preset)
        short_curve = short_curve_by_preset.get(item.summary.preset)
        long_curve = long_curve_by_preset.get(item.summary.preset)

        short_observed_seconds = (
            short_curve.curve_points[-1].actual_training_seconds
            if short_curve is not None and short_curve.curve_points
            else None
        )
        short_observed_tokens = (
            short_curve.curve_points[-1].total_tokens
            if short_curve is not None and short_curve.curve_points
            else None
        )
        long_observed_seconds = (
            long_curve.curve_points[-1].actual_training_seconds
            if long_curve is not None and long_curve.curve_points
            else None
        )
        long_observed_tokens = (
            long_curve.curve_points[-1].total_tokens
            if long_curve is not None and long_curve.curve_points
            else None
        )
        short_projected_val_bpb = short_item.corrected_val_bpb if short_item is not None else None
        fit_quality_min = None
        horizon_alpha = None
        effective_damping = None
        horizon_correction = None
        projection_delta = None
        projection_sigma = None
        projection_snr = None
        stability_gap = None
        stability_snr = None
        stable_projection = True

        if (
            short_item is not None
            and short_observed_tokens is not None
            and long_observed_tokens is not None
            and long_observed_tokens > short_observed_tokens
            and short_item.corrected_val_bpb > 0
            and item.corrected_val_bpb > 0
        ):
            fit_values = [value for value in (short_item.fit_r2, item.fit_r2) if value is not None]
            fit_quality_min = min(fit_values) if fit_values else None
            horizon_alpha = (
                math.log(item.corrected_val_bpb) - math.log(short_item.corrected_val_bpb)
            ) / (math.log(long_observed_tokens) - math.log(short_observed_tokens))
            if fit_quality_min is not None:
                effective_damping = EXTRAPOLATION_BASE_DAMPING * max(0.25, min(1.0, fit_quality_min))
            else:
                effective_damping = EXTRAPOLATION_BASE_DAMPING
            target_tokens = item.summary.projected_tokens
            if (
                target_tokens is not None
                and long_observed_tokens > 0
                and target_tokens > long_observed_tokens
                and horizon_alpha < 0
            ):
                horizon_correction = (target_tokens / long_observed_tokens) ** (horizon_alpha * effective_damping)
            else:
                horizon_correction = 1.0
            projection_delta = abs(item.corrected_val_bpb - short_item.corrected_val_bpb)
            projection_sigma = max(
                PROJECTION_STD_FLOOR,
                math.sqrt(short_item.projection_std**2 + item.projection_std**2),
            )
            projection_snr = projection_delta / projection_sigma
            stability_gap = abs(item.corrected_val_bpb - short_item.corrected_val_bpb)
            stability_snr = stability_gap / projection_sigma
            stable_projection = stability_gap <= max(projected_margin_threshold, projection_sigma)

        stability_reason = classify_multi_horizon_reason(
            short_item,
            item,
            horizon_alpha=horizon_alpha,
            projection_snr=projection_snr,
            stability_snr=stability_snr,
            stable_projection=stable_projection,
        )

        diag = MultiHorizonProjectionDiagnostics(
            preset=item.summary.preset,
            short_observed_seconds=short_observed_seconds,
            long_observed_seconds=long_observed_seconds,
            short_observed_tokens=short_observed_tokens,
            long_observed_tokens=long_observed_tokens,
            short_projected_val_bpb=short_projected_val_bpb,
            long_projected_val_bpb=item.corrected_val_bpb,
            short_fit_r2=short_item.fit_r2 if short_item is not None else None,
            long_fit_r2=item.fit_r2,
            short_fit_sigma=short_item.fit_sigma if short_item is not None else None,
            long_fit_sigma=item.fit_sigma,
            fit_quality_min=fit_quality_min,
            horizon_alpha=horizon_alpha,
            effective_damping=effective_damping,
            horizon_correction=horizon_correction,
            projection_delta=projection_delta,
            projection_sigma=projection_sigma,
            projection_snr=projection_snr,
            stability_gap=stability_gap,
            stability_snr=stability_snr,
            stable_projection=stable_projection,
            stability_reason=stability_reason,
        )
        diagnostics.append(diag)
        diagnostics_by_preset[diag.preset] = diag

    top_diag = diagnostics_by_preset.get(long_decision.top_preset)
    stability_reason = top_diag.stability_reason if top_diag is not None else None
    enough_signal = (
        long_decision.enough_signal
        and (top_diag.stable_projection if top_diag is not None else True)
        and stability_reason not in {"low_r2", "insufficient_projection_snr", "insufficient_snr", "unstable"}
    )
    confidence_reason = long_decision.confidence_reason
    if stability_reason is not None and stability_reason not in {"single_horizon"}:
        confidence_reason = stability_reason
    decision = MultiHorizonProjectionDecision(
        target_seconds=target_seconds,
        short_horizon_seconds=top_diag.short_observed_seconds if top_diag is not None else None,
        long_horizon_seconds=top_diag.long_observed_seconds if top_diag is not None else None,
        top_preset=long_decision.top_preset,
        top_winner_probability=long_decision.top_winner_probability,
        top_margin_to_second=long_decision.top_margin_to_second,
        top_margin_snr=long_decision.top_margin_snr,
        top_stability_gap=top_diag.stability_gap if top_diag is not None else None,
        top_stability_snr=top_diag.stability_snr if top_diag is not None else None,
        enough_signal=enough_signal,
        stability_reason=stability_reason,
        confidence_reason=confidence_reason,
    )

    enriched: list[ProjectedCurveEstimate] = []
    for item in long_estimates:
        enriched.append(
            ProjectedCurveEstimate(
                summary=item.summary,
                corrected_val_bpb=item.corrected_val_bpb,
                projection_std=item.projection_std,
                fit_r2=item.fit_r2,
                fit_sigma=item.fit_sigma,
                calibration_horizon_seconds=item.calibration_horizon_seconds,
                calibration_horizon_tokens=item.calibration_horizon_tokens,
                calibration_sample_count=item.calibration_sample_count,
                correction_mean=item.correction_mean,
                projection_source=item.projection_source,
                matched_truth_count=item.matched_truth_count,
                winner_probability=item.winner_probability,
                enough_signal=enough_signal if item.summary.preset == long_decision.top_preset else False,
                confidence_reason=long_decision.confidence_reason if item.summary.preset == long_decision.top_preset else None,
            )
        )

    return enriched, decision, diagnostics
