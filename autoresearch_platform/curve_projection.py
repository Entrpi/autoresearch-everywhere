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


@dataclass(frozen=True)
class CurveSummary:
    engine: str | None
    preset: str
    hardware_key: str | None
    device_batch_size: int | None
    total_batch_size: int | None
    curve_points: int
    target_seconds: float
    projected_val_bpb: float | None
    projected_tokens: float | None
    projection_method: str | None
    final_val_bpb: float | None
    final_eval_seconds: float | None
    final_training_seconds: float | None
    projected_margin_to_best: float | None = None
    final_margin_to_best: float | None = None


@dataclass(frozen=True)
class ProjectionCalibrationPoint:
    horizon_seconds: float
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
    calibration_horizon_seconds: float | None
    calibration_sample_count: int
    correction_mean: float
    projection_source: str = "generic-projection"
    matched_truth_count: int = 0
    winner_probability: float | None = None
    enough_signal: bool | None = None


@dataclass(frozen=True)
class ProjectionDecision:
    target_seconds: float
    top_preset: str
    top_winner_probability: float
    top_margin_to_second: float | None
    enough_signal: bool


@dataclass(frozen=True)
class MultiHorizonProjectionDiagnostics:
    preset: str
    short_observed_seconds: float | None
    long_observed_seconds: float | None
    short_projected_val_bpb: float | None
    long_projected_val_bpb: float
    horizon_alpha: float | None
    stability_gap: float | None
    stability_snr: float | None
    stable_projection: bool


@dataclass(frozen=True)
class MultiHorizonProjectionDecision:
    target_seconds: float
    short_horizon_seconds: float | None
    long_horizon_seconds: float | None
    top_preset: str
    top_winner_probability: float
    top_margin_to_second: float | None
    top_stability_gap: float | None
    top_stability_snr: float | None
    enough_signal: bool


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


def _fit_log_token_projection(points: tuple[CurvePoint, ...], target_tokens: float) -> float:
    fit_points = list(points[1:] if len(points) > 2 else points)
    xs = [math.log(max(point.total_tokens, 1.0)) for point in fit_points]
    ys = [point.val_bpb for point in fit_points]
    if len(fit_points) == 1 or math.isclose(max(xs), min(xs)):
        return fit_points[-1].val_bpb
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(denominator, 0.0):
        return fit_points[-1].val_bpb
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    projected = intercept + slope * math.log(max(target_tokens, 1.0))
    last_observed = fit_points[-1].val_bpb
    return min(last_observed, projected)


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
    projected_val_bpb = _fit_log_token_projection(points, projected_tokens)
    return CurveProjection(
        target_seconds=target_seconds,
        projected_seconds=target_seconds,
        projected_tokens=projected_tokens,
        projected_val_bpb=projected_val_bpb,
        method="log-token-fit",
        observed_point_count=len(points),
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
        "projected_val_bpb": projection.projected_val_bpb if projection else None,
        "projected_tokens": projection.projected_tokens if projection else None,
        "projection_method": projection.method if projection else None,
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
                projected_val_bpb=row.projected_val_bpb,
                projected_tokens=row.projected_tokens,
                projection_method=row.projection_method,
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
    residuals_by_horizon: dict[float, list[float]] = {}
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
            projection = project_curve(truncated, target_seconds=target_seconds)
            if projection is None:
                continue
            residual = float(final_val_bpb) - projection.projected_val_bpb
            residuals_by_horizon.setdefault(float(horizon_seconds), []).append(residual)

    points: list[ProjectionCalibrationPoint] = []
    for horizon_seconds in sorted(residuals_by_horizon):
        residuals = residuals_by_horizon[horizon_seconds]
        points.append(
            ProjectionCalibrationPoint(
                horizon_seconds=horizon_seconds,
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
    observed_seconds: float,
) -> ProjectionCalibrationPoint | None:
    if not calibration.points:
        return None
    return min(
        calibration.points,
        key=lambda point: (
            abs(point.horizon_seconds - observed_seconds),
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
            calibration_horizon_seconds=None,
            calibration_sample_count=0,
            correction_mean=0.0,
        )
    truth_estimate = estimate_from_matching_truth(curve, truth_curves=truth_curves or [])
    if truth_estimate is not None:
        corrected_val_bpb, projection_std, matched_truth_count = truth_estimate
        return ProjectedCurveEstimate(
            summary=summary,
            corrected_val_bpb=corrected_val_bpb,
            projection_std=projection_std,
            calibration_horizon_seconds=curve.curve_points[-1].actual_training_seconds if curve.curve_points else None,
            calibration_sample_count=matched_truth_count,
            correction_mean=corrected_val_bpb - summary.projected_val_bpb,
            projection_source="truth-match",
            matched_truth_count=matched_truth_count,
        )
    observed_seconds = curve.curve_points[-1].actual_training_seconds if curve.curve_points else 0.0
    calibration_point = (
        nearest_calibration_point(calibration, observed_seconds=observed_seconds) if calibration is not None else None
    )
    if calibration_point is None:
        corrected_val_bpb = summary.projected_val_bpb
        projection_std = PROJECTION_STD_FLOOR
        calibration_horizon_seconds = None
        calibration_sample_count = 0
        correction_mean = 0.0
        projection_source = "generic-projection"
        matched_truth_count = 0
    else:
        corrected_val_bpb = summary.projected_val_bpb + calibration_point.residual_mean
        projection_std = calibration_point.residual_std
        calibration_horizon_seconds = calibration_point.horizon_seconds
        calibration_sample_count = calibration_point.sample_count
        correction_mean = calibration_point.residual_mean
        projection_source = "calibrated-projection"
        matched_truth_count = 0
    return ProjectedCurveEstimate(
        summary=summary,
        corrected_val_bpb=corrected_val_bpb,
        projection_std=projection_std,
        calibration_horizon_seconds=calibration_horizon_seconds,
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
                    calibration_horizon_seconds=item.calibration_horizon_seconds,
                    calibration_sample_count=item.calibration_sample_count,
                    correction_mean=item.correction_mean,
                    winner_probability=win_probabilities[item.summary.preset],
                )
            )
        else:
            enriched.append(item)
    estimates = enriched
    estimates.sort(key=lambda item: item.corrected_val_bpb)
    top = estimates[0]
    second = estimates[1] if len(estimates) > 1 else None
    top_margin = None
    if second is not None:
        top_margin = second.corrected_val_bpb - top.corrected_val_bpb
    enough_signal = (
        (top.winner_probability or 0.0) >= winner_probability_threshold
        and (top_margin is None or top_margin >= projected_margin_threshold)
    )
    decision = ProjectionDecision(
        target_seconds=target_seconds,
        top_preset=top.summary.preset,
        top_winner_probability=top.winner_probability or 0.0,
        top_margin_to_second=top_margin,
        enough_signal=enough_signal,
    )
    enriched = []
    for item in estimates:
        enriched.append(
            ProjectedCurveEstimate(
                summary=item.summary,
                corrected_val_bpb=item.corrected_val_bpb,
                projection_std=item.projection_std,
                calibration_horizon_seconds=item.calibration_horizon_seconds,
                calibration_sample_count=item.calibration_sample_count,
                correction_mean=item.correction_mean,
                winner_probability=item.winner_probability,
                enough_signal=enough_signal if item.summary.preset == top.summary.preset else False,
            )
        )
    return enriched, decision


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
        long_observed_seconds = (
            long_curve.curve_points[-1].actual_training_seconds
            if long_curve is not None and long_curve.curve_points
            else None
        )
        short_projected_val_bpb = short_item.corrected_val_bpb if short_item is not None else None
        horizon_alpha = None
        stability_gap = None
        stability_snr = None
        stable_projection = True

        if (
            short_item is not None
            and short_observed_seconds is not None
            and long_observed_seconds is not None
            and long_observed_seconds > short_observed_seconds
            and short_item.corrected_val_bpb > 0
            and item.corrected_val_bpb > 0
        ):
            horizon_alpha = (
                math.log(item.corrected_val_bpb) - math.log(short_item.corrected_val_bpb)
            ) / (math.log(long_observed_seconds) - math.log(short_observed_seconds))
            stability_gap = abs(item.corrected_val_bpb - short_item.corrected_val_bpb)
            pooled_std = max(
                PROJECTION_STD_FLOOR,
                math.sqrt(short_item.projection_std**2 + item.projection_std**2),
            )
            stability_snr = stability_gap / pooled_std
            stable_projection = stability_gap <= max(projected_margin_threshold, pooled_std)

        diag = MultiHorizonProjectionDiagnostics(
            preset=item.summary.preset,
            short_observed_seconds=short_observed_seconds,
            long_observed_seconds=long_observed_seconds,
            short_projected_val_bpb=short_projected_val_bpb,
            long_projected_val_bpb=item.corrected_val_bpb,
            horizon_alpha=horizon_alpha,
            stability_gap=stability_gap,
            stability_snr=stability_snr,
            stable_projection=stable_projection,
        )
        diagnostics.append(diag)
        diagnostics_by_preset[diag.preset] = diag

    top_diag = diagnostics_by_preset.get(long_decision.top_preset)
    enough_signal = long_decision.enough_signal and (top_diag.stable_projection if top_diag is not None else True)
    decision = MultiHorizonProjectionDecision(
        target_seconds=target_seconds,
        short_horizon_seconds=top_diag.short_observed_seconds if top_diag is not None else None,
        long_horizon_seconds=top_diag.long_observed_seconds if top_diag is not None else None,
        top_preset=long_decision.top_preset,
        top_winner_probability=long_decision.top_winner_probability,
        top_margin_to_second=long_decision.top_margin_to_second,
        top_stability_gap=top_diag.stability_gap if top_diag is not None else None,
        top_stability_snr=top_diag.stability_snr if top_diag is not None else None,
        enough_signal=enough_signal,
    )

    enriched: list[ProjectedCurveEstimate] = []
    for item in long_estimates:
        enriched.append(
            ProjectedCurveEstimate(
                summary=item.summary,
                corrected_val_bpb=item.corrected_val_bpb,
                projection_std=item.projection_std,
                calibration_horizon_seconds=item.calibration_horizon_seconds,
                calibration_sample_count=item.calibration_sample_count,
                correction_mean=item.correction_mean,
                projection_source=item.projection_source,
                matched_truth_count=item.matched_truth_count,
                winner_probability=item.winner_probability,
                enough_signal=enough_signal if item.summary.preset == long_decision.top_preset else False,
            )
        )

    return enriched, decision, diagnostics
