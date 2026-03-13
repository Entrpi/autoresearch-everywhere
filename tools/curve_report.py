#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_platform.curve_projection import (
    HorizonProjectionDecision,
    MultiHorizonProjectionDecision,
    MultiHorizonProjectionDiagnostics,
    ProjectionDecision,
    ProjectedCurveEstimate,
    build_horizon_projection_table,
    build_projection_calibration,
    compare_multi_horizon_curves,
    compare_projected_curves,
    estimate_confidence_interval,
    load_curve_artifact,
    load_curve_artifacts_from_dir,
    summarize_curves,
)


def summary_to_dict(row) -> dict:
    return {
        "engine": row.engine,
        "preset": row.preset,
        "hardware_key": row.hardware_key,
        "device_batch_size": row.device_batch_size,
        "total_batch_size": row.total_batch_size,
        "curve_points": row.curve_points,
        "target_seconds": row.target_seconds,
        "observed_seconds": row.observed_seconds,
        "observed_tokens": row.observed_tokens,
        "projected_val_bpb": row.projected_val_bpb,
        "projected_tokens": row.projected_tokens,
        "projection_method": row.projection_method,
        "projection_fit_r2": row.projection_fit_r2,
        "projection_fit_sigma": row.projection_fit_sigma,
        "final_val_bpb": row.final_val_bpb,
        "final_eval_seconds": row.final_eval_seconds,
        "final_training_seconds": row.final_training_seconds,
        "projected_margin_to_best": row.projected_margin_to_best,
        "final_margin_to_best": row.final_margin_to_best,
    }


def projection_to_dict(row: ProjectedCurveEstimate) -> dict:
    interval_low, interval_high = estimate_confidence_interval(row)
    return {
        "engine": row.summary.engine,
        "preset": row.summary.preset,
        "hardware_key": row.summary.hardware_key,
        "device_batch_size": row.summary.device_batch_size,
        "total_batch_size": row.summary.total_batch_size,
        "curve_points": row.summary.curve_points,
        "target_seconds": row.summary.target_seconds,
        "observed_seconds": row.summary.observed_seconds,
        "observed_tokens": row.summary.observed_tokens,
        "projected_val_bpb": row.summary.projected_val_bpb,
        "projected_tokens": row.summary.projected_tokens,
        "projection_method": row.summary.projection_method,
        "final_val_bpb": row.summary.final_val_bpb,
        "final_eval_seconds": row.summary.final_eval_seconds,
        "final_training_seconds": row.summary.final_training_seconds,
        "corrected_val_bpb": row.corrected_val_bpb,
        "projection_std": row.projection_std,
        "fit_r2": row.fit_r2,
        "fit_sigma": row.fit_sigma,
        "confidence_interval_low": interval_low,
        "confidence_interval_high": interval_high,
        "calibration_horizon_seconds": row.calibration_horizon_seconds,
        "calibration_horizon_tokens": row.calibration_horizon_tokens,
        "calibration_sample_count": row.calibration_sample_count,
        "correction_mean": row.correction_mean,
        "projection_source": row.projection_source,
        "matched_truth_count": row.matched_truth_count,
        "winner_probability": row.winner_probability,
        "enough_signal": row.enough_signal,
        "confidence_reason": row.confidence_reason,
    }


def multi_horizon_row_to_dict(
    row: ProjectedCurveEstimate,
    diagnostics: MultiHorizonProjectionDiagnostics | None,
) -> dict:
    data = projection_to_dict(row)
    data.update(
        {
            "short_observed_seconds": diagnostics.short_observed_seconds if diagnostics is not None else None,
            "long_observed_seconds": diagnostics.long_observed_seconds if diagnostics is not None else None,
            "short_observed_tokens": diagnostics.short_observed_tokens if diagnostics is not None else None,
            "long_observed_tokens": diagnostics.long_observed_tokens if diagnostics is not None else None,
            "short_projected_val_bpb": diagnostics.short_projected_val_bpb if diagnostics is not None else None,
            "long_projected_val_bpb": diagnostics.long_projected_val_bpb if diagnostics is not None else None,
            "short_fit_r2": diagnostics.short_fit_r2 if diagnostics is not None else None,
            "long_fit_r2": diagnostics.long_fit_r2 if diagnostics is not None else None,
            "short_fit_sigma": diagnostics.short_fit_sigma if diagnostics is not None else None,
            "long_fit_sigma": diagnostics.long_fit_sigma if diagnostics is not None else None,
            "fit_quality_min": diagnostics.fit_quality_min if diagnostics is not None else None,
            "horizon_alpha": diagnostics.horizon_alpha if diagnostics is not None else None,
            "effective_damping": diagnostics.effective_damping if diagnostics is not None else None,
            "horizon_correction": diagnostics.horizon_correction if diagnostics is not None else None,
            "projection_delta": diagnostics.projection_delta if diagnostics is not None else None,
            "projection_sigma": diagnostics.projection_sigma if diagnostics is not None else None,
            "projection_snr": diagnostics.projection_snr if diagnostics is not None else None,
            "stability_gap": diagnostics.stability_gap if diagnostics is not None else None,
            "stability_snr": diagnostics.stability_snr if diagnostics is not None else None,
            "stable_projection": diagnostics.stable_projection if diagnostics is not None else None,
            "stability_reason": diagnostics.stability_reason if diagnostics is not None else None,
        }
    )
    return data


def horizon_row_to_dict(row) -> dict:
    return {
        "target_seconds": row.target_seconds,
        "preset": row.preset,
        "engine": row.engine,
        "hardware_key": row.hardware_key,
        "device_batch_size": row.device_batch_size,
        "total_batch_size": row.total_batch_size,
        "observed_seconds": row.observed_seconds,
        "observed_tokens": row.observed_tokens,
        "target_tokens": row.target_tokens,
        "corrected_val_bpb": row.corrected_val_bpb,
        "correction_mean": row.correction_mean,
        "projection_std": row.projection_std,
        "fit_r2": row.fit_r2,
        "fit_sigma": row.fit_sigma,
        "interval_low": row.interval_low,
        "interval_high": row.interval_high,
        "winner_probability": row.winner_probability,
        "enough_signal": row.enough_signal,
        "projection_source": row.projection_source,
        "matched_truth_count": row.matched_truth_count,
        "truth_anchor_seconds": row.truth_anchor_seconds,
        "truth_anchor_tokens": row.truth_anchor_tokens,
        "extrapolation_ratio": row.extrapolation_ratio,
        "calibration_sample_count": row.calibration_sample_count,
        "confidence_label": row.confidence_label,
        "confidence_reason": row.confidence_reason,
    }


def _format_float(value: float | None, *, digits: int = 6, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _format_batch(row: dict) -> str:
    return f"{row['device_batch_size']} / {row['total_batch_size']}"


def _format_tokens(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    return f"{value / 1e6:.1f}M"


def print_summary_markdown(rows: list[dict]) -> None:
    headers = [
        "Preset",
        "Engine",
        "Hardware",
        "Batch",
        "Curve points",
        "Observed tokens",
        "Target sec",
        "Target tokens",
        "Projected val_bpb",
        "Proj Δ best",
        "Method",
        "Final val_bpb",
        "Final Δ best",
        "Final eval sec",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["preset"]),
                    str(row["engine"]),
                    str(row["hardware_key"]),
                    _format_batch(row),
                    str(row["curve_points"]),
                    _format_tokens(row["observed_tokens"]),
                    _format_float(row["target_seconds"], digits=1),
                    _format_tokens(row["projected_tokens"]),
                    _format_float(row["projected_val_bpb"]),
                    _format_float(row["projected_margin_to_best"], digits=6, suffix=""),
                    str(row["projection_method"]),
                    _format_float(row["final_val_bpb"]),
                    _format_float(row["final_margin_to_best"], digits=6, suffix=""),
                    _format_float(row["final_eval_seconds"], digits=1),
                ]
            )
            + " |"
        )


def print_projection_markdown(rows: list[dict], decision: ProjectionDecision | None) -> None:
    if decision is not None:
        print(
            f"Projection target: `{decision.target_seconds:.0f}s`  \n"
            f"Projected winner: `{decision.top_preset}`  \n"
            f"Winner projected tokens: `{_format_tokens(decision.top_projected_tokens)}`  \n"
            f"Winner probability: `{decision.top_winner_probability:.3f}`  \n"
            f"Margin to second: `{decision.top_margin_to_second:.6f}`  \n"
            f"Margin SNR: `{_format_float(decision.top_margin_snr, digits=3)}`  \n"
            f"Confidence reason: `{decision.confidence_reason or 'n/a'}`  \n"
            f"Enough signal: `{str(decision.enough_signal).lower()}`"
        )
        print()
    headers = [
        "Preset",
        "Engine",
        "Hardware",
        "Batch",
        "Pts",
        "Observed tokens",
        "Target tokens",
        "Proj",
        "Corrected",
        "95% CI",
        "Fit R²",
        "Fit σ",
        "Winner p",
        "Enough?",
        "Reason",
        "Source",
        "Truth",
        "Cal horizon",
        "Cal tokens",
        "Samples",
        "Final",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["preset"]),
                    str(row["engine"]),
                    str(row["hardware_key"]),
                    _format_batch(row),
                    str(row["curve_points"]),
                    _format_tokens(row["observed_tokens"]),
                    _format_tokens(row["projected_tokens"]),
                    _format_float(row["projected_val_bpb"]),
                    _format_float(row["corrected_val_bpb"]),
                    f"{_format_float(row['confidence_interval_low'])}..{_format_float(row['confidence_interval_high'])}",
                    _format_float(row["fit_r2"], digits=3),
                    _format_float(row["fit_sigma"]),
                    _format_float(row["winner_probability"], digits=3),
                    "yes" if row["enough_signal"] else "no",
                    str(row["confidence_reason"] or "n/a"),
                    str(row["projection_source"]),
                    str(row["matched_truth_count"]),
                    f"{row['calibration_horizon_seconds']:.0f}s"
                    if row["calibration_horizon_seconds"] is not None
                    else "n/a",
                    _format_tokens(row["calibration_horizon_tokens"]),
                    str(row["calibration_sample_count"]),
                    _format_float(row["final_val_bpb"]),
                ]
            )
            + " |"
        )


def print_multi_horizon_markdown(
    rows: list[dict],
    decision: MultiHorizonProjectionDecision | None,
) -> None:
    if decision is not None:
        margin = _format_float(decision.top_margin_to_second)
        short_horizon = (
            f"{decision.short_horizon_seconds:.0f}s"
            if decision.short_horizon_seconds is not None
            else "n/a"
        )
        long_horizon = (
            f"{decision.long_horizon_seconds:.0f}s"
            if decision.long_horizon_seconds is not None
            else "n/a"
        )
        stability_gap = _format_float(decision.top_stability_gap)
        stability_snr = _format_float(decision.top_stability_snr, digits=3)
        print(
            f"Projection target: `{decision.target_seconds:.0f}s`  \n"
            f"Projected winner: `{decision.top_preset}`  \n"
            f"Winner probability: `{decision.top_winner_probability:.3f}`  \n"
            f"Margin to second: `{margin}`  \n"
            f"Margin SNR: `{_format_float(decision.top_margin_snr, digits=3)}`  \n"
            f"Short horizon: `{short_horizon}`  \n"
            f"Long horizon: `{long_horizon}`  \n"
            f"Top stability gap: `{stability_gap}`  \n"
            f"Top stability SNR: `{stability_snr}`  \n"
            f"Stability reason: `{decision.stability_reason or 'n/a'}`  \n"
            f"Confidence reason: `{decision.confidence_reason or 'n/a'}`  \n"
            f"Enough signal: `{str(decision.enough_signal).lower()}`"
        )
        print()

    headers = [
        "Preset",
        "Batch",
        "Short sec",
        "Long sec",
        "Short tok",
        "Long tok",
        "Observed tokens",
        "Target tokens",
        "Short proj",
        "Long proj",
        "Corrected",
        "Correction",
        "95% CI",
        "Short R²",
        "Long R²",
        "Min R²",
        "Short σ",
        "Long σ",
        "Alpha",
        "Damp",
        "Corr",
        "Δ",
        "Proj σ",
        "Proj SNR",
        "Winner p",
        "Stable?",
        "Gap",
        "SNR",
        "Reason",
        "Source",
        "Truth",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["preset"]),
                    _format_batch(row),
                    _format_float(row["short_observed_seconds"], digits=1),
                    _format_float(row["long_observed_seconds"], digits=1),
                    _format_tokens(row["short_observed_tokens"]),
                    _format_tokens(row["long_observed_tokens"]),
                    _format_tokens(row["observed_tokens"]),
                    _format_tokens(row["projected_tokens"]),
                    _format_float(row["short_projected_val_bpb"]),
                    _format_float(row["long_projected_val_bpb"]),
                    _format_float(row["corrected_val_bpb"]),
                    _format_float(row["correction_mean"]),
                    f"{_format_float(row['confidence_interval_low'])}..{_format_float(row['confidence_interval_high'])}",
                    _format_float(row["short_fit_r2"], digits=3),
                    _format_float(row["long_fit_r2"], digits=3),
                    _format_float(row["fit_quality_min"], digits=3),
                    _format_float(row["short_fit_sigma"]),
                    _format_float(row["long_fit_sigma"]),
                    _format_float(row["horizon_alpha"], digits=3),
                    _format_float(row["effective_damping"], digits=3),
                    _format_float(row["horizon_correction"], digits=3),
                    _format_float(row["projection_delta"]),
                    _format_float(row["projection_sigma"]),
                    _format_float(row["projection_snr"], digits=3),
                    _format_float(row["winner_probability"], digits=3),
                    "yes" if row["stable_projection"] else "no",
                    _format_float(row["stability_gap"]),
                    _format_float(row["stability_snr"], digits=3),
                    str(row["stability_reason"] or row["confidence_reason"] or "n/a"),
                    str(row["projection_source"]),
                    str(row["matched_truth_count"]),
                ]
            )
            + " |"
        )


def print_horizon_table_markdown(
    rows: list[dict],
    decisions: list[HorizonProjectionDecision],
) -> None:
    if decisions:
        print("| Horizon | Projected winner | Winner tokens | Winner p | Margin to second | Margin SNR | Reason | Enough signal |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for decision in decisions:
            print(
                f"| {decision.target_seconds:.0f}s | "
                f"{decision.top_preset or 'n/a'} | "
                f"{_format_tokens(decision.top_projected_tokens)} | "
                f"{_format_float(decision.top_winner_probability, digits=3)} | "
                f"{_format_float(decision.top_margin_to_second)} | "
                f"{_format_float(decision.top_margin_snr, digits=3)} | "
                f"{getattr(decision, 'stability_reason', None) or decision.confidence_reason or 'n/a'} | "
                f"{'yes' if decision.enough_signal else 'no'} |"
            )
        print()

    headers = [
        "Horizon",
        "Preset",
        "Batch",
        "Observed tokens",
        "Target tokens",
        "Anchor",
        "Anchor tokens",
        "Ratio",
        "Projected",
        "Correction",
        "95% CI",
        "Std",
        "Fit R²",
        "Fit σ",
        "Winner p",
        "Enough?",
        "Reason",
        "Source",
        "Truth",
        "Samples",
        "Confidence",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    f"{row['target_seconds']:.0f}s",
                    str(row["preset"]),
                    _format_batch(row),
                    _format_tokens(row["observed_tokens"]),
                    _format_tokens(row["target_tokens"]),
                    _format_float(row["truth_anchor_seconds"], digits=0, suffix="s"),
                    _format_tokens(row["truth_anchor_tokens"]),
                    _format_float(row["extrapolation_ratio"], digits=2),
                    _format_float(row["corrected_val_bpb"]),
                    _format_float(row["correction_mean"]),
                    f"[{_format_float(row['interval_low'])}, {_format_float(row['interval_high'])}]",
                    _format_float(row["projection_std"]),
                    _format_float(row["fit_r2"], digits=3),
                    _format_float(row["fit_sigma"]),
                    _format_float(row["winner_probability"], digits=3),
                    "yes" if row["enough_signal"] else "no",
                    str(row["confidence_reason"]),
                    str(row["projection_source"]),
                    str(row["matched_truth_count"]),
                    str(row["calibration_sample_count"]),
                    str(row["confidence_label"]),
                ]
            )
            + " |"
        )


def _collect_curve_artifacts(
    raw_inputs: list[str],
    *,
    engine: str | None = None,
    hardware_key: str | None = None,
) -> list:
    curves = []
    seen: set[tuple[str | None, str, str | None, int | None, int | None, int]] = set()
    for raw_input in raw_inputs:
        path = Path(raw_input)
        input_curves = []
        if path.is_dir():
            input_curves = load_curve_artifacts_from_dir(path, engine=engine, hardware_key=hardware_key)
        else:
            input_curves = [load_curve_artifact(path)]
        for curve in input_curves:
            key = (
                curve.engine,
                curve.preset,
                curve.hardware_key,
                curve.device_batch_size,
                curve.total_batch_size,
                len(curve.curve_points),
            )
            if key in seen:
                continue
            seen.add(key)
            curves.append(curve)
    return curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize or compare curve artifacts, with optional truth-backed confidence."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Short-horizon curve JSON paths or directories containing curve artifacts.",
    )
    parser.add_argument(
        "--long-inputs",
        nargs="*",
        default=None,
        help="Optional longer-horizon curve JSON paths or directories for multi-horizon comparison.",
    )
    parser.add_argument("--engine", help="Optional engine filter when scanning directories.")
    parser.add_argument("--hardware-key", help="Optional hardware-key filter when scanning directories.")
    parser.add_argument("--target-seconds", type=float, default=300.0, help="Training horizon to estimate.")
    parser.add_argument(
        "--truth-curves-dir",
        help="Optional directory of completed truth-curve artifacts used to calibrate projection uncertainty.",
    )
    parser.add_argument(
        "--winner-probability-threshold",
        type=float,
        default=0.9,
        help="Winner probability threshold used when deciding if there is enough signal.",
    )
    parser.add_argument(
        "--projected-margin-threshold",
        type=float,
        default=0.01,
        help="Minimum projected margin between first and second place to treat the signal as sufficient.",
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Emit simple per-curve summaries instead of confidence-bearing projections.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown.")
    parser.add_argument(
        "--horizons",
        help="Optional comma-separated list of horizons to compare in one table, e.g. 60,120,300,900.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.inputs:
        raise SystemExit("curve_report.py requires at least one input path or directory")

    short_curves = _collect_curve_artifacts(
        args.inputs,
        engine=args.engine,
        hardware_key=args.hardware_key,
    )
    if not short_curves:
        raise SystemExit("no curve artifacts found in the supplied inputs")

    truth_curves = None
    calibration = None
    if args.truth_curves_dir:
        truth_engine = args.engine if args.engine is not None else short_curves[0].engine
        truth_hardware_key = args.hardware_key if args.hardware_key is not None else short_curves[0].hardware_key
        truth_curves = load_curve_artifacts_from_dir(
            Path(args.truth_curves_dir),
            engine=truth_engine,
            hardware_key=truth_hardware_key,
            require_target_seconds=args.target_seconds,
        )
        calibration = build_projection_calibration(truth_curves, target_seconds=args.target_seconds)

    if args.horizons:
        horizons = [float(item.strip()) for item in args.horizons.split(",") if item.strip()]
        if not horizons:
            raise SystemExit("--horizons was provided but no numeric horizons were parsed")
        rows, decisions = build_horizon_projection_table(
            short_curves,
            horizons_seconds=horizons,
            calibration=calibration,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        payload_rows = [horizon_row_to_dict(row) for row in rows]
        if args.json:
            print(
                json.dumps(
                    {
                        "mode": "horizon-table",
                        "decisions": [decision.__dict__ for decision in decisions],
                        "rows": payload_rows,
                    },
                    indent=2,
                )
            )
        else:
            print_horizon_table_markdown(payload_rows, decisions)
        return 0

    if args.long_inputs:
        long_curves = _collect_curve_artifacts(
            args.long_inputs,
            engine=args.engine if args.engine is not None else short_curves[0].engine,
            hardware_key=args.hardware_key if args.hardware_key is not None else short_curves[0].hardware_key,
        )
        projections, decision, diagnostics = compare_multi_horizon_curves(
            short_curves,
            long_curves,
            target_seconds=args.target_seconds,
            calibration=calibration,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        diag_by_preset = {item.preset: item for item in diagnostics}
        rows = [multi_horizon_row_to_dict(item, diag_by_preset.get(item.summary.preset)) for item in projections]
        if args.json:
            print(
                json.dumps(
                    {
                        "mode": "multi-horizon",
                        "decision": None if decision is None else decision.__dict__,
                        "rows": rows,
                    },
                    indent=2,
                )
            )
        else:
            print_multi_horizon_markdown(rows, decision)
        return 0

    if args.summaries_only:
        summaries = summarize_curves(short_curves, target_seconds=args.target_seconds)
        rows = [summary_to_dict(row) for row in summaries]
        if args.json:
            print(json.dumps({"mode": "summary", "rows": rows}, indent=2))
        else:
            print_summary_markdown(rows)
        return 0

    projections, decision = compare_projected_curves(
        short_curves,
        target_seconds=args.target_seconds,
        calibration=calibration,
        truth_curves=truth_curves,
        winner_probability_threshold=args.winner_probability_threshold,
        projected_margin_threshold=args.projected_margin_threshold,
    )
    rows = [projection_to_dict(row) for row in projections]
    if args.json:
        print(
            json.dumps(
                {
                    "mode": "projection",
                    "decision": None if decision is None else decision.__dict__,
                    "rows": rows,
                },
                indent=2,
            )
        )
    else:
        print_projection_markdown(rows, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
