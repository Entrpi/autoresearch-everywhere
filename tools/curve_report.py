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
    build_projection_calibration,
    compare_projected_curves,
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
        "projected_val_bpb": row.projected_val_bpb,
        "projected_tokens": row.projected_tokens,
        "projection_method": row.projection_method,
        "final_val_bpb": row.final_val_bpb,
        "final_eval_seconds": row.final_eval_seconds,
        "final_training_seconds": row.final_training_seconds,
        "projected_margin_to_best": row.projected_margin_to_best,
        "final_margin_to_best": row.final_margin_to_best,
    }


def projection_to_dict(row) -> dict:
    return {
        "engine": row.summary.engine,
        "preset": row.summary.preset,
        "hardware_key": row.summary.hardware_key,
        "device_batch_size": row.summary.device_batch_size,
        "total_batch_size": row.summary.total_batch_size,
        "curve_points": row.summary.curve_points,
        "target_seconds": row.summary.target_seconds,
        "projected_val_bpb": row.summary.projected_val_bpb,
        "projected_tokens": row.summary.projected_tokens,
        "projection_method": row.summary.projection_method,
        "final_val_bpb": row.summary.final_val_bpb,
        "final_eval_seconds": row.summary.final_eval_seconds,
        "final_training_seconds": row.summary.final_training_seconds,
        "corrected_val_bpb": row.corrected_val_bpb,
        "projection_std": row.projection_std,
        "calibration_horizon_seconds": row.calibration_horizon_seconds,
        "calibration_sample_count": row.calibration_sample_count,
        "correction_mean": row.correction_mean,
        "winner_probability": row.winner_probability,
        "enough_signal": row.enough_signal,
    }


def print_markdown(rows) -> None:
    headers = [
        "Preset",
        "Engine",
        "Hardware",
        "Batch",
        "Curve points",
        "Target sec",
        "Projected val_bpb",
        "Proj Δ best",
        "Projected tokens",
        "Method",
        "Final val_bpb",
        "Final Δ best",
        "Final eval sec",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        batch = f"{row['device_batch_size']} / {row['total_batch_size']}"
        projected_val_bpb = f"{row['projected_val_bpb']:.6f}" if row["projected_val_bpb"] is not None else "n/a"
        projected_margin = (
            f"{row['projected_margin_to_best']:+.6f}" if row["projected_margin_to_best"] is not None else "n/a"
        )
        projected_tokens = f"{row['projected_tokens'] / 1e6:.1f}M" if row["projected_tokens"] is not None else "n/a"
        final_val_bpb = f"{row['final_val_bpb']:.6f}" if row["final_val_bpb"] is not None else "n/a"
        final_margin = f"{row['final_margin_to_best']:+.6f}" if row["final_margin_to_best"] is not None else "n/a"
        final_eval_seconds = f"{row['final_eval_seconds']:.1f}" if row["final_eval_seconds"] is not None else "n/a"
        print(
            "| "
            + " | ".join(
                [
                    str(row["preset"]),
                    str(row["engine"]),
                    str(row["hardware_key"]),
                    batch,
                    str(row["curve_points"]),
                    f"{row['target_seconds']:.1f}",
                    projected_val_bpb,
                    projected_margin,
                    projected_tokens,
                    str(row["projection_method"]),
                    final_val_bpb,
                    final_margin,
                    final_eval_seconds,
                ]
            )
            + " |"
        )


def print_projection_markdown(rows, decision) -> None:
    if decision is not None:
        print(
            f"Projection target: `{decision.target_seconds:.0f}s`  \n"
            f"Projected winner: `{decision.top_preset}`  \n"
            f"Winner probability: `{decision.top_winner_probability:.3f}`  \n"
            f"Margin to second: `{decision.top_margin_to_second:.6f}`  \n"
            f"Enough signal: `{str(decision.enough_signal).lower()}`"
        )
        print()
    headers = [
        "Preset",
        "Batch",
        "Observed points",
        "Projected val_bpb",
        "Corrected val_bpb",
        "Std",
        "Winner p",
        "Cal horizon",
        "Samples",
        "Final val_bpb",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        batch = f"{row['device_batch_size']} / {row['total_batch_size']}"
        projected = f"{row['projected_val_bpb']:.6f}" if row["projected_val_bpb"] is not None else "n/a"
        corrected = f"{row['corrected_val_bpb']:.6f}" if row["corrected_val_bpb"] is not None else "n/a"
        std = f"{row['projection_std']:.6f}" if row["projection_std"] is not None else "n/a"
        winner_p = f"{row['winner_probability']:.3f}" if row["winner_probability"] is not None else "n/a"
        cal_horizon = (
            f"{row['calibration_horizon_seconds']:.0f}s"
            if row["calibration_horizon_seconds"] is not None
            else "n/a"
        )
        final_val = f"{row['final_val_bpb']:.6f}" if row["final_val_bpb"] is not None else "n/a"
        print(
            "| "
            + " | ".join(
                [
                    str(row["preset"]),
                    batch,
                    str(row["curve_points"]),
                    projected,
                    corrected,
                    std,
                    winner_p,
                    cal_horizon,
                    str(row["calibration_sample_count"]),
                    final_val,
                ]
            )
            + " |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize backend-agnostic curve artifacts.")
    parser.add_argument("inputs", nargs="+", help="Curve JSON artifact paths.")
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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    curves = [load_curve_artifact(Path(raw_path)) for raw_path in args.inputs]
    if args.truth_curves_dir:
        truth_curves = load_curve_artifacts_from_dir(
            Path(args.truth_curves_dir),
            engine=curves[0].engine if curves else None,
            hardware_key=curves[0].hardware_key if curves else None,
            require_target_seconds=args.target_seconds,
        )
        calibration = build_projection_calibration(truth_curves, target_seconds=args.target_seconds)
        projections, decision = compare_projected_curves(
            curves,
            target_seconds=args.target_seconds,
            calibration=calibration,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        rows = [projection_to_dict(row) for row in projections]
        if args.json:
            print(json.dumps({"decision": None if decision is None else decision.__dict__, "rows": rows}, indent=2))
        else:
            print_projection_markdown(rows, None if decision is None else decision)
        return 0

    summaries = summarize_curves(curves, target_seconds=args.target_seconds)
    rows = [summary_to_dict(row) for row in summaries]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
