#!/usr/bin/env python3
"""
Checkpoint interval tradeoff analysis for the MLX port.

This models the classic checkpointing tradeoff:

- checkpoint too often: lose time to repeated saves
- checkpoint too rarely: lose more work when a resume is needed

For each checkpoint profile we compute the Young/Daly-style optimal interval:

    interval* = max(C, sqrt(2 * C * M) - C)

where:
- C is checkpoint save cost in seconds
- M is mean time between resume-needed events in seconds

We also compute two separate quantities:

    fixed_checkpoint_overhead = C / interval
    projected_total_waste = fixed_checkpoint_overhead + interval / (2 * M) + R / M

where R is an optional fixed resume penalty in seconds.

The default profiles are grounded in measured save costs from this repo's
checkpoint/resume work, including repeated "resume ready" latency trials.
The tool is structured to accept more profiles later, for example if lighter
checkpoint modes are added and benchmarked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoresearch_mlx.checkpoint_policy import (
    DEFAULT_CHECKPOINT_CALIBRATIONS,
    DEFAULT_HUMAN_INTERVAL_POLICY,
    HumanIntervalPolicy,
    format_interval_label,
    save_only_overhead_fraction,
)


@dataclass(frozen=True)
class CheckpointProfile:
    key: str
    label: str
    robustness: str
    checkpoint_cost_sec: float
    resume_penalty_sec: float
    source: str
    notes: str = ""


DEFAULT_PROFILES = tuple(
    CheckpointProfile(
        key=calibration.key,
        label=calibration.label,
        robustness="exact step-boundary full-state resume",
        checkpoint_cost_sec=calibration.checkpoint_cost_sec,
        resume_penalty_sec=calibration.resume_ready_penalty_sec,
        source=calibration.source,
        notes=" ".join(
            note
            for note in (calibration.notes, calibration.resume_ready_source)
            if note
        ),
    )
    for calibration in DEFAULT_CHECKPOINT_CALIBRATIONS
)

DEFAULT_SCENARIOS_PER_DAY = (0.25, 1.0, 2.0, 4.0, 8.0, 24.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        default="results/analysis/checkpoint_tradeoff",
        help="Output path prefix for generated artifacts.",
    )
    parser.add_argument(
        "--min-events-per-day",
        type=float,
        default=0.1,
        help="Lowest expected resume-needed frequency to model.",
    )
    parser.add_argument(
        "--max-events-per-day",
        type=float,
        default=48.0,
        help="Highest expected resume-needed frequency to model.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=240,
        help="Number of log-spaced points in the frequency grid.",
    )
    parser.add_argument(
        "--scenarios-per-day",
        default=",".join(str(value) for value in DEFAULT_SCENARIOS_PER_DAY),
        help="Comma-separated scenario frequencies to summarize in the Markdown table.",
    )
    return parser.parse_args()


def optimal_interval_seconds(checkpoint_cost_sec: float, mean_between_events_sec: float) -> float:
    if checkpoint_cost_sec <= 0:
        raise ValueError("checkpoint_cost_sec must be positive")
    if mean_between_events_sec <= 0:
        raise ValueError("mean_between_events_sec must be positive")
    return max(checkpoint_cost_sec, math.sqrt(2.0 * checkpoint_cost_sec * mean_between_events_sec) - checkpoint_cost_sec)


def fixed_checkpoint_overhead_fraction(
    *,
    checkpoint_cost_sec: float,
    interval_sec: float,
) -> float:
    return checkpoint_cost_sec / interval_sec


def expected_total_waste_fraction(
    *,
    checkpoint_cost_sec: float,
    resume_penalty_sec: float,
    interval_sec: float,
    mean_between_events_sec: float,
) -> float:
    fixed_overhead_fraction = fixed_checkpoint_overhead_fraction(
        checkpoint_cost_sec=checkpoint_cost_sec,
        interval_sec=interval_sec,
    )
    lost_work_fraction = interval_sec / (2.0 * mean_between_events_sec)
    resume_penalty_fraction = resume_penalty_sec / mean_between_events_sec
    return fixed_overhead_fraction + lost_work_fraction + resume_penalty_fraction


def scenario_rates(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise ValueError("Scenario frequencies must be positive.")
        values.append(value)
    return values


def compute_grid(
    profiles: tuple[CheckpointProfile, ...],
    *,
    min_events_per_day: float,
    max_events_per_day: float,
    num_points: int,
) -> list[dict]:
    events_per_day = np.logspace(np.log10(min_events_per_day), np.log10(max_events_per_day), num=num_points)
    rows: list[dict] = []
    for profile in profiles:
        for events_day in events_per_day:
            mean_between_sec = 86400.0 / float(events_day)
            interval_sec = optimal_interval_seconds(profile.checkpoint_cost_sec, mean_between_sec)
            fixed_overhead_fraction = fixed_checkpoint_overhead_fraction(
                checkpoint_cost_sec=profile.checkpoint_cost_sec,
                interval_sec=interval_sec,
            )
            total_waste_fraction = expected_total_waste_fraction(
                checkpoint_cost_sec=profile.checkpoint_cost_sec,
                resume_penalty_sec=profile.resume_penalty_sec,
                interval_sec=interval_sec,
                mean_between_events_sec=mean_between_sec,
            )
            rows.append(
                {
                    "profile_key": profile.key,
                    "label": profile.label,
                    "robustness": profile.robustness,
                    "events_per_day": float(events_day),
                    "events_per_hour": float(events_day / 24.0),
                    "mean_hours_between_events": float(24.0 / events_day),
                    "checkpoint_cost_sec": profile.checkpoint_cost_sec,
                    "resume_penalty_sec": profile.resume_penalty_sec,
                    "optimal_interval_sec": interval_sec,
                    "optimal_interval_min": interval_sec / 60.0,
                    "fixed_checkpoint_overhead_fraction": fixed_overhead_fraction,
                    "fixed_checkpoint_overhead_percent": fixed_overhead_fraction * 100.0,
                    "total_waste_fraction": total_waste_fraction,
                    "total_waste_percent": total_waste_fraction * 100.0,
                    "resume_penalty_ms": profile.resume_penalty_sec * 1000.0,
                    "source": profile.source,
                    "notes": profile.notes,
                }
            )
    return rows


def compute_scenarios(profiles: tuple[CheckpointProfile, ...], frequencies_per_day: list[float]) -> list[dict]:
    rows: list[dict] = []
    for profile in profiles:
        for events_day in frequencies_per_day:
            mean_between_sec = 86400.0 / events_day
            interval_sec = optimal_interval_seconds(profile.checkpoint_cost_sec, mean_between_sec)
            fixed_overhead_fraction = fixed_checkpoint_overhead_fraction(
                checkpoint_cost_sec=profile.checkpoint_cost_sec,
                interval_sec=interval_sec,
            )
            total_waste_fraction = expected_total_waste_fraction(
                checkpoint_cost_sec=profile.checkpoint_cost_sec,
                resume_penalty_sec=profile.resume_penalty_sec,
                interval_sec=interval_sec,
                mean_between_events_sec=mean_between_sec,
            )
            rows.append(
                {
                    "profile_key": profile.key,
                    "label": profile.label,
                    "robustness": profile.robustness,
                    "events_per_day": events_day,
                    "mean_hours_between_events": 24.0 / events_day,
                    "optimal_interval_min": interval_sec / 60.0,
                    "fixed_checkpoint_overhead_percent": fixed_overhead_fraction * 100.0,
                    "total_waste_percent": total_waste_fraction * 100.0,
                    "checkpoint_cost_ms": profile.checkpoint_cost_sec * 1000.0,
                    "resume_penalty_ms": profile.resume_penalty_sec * 1000.0,
                }
            )
    return rows


def compute_human_interval_rows(
    profiles: tuple[CheckpointProfile, ...],
    policy: HumanIntervalPolicy,
) -> list[dict]:
    rows: list[dict] = []
    for profile in profiles:
        for interval_sec in policy.friendly_intervals_sec:
            save_overhead_fraction = save_only_overhead_fraction(
                checkpoint_cost_sec=profile.checkpoint_cost_sec,
                interval_sec=interval_sec,
            )
            rows.append(
                {
                    "profile_key": profile.key,
                    "label": profile.label,
                    "robustness": profile.robustness,
                    "policy_label": policy.label,
                    "interval_sec": interval_sec,
                    "interval_label": format_interval_label(interval_sec),
                    "interval_min": interval_sec / 60.0,
                    "save_only_overhead_percent": save_overhead_fraction * 100.0,
                    "max_allowed_overhead_percent": policy.save_only_overhead_cap_fraction * 100.0,
                    "passes": save_overhead_fraction <= policy.save_only_overhead_cap_fraction,
                }
            )
    return rows


def compute_human_interval_boundary_rows(human_interval_rows: list[dict]) -> list[dict]:
    by_key: dict[str, list[dict]] = {}
    for row in human_interval_rows:
        by_key.setdefault(row["profile_key"], []).append(row)

    boundary_rows: list[dict] = []
    for profile_rows in by_key.values():
        ordered = sorted(profile_rows, key=lambda row: row["interval_sec"])
        last_fail = None
        first_pass = None
        for row in ordered:
            if row["passes"]:
                first_pass = row
                break
            last_fail = row
        if last_fail is not None:
            boundary_rows.append(last_fail)
        if first_pass is not None:
            boundary_rows.append(first_pass)
        if last_fail is None and first_pass is None and ordered:
            boundary_rows.append(ordered[0])
    return boundary_rows


def compute_human_interval_recommendations(
    profiles: tuple[CheckpointProfile, ...],
    policy: HumanIntervalPolicy,
) -> list[dict]:
    rows: list[dict] = []
    for profile in profiles:
        chosen_interval_sec = None
        chosen_overhead_fraction = None
        for interval_sec in policy.friendly_intervals_sec:
            save_overhead_fraction = save_only_overhead_fraction(
                checkpoint_cost_sec=profile.checkpoint_cost_sec,
                interval_sec=interval_sec,
            )
            if save_overhead_fraction <= policy.save_only_overhead_cap_fraction:
                chosen_interval_sec = interval_sec
                chosen_overhead_fraction = save_overhead_fraction
                break
        hourly_overhead_fraction = save_only_overhead_fraction(
            checkpoint_cost_sec=profile.checkpoint_cost_sec,
            interval_sec=policy.anchor_interval_sec,
        )
        if chosen_interval_sec is None:
            minimum_interval_sec = profile.checkpoint_cost_sec / policy.save_only_overhead_cap_fraction
            rows.append(
                {
                    "profile_key": profile.key,
                    "label": profile.label,
                    "recommended_max_interval_min": minimum_interval_sec / 60.0,
                    "recommended_interval_label": f">= {minimum_interval_sec / 60.0:.2f} min",
                    "reason": (
                        f"No friendly interval stays under the fixed {policy.save_only_overhead_cap_fraction * 100.0:.3f}% "
                        "save-only overhead cap; a custom interval this large would be required."
                    ),
                }
            )
        else:
            failing_intervals = [
                format_interval_label(interval_sec)
                for interval_sec in policy.friendly_intervals_sec
                if interval_sec < chosen_interval_sec
            ]
            if failing_intervals:
                failure_note = f"{', '.join(failing_intervals)} fail"
            else:
                failure_note = "no shorter friendly interval was tested"
            rows.append(
                {
                    "profile_key": profile.key,
                    "label": profile.label,
                    "recommended_max_interval_min": chosen_interval_sec / 60.0,
                    "recommended_interval_label": format_interval_label(chosen_interval_sec),
                    "reason": (
                        f"{format_interval_label(chosen_interval_sec)} is the shortest friendly interval under the fixed "
                        f"{policy.save_only_overhead_cap_fraction * 100.0:.3f}% save-only overhead cap "
                        f"({failure_note}); hourly anchor overhead is {hourly_overhead_fraction * 100.0:.4f}%."
                    ),
                }
            )
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown(
    path: Path,
    scenarios: list[dict],
    human_interval_boundary_rows: list[dict],
    human_interval_recommendations: list[dict],
) -> None:
    ensure_parent(path)
    lines = [
        "# Checkpoint Tradeoff Scenarios",
        "",
        "| Profile | Robustness | Events/day | Mean hours between resumes | Optimal interval (min) | Fixed checkpoint overhead (%) | Projected total waste (%) | Save cost (ms) | Resume penalty (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenarios:
        lines.append(
            "| {label} | {robustness} | {events_per_day:.2f} | {mean_hours_between_events:.2f} | {optimal_interval_min:.2f} | {fixed_checkpoint_overhead_percent:.3f} | {total_waste_percent:.3f} | {checkpoint_cost_ms:.0f} | {resume_penalty_ms:.0f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "# Human-Factors Boundary Checks",
            "",
            "| Profile | Interval | Save-only overhead (%) | Allowed overhead (%) | Pass |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in human_interval_boundary_rows:
        lines.append(
            "| {label} | {interval_label} | {save_only_overhead_percent:.4f} | {max_allowed_overhead_percent:.3f} | {passes} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "# Human-Factors Recommendations",
            "",
            "| Profile | Recommended max interval | Reason |",
            "| --- | ---: | --- |",
        ]
    )
    for row in human_interval_recommendations:
        lines.append(
            f"| {row['label']} | {row['recommended_interval_label']} | {row['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def make_plot(path: Path, rows: list[dict], profiles: tuple[CheckpointProfile, ...]) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    by_key: dict[str, list[dict]] = {profile.key: [] for profile in profiles}
    for row in rows:
        by_key[row["profile_key"]].append(row)

    for profile in profiles:
        series = by_key[profile.key]
        events_per_day = np.array([row["events_per_day"] for row in series])
        optimal_minutes = np.array([row["optimal_interval_min"] for row in series])
        fixed_checkpoint_overhead_percent = np.array([row["fixed_checkpoint_overhead_percent"] for row in series])
        total_waste_percent = np.array([row["total_waste_percent"] for row in series])
        label = (
            f"{profile.label} "
            f"({profile.checkpoint_cost_sec * 1000.0:.0f} ms/save, "
            f"{profile.resume_penalty_sec * 1000.0:.0f} ms/resume)"
        )
        axes[0].plot(events_per_day, optimal_minutes, label=label, linewidth=2.0)
        axes[1].plot(events_per_day, total_waste_percent, label=f"{label} total", linewidth=2.0)
        axes[1].plot(
            events_per_day,
            fixed_checkpoint_overhead_percent,
            label=f"{label} fixed overhead",
            linewidth=1.5,
            linestyle="--",
        )

    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_title("Checkpoint Policy Tradeoff on Apple Silicon")
    axes[0].set_ylabel("Optimal checkpoint interval (minutes)")
    axes[1].set_ylabel("Expected wasted wall-clock (%)")
    axes[1].set_xlabel("Expected resume-needed events per day")
    axes[0].legend(loc="upper right")

    def events_per_day_to_hours_between(values):
        array = np.asarray(values, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        np.divide(24.0, array, out=result, where=array > 0.0)
        return result

    def hours_between_to_events_per_day(values):
        array = np.asarray(values, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        np.divide(24.0, array, out=result, where=array > 0.0)
        return result

    top_axis = axes[0].secondary_xaxis(
        "top",
        functions=(events_per_day_to_hours_between, hours_between_to_events_per_day),
    )
    top_axis.set_xlabel("Mean hours between resume-needed events")

    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequencies_per_day = scenario_rates(args.scenarios_per_day)
    profiles = DEFAULT_PROFILES
    human_policy = DEFAULT_HUMAN_INTERVAL_POLICY
    output_prefix = Path(args.output_prefix)

    grid_rows = compute_grid(
        profiles,
        min_events_per_day=args.min_events_per_day,
        max_events_per_day=args.max_events_per_day,
        num_points=args.num_points,
    )
    scenario_rows = compute_scenarios(profiles, frequencies_per_day)
    human_interval_rows = compute_human_interval_rows(profiles, human_policy)
    human_interval_boundary_rows = compute_human_interval_boundary_rows(human_interval_rows)
    human_interval_recommendations = compute_human_interval_recommendations(profiles, human_policy)

    plot_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path = output_prefix.with_suffix(".json")

    make_plot(plot_path, grid_rows, profiles)
    write_csv(csv_path, grid_rows)
    write_markdown(md_path, scenario_rows, human_interval_boundary_rows, human_interval_recommendations)
    write_json(
        json_path,
        {
            "profiles": [asdict(profile) for profile in profiles],
            "scenarios": scenario_rows,
            "human_interval_policy": asdict(human_policy),
            "human_interval_checks": human_interval_rows,
            "human_interval_boundary_checks": human_interval_boundary_rows,
            "human_interval_recommendations": human_interval_recommendations,
            "plot": str(plot_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        },
    )

    print(f"plot: {plot_path}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(f"json: {json_path}")
    print()
    print("| Profile | Events/day | Hours between resumes | Optimal interval (min) | Fixed checkpoint overhead (%) | Projected total waste (%) | Save cost (ms) | Resume penalty (ms) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in scenario_rows:
        print(
            f"| {row['label']} | {row['events_per_day']:.2f} | {row['mean_hours_between_events']:.2f} | "
            f"{row['optimal_interval_min']:.2f} | {row['fixed_checkpoint_overhead_percent']:.3f} | "
            f"{row['total_waste_percent']:.3f} | "
            f"{row['checkpoint_cost_ms']:.0f} | {row['resume_penalty_ms']:.0f} |"
        )
    print()
    print("| Profile | Recommended max interval | Reason |")
    print("| --- | ---: | --- |")
    for row in human_interval_recommendations:
        print(
            f"| {row['label']} | {row['recommended_interval_label']} | {row['reason']} |"
        )


if __name__ == "__main__":
    main()
