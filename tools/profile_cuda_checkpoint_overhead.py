#!/usr/bin/env python3
"""Measure repeated-save CUDA checkpoint overhead against a matched no-checkpoint baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_platform.summary import parse_final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="m5-balanced")
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument(
        "--checkpoint-interval",
        default="5s",
        help="Aggressive explicit interval used to generate many saves during the benchmark window.",
    )
    parser.add_argument(
        "--save-modes",
        default="sync,async",
        help="Comma-separated checkpoint save modes to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--window-pattern")
    parser.add_argument("--total-batch-size", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--device-batch-size", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", help="Directory for logs and summaries.")
    parser.add_argument("--json-out", help="Optional explicit JSON summary path.")
    return parser.parse_args()


def _add_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_train_command(
    args: argparse.Namespace,
    *,
    checkpoint_enabled: bool,
    checkpoint_save_mode: str | None = None,
    checkpoint_path: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "autoresearch_cuda.train",
        "--preset",
        args.preset,
        "--time-budget",
        f"{args.time_budget:g}",
        "--benchmark-skip-eval",
    ]
    _add_optional_arg(command, "--seq-len", args.seq_len)
    _add_optional_arg(command, "--window-pattern", args.window_pattern)
    _add_optional_arg(command, "--total-batch-size", args.total_batch_size)
    _add_optional_arg(command, "--depth", args.depth)
    _add_optional_arg(command, "--device-batch-size", args.device_batch_size)
    if not checkpoint_enabled:
        command.append("--no-checkpoint")
        return command
    command.extend(["--checkpoint-interval", args.checkpoint_interval])
    if checkpoint_save_mode is not None:
        command.extend(["--checkpoint-save-mode", checkpoint_save_mode])
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpointing is enabled")
    command.extend(["--checkpoint-path", str(checkpoint_path)])
    return command


def run_arm(name: str, command: list[str], *, output_dir: Path) -> dict[str, object]:
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}. See {stdout_path} and {stderr_path}."
        )
    summary = dict(parse_final_summary(completed.stdout))
    summary["command"] = command
    summary["stdout_log"] = str(stdout_path)
    summary["stderr_log"] = str(stderr_path)
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric(summary: dict[str, object], key: str, *, default: float = 0.0) -> float:
    value = summary.get(key, default)
    return float(value) if value is not None else default


def summarize_mode(
    *,
    save_mode: str,
    baseline_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    baseline_total_seconds = [_metric(row, "total_seconds") for row in baseline_rows]
    baseline_tokens = [_metric(row, "total_tokens_M") for row in baseline_rows]
    baseline_toks = [_metric(row, "steady_state_tok_per_sec") for row in baseline_rows]

    candidate_total_seconds = [_metric(row, "total_seconds") for row in candidate_rows]
    candidate_tokens = [_metric(row, "total_tokens_M") for row in candidate_rows]
    candidate_toks = [_metric(row, "steady_state_tok_per_sec") for row in candidate_rows]
    checkpoint_percent = [_metric(row, "checkpoint_percent") for row in candidate_rows]
    checkpoint_write_percent = [_metric(row, "checkpoint_write_percent") for row in candidate_rows]
    checkpoint_count = [max(0.0, _metric(row, "checkpoint_count")) for row in candidate_rows]
    cumulative_checkpoint_seconds = [_metric(row, "cumulative_checkpoint_seconds") for row in candidate_rows]
    cumulative_checkpoint_write_seconds = [
        _metric(row, "cumulative_checkpoint_write_seconds") for row in candidate_rows
    ]

    per_save_costs = [
        seconds / count
        for seconds, count in zip(cumulative_checkpoint_seconds, checkpoint_count)
        if count > 0
    ]
    per_save_write_costs = [
        seconds / count
        for seconds, count in zip(cumulative_checkpoint_write_seconds, checkpoint_count)
        if count > 0
    ]

    return {
        "save_mode": save_mode,
        "baseline": baseline_rows,
        "candidate": candidate_rows,
        "mean_total_seconds": _mean(candidate_total_seconds),
        "mean_added_wall_seconds": _mean(candidate_total_seconds) - _mean(baseline_total_seconds),
        "mean_blocking_checkpoint_percent": _mean(checkpoint_percent),
        "mean_total_checkpoint_write_percent": _mean(checkpoint_write_percent),
        "mean_session_tokens_M": _mean(candidate_tokens),
        "mean_baseline_tokens_M": _mean(baseline_tokens),
        "mean_steady_state_tok_per_sec": _mean(candidate_toks),
        "mean_baseline_steady_state_tok_per_sec": _mean(baseline_toks),
        "mean_checkpoint_count": _mean(checkpoint_count),
        "mean_checkpoint_cost_sec": _mean(per_save_costs),
        "mean_checkpoint_write_cost_sec": _mean(per_save_write_costs),
    }


def build_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# CUDA Checkpoint Overhead",
        "",
        f"- Preset: `{summary['preset']}`",
        f"- Time budget: `{summary['time_budget']}`",
        f"- Checkpoint interval: `{summary['checkpoint_interval']}`",
        f"- Repeats: `{summary['repeats']}`",
        "",
        "| Save mode | mean total seconds | mean added wall seconds | mean blocking checkpoint % | mean total checkpoint write % | mean checkpoint count | mean checkpoint cost (ms/save) | mean checkpoint write cost (ms/save) | mean session tokens (M) | mean steady tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {save_mode} | {mean_total_seconds:.2f} | {mean_added_wall_seconds:.3f} | {mean_blocking_checkpoint_percent:.3f} | {mean_total_checkpoint_write_percent:.3f} | {mean_checkpoint_count:.2f} | {mean_checkpoint_cost_ms:.2f} | {mean_checkpoint_write_cost_ms:.2f} | {mean_session_tokens_M:.3f} | {mean_steady_state_tok_per_sec:.1f} |".format(
                save_mode=row["save_mode"],
                mean_total_seconds=row["mean_total_seconds"],
                mean_added_wall_seconds=row["mean_added_wall_seconds"],
                mean_blocking_checkpoint_percent=row["mean_blocking_checkpoint_percent"],
                mean_total_checkpoint_write_percent=row["mean_total_checkpoint_write_percent"],
                mean_checkpoint_count=row["mean_checkpoint_count"],
                mean_checkpoint_cost_ms=row["mean_checkpoint_cost_sec"] * 1000.0,
                mean_checkpoint_write_cost_ms=row["mean_checkpoint_write_cost_sec"] * 1000.0,
                mean_session_tokens_M=row["mean_session_tokens_M"],
                mean_steady_state_tok_per_sec=row["mean_steady_state_tok_per_sec"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    save_modes = [token.strip() for token in args.save_modes.split(",") if token.strip()]
    if not save_modes:
        raise ValueError("--save-modes cannot be empty.")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        tempfile.mkdtemp(prefix="autoresearch-cuda-checkpoint-overhead-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict[str, object]] = []
    candidate_rows_by_mode: dict[str, list[dict[str, object]]] = {mode: [] for mode in save_modes}

    for repeat in range(args.repeats):
        baseline_rows.append(
            run_arm(
                f"baseline_r{repeat + 1}",
                build_train_command(args, checkpoint_enabled=False),
                output_dir=output_dir,
            )
        )
        for save_mode in save_modes:
            checkpoint_dir = checkpoint_root / f"{save_mode}_r{repeat + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            candidate_rows_by_mode[save_mode].append(
                run_arm(
                    f"{save_mode}_r{repeat + 1}",
                    build_train_command(
                        args,
                        checkpoint_enabled=True,
                        checkpoint_save_mode=save_mode,
                        checkpoint_path=checkpoint_dir,
                    ),
                    output_dir=output_dir,
                )
            )

    rows = [
        summarize_mode(
            save_mode=save_mode,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows_by_mode[save_mode],
        )
        for save_mode in save_modes
    ]

    summary = {
        "preset": args.preset,
        "time_budget": args.time_budget,
        "checkpoint_interval": args.checkpoint_interval,
        "repeats": args.repeats,
        "output_dir": str(output_dir),
        "rows": rows,
    }

    summary_path = Path(args.json_out) if args.json_out else output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "summary.md").write_text(build_markdown(summary) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
