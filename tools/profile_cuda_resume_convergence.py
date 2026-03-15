#!/usr/bin/env python3
"""Compare uninterrupted CUDA training against an exact checkpoint/resume split."""

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
    parser.add_argument("--continuous-seconds", type=float, default=601.0)
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=301.0,
        help="Time budget for the first leg that should auto-produce the resume snapshot.",
    )
    parser.add_argument(
        "--resume-seconds",
        type=float,
        help="Final cumulative training-time target for the resumed leg. Defaults to --continuous-seconds.",
    )
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--window-pattern")
    parser.add_argument("--total-batch-size", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--device-batch-size", type=int)
    parser.add_argument("--eval-seq-len", type=int)
    parser.add_argument("--eval-tokens", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument(
        "--continuous-with-checkpoint",
        action="store_true",
        help="Leave checkpointing enabled on the uninterrupted baseline. By default the baseline uses --no-checkpoint.",
    )
    parser.add_argument(
        "--resume-with-checkpoint",
        action="store_true",
        help="Allow the resumed leg to continue writing checkpoints. By default it resumes once and finishes without further saves.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-dir",
        help="Directory to keep logs and summary artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument("--json-out", help="Optional explicit JSON summary path.")
    return parser.parse_args()


def _add_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_train_command(
    args: argparse.Namespace,
    *,
    time_budget: float,
    checkpoint_enabled: bool,
    resume_from: str | None = None,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "autoresearch_cuda.train",
        "--preset",
        args.preset,
        "--time-budget",
        f"{time_budget:g}",
    ]
    _add_optional_arg(command, "--resume-from", resume_from)
    _add_optional_arg(command, "--seq-len", args.seq_len)
    _add_optional_arg(command, "--window-pattern", args.window_pattern)
    _add_optional_arg(command, "--total-batch-size", args.total_batch_size)
    _add_optional_arg(command, "--depth", args.depth)
    _add_optional_arg(command, "--device-batch-size", args.device_batch_size)
    _add_optional_arg(command, "--eval-seq-len", args.eval_seq_len)
    _add_optional_arg(command, "--eval-tokens", args.eval_tokens)
    _add_optional_arg(command, "--eval-batch-size", args.eval_batch_size)
    if not checkpoint_enabled:
        command.append("--no-checkpoint")
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


def build_markdown(summary: dict[str, object]) -> str:
    continuous = summary["continuous"]
    phase1 = summary["phase1"]
    resumed = summary["resumed"]
    comparison = summary["comparison"]
    lines = [
        f"# CUDA Resume Convergence",
        "",
        f"- Preset: `{summary['preset']}`",
        f"- Continuous seconds: `{summary['continuous_seconds']}`",
        f"- Checkpoint seconds: `{summary['checkpoint_seconds']}`",
        f"- Resume seconds: `{summary['resume_seconds']}`",
        f"- Checkpoint directory: `{summary['checkpoint_dir']}`",
        "",
        "| Arm | val_bpb | last_train_loss | smoothed_train_loss | training_seconds | total_tokens_M | num_steps | checkpoint_path |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| continuous | {val_bpb:.6f} | {last_train_loss:.6f} | {smoothed_train_loss:.6f} | {training_seconds:.1f} | {total_tokens_M:.1f} | {num_steps} | {checkpoint_path} |".format(
            val_bpb=float(continuous["val_bpb"]),
            last_train_loss=float(continuous["last_train_loss"]),
            smoothed_train_loss=float(continuous["smoothed_train_loss"]),
            training_seconds=float(continuous["training_seconds"]),
            total_tokens_M=float(continuous["total_tokens_M"]),
            num_steps=int(continuous["num_steps"]),
            checkpoint_path=continuous.get("checkpoint_path", "-"),
        ),
        "| phase1 | {val_bpb:.6f} | {last_train_loss:.6f} | {smoothed_train_loss:.6f} | {training_seconds:.1f} | {total_tokens_M:.1f} | {num_steps} | {checkpoint_path} |".format(
            val_bpb=float(phase1["val_bpb"]),
            last_train_loss=float(phase1["last_train_loss"]),
            smoothed_train_loss=float(phase1["smoothed_train_loss"]),
            training_seconds=float(phase1["training_seconds"]),
            total_tokens_M=float(phase1["total_tokens_M"]),
            num_steps=int(phase1["num_steps"]),
            checkpoint_path=phase1.get("checkpoint_path", "-"),
        ),
        "| resumed | {val_bpb:.6f} | {last_train_loss:.6f} | {smoothed_train_loss:.6f} | {training_seconds:.1f} | {total_tokens_M:.1f} | {num_steps} | {checkpoint_path} |".format(
            val_bpb=float(resumed["val_bpb"]),
            last_train_loss=float(resumed["last_train_loss"]),
            smoothed_train_loss=float(resumed["smoothed_train_loss"]),
            training_seconds=float(resumed["training_seconds"]),
            total_tokens_M=float(resumed["total_tokens_M"]),
            num_steps=int(resumed["num_steps"]),
            checkpoint_path=resumed.get("checkpoint_path", "-"),
        ),
        "",
        "| Comparison | Delta |",
        "| --- | ---: |",
        f"| resumed - continuous val_bpb | {comparison['val_bpb_delta']:.6f} |",
        f"| resumed - continuous last_train_loss | {comparison['last_train_loss_delta']:.6f} |",
        f"| resumed - continuous smoothed_train_loss | {comparison['smoothed_train_loss_delta']:.6f} |",
        f"| resumed - continuous total_tokens_M | {comparison['total_tokens_M_delta']:.1f} |",
        f"| resumed - continuous num_steps | {comparison['num_steps_delta']} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    resume_seconds = args.resume_seconds if args.resume_seconds is not None else args.continuous_seconds
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="autoresearch-cuda-resume-convergence-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path | None = None
    try:
        continuous = run_arm(
            "continuous",
            build_train_command(
                args,
                time_budget=args.continuous_seconds,
                checkpoint_enabled=args.continuous_with_checkpoint,
            ),
            output_dir=output_dir,
        )
        phase1 = run_arm(
            "phase1",
            build_train_command(
                args,
                time_budget=args.checkpoint_seconds,
                checkpoint_enabled=True,
            ),
            output_dir=output_dir,
        )
        checkpoint_path = phase1.get("checkpoint_path")
        if not checkpoint_path:
            raise RuntimeError("Phase1 did not emit a checkpoint_path in its final summary.")
        checkpoint_dir = Path(str(checkpoint_path))
        resumed = run_arm(
            "resumed",
            build_train_command(
                args,
                time_budget=resume_seconds,
                checkpoint_enabled=args.resume_with_checkpoint,
                resume_from=str(checkpoint_dir),
            ),
            output_dir=output_dir,
        )

        comparison = {
            "val_bpb_delta": float(resumed["val_bpb"]) - float(continuous["val_bpb"]),
            "last_train_loss_delta": float(resumed["last_train_loss"]) - float(continuous["last_train_loss"]),
            "smoothed_train_loss_delta": float(resumed["smoothed_train_loss"]) - float(continuous["smoothed_train_loss"]),
            "total_tokens_M_delta": float(resumed["total_tokens_M"]) - float(continuous["total_tokens_M"]),
            "num_steps_delta": int(resumed["num_steps"]) - int(continuous["num_steps"]),
        }
        summary = {
            "preset": args.preset,
            "continuous_seconds": args.continuous_seconds,
            "checkpoint_seconds": args.checkpoint_seconds,
            "resume_seconds": resume_seconds,
            "checkpoint_dir": str(checkpoint_dir),
            "continuous": continuous,
            "phase1": phase1,
            "resumed": resumed,
            "comparison": comparison,
        }

        json_path = Path(args.json_out) if args.json_out else output_dir / "summary.json"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        markdown_path = output_dir / "summary.md"
        markdown_path.write_text(build_markdown(summary) + "\n")
        print(f"json: {json_path}")
        print(f"markdown: {markdown_path}")
    finally:
        print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
