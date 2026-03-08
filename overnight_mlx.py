#!/usr/bin/env python3
"""
Round-robin overnight sweep runner for the MLX autoresearch port.

This does not mutate code. It executes a curated set of M5-safe training shapes,
captures per-run logs, and appends a compact summary to results.tsv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_mlx.py"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
RESULTS_FILE = ROOT / "results.tsv"
ARTIFACTS_DIR = ROOT / "results" / "overnight"
RESULTS_HEADER = ["commit", "val_bpb", "memory_gb", "status", "description"]
SUMMARY_PATTERN = re.compile(r"^([a-z_]+):\s+(.+)$")


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    args: tuple[str, ...]


@dataclass
class RunState:
    run_tag: str
    plan: str
    started_at: str
    deadline: str
    branch: str
    commit: str
    completed: int = 0
    best_bpb: float | None = None
    best_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an overnight MLX experiment sweep.")
    parser.add_argument("--run-tag", help="Run tag used for logs and state files.")
    parser.add_argument("--plan", default="m5-overnight", choices=("m5-overnight",), help="Experiment plan.")
    parser.add_argument("--duration-hours", type=float, default=8.0, help="Wall-clock duration for the sweep.")
    parser.add_argument("--max-experiments", type=int, help="Optional cap on the number of experiments.")
    parser.add_argument("--base-seed", type=int, default=1000, help="Seed assigned to the first run.")
    parser.add_argument("--time-budget-override", type=float, help="Optional override for train_mlx.py --time-budget.")
    parser.add_argument("--eval-tokens-override", type=int, help="Optional override for train_mlx.py --eval-tokens.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running.")
    return parser.parse_args()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_results_tsv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(RESULTS_HEADER)


def build_plan(name: str) -> list[Experiment]:
    if name != "m5-overnight":
        raise ValueError(f"Unknown plan: {name}")
    return [
        Experiment(
            name="balanced_baseline",
            description="Default M5 baseline.",
            args=("--preset", "m5-balanced"),
        ),
        Experiment(
            name="balanced_bigbatch",
            description="Balanced shape with more gradient accumulation.",
            args=("--preset", "m5-balanced", "--total-batch-size", "4096"),
        ),
        Experiment(
            name="balanced_smallbatch",
            description="Balanced shape with lower total batch size.",
            args=("--preset", "m5-balanced", "--total-batch-size", "1024"),
        ),
        Experiment(
            name="mid_context",
            description="Slightly shorter context to trade context for updates.",
            args=(
                "--preset",
                "m5-balanced",
                "--seq-len",
                "384",
                "--device-batch-size",
                "4",
                "--total-batch-size",
                "1536",
            ),
        ),
        Experiment(
            name="large_768",
            description="Larger model with moderated context length.",
            args=(
                "--preset",
                "m5-large",
                "--seq-len",
                "768",
                "--device-batch-size",
                "2",
                "--total-batch-size",
                "3072",
            ),
        ),
        Experiment(
            name="large_1024",
            description="Largest M5-safe baseline from the initial profiling pass.",
            args=("--preset", "m5-large"),
        ),
    ]


def parse_summary_metrics(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    in_summary = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "---":
            in_summary = True
            continue
        if not in_summary or not line:
            continue
        match = SUMMARY_PATTERN.match(line)
        if match:
            metrics[match.group(1)] = match.group(2)
    return metrics


def append_result(path: Path, row: list[str]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def experiment_command(experiment: Experiment, seed: int, args: argparse.Namespace) -> list[str]:
    command = [str(PYTHON_BIN), str(TRAIN_SCRIPT), *experiment.args, "--seed", str(seed)]
    if args.time_budget_override is not None:
        command.extend(["--time-budget", str(args.time_budget_override)])
    if args.eval_tokens_override is not None:
        command.extend(["--eval-tokens", str(args.eval_tokens_override)])
    return command


def stream_command(command: list[str], log_path: Path) -> tuple[int, str]:
    output_lines: list[str] = []
    with log_path.open("w") as log_file:
        log_file.write(f"$ {' '.join(shlex.quote(part) for part in command)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
            output_lines.append(line)
        return_code = process.wait()
    return return_code, "".join(output_lines)


def iter_round_robin(items: Iterable[Experiment]) -> Iterable[Experiment]:
    sequence = tuple(items)
    while True:
        yield from sequence


def main() -> None:
    args = parse_args()
    if not PYTHON_BIN.exists():
        raise SystemExit(f"Missing virtualenv interpreter: {PYTHON_BIN}")

    start = datetime.now().astimezone()
    deadline = start + timedelta(hours=args.duration_hours)
    run_tag = args.run_tag or f"overnight-{start.strftime('%Y%m%d-%H%M%S')}"
    branch = run_git("branch", "--show-current")
    commit = run_git("rev-parse", "--short", "HEAD")
    experiments = build_plan(args.plan)

    if args.dry_run:
        print(f"run_tag: {run_tag}")
        print(f"branch:  {branch}")
        print(f"commit:  {commit}")
        print(f"start:   {start.isoformat()}")
        print(f"stop:    {deadline.isoformat()}")
        for index, experiment in enumerate(experiments, start=1):
            print(f"{index:02d}. {experiment.name} :: {experiment.description}")
            print(f"    {' '.join(experiment.args)}")
        return

    ensure_results_tsv(RESULTS_FILE)
    run_dir = ARTIFACTS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    state = RunState(
        run_tag=run_tag,
        plan=args.plan,
        started_at=start.isoformat(),
        deadline=deadline.isoformat(),
        branch=branch,
        commit=commit,
    )
    metadata = {
        "run_tag": run_tag,
        "plan": args.plan,
        "branch": branch,
        "commit": commit,
        "started_at": start.isoformat(),
        "deadline": deadline.isoformat(),
        "duration_hours": args.duration_hours,
        "base_seed": args.base_seed,
        "time_budget_override": args.time_budget_override,
        "eval_tokens_override": args.eval_tokens_override,
        "experiments": [asdict(experiment) for experiment in experiments],
    }
    write_json(run_dir / "metadata.json", metadata)

    for experiment in iter_round_robin(experiments):
        if args.max_experiments is not None and state.completed >= args.max_experiments:
            break
        if datetime.now().astimezone() >= deadline:
            break

        seed = args.base_seed + state.completed
        log_name = f"{state.completed:03d}_{experiment.name}_seed{seed}.log"
        log_path = run_dir / log_name
        command = experiment_command(experiment, seed, args)
        print()
        print(
            f"[{state.completed:03d}] {experiment.name} | seed={seed} | "
            f"{datetime.now().astimezone().isoformat()}"
        )
        return_code, output = stream_command(command, log_path)
        metrics = parse_summary_metrics(output)

        status = "crash"
        val_bpb = ""
        memory_gb = ""
        if return_code == 0 and "val_bpb" in metrics and "peak_vram_mb" in metrics:
            val = float(metrics["val_bpb"])
            memory = float(metrics["peak_vram_mb"]) / 1024.0
            val_bpb = f"{val:.6f}"
            memory_gb = f"{memory:.3f}"
            if state.best_bpb is None or val < state.best_bpb:
                status = "keep"
                state.best_bpb = val
                state.best_name = experiment.name
            else:
                status = "discard"

        description = (
            f"run_tag={run_tag} branch={branch} exp={experiment.name} seed={seed} "
            f"desc={experiment.description} args=\"{' '.join(experiment.args)}\""
        )
        append_result(RESULTS_FILE, [commit, val_bpb, memory_gb, status, description])

        state.completed += 1
        write_json(run_dir / "state.json", asdict(state))
        if status == "crash":
            print(f"[{state.completed:03d}] crash recorded for {experiment.name}")
        else:
            print(
                f"[{state.completed:03d}] {experiment.name} -> val_bpb={val_bpb} "
                f"memory_gb={memory_gb} status={status}"
            )

        time.sleep(1)

    write_json(run_dir / "state.json", asdict(state))
    print()
    print(f"run_tag:   {run_tag}")
    print(f"completed: {state.completed}")
    print(f"best_bpb:  {state.best_bpb if state.best_bpb is not None else 'n/a'}")
    print(f"best_name: {state.best_name or 'n/a'}")
    print(f"run_dir:   {run_dir}")


if __name__ == "__main__":
    main()
