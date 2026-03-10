#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from autoresearch_mlx.checkpoints import load_checkpoint_metadata, restore_checkpoint
from autoresearch_mlx.constants import CANONICAL_EVAL_SEQ_LEN, CANONICAL_EVAL_TOKENS
from autoresearch_mlx.data import Tokenizer, evaluate_bpb, make_dataloader
from autoresearch_mlx.eval_policy import (
    DEFAULT_EVAL_HARDWARE_KEY,
    DEFAULT_EVAL_RUNGS,
    EVAL_POLICY_VERSION,
    EvalRungSpec,
    detect_current_hardware_key,
    default_eval_batch_size,
)
from autoresearch_mlx.eval_telemetry import summarize_eval_telemetry
from autoresearch_mlx.model import GPT, GPTConfig
from autoresearch_mlx.optim import MuonAdamW
from train_mlx import PRESETS


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_string_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_summary(stdout: str) -> dict[str, str | float | int]:
    result: dict[str, str | float | int] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        value = raw.strip()
        if not value:
            continue
        try:
            if any(char in value for char in ".eE"):
                parsed: str | float | int = float(value)
            else:
                parsed = int(value)
        except ValueError:
            parsed = value
        result[key] = parsed
    return result


def ensure_checkpoint(
    *,
    preset: str,
    checkpoint: str | None,
    train_seconds: float,
    force_retrain: bool,
) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)
    checkpoint_path = Path(f"/tmp/autoresearch_{preset.replace('-', '_')}_{int(train_seconds)}s_eval_calibration")
    metadata = checkpoint_path / "checkpoint.json"
    if metadata.exists() and not force_retrain:
        return checkpoint_path
    cmd = [
        sys.executable,
        "train_mlx.py",
        "--preset",
        preset,
        "--time-budget",
        str(train_seconds),
        "--benchmark-skip-eval",
        "--checkpoint-path",
        str(checkpoint_path),
    ]
    subprocess.run(cmd, check=True)
    return checkpoint_path


def restore_checkpoint_runtime(checkpoint_dir: Path, *, min_sequence_len: int | None = None):
    metadata = load_checkpoint_metadata(checkpoint_dir)
    run_config = metadata["run_config"]
    tokenizer = Tokenizer.from_directory()
    model_config = GPTConfig(**metadata["model_config"])
    if min_sequence_len is not None and model_config.sequence_len < min_sequence_len:
        model_config.sequence_len = min_sequence_len
    model = GPT(model_config)
    model.init_weights()
    model.ensure_runtime_caches(model_config.sequence_len)
    optimizer = MuonAdamW(model)
    train_loader = make_dataloader(
        tokenizer,
        int(run_config["device_batch_size"]),
        int(run_config["seq_len"]),
        "train",
        prefer_prepacked_cache=bool(run_config.get("prefer_prepacked_cache", True)),
    )
    restore_checkpoint(str(checkpoint_dir), model=model, optimizer=optimizer, train_loader=train_loader)
    return metadata, run_config, tokenizer, model


def write_json(path: str | None, payload: dict) -> None:
    if path is None:
        return
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: str | None, rows: list[dict], *, columns: list[tuple[str, str]]) -> None:
    if path is None:
        return
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    Path(path).write_text("\n".join([header, separator, *body]) + "\n")


def run_eval_batch_sweep(args) -> dict:
    checkpoint_dir = Path(args.checkpoint)
    _, _, tokenizer, model = restore_checkpoint_runtime(
        checkpoint_dir,
        min_sequence_len=args.seq_len,
    )
    rows: list[dict] = []
    for batch_size in args.batches:
        mx.clear_cache()
        t0 = time.perf_counter()
        val_bpb = evaluate_bpb(
            model,
            tokenizer,
            batch_size,
            seq_len=args.seq_len,
            eval_tokens=args.eval_tokens,
            prefer_prepacked_cache=not args.no_prepacked_cache,
            eval_slices=args.eval_slices,
            reference_eval_tokens=args.reference_eval_tokens,
        )
        eval_seconds = time.perf_counter() - t0
        rows.append(
            {
                "batch_size": batch_size,
                "seq_len": args.seq_len,
                "eval_tokens": args.eval_tokens,
                "eval_slices": args.eval_slices,
                "reference_eval_tokens": args.reference_eval_tokens,
                "val_bpb": float(val_bpb),
                "eval_seconds": eval_seconds,
            }
        )
    best = min(rows, key=lambda row: row["eval_seconds"])
    return {
        "mode": "eval-batch",
        "checkpoint": str(checkpoint_dir),
        "rows": rows,
        "best_by_time": best,
    }


def selected_rungs(keys: list[str]) -> list[EvalRungSpec]:
    order = {spec.key: spec for spec in DEFAULT_EVAL_RUNGS}
    missing = [key for key in keys if key not in order]
    if missing:
        raise ValueError(f"Unknown rung keys: {missing}")
    return [order[key] for key in keys]


def run_eval_rungs(args) -> dict:
    checkpoint_dir = Path(args.checkpoint)
    _, _, tokenizer, model = restore_checkpoint_runtime(
        checkpoint_dir,
        min_sequence_len=args.seq_len,
    )
    rows: list[dict] = []
    for spec in selected_rungs(args.rungs):
        mx.clear_cache()
        t0 = time.perf_counter()
        val_bpb = evaluate_bpb(
            model,
            tokenizer,
            args.batch_size,
            seq_len=args.seq_len,
            eval_tokens=spec.eval_tokens,
            prefer_prepacked_cache=not args.no_prepacked_cache,
            eval_slices=spec.eval_slices,
            reference_eval_tokens=spec.reference_eval_tokens,
        )
        eval_seconds = time.perf_counter() - t0
        rows.append(
            {
                "rung": spec.key,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "eval_tokens": spec.eval_tokens,
                "eval_slices": spec.eval_slices,
                "reference_eval_tokens": spec.reference_eval_tokens,
                "val_bpb": float(val_bpb),
                "eval_seconds": eval_seconds,
            }
        )
    full_row = next((row for row in rows if row["rung"] == "full"), None)
    if full_row is not None:
        full_bpb = full_row["val_bpb"]
        full_seconds = full_row["eval_seconds"]
        for row in rows:
            row["abs_error_vs_full"] = abs(row["val_bpb"] - full_bpb)
            row["speedup_vs_full"] = full_seconds / row["eval_seconds"] if row["eval_seconds"] > 0 else None
            for budget in args.budget_seconds:
                row[f"overhead_{int(budget)}s"] = row["eval_seconds"] / budget
    return {
        "mode": "eval-rungs",
        "preset": args.preset,
        "hardware_key": args.hardware_key,
        "checkpoint": str(checkpoint_dir),
        "rows": rows,
    }


def resolve_train_grid_axes(args) -> tuple[list[int], list[int], list[int], list[str]]:
    preset = PRESETS[args.preset]
    device_batches = args.device_batches or [preset.device_batch_size]
    if args.total_batch_size is not None and args.total_batches is not None:
        raise ValueError("Use either --total-batch-size or --total-batches, not both.")
    if args.total_batch_size is not None:
        total_batches = [args.total_batch_size]
    elif args.total_batches is not None:
        total_batches = args.total_batches
    else:
        total_batches = [preset.total_batch_size]
    seq_lens = args.seq_lens or [preset.seq_len]
    window_patterns = args.window_patterns or [preset.window_pattern]
    return device_batches, total_batches, seq_lens, window_patterns


def run_train_grid_sweep(args) -> dict:
    device_batches, total_batches, seq_lens, window_patterns = resolve_train_grid_axes(args)
    rows: list[dict] = []
    for seq_len, window_pattern, total_batch_size, batch_size in product(
        seq_lens,
        window_patterns,
        total_batches,
        device_batches,
    ):
        tokens_per_fwdbwd = batch_size * seq_len
        grad_accum_steps = total_batch_size // tokens_per_fwdbwd if total_batch_size % tokens_per_fwdbwd == 0 else None
        cmd = [
            sys.executable,
            "train_mlx.py",
            "--preset",
            args.preset,
            "--time-budget",
            str(args.time_budget),
            "--benchmark-skip-eval",
            "--no-checkpoint",
            "--seq-len",
            str(seq_len),
            "--window-pattern",
            window_pattern,
            "--device-batch-size",
            str(batch_size),
            "--total-batch-size",
            str(total_batch_size),
        ]
        started = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True)
        wall_seconds = time.perf_counter() - started
        row = {
            "preset": args.preset,
            "seq_len": seq_len,
            "window_pattern": window_pattern,
            "device_batch_size": batch_size,
            "total_batch_size": total_batch_size,
            "tokens_per_fwdbwd": tokens_per_fwdbwd,
            "grad_accum_steps": grad_accum_steps,
            "status": "ok" if completed.returncode == 0 else "error",
            "returncode": completed.returncode,
            "wall_seconds": wall_seconds,
        }
        if completed.returncode == 0:
            summary = parse_summary(completed.stdout)
            row.update(
                {
                    "steady_state_tok_per_sec": summary.get("steady_state_tok_per_sec"),
                    "peak_vram_mb": summary.get("peak_vram_mb"),
                    "training_seconds": summary.get("training_seconds"),
                    "num_steps": summary.get("num_steps"),
                    "mfu_percent": summary.get("mfu_percent"),
                }
            )
        else:
            row["error_tail"] = "\n".join(completed.stderr.splitlines()[-8:])
        rows.append(row)
    ok_rows = [row for row in rows if row["status"] == "ok" and row.get("steady_state_tok_per_sec") is not None]
    best = max(ok_rows, key=lambda row: row["steady_state_tok_per_sec"]) if ok_rows else None
    return {
        "mode": "train-grid",
        "preset": args.preset,
        "time_budget": args.time_budget,
        "device_batches": device_batches,
        "total_batches": total_batches,
        "seq_lens": seq_lens,
        "window_patterns": window_patterns,
        "rows": rows,
        "best_by_throughput": best,
    }


def run_telemetry_summary(args) -> dict:
    summary = summarize_eval_telemetry(
        args.preset,
        hardware_key=args.hardware_key,
        policy_version=args.policy_version,
    )
    return {
        "mode": "telemetry-summary",
        "preset": args.preset,
        "hardware_key": args.hardware_key,
        "policy_version": args.policy_version,
        "eligible_count": summary.eligible_count,
        "commit_count": summary.commit_count,
        "day_count": summary.day_count,
        "observed_rungs": list(summary.observed_rungs),
        "stable_rungs": list(summary.stable_rungs),
        "last_seen_on": summary.last_seen_on,
        "last_seen_age_days": summary.last_seen_age_days,
        "rung_stats": [
            {
                "rung": stat.rung_key,
                "count": stat.count,
                "commit_count": stat.commit_count,
                "day_count": stat.day_count,
                "median_eval_seconds": stat.median_eval_seconds,
                "rel_mad_eval_seconds": stat.rel_mad_eval_seconds,
                "stable_timing": stat.stable_timing,
            }
            for stat in summary.rung_stats
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate preset- and hardware-specific training/eval policy values.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_batch = subparsers.add_parser("eval-batch", help="Sweep eval batch sizes on an existing checkpoint.")
    eval_batch.add_argument("--checkpoint", required=True)
    eval_batch.add_argument("--seq-len", type=int, default=CANONICAL_EVAL_SEQ_LEN)
    eval_batch.add_argument("--eval-tokens", type=int, default=CANONICAL_EVAL_TOKENS)
    eval_batch.add_argument("--eval-slices", type=int, default=1)
    eval_batch.add_argument("--reference-eval-tokens", type=int)
    eval_batch.add_argument("--batches", type=parse_int_list, default=parse_int_list("1,2,4,8,16,32,64,128,256"))
    eval_batch.add_argument("--no-prepacked-cache", action="store_true")
    eval_batch.add_argument("--json-out")
    eval_batch.add_argument("--markdown-out")

    eval_rungs = subparsers.add_parser("eval-rungs", help="Measure cheap/reference/full eval rungs on a checkpoint.")
    eval_rungs.add_argument("--preset", required=True, choices=tuple(PRESETS))
    eval_rungs.add_argument("--checkpoint")
    eval_rungs.add_argument("--train-seconds", type=float, default=120.0)
    eval_rungs.add_argument("--force-retrain", action="store_true")
    eval_rungs.add_argument("--seq-len", type=int, default=CANONICAL_EVAL_SEQ_LEN)
    eval_rungs.add_argument("--batch-size", type=int, default=default_eval_batch_size(CANONICAL_EVAL_SEQ_LEN))
    eval_rungs.add_argument("--rungs", type=lambda value: [item.strip() for item in value.split(",") if item.strip()], default=["cheap", "reference", "full"])
    eval_rungs.add_argument("--budget-seconds", type=parse_float_list, default=parse_float_list("300,28800"))
    eval_rungs.add_argument("--hardware-key", default=detect_current_hardware_key())
    eval_rungs.add_argument("--no-prepacked-cache", action="store_true")
    eval_rungs.add_argument("--json-out")
    eval_rungs.add_argument("--markdown-out")

    train_grid = subparsers.add_parser(
        "train-grid",
        aliases=["train-batch"],
        help="Sweep short training operating points across constrained batch and shape axes.",
    )
    train_grid.add_argument("--preset", required=True, choices=tuple(PRESETS))
    train_grid.add_argument("--time-budget", type=float, default=5.0)
    train_grid.add_argument(
        "--device-batches",
        type=parse_int_list,
        help="Comma-separated device_batch_size values. Defaults to the preset value.",
    )
    train_grid.add_argument(
        "--total-batches",
        type=parse_int_list,
        help="Comma-separated total_batch_size values. Defaults to the preset value.",
    )
    train_grid.add_argument(
        "--total-batch-size",
        type=int,
        help="Single total_batch_size value. Use --total-batches for a grid.",
    )
    train_grid.add_argument(
        "--seq-lens",
        type=parse_int_list,
        help="Comma-separated training seq_len values. Defaults to the preset value.",
    )
    train_grid.add_argument(
        "--window-patterns",
        type=parse_string_list,
        help="Comma-separated window_pattern values. Defaults to the preset value.",
    )
    train_grid.add_argument("--json-out")
    train_grid.add_argument("--markdown-out")

    telemetry_summary = subparsers.add_parser(
        "telemetry-summary",
        help="Summarize passive eval telemetry coverage for a preset/hardware/policy combination.",
    )
    telemetry_summary.add_argument("--preset", required=True, choices=tuple(PRESETS))
    telemetry_summary.add_argument("--hardware-key", default=detect_current_hardware_key())
    telemetry_summary.add_argument("--policy-version", type=int, default=EVAL_POLICY_VERSION)
    telemetry_summary.add_argument("--json-out")
    telemetry_summary.add_argument("--markdown-out")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "eval-batch":
        payload = run_eval_batch_sweep(args)
        rows = payload["rows"]
        write_markdown(
            args.markdown_out,
            rows,
            columns=[
                ("batch_size", "batch"),
                ("eval_seconds", "eval sec"),
                ("val_bpb", "val_bpb"),
            ],
        )
    elif args.command == "eval-rungs":
        checkpoint_dir = ensure_checkpoint(
            preset=args.preset,
            checkpoint=args.checkpoint,
            train_seconds=args.train_seconds,
            force_retrain=args.force_retrain,
        )
        args.checkpoint = str(checkpoint_dir)
        payload = run_eval_rungs(args)
        rows = payload["rows"]
        write_markdown(
            args.markdown_out,
            rows,
            columns=[
                ("rung", "rung"),
                ("eval_tokens", "eval tokens"),
                ("eval_seconds", "eval sec"),
                ("val_bpb", "val_bpb"),
                ("abs_error_vs_full", "abs error vs full"),
            ],
        )
    elif args.command == "telemetry-summary":
        payload = run_telemetry_summary(args)
        rows = payload["rung_stats"]
        write_markdown(
            args.markdown_out,
            rows,
            columns=[
                ("rung", "rung"),
                ("count", "count"),
                ("commit_count", "commits"),
                ("day_count", "days"),
                ("median_eval_seconds", "median eval sec"),
                ("rel_mad_eval_seconds", "rel MAD"),
                ("stable_timing", "stable"),
            ],
        )
    else:
        payload = run_train_grid_sweep(args)
        rows = payload["rows"]
        write_markdown(
            args.markdown_out,
            rows,
            columns=[
                ("seq_len", "seq len"),
                ("window_pattern", "window"),
                ("device_batch_size", "device batch"),
                ("total_batch_size", "total batch"),
                ("grad_accum_steps", "grad accum"),
                ("status", "status"),
                ("steady_state_tok_per_sec", "steady tok/s"),
                ("peak_vram_mb", "peak MB"),
            ],
        )

    write_json(getattr(args, "json_out", None), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
