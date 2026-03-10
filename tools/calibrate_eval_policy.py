#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    EvalRungSpec,
    default_eval_batch_size,
)
from autoresearch_mlx.model import GPT, GPTConfig
from autoresearch_mlx.optim import MuonAdamW
from train_mlx import PRESETS


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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


def run_train_batch_sweep(args) -> dict:
    rows: list[dict] = []
    for batch_size in args.device_batches:
        cmd = [
            sys.executable,
            "train_mlx.py",
            "--preset",
            args.preset,
            "--time-budget",
            str(args.time_budget),
            "--benchmark-skip-eval",
            "--no-checkpoint",
            "--device-batch-size",
            str(batch_size),
        ]
        if args.total_batch_size is not None:
            cmd.extend(["--total-batch-size", str(args.total_batch_size)])
        started = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True)
        wall_seconds = time.perf_counter() - started
        row = {
            "preset": args.preset,
            "device_batch_size": batch_size,
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
        "mode": "train-batch",
        "preset": args.preset,
        "time_budget": args.time_budget,
        "total_batch_size": args.total_batch_size,
        "rows": rows,
        "best_by_throughput": best,
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
    eval_rungs.add_argument("--hardware-key", default=DEFAULT_EVAL_HARDWARE_KEY)
    eval_rungs.add_argument("--no-prepacked-cache", action="store_true")
    eval_rungs.add_argument("--json-out")
    eval_rungs.add_argument("--markdown-out")

    train_batch = subparsers.add_parser("train-batch", help="Sweep device batch sizes with short training runs.")
    train_batch.add_argument("--preset", required=True, choices=tuple(PRESETS))
    train_batch.add_argument("--time-budget", type=float, default=5.0)
    train_batch.add_argument("--device-batches", type=parse_int_list, required=True)
    train_batch.add_argument("--total-batch-size", type=int)
    train_batch.add_argument("--json-out")
    train_batch.add_argument("--markdown-out")

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
    else:
        payload = run_train_batch_sweep(args)
        rows = payload["rows"]
        write_markdown(
            args.markdown_out,
            rows,
            columns=[
                ("device_batch_size", "device batch"),
                ("status", "status"),
                ("steady_state_tok_per_sec", "steady tok/s"),
                ("peak_vram_mb", "peak MB"),
            ],
        )

    write_json(getattr(args, "json_out", None), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
