#!/usr/bin/env python3
"""Measure resumed-run readiness as wall time to the first completed optimizer step."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from autoresearch_mlx.checkpoints import (
    CHECKPOINT_MODE_EXACT,
    CHECKPOINT_MODES,
    restore_checkpoint,
    save_checkpoint,
)
from autoresearch_mlx.data import Tokenizer, make_dataloader
from autoresearch_mlx.model import GPT
from autoresearch_mlx.optim import MuonAdamW
from autoresearch_mlx.train import (
    ADAM_BETAS,
    EMBEDDING_LR,
    MATRIX_LR,
    PRESETS,
    PRESET_CHOICES,
    SCALAR_LR,
    UNEMBEDDING_LR,
    WEIGHT_DECAY,
    build_model_config,
    make_apply_grads_fn,
    make_grad_step_fn,
    run_train_step,
    verify_mlx_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="m5-balanced")
    parser.add_argument(
        "--checkpoint-train-steps",
        type=int,
        default=5,
        help="Optimizer steps to run before saving the checkpoint that will be resumed.",
    )
    parser.add_argument(
        "--resume-steps",
        type=int,
        default=2,
        help="Number of optimizer steps to run after resuming. The first is the resume-ready target; the second is steady-state context.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent checkpoint-and-resume trials to run.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=CHECKPOINT_MODES,
        default=CHECKPOINT_MODE_EXACT,
        help="Checkpoint semantics to benchmark.",
    )
    parser.add_argument(
        "--no-prepacked-cache",
        action="store_true",
        help="Disable prepacked caches and use the live-packed loader path.",
    )
    parser.add_argument("--json-out", help="Optional path to write the summary JSON.")
    parser.add_argument(
        "--checkpoint-dir",
        help="Optional directory to reuse instead of an auto-created temporary checkpoint directory.",
    )
    parser.add_argument(
        "--keep-checkpoint-dir",
        action="store_true",
        help="Keep the checkpoint directory on disk after profiling.",
    )
    return parser.parse_args()


def build_runtime(preset_name: str, *, prefer_prepacked_cache: bool, compile_fns: bool = True):
    preset = PRESETS[preset_name]
    tokenizer = Tokenizer.from_directory()
    config = build_model_config(
        preset.depth,
        tokenizer.get_vocab_size(),
        sequence_len=preset.seq_len,
        window_pattern=preset.window_pattern,
    )
    model = GPT(config)
    model.init_weights()
    model.ensure_runtime_caches(preset.seq_len)
    mx.eval(model.state)

    optimizer = MuonAdamW(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.set_schedule(lr_multiplier=1.0, muon_momentum=0.95, muon_weight_decay=WEIGHT_DECAY)

    train_loader = make_dataloader(
        tokenizer,
        preset.device_batch_size,
        preset.seq_len,
        "train",
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    grad_step = make_grad_step_fn(model) if compile_fns else None
    apply_grads = make_apply_grads_fn(model, optimizer) if compile_fns else None
    grad_accum_steps = preset.total_batch_size // (preset.device_batch_size * preset.seq_len)
    return tokenizer, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps


def run_checkpoint_seed_steps(
    *,
    train_steps: int,
    train_loader,
    grad_step,
    apply_grads,
    grad_accum_steps: int,
    model,
    optimizer,
) -> tuple[int, float]:
    smooth_train_loss = 0.0
    for step in range(train_steps):
        loss, _, _ = run_train_step(
            train_loader,
            grad_step,
            apply_grads,
            grad_accum_steps,
            model,
            optimizer,
        )
        smooth_train_loss = float(loss.item())
    return train_steps, smooth_train_loss


def timed(callable_):
    t_start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - t_start


def percentile(sorted_samples: list[float], pct: float) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (len(sorted_samples) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_samples) - 1)
    frac = rank - low
    return sorted_samples[low] * (1.0 - frac) + sorted_samples[high] * frac


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": sum(ordered) / len(ordered),
        "median": percentile(ordered, 0.5),
        "p95": percentile(ordered, 0.95),
    }


def measure_once(args: argparse.Namespace, *, repeat_index: int) -> dict:
    mx.random.seed(42)

    prefer_prepacked_cache = not args.no_prepacked_cache
    preset = PRESETS[args.preset]

    seed_runtime = build_runtime(args.preset, prefer_prepacked_cache=prefer_prepacked_cache)
    _, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps = seed_runtime
    step, smooth_train_loss = run_checkpoint_seed_steps(
        train_steps=args.checkpoint_train_steps,
        train_loader=train_loader,
        grad_step=grad_step,
        apply_grads=apply_grads,
        grad_accum_steps=grad_accum_steps,
        model=model,
        optimizer=optimizer,
    )

    temp_root = None
    if args.checkpoint_dir:
        checkpoint_root = Path(args.checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="autoresearch-resume-ready-"))
        checkpoint_root = temp_root / "checkpoint"

    save_seconds = save_checkpoint(
        checkpoint_root,
        checkpoint_mode=args.checkpoint_mode,
        run_config={"preset": args.preset},
        model_config=asdict(config),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        step=step,
        total_training_time=0.0,
        smooth_train_loss=smooth_train_loss,
        step_telemetry=None,
        total_checkpoint_time=0.0,
        checkpoint_count=0,
    )

    mx.random.seed(42)
    t_resume_start = time.perf_counter()
    (
        _,
        _restore_config,
        restore_model,
        restore_optimizer,
        restore_loader,
        _restore_grad_step,
        _restore_apply_grads,
        restore_grad_accum_steps,
    ), runtime_build_seconds = timed(
        lambda: build_runtime(args.preset, prefer_prepacked_cache=prefer_prepacked_cache, compile_fns=False)
    )
    (_, restore_seconds) = timed(
        lambda: restore_checkpoint(
            checkpoint_root,
            model=restore_model,
            optimizer=restore_optimizer,
            train_loader=restore_loader,
        )
    )
    (restore_grad_step, restore_apply_grads), function_setup_seconds = timed(
        lambda: (
            make_grad_step_fn(restore_model),
            make_apply_grads_fn(restore_model, restore_optimizer),
        )
    )

    resumed_steps: list[dict[str, float | int]] = []
    for _ in range(args.resume_steps):
        (step_result, step_seconds) = timed(
            lambda: run_train_step(
                restore_loader,
                restore_grad_step,
                restore_apply_grads,
                restore_grad_accum_steps,
                restore_model,
                restore_optimizer,
            )
        )
        loss, epoch, step_timing = step_result
        resumed_steps.append(
            {
                "loss": float(loss.item()),
                "epoch": int(epoch),
                "wall_seconds": step_seconds,
                "train_step_seconds": step_timing.total_seconds,
                "loader_seconds": step_timing.loader_seconds,
                "grad_seconds": step_timing.grad_seconds,
                "accumulate_seconds": step_timing.accumulate_seconds,
                "optimizer_seconds": step_timing.optimizer_seconds,
            }
        )

    first_step = resumed_steps[0]
    second_step = resumed_steps[1] if len(resumed_steps) > 1 else None
    trial_wall_seconds = time.perf_counter() - t_resume_start

    summary = {
        "repeat_index": repeat_index,
        "preset": args.preset,
        "prefer_prepacked_cache": prefer_prepacked_cache,
        "checkpoint_mode": args.checkpoint_mode,
        "loader_type": type(restore_loader).__name__,
        "seq_len": preset.seq_len,
        "total_batch_size": preset.total_batch_size,
        "checkpoint_train_steps": args.checkpoint_train_steps,
        "resume_steps": args.resume_steps,
        "checkpoint_save_seconds": save_seconds,
        "resume_ready": {
            "runtime_build_seconds": runtime_build_seconds,
            "restore_seconds": restore_seconds,
            "function_setup_seconds": function_setup_seconds,
            "first_step_wall_seconds": first_step["wall_seconds"],
            "first_step_train_seconds": first_step["train_step_seconds"],
            "first_step_loader_seconds": first_step["loader_seconds"],
            "first_step_grad_seconds": first_step["grad_seconds"],
            "first_step_accumulate_seconds": first_step["accumulate_seconds"],
            "first_step_optimizer_seconds": first_step["optimizer_seconds"],
            "resume_ready_seconds": runtime_build_seconds + restore_seconds + function_setup_seconds + first_step["wall_seconds"],
            "trial_wall_seconds": trial_wall_seconds,
            "first_step_loss": first_step["loss"],
        },
        "steady_step": second_step,
        "checkpoint_dir": str(checkpoint_root),
    }

    if temp_root is not None and not args.keep_checkpoint_dir:
        shutil.rmtree(temp_root)
    return summary


def aggregate_trials(trials: list[dict]) -> dict:
    return {
        "checkpoint_save_seconds": summarize([trial["checkpoint_save_seconds"] for trial in trials]),
        "resume_ready_seconds": summarize([trial["resume_ready"]["resume_ready_seconds"] for trial in trials]),
        "trial_wall_seconds": summarize([trial["resume_ready"]["trial_wall_seconds"] for trial in trials]),
        "runtime_build_seconds": summarize([trial["resume_ready"]["runtime_build_seconds"] for trial in trials]),
        "restore_seconds": summarize([trial["resume_ready"]["restore_seconds"] for trial in trials]),
        "function_setup_seconds": summarize([trial["resume_ready"]["function_setup_seconds"] for trial in trials]),
        "first_step_wall_seconds": summarize([trial["resume_ready"]["first_step_wall_seconds"] for trial in trials]),
        "first_step_train_seconds": summarize([trial["resume_ready"]["first_step_train_seconds"] for trial in trials]),
        "steady_step_wall_seconds": summarize([trial["steady_step"]["wall_seconds"] for trial in trials if trial["steady_step"]]),
        "steady_step_train_seconds": summarize([trial["steady_step"]["train_step_seconds"] for trial in trials if trial["steady_step"]]),
        "resume_tax_vs_steady_wall_seconds": summarize(
            [
                trial["resume_ready"]["resume_ready_seconds"] - trial["steady_step"]["wall_seconds"]
                for trial in trials
                if trial["steady_step"]
            ]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    verify_mlx_env()

    trials = [measure_once(args, repeat_index=index + 1) for index in range(args.repeats)]
    summary = {
        "preset": args.preset,
        "prefer_prepacked_cache": not args.no_prepacked_cache,
        "checkpoint_mode": args.checkpoint_mode,
        "repeats": args.repeats,
        "trials": trials,
        "aggregate": aggregate_trials(trials),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
