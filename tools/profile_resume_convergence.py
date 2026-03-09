#!/usr/bin/env python3
"""Compare uninterrupted, exact-resume, and weights-only-resume end states."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
from mlx.utils import tree_flatten

from autoresearch_mlx.checkpoints import (
    CHECKPOINT_MODE_EXACT,
    CHECKPOINT_MODE_WEIGHTS_ONLY,
    CHECKPOINT_SAVE_MODE_SYNC,
    CHECKPOINT_SAVE_MODES,
    restore_checkpoint,
    save_checkpoint,
)
from autoresearch_mlx.data import Tokenizer, evaluate_bpb, make_dataloader
from autoresearch_mlx.model import GPT
from autoresearch_mlx.optim import MuonAdamW
from train_mlx import (
    ADAM_BETAS,
    EMBEDDING_LR,
    MATRIX_LR,
    PRESETS,
    SCALAR_LR,
    UNEMBEDDING_LR,
    WEIGHT_DECAY,
    build_model_config,
    get_lr_multiplier,
    get_muon_momentum,
    get_weight_decay,
    make_apply_grads_fn,
    make_grad_step_fn,
    run_train_step,
    verify_mlx_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="m5-large")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=80,
        help="Total optimizer steps in each trajectory.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="Optimizer step at which to save and resume. Defaults to total_steps // 2.",
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=4096,
        help="Canonical evaluation tokens for the final comparison. Set to 0 to skip eval.",
    )
    parser.add_argument(
        "--trailing-loss-window",
        type=int,
        default=8,
        help="Number of final losses to average for a less noisy convergence readout.",
    )
    parser.add_argument(
        "--no-prepacked-cache",
        action="store_true",
        help="Disable prepacked caches and use the live-packed loader path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exact-checkpoint-save-mode",
        choices=CHECKPOINT_SAVE_MODES,
        default=CHECKPOINT_SAVE_MODE_SYNC,
        help="Checkpoint write path to use for the exact midpoint resume arm.",
    )
    parser.add_argument("--json-out", help="Optional path to write the summary JSON.")
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
    model.ensure_runtime_caches(max(preset.seq_len, preset.canonical_eval_seq_len))
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
    return tokenizer, preset, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps


def set_step_schedule(optimizer, *, step: int, total_steps: int) -> None:
    progress = min(step / total_steps, 1.0)
    optimizer.set_schedule(
        lr_multiplier=get_lr_multiplier(progress),
        muon_momentum=get_muon_momentum(step),
        muon_weight_decay=get_weight_decay(progress),
    )


def run_step_budget(
    *,
    target_step: int,
    schedule_total_steps: int,
    start_step: int,
    model,
    optimizer,
    train_loader,
    grad_step,
    apply_grads,
    grad_accum_steps: int,
) -> dict:
    losses: list[float] = []
    total_training_seconds = 0.0
    last_epoch = 1
    step = start_step
    while step < target_step:
        set_step_schedule(optimizer, step=step, total_steps=schedule_total_steps)
        loss, epoch, step_timing = run_train_step(
            train_loader,
            grad_step,
            apply_grads,
            grad_accum_steps,
            model,
            optimizer,
        )
        losses.append(float(loss.item()))
        total_training_seconds += step_timing.total_seconds
        last_epoch = epoch
        step += 1
    return {
        "step": step,
        "losses": losses,
        "training_seconds": total_training_seconds,
        "epoch": last_epoch,
    }


def evaluate_final_state(model, tokenizer, preset, *, eval_tokens: int, prefer_prepacked_cache: bool) -> float | None:
    if eval_tokens <= 0:
        return None
    return evaluate_bpb(
        model,
        tokenizer,
        preset.canonical_eval_batch_size,
        seq_len=preset.canonical_eval_seq_len,
        eval_tokens=eval_tokens,
        prefer_prepacked_cache=prefer_prepacked_cache,
    )


def summarize_losses(losses: list[float], trailing_window: int) -> dict[str, float]:
    if not losses:
        return {"final_loss": 0.0, "trailing_mean_loss": 0.0}
    window = losses[-min(trailing_window, len(losses)) :]
    return {
        "final_loss": losses[-1],
        "trailing_mean_loss": sum(window) / len(window),
    }


def flatten_params(model) -> dict[str, mx.array]:
    return tree_flatten(model.trainable_parameters(), destination={})


def compare_param_states(reference: dict[str, mx.array], candidate: dict[str, mx.array]) -> dict[str, float | bool]:
    if set(reference) != set(candidate):
        raise ValueError("Parameter keys do not match between reference and candidate states.")
    max_abs_diff = 0.0
    diff_sq_sum = 0.0
    ref_sq_sum = 0.0
    numel = 0
    for key in reference:
        ref = reference[key]
        cand = candidate[key]
        diff = ref - cand
        abs_diff = mx.abs(diff)
        max_abs_diff = max(max_abs_diff, float(mx.max(abs_diff).item()))
        diff_sq_sum += float(mx.sum(diff * diff).item())
        ref_sq_sum += float(mx.sum(ref * ref).item())
        numel += ref.size
    rms_diff = (diff_sq_sum / numel) ** 0.5 if numel > 0 else 0.0
    rel_rms_diff = (diff_sq_sum / ref_sq_sum) ** 0.5 if ref_sq_sum > 0.0 else 0.0
    return {
        "param_match": max_abs_diff == 0.0,
        "max_abs_diff": max_abs_diff,
        "rms_diff": rms_diff,
        "relative_rms_diff": rel_rms_diff,
    }


def run_continuous(args: argparse.Namespace, *, checkpoint_step: int) -> tuple[dict, dict[str, mx.array]]:
    prefer_prepacked_cache = not args.no_prepacked_cache
    mx.random.seed(args.seed)
    tokenizer, preset, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps = build_runtime(
        args.preset,
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    run = run_step_budget(
        target_step=args.total_steps,
        schedule_total_steps=args.total_steps,
        start_step=0,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        grad_step=grad_step,
        apply_grads=apply_grads,
        grad_accum_steps=grad_accum_steps,
    )
    val_bpb = evaluate_final_state(
        model,
        tokenizer,
        preset,
        eval_tokens=args.eval_tokens,
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    summary = {
        "trajectory": "continuous",
        "checkpoint_mode": None,
        "checkpoint_step": checkpoint_step,
        "step": run["step"],
        "training_seconds": run["training_seconds"],
        "val_bpb": val_bpb,
        **summarize_losses(run["losses"], args.trailing_loss_window),
    }
    return summary, flatten_params(model)


def run_resumed(
    args: argparse.Namespace,
    *,
    checkpoint_mode: str,
    checkpoint_step: int,
    checkpoint_save_mode: str = CHECKPOINT_SAVE_MODE_SYNC,
) -> tuple[dict, dict[str, mx.array]]:
    prefer_prepacked_cache = not args.no_prepacked_cache
    mx.random.seed(args.seed)
    tokenizer, preset, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps = build_runtime(
        args.preset,
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    first_half = run_step_budget(
        target_step=checkpoint_step,
        schedule_total_steps=args.total_steps,
        start_step=0,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        grad_step=grad_step,
        apply_grads=apply_grads,
        grad_accum_steps=grad_accum_steps,
    )
    temp_root = Path(tempfile.mkdtemp(prefix="autoresearch-resume-convergence-"))
    checkpoint_dir = temp_root / "checkpoint"
    try:
        save_seconds = save_checkpoint(
            checkpoint_dir,
            checkpoint_mode=checkpoint_mode,
            checkpoint_save_mode=checkpoint_save_mode,
            run_config={"preset": args.preset},
            model_config=asdict(config),
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            step=checkpoint_step,
            total_training_time=first_half["training_seconds"],
            smooth_train_loss=first_half["losses"][-1],
            step_telemetry=None,
            total_checkpoint_time=0.0,
            checkpoint_count=0,
        )

        mx.random.seed(args.seed)
        # Build a fresh runtime to match actual resume semantics.
        (
            tokenizer,
            preset,
            config,
            restore_model,
            restore_optimizer,
            restore_loader,
            _restore_grad_step,
            _restore_apply_grads,
            restore_grad_accum_steps,
        ) = build_runtime(
            args.preset,
            prefer_prepacked_cache=prefer_prepacked_cache,
            compile_fns=False,
        )
        restored = restore_checkpoint(
            checkpoint_dir,
            model=restore_model,
            optimizer=restore_optimizer,
            train_loader=restore_loader,
        )
        restore_grad_step = make_grad_step_fn(restore_model)
        restore_apply_grads = make_apply_grads_fn(restore_model, restore_optimizer)
        second_half = run_step_budget(
            target_step=args.total_steps,
            schedule_total_steps=args.total_steps,
            start_step=int(restored["step"]),
            model=restore_model,
            optimizer=restore_optimizer,
            train_loader=restore_loader,
            grad_step=restore_grad_step,
            apply_grads=restore_apply_grads,
            grad_accum_steps=restore_grad_accum_steps,
        )
        losses = first_half["losses"] + second_half["losses"]
        val_bpb = evaluate_final_state(
            restore_model,
            tokenizer,
            preset,
            eval_tokens=args.eval_tokens,
            prefer_prepacked_cache=prefer_prepacked_cache,
        )
        summary = {
            "trajectory": "resumed",
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_save_mode": checkpoint_save_mode,
            "checkpoint_step": checkpoint_step,
            "step": second_half["step"],
            "save_seconds": save_seconds,
            "phase1_training_seconds": first_half["training_seconds"],
            "phase2_training_seconds": second_half["training_seconds"],
            "training_seconds": first_half["training_seconds"] + second_half["training_seconds"],
            "restored_optimizer_state": restored["restored_optimizer_state"],
            "restored_loader_state": restored["restored_loader_state"],
            "val_bpb": val_bpb,
            **summarize_losses(losses, args.trailing_loss_window),
        }
        return summary, flatten_params(restore_model)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def print_table(rows: list[dict]) -> None:
    print(
        "| Trajectory | Final loss | Trailing loss | Canonical val_bpb | Train seconds | Save seconds | Max abs param diff | Rel RMS diff |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        label = row["trajectory"]
        if row["checkpoint_mode"] is not None:
            label = f"{label} ({row['checkpoint_mode']})"
        if row.get("checkpoint_save_mode") not in (None, CHECKPOINT_SAVE_MODE_SYNC):
            label = f"{label}, {row['checkpoint_save_mode']}"
        val_bpb = "skipped" if row["val_bpb"] is None else f"{row['val_bpb']:.6f}"
        save_seconds = row.get("save_seconds")
        save_value = "-" if save_seconds is None else f"{save_seconds:.3f}"
        print(
            f"| {label} | {row['final_loss']:.6f} | {row['trailing_mean_loss']:.6f} | "
            f"{val_bpb} | {row['training_seconds']:.3f} | {save_value} | "
            f"{row['max_abs_diff']:.3e} | {row['relative_rms_diff']:.3e} |"
        )


def main() -> None:
    args = parse_args()
    if args.total_steps <= 1:
        raise ValueError("--total-steps must be greater than 1.")
    checkpoint_step = args.checkpoint_step if args.checkpoint_step is not None else args.total_steps // 2
    if checkpoint_step <= 0 or checkpoint_step >= args.total_steps:
        raise ValueError("--checkpoint-step must be between 1 and total_steps - 1.")

    verify_mlx_env()
    continuous, reference_params = run_continuous(args, checkpoint_step=checkpoint_step)
    exact, exact_params = run_resumed(
        args,
        checkpoint_mode=CHECKPOINT_MODE_EXACT,
        checkpoint_step=checkpoint_step,
        checkpoint_save_mode=args.exact_checkpoint_save_mode,
    )
    weights_only, weights_only_params = run_resumed(
        args,
        checkpoint_mode=CHECKPOINT_MODE_WEIGHTS_ONLY,
        checkpoint_step=checkpoint_step,
    )

    rows = [continuous, exact, weights_only]
    rows[0].update({"param_match": True, "max_abs_diff": 0.0, "rms_diff": 0.0, "relative_rms_diff": 0.0})
    rows[1].update(compare_param_states(reference_params, exact_params))
    rows[2].update(compare_param_states(reference_params, weights_only_params))

    print(f"Preset: {args.preset}")
    print(f"Total steps: {args.total_steps}")
    print(f"Checkpoint step: {checkpoint_step}")
    print(f"Eval tokens: {args.eval_tokens if args.eval_tokens > 0 else 'skipped'}")
    print_table(rows)

    summary = {
        "preset": args.preset,
        "total_steps": args.total_steps,
        "checkpoint_step": checkpoint_step,
        "eval_tokens": args.eval_tokens,
        "exact_checkpoint_save_mode": args.exact_checkpoint_save_mode,
        "prefer_prepacked_cache": not args.no_prepacked_cache,
        "rows": rows,
    }
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"json: {output_path}")


if __name__ == "__main__":
    main()
