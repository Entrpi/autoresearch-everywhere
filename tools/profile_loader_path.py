#!/usr/bin/env python3
"""Profile MLX train-step boundaries for prepacked vs live-packed dataloaders."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
from mlx.utils import tree_map

from autoresearch_mlx.data import Tokenizer, make_dataloader
from autoresearch_mlx.model import GPT
from autoresearch_mlx.optim import MuonAdamW
from autoresearch_mlx.train import (
    ADAM_BETAS,
    EMBEDDING_LR,
    MATRIX_LR,
    PRESETS,
    SCALAR_LR,
    UNEMBEDDING_LR,
    WEIGHT_DECAY,
    build_model_config,
    make_apply_grads_fn,
    make_grad_step_fn,
    verify_mlx_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="m5-xlarge")
    parser.add_argument("--steps", type=int, default=80, help="Measured optimizer steps after warmup.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup optimizer steps to exclude from the summary.")
    parser.add_argument(
        "--no-prepacked-cache",
        action="store_true",
        help="Disable prepacked caches and use the live packing path.",
    )
    parser.add_argument("--json-out", help="Optional path to write the summary JSON.")
    return parser.parse_args()


def percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    rank = (len(samples) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(samples) - 1)
    frac = rank - low
    ordered = sorted(samples)
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": 1000.0 * statistics.fmean(samples),
        "median_ms": 1000.0 * statistics.median(samples),
        "p95_ms": 1000.0 * percentile(samples, 0.95),
    }


def main() -> None:
    args = parse_args()
    verify_mlx_env()
    mx.random.seed(42)

    preset = PRESETS[args.preset]
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

    grad_step = make_grad_step_fn(model)
    apply_grads = make_apply_grads_fn(model, optimizer)

    loader = make_dataloader(
        tokenizer,
        preset.device_batch_size,
        preset.seq_len,
        "train",
        prefer_prepacked_cache=not args.no_prepacked_cache,
    )
    grad_accum_steps = preset.total_batch_size // (preset.device_batch_size * preset.seq_len)

    def empty_totals():
        return {
            "loader_call": [],
            "materialize": [],
            "grad": [],
            "accumulate": [],
            "optimizer": [],
            "total": [],
        }

    totals = empty_totals()
    warmup_totals = empty_totals()

    measured_tokens = 0
    measured_steps = 0
    last_epoch = None
    for step_idx in range(args.warmup_steps + args.steps):
        total_loss = None
        total_grads = None
        t_total_start = time.perf_counter()
        last_epoch = 1
        loader_call_seconds = 0.0
        materialize_seconds = 0.0
        grad_seconds = 0.0
        accumulate_seconds = 0.0

        for _ in range(grad_accum_steps):
            t_loader = time.perf_counter()
            batch_inputs, batch_targets, last_epoch = next(loader)
            loader_call_seconds += time.perf_counter() - t_loader

            t_materialize = time.perf_counter()
            mx.eval(batch_inputs, batch_targets)
            materialize_seconds += time.perf_counter() - t_materialize

            t_grad = time.perf_counter()
            loss, grads = grad_step(batch_inputs, batch_targets)
            mx.eval(loss, grads)
            grad_seconds += time.perf_counter() - t_grad

            t_accumulate = time.perf_counter()
            scaled_loss = loss / grad_accum_steps
            scaled_grads = tree_map(lambda grad: grad / grad_accum_steps, grads)
            if total_grads is None:
                total_loss = scaled_loss
                total_grads = scaled_grads
            else:
                total_loss = total_loss + scaled_loss
                total_grads = tree_map(lambda left, right: left + right, total_grads, scaled_grads)
            mx.eval(total_loss, total_grads)
            accumulate_seconds += time.perf_counter() - t_accumulate

        t_optimizer = time.perf_counter()
        optimizer_step = apply_grads(total_grads)
        mx.eval(optimizer_step, model.state, optimizer.state)
        optimizer_seconds = time.perf_counter() - t_optimizer
        total_seconds = time.perf_counter() - t_total_start

        target = totals if step_idx >= args.warmup_steps else warmup_totals
        target["loader_call"].append(loader_call_seconds)
        target["materialize"].append(materialize_seconds)
        target["grad"].append(grad_seconds)
        target["accumulate"].append(accumulate_seconds)
        target["optimizer"].append(optimizer_seconds)
        target["total"].append(total_seconds)
        if step_idx >= args.warmup_steps:
            measured_steps += 1
            measured_tokens += preset.total_batch_size

    summary = {
        "preset": args.preset,
        "prefer_prepacked_cache": not args.no_prepacked_cache,
        "loader_type": type(loader).__name__,
        "seq_len": preset.seq_len,
        "device_batch_size": preset.device_batch_size,
        "total_batch_size": preset.total_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "warmup_steps": args.warmup_steps,
        "measured_steps": measured_steps,
        "measured_tokens": measured_tokens,
        "epoch": last_epoch,
        "tok_per_sec": measured_tokens / max(sum(totals["total"]), 1e-9),
        "warmup": {
            "steps": args.warmup_steps,
            "total": summarize(warmup_totals["total"]),
            "loader_call": summarize(warmup_totals["loader_call"]),
            "materialize": summarize(warmup_totals["materialize"]),
            "grad": summarize(warmup_totals["grad"]),
            "accumulate": summarize(warmup_totals["accumulate"]),
            "optimizer": summarize(warmup_totals["optimizer"]),
        },
        "total": summarize(totals["total"]),
        "loader_call": summarize(totals["loader_call"]),
        "materialize": summarize(totals["materialize"]),
        "grad": summarize(totals["grad"]),
        "accumulate": summarize(totals["accumulate"]),
        "optimizer": summarize(totals["optimizer"]),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
