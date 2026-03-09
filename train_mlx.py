"""
Autoresearch pretraining script for MLX on Apple Silicon.
Usage: uv run train_mlx.py
"""

import argparse
import gc
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from autoresearch_mlx.constants import (
    CANONICAL_EVAL_BATCH_SIZE,
    CANONICAL_EVAL_SEQ_LEN,
    CANONICAL_EVAL_TOKENS,
    MAX_SEQ_LEN,
    PROXY_EVAL_TOKENS,
    TIME_BUDGET,
)
from autoresearch_mlx.checkpoints import load_checkpoint_metadata, restore_checkpoint, save_checkpoint
from autoresearch_mlx.data import Tokenizer, evaluate_bpb, make_dataloader
from autoresearch_mlx.model import GPT, GPTConfig
from autoresearch_mlx.optim import MuonAdamW


@dataclass(frozen=True)
class RunPreset:
    description: str
    seq_len: int
    eval_tokens: int
    canonical_eval_seq_len: int
    canonical_eval_tokens: int
    canonical_eval_batch_size: int
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int


@dataclass(frozen=True)
class RunConfig:
    preset: str
    time_budget: float
    seq_len: int
    eval_tokens: int
    canonical_eval_seq_len: int
    canonical_eval_tokens: int
    canonical_eval_batch_size: int
    depth: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int
    seed: int
    smoke: bool
    prefer_prepacked_cache: bool
    checkpoint_path: str | None
    checkpoint_interval: float | None
    resume_from: str | None


def verify_mlx_env() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(f"train_mlx.py requires macOS. Detected platform: {sys.platform}")
    if not mx.is_available(mx.gpu) or not mx.metal.is_available():
        raise RuntimeError("MLX GPU/Metal is not available. This script expects Apple Silicon with Metal.")
    print("Environment verified: macOS detected with MLX Metal GPU support available.")
    print()


def build_model_config(depth: int, vocab_size: int, *, sequence_len: int, window_pattern: str) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    if WARMDOWN_RATIO == 0:
        return FINAL_LR_FRAC
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float) -> float:
    return WEIGHT_DECAY * (1.0 - progress)


def make_grad_step_fn(model):
    def loss_fn(model, inputs, targets):
        return model(inputs, targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def grad_step(inputs, targets):
        return loss_and_grad(model, inputs, targets)

    return grad_step


def make_apply_grads_fn(model, optimizer):
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def apply_grads(grads):
        optimizer.update(model, grads)
        return optimizer.state["step"]

    return apply_grads


def run_train_step(loader, grad_step, apply_grads, grad_accum_steps: int, model, optimizer):
    total_loss = None
    total_grads = None
    epoch = 1

    for _ in range(grad_accum_steps):
        batch_inputs, batch_targets, epoch = next(loader)
        loss, grads = grad_step(batch_inputs, batch_targets)
        mx.eval(loss, grads)

        scaled_loss = loss / grad_accum_steps
        scaled_grads = tree_map(lambda grad: grad / grad_accum_steps, grads)
        if total_grads is None:
            total_loss = scaled_loss
            total_grads = scaled_grads
        else:
            total_loss = total_loss + scaled_loss
            total_grads = tree_map(lambda left, right: left + right, total_grads, scaled_grads)
        mx.eval(total_loss, total_grads)

    optimizer_step = apply_grads(total_grads)
    mx.eval(optimizer_step, model.state, optimizer.state)
    return total_loss, epoch


# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128

# Optimization
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

PRESETS = {
    "m5-fast": RunPreset(
        description="Fast local iteration on Apple Silicon.",
        seq_len=256,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=CANONICAL_EVAL_BATCH_SIZE,
        depth=2,
        window_pattern="L",
        device_batch_size=2,
        total_batch_size=512,
    ),
    "m5-balanced": RunPreset(
        description="Default M5 baseline with materially better throughput than the upstream shape.",
        seq_len=512,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=CANONICAL_EVAL_BATCH_SIZE,
        depth=4,
        window_pattern="L",
        device_batch_size=4,
        total_batch_size=2048,
    ),
    "m5-large": RunPreset(
        description="Larger M5 run when you want more model capacity and can accept slower updates.",
        seq_len=1024,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=CANONICAL_EVAL_BATCH_SIZE,
        depth=6,
        window_pattern="L",
        device_batch_size=2,
        total_batch_size=4096,
    ),
    "m5-xlarge": RunPreset(
        description="Upstream-scale model shape with an M5-sized batch and dense attention.",
        seq_len=2048,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=CANONICAL_EVAL_BATCH_SIZE,
        depth=8,
        window_pattern="L",
        device_batch_size=2,
        total_batch_size=4096,
    ),
    "upstream": RunPreset(
        description="Original upstream-shaped MLX port for reference, closest to the H100-oriented defaults.",
        seq_len=MAX_SEQ_LEN,
        eval_tokens=PROXY_EVAL_TOKENS,
        canonical_eval_seq_len=CANONICAL_EVAL_SEQ_LEN,
        canonical_eval_tokens=CANONICAL_EVAL_TOKENS,
        canonical_eval_batch_size=CANONICAL_EVAL_BATCH_SIZE,
        depth=8,
        window_pattern="SSSL",
        device_batch_size=8,
        total_batch_size=2**16,
    ),
}
DEFAULT_PRESET = "m5-balanced"


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    preset_name = args.preset or DEFAULT_PRESET
    preset = PRESETS[preset_name]
    config = RunConfig(
        preset=preset_name,
        time_budget=TIME_BUDGET,
        seq_len=preset.seq_len,
        eval_tokens=preset.eval_tokens,
        canonical_eval_seq_len=preset.canonical_eval_seq_len,
        canonical_eval_tokens=preset.canonical_eval_tokens,
        canonical_eval_batch_size=preset.canonical_eval_batch_size,
        depth=preset.depth,
        window_pattern=preset.window_pattern,
        device_batch_size=preset.device_batch_size,
        total_batch_size=preset.total_batch_size,
        seed=42 if args.seed is None else args.seed,
        smoke=args.smoke,
        prefer_prepacked_cache=not args.no_prepacked_cache,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )

    if args.smoke:
        smoke_seq_len = min(config.seq_len, 256)
        smoke_depth = min(config.depth, 2)
        smoke_device_batch_size = min(config.device_batch_size, 2)
        config = replace(
            config,
            time_budget=1.0,
            seq_len=smoke_seq_len,
            eval_tokens=smoke_seq_len * smoke_device_batch_size,
            canonical_eval_seq_len=smoke_seq_len,
            canonical_eval_tokens=smoke_seq_len * smoke_device_batch_size,
            canonical_eval_batch_size=smoke_device_batch_size,
            depth=smoke_depth,
            window_pattern="L",
            device_batch_size=smoke_device_batch_size,
            total_batch_size=smoke_seq_len * smoke_device_batch_size,
        )

    overrides = {}
    for field in (
        "time_budget",
        "seq_len",
        "eval_tokens",
        "canonical_eval_seq_len",
        "canonical_eval_tokens",
        "canonical_eval_batch_size",
        "depth",
        "window_pattern",
        "device_batch_size",
        "total_batch_size",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if overrides:
        config = replace(config, **overrides)

    return config


def resolve_resume_config(args: argparse.Namespace) -> RunConfig:
    disallowed = []
    for field in (
        "preset",
        "seq_len",
        "eval_tokens",
        "canonical_eval_seq_len",
        "canonical_eval_tokens",
        "canonical_eval_batch_size",
        "depth",
        "window_pattern",
        "device_batch_size",
        "total_batch_size",
        "seed",
    ):
        if getattr(args, field) is not None:
            disallowed.append(field)
    if args.smoke:
        disallowed.append("smoke")
    if args.no_prepacked_cache:
        disallowed.append("no_prepacked_cache")
    if disallowed:
        raise ValueError(
            "--resume-from restores the saved run configuration. Only "
            "--time-budget, --checkpoint-path, and --checkpoint-interval may be overridden. "
            f"Got overrides for: {', '.join(disallowed)}"
        )

    metadata = load_checkpoint_metadata(args.resume_from)
    run_config = dict(metadata["run_config"])
    run_config.setdefault("checkpoint_path", None)
    run_config.setdefault("checkpoint_interval", None)
    run_config["time_budget"] = args.time_budget if args.time_budget is not None else run_config["time_budget"]
    run_config["checkpoint_path"] = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else run_config["checkpoint_path"] or args.resume_from
    )
    run_config["checkpoint_interval"] = (
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else run_config["checkpoint_interval"]
    )
    run_config["resume_from"] = args.resume_from
    return RunConfig(**run_config)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run autoresearch pretraining with MLX on Apple Silicon.")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        help="Named runtime preset. Defaults to the M5-friendly balanced preset.",
    )
    parser.add_argument("--time-budget", type=float, help="Training budget in seconds.")
    parser.add_argument("--seq-len", type=int, help="Sequence length for training and proxy evaluation.")
    parser.add_argument("--eval-tokens", type=int, help="Proxy validation token budget.")
    parser.add_argument(
        "--canonical-eval-seq-len",
        type=int,
        help="Fixed evaluation sequence length used for cross-preset comparisons.",
    )
    parser.add_argument(
        "--canonical-eval-tokens",
        type=int,
        help="Fixed evaluation token budget used for cross-preset comparisons.",
    )
    parser.add_argument(
        "--canonical-eval-batch-size",
        type=int,
        help="Batch size for fixed canonical evaluation.",
    )
    parser.add_argument("--depth", type=int, help="Number of transformer blocks.")
    parser.add_argument(
        "--window-pattern",
        help="Sliding-window pattern (S/L) repeated across layers.",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        help="Per-step device batch size.",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        help="Total batch size in tokens across gradient accumulation.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short end-to-end sanity check with smaller defaults.",
    )
    parser.add_argument(
        "--no-prepacked-cache",
        action="store_true",
        help="Disable optional prepacked row caches and use the live packing path instead.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Directory to save a resumable training checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        help="Save a checkpoint every N training seconds. Requires --checkpoint-path.",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume training from a checkpoint directory.",
    )
    args = parser.parse_args()
    if args.resume_from:
        return resolve_resume_config(args)
    return resolve_run_config(args)


def main() -> None:
    args = parse_args()
    if args.canonical_eval_seq_len > args.seq_len:
        raise ValueError(
            "canonical_eval_seq_len cannot exceed training seq_len. "
            "Lower --canonical-eval-seq-len or increase --seq-len."
        )
    verify_mlx_env()
    t_start = time.time()
    mx.random.seed(args.seed)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = build_model_config(
        args.depth,
        vocab_size,
        sequence_len=args.seq_len,
        window_pattern=args.window_pattern,
    )
    print(f"Model config: {asdict(config)}")
    print(f"Run preset: {args.preset} ({PRESETS[args.preset].description})")
    print(
        "Run config: "
        f"time_budget={args.time_budget}s, seq_len={args.seq_len}, eval_tokens={args.eval_tokens}, "
        f"canonical_eval_seq_len={args.canonical_eval_seq_len}, "
        f"canonical_eval_tokens={args.canonical_eval_tokens}, "
        f"canonical_eval_batch_size={args.canonical_eval_batch_size}, "
        f"device_batch_size={args.device_batch_size}, total_batch_size={args.total_batch_size}, "
        f"smoke={args.smoke}, prefer_prepacked_cache={args.prefer_prepacked_cache}, "
        f"checkpoint_path={args.checkpoint_path}, checkpoint_interval={args.checkpoint_interval}, "
        f"resume_from={args.resume_from}"
    )

    model = GPT(config)
    model.init_weights()
    model.ensure_runtime_caches(max(args.seq_len, args.canonical_eval_seq_len))
    mx.eval(model.state)

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = args.device_batch_size * args.seq_len
    if args.total_batch_size % tokens_per_fwdbwd != 0:
        raise ValueError("total_batch_size must be divisible by device_batch_size * seq_len")
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

    optimizer = MuonAdamW(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_dataloader(
        tokenizer,
        args.device_batch_size,
        args.seq_len,
        "train",
        prefer_prepacked_cache=args.prefer_prepacked_cache,
    )
    if args.resume_from:
        restored = restore_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        print(
            f"Resumed from {args.resume_from}: step={restored['step']}, "
            f"training_seconds={restored['total_training_time']:.1f}"
        )
    else:
        restored = {
            "step": 0,
            "total_training_time": 0.0,
            "smooth_train_loss": 0.0,
        }
    grad_step = make_grad_step_fn(model)
    apply_grads = make_apply_grads_fn(model, optimizer)

    print(f"Time budget: {args.time_budget}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    if args.checkpoint_interval is not None and args.checkpoint_path is None:
        raise ValueError("--checkpoint-interval requires --checkpoint-path")

    mx.reset_peak_memory()
    t_start_training = time.time()
    smooth_train_loss = float(restored["smooth_train_loss"])
    total_training_time = float(restored["total_training_time"])
    step = int(restored["step"])
    last_checkpoint_time = total_training_time
    checkpoint_run_config = asdict(replace(args, resume_from=None))

    def maybe_save_checkpoint(*, force: bool = False) -> None:
        nonlocal last_checkpoint_time
        if args.checkpoint_path is None:
            return
        if force and total_training_time == last_checkpoint_time:
            return
        if not force:
            if args.checkpoint_interval is None:
                return
            if (total_training_time - last_checkpoint_time) < args.checkpoint_interval:
                return
        save_checkpoint(
            args.checkpoint_path,
            run_config=checkpoint_run_config,
            model_config=asdict(config),
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            step=step,
            total_training_time=total_training_time,
            smooth_train_loss=smooth_train_loss,
        )
        last_checkpoint_time = total_training_time
        print(f"\nCheckpoint saved to {args.checkpoint_path} at step {step}.")

    while total_training_time < args.time_budget:
        progress = min(total_training_time / args.time_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        optimizer.set_schedule(
            lr_multiplier=lrm,
            muon_momentum=muon_momentum,
            muon_weight_decay=muon_weight_decay,
        )

        t0 = time.time()
        loss, epoch = run_train_step(
            train_loader,
            grad_step,
            apply_grads,
            grad_accum_steps,
            model,
            optimizer,
        )
        t1 = time.time()
        dt = t1 - t0

        train_loss = loss.item()
        if train_loss > 100:
            print("FAIL")
            raise SystemExit(1)

        total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(args.total_batch_size / dt)
        remaining = max(0.0, args.time_budget - total_training_time)
        mfu = 0.0
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        maybe_save_checkpoint()

    print()
    maybe_save_checkpoint(force=True)

    total_tokens = step * args.total_batch_size
    model.eval()
    proxy_val_bpb = evaluate_bpb(
        model,
        tokenizer,
        args.device_batch_size,
        seq_len=args.seq_len,
        eval_tokens=args.eval_tokens,
        prefer_prepacked_cache=args.prefer_prepacked_cache,
    )
    val_bpb = evaluate_bpb(
        model,
        tokenizer,
        args.canonical_eval_batch_size,
        seq_len=args.canonical_eval_seq_len,
        eval_tokens=args.canonical_eval_tokens,
        prefer_prepacked_cache=args.prefer_prepacked_cache,
    )
    t_end = time.time()
    steady_state_mfu = 0.0
    peak_vram_mb = mx.get_peak_memory() / 1024 / 1024

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"proxy_val_bpb:    {proxy_val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {args.depth}")
    print(f"proxy_eval_tokens: {args.eval_tokens}")
    print(f"canonical_seq_len: {args.canonical_eval_seq_len}")
    print(f"canonical_tokens: {args.canonical_eval_tokens}")
    print(f"canonical_batch:  {args.canonical_eval_batch_size}")


if __name__ == "__main__":
    main()
