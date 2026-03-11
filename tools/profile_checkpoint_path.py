#!/usr/bin/env python3
"""Profile MLX checkpoint save/load cost by phase."""

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
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from autoresearch_mlx.checkpoints import (
    CHECKPOINT_VERSION,
    CHECKPOINT_MODE_EXACT,
    CHECKPOINT_MODES,
    _checkpoint_paths,
    _load_trainable_weights,
    _save_npz,
    _save_safetensors,
    _write_json,
    load_checkpoint_metadata,
)
from autoresearch_mlx.data import Tokenizer, make_dataloader, restore_loader_state, serialize_loader_state
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
    parser.add_argument("--preset", choices=tuple(PRESETS), default="m5-large")
    parser.add_argument("--train-steps", type=int, default=5, help="Optimizer steps to run before profiling save/load.")
    parser.add_argument(
        "--checkpoint-mode",
        choices=CHECKPOINT_MODES,
        default=CHECKPOINT_MODE_EXACT,
        help="Checkpoint semantics to profile.",
    )
    parser.add_argument(
        "--no-prepacked-cache",
        action="store_true",
        help="Disable prepacked caches and profile the live-packed loader path instead.",
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


def build_runtime(preset_name: str, *, prefer_prepacked_cache: bool):
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
    grad_step = make_grad_step_fn(model)
    apply_grads = make_apply_grads_fn(model, optimizer)
    grad_accum_steps = preset.total_batch_size // (preset.device_batch_size * preset.seq_len)
    return tokenizer, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps


def run_training_steps(
    *,
    train_steps: int,
    grad_step,
    apply_grads,
    train_loader,
    grad_accum_steps: int,
):
    smooth_train_loss = 0.0
    last_epoch = 1
    for _ in range(train_steps):
        total_loss = None
        total_grads = None
        for _ in range(grad_accum_steps):
            batch_inputs, batch_targets, last_epoch = next(train_loader)
            loss, grads = grad_step(batch_inputs, batch_targets)
            scaled_loss = loss / grad_accum_steps
            scaled_grads = tree_map(lambda grad: grad / grad_accum_steps, grads)
            if total_grads is None:
                total_loss = scaled_loss
                total_grads = scaled_grads
            else:
                total_loss = total_loss + scaled_loss
                total_grads = tree_map(lambda left, right: left + right, total_grads, scaled_grads)
        optimizer_step = apply_grads(total_grads)
        mx.eval(total_loss, optimizer_step)
        smooth_train_loss = float(total_loss.item())
    return smooth_train_loss, last_epoch


def timed(callable_):
    t_start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - t_start


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> None:
    args = parse_args()
    verify_mlx_env()
    mx.random.seed(42)

    prefer_prepacked_cache = not args.no_prepacked_cache
    preset = PRESETS[args.preset]
    tokenizer, config, model, optimizer, train_loader, grad_step, apply_grads, grad_accum_steps = build_runtime(
        args.preset,
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    smooth_train_loss, last_epoch = run_training_steps(
        train_steps=args.train_steps,
        grad_step=grad_step,
        apply_grads=apply_grads,
        train_loader=train_loader,
        grad_accum_steps=grad_accum_steps,
    )

    temp_root = None
    if args.checkpoint_dir:
        checkpoint_root = Path(args.checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="autoresearch-checkpoint-profile-"))
        checkpoint_root = temp_root / "checkpoint"

    paths = _checkpoint_paths(checkpoint_root)
    paths["root"].mkdir(parents=True, exist_ok=True)

    (model_arrays, flatten_model_seconds) = timed(lambda: tree_flatten(model.trainable_parameters(), destination={}))
    if args.checkpoint_mode == CHECKPOINT_MODE_EXACT:
        (optimizer_arrays, flatten_optimizer_seconds) = timed(lambda: tree_flatten(optimizer.state, destination={}))
        ((loader_metadata, loader_arrays), serialize_loader_seconds) = timed(lambda: serialize_loader_state(train_loader))
    else:
        optimizer_arrays = {}
        flatten_optimizer_seconds = 0.0
        loader_metadata = None
        loader_arrays = {}
        serialize_loader_seconds = 0.0
    (_, model_write_seconds) = timed(lambda: _save_safetensors(paths["model"], model_arrays))
    if args.checkpoint_mode == CHECKPOINT_MODE_EXACT:
        (_, optimizer_write_seconds) = timed(lambda: _save_safetensors(paths["optimizer"], optimizer_arrays))
        (_, loader_write_seconds) = timed(lambda: _save_npz(paths["loader"], loader_arrays))
    else:
        optimizer_write_seconds = 0.0
        loader_write_seconds = 0.0
    (_, metadata_write_seconds) = timed(
        lambda: _write_json(
            paths["metadata"],
            {
                "version": CHECKPOINT_VERSION,
                "checkpoint_mode": args.checkpoint_mode,
                "run_config": {"preset": args.preset},
                "model_config": asdict(config),
                "training_state": {
                    "step": args.train_steps,
                    "total_training_time": 0.0,
                    "smooth_train_loss": smooth_train_loss,
                    "step_telemetry": None,
                    "total_checkpoint_time": 0.0,
                    "checkpoint_count": 1,
                },
                "loader_state": loader_metadata,
            },
        )
    )
    save_total_seconds = (
        flatten_model_seconds
        + flatten_optimizer_seconds
        + serialize_loader_seconds
        + model_write_seconds
        + optimizer_write_seconds
        + loader_write_seconds
        + metadata_write_seconds
    )

    (
        _,
        _restore_config,
        restore_model,
        restore_optimizer,
        restore_loader,
        restore_grad_step,
        restore_apply_grads,
        restore_grad_accum_steps,
    ) = build_runtime(
        args.preset,
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    # Compile the restored runtime before measuring resumed-step readiness.
    run_training_steps(
        train_steps=1,
        grad_step=restore_grad_step,
        apply_grads=restore_apply_grads,
        train_loader=restore_loader,
        grad_accum_steps=restore_grad_accum_steps,
    )

    (metadata, metadata_read_seconds) = timed(lambda: load_checkpoint_metadata(checkpoint_root))
    (restored_model_arrays, model_load_seconds) = timed(lambda: mx.load(str(paths["model"])))
    if args.checkpoint_mode == CHECKPOINT_MODE_EXACT:
        (restored_optimizer_arrays, optimizer_load_seconds) = timed(lambda: mx.load(str(paths["optimizer"])))

        def load_loader_arrays():
            with np.load(paths["loader"]) as loader_npz:
                return {key: loader_npz[key] for key in loader_npz.files}

        (loader_arrays_dict, loader_load_seconds) = timed(load_loader_arrays)
    else:
        restored_optimizer_arrays = {}
        optimizer_load_seconds = 0.0
        loader_arrays_dict = {}
        loader_load_seconds = 0.0
    (_, apply_model_seconds) = timed(lambda: _load_trainable_weights(restore_model, restored_model_arrays))
    if args.checkpoint_mode == CHECKPOINT_MODE_EXACT:
        (_, apply_optimizer_seconds) = timed(lambda: setattr(restore_optimizer, "state", tree_unflatten(restored_optimizer_arrays)))
        (_, restore_loader_seconds) = timed(
            lambda: restore_loader_state(restore_loader, metadata["loader_state"], loader_arrays_dict)
        )
    else:
        apply_optimizer_seconds = 0.0
        restore_loader_seconds = 0.0
    (_, eval_restore_seconds) = timed(lambda: mx.eval(restore_model.state, restore_optimizer.state))
    (_, post_restore_step_seconds) = timed(
        lambda: run_training_steps(
            train_steps=1,
            grad_step=restore_grad_step,
            apply_grads=restore_apply_grads,
            train_loader=restore_loader,
            grad_accum_steps=restore_grad_accum_steps,
        )
    )
    (_, steady_step_seconds) = timed(
        lambda: run_training_steps(
            train_steps=1,
            grad_step=restore_grad_step,
            apply_grads=restore_apply_grads,
            train_loader=restore_loader,
            grad_accum_steps=restore_grad_accum_steps,
        )
    )
    restore_total_seconds = (
        metadata_read_seconds
        + model_load_seconds
        + optimizer_load_seconds
        + loader_load_seconds
        + apply_model_seconds
        + apply_optimizer_seconds
        + restore_loader_seconds
        + eval_restore_seconds
        + post_restore_step_seconds
    )

    summary = {
        "preset": args.preset,
        "checkpoint_mode": args.checkpoint_mode,
        "prefer_prepacked_cache": prefer_prepacked_cache,
        "loader_type": type(train_loader).__name__,
        "train_steps": args.train_steps,
        "seq_len": preset.seq_len,
        "total_batch_size": preset.total_batch_size,
        "save_seconds": {
            "flatten_model": flatten_model_seconds,
            "flatten_optimizer": flatten_optimizer_seconds,
            "serialize_loader": serialize_loader_seconds,
            "write_model": model_write_seconds,
            "write_optimizer": optimizer_write_seconds,
            "write_loader": loader_write_seconds,
            "write_metadata": metadata_write_seconds,
            "total": save_total_seconds,
        },
        "restore_seconds": {
            "read_metadata": metadata_read_seconds,
            "load_model": model_load_seconds,
            "load_optimizer": optimizer_load_seconds,
            "load_loader": loader_load_seconds,
            "apply_model": apply_model_seconds,
            "apply_optimizer": apply_optimizer_seconds,
            "restore_loader": restore_loader_seconds,
            "eval": eval_restore_seconds,
            "post_restore_step": post_restore_step_seconds,
            "steady_step": steady_step_seconds,
            "total": restore_total_seconds,
        },
        "artifacts": {
            "model_bytes": file_size(paths["model"]),
            "optimizer_bytes": file_size(paths["optimizer"]),
            "loader_bytes": file_size(paths["loader"]),
            "metadata_bytes": file_size(paths["metadata"]),
            "checkpoint_dir": str(paths["root"]),
        },
        "loader_state": {
            "epoch": last_epoch,
            "array_count": len(loader_arrays),
            "array_bytes": int(sum(array.nbytes for array in loader_arrays.values())),
        },
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if temp_root is not None and not args.keep_checkpoint_dir:
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
