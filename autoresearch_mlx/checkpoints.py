import time
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .data import restore_loader_state, serialize_loader_state


CHECKPOINT_VERSION = 1
CHECKPOINT_METADATA = "checkpoint.json"
MODEL_WEIGHTS = "model.safetensors"
OPTIMIZER_STATE = "optimizer.safetensors"
LOADER_STATE = "loader_state.npz"


def _checkpoint_paths(checkpoint_dir: str | Path) -> dict[str, Path]:
    root = Path(checkpoint_dir)
    return {
        "root": root,
        "metadata": root / CHECKPOINT_METADATA,
        "model": root / MODEL_WEIGHTS,
        "optimizer": root / OPTIMIZER_STATE,
        "loader": root / LOADER_STATE,
    }


def _write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
    temp.replace(path)


def _save_safetensors(path: Path, arrays: dict[str, mx.array]) -> None:
    temp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    mx.save_safetensors(str(temp), arrays)
    temp.replace(path)


def _load_trainable_weights(model, arrays: dict[str, mx.array]) -> None:
    current = tree_flatten(model.trainable_parameters(), destination={})
    if set(arrays) != set(current):
        missing = sorted(set(current) - set(arrays))
        extra = sorted(set(arrays) - set(current))
        raise ValueError(
            f"Checkpoint/model parameter mismatch. Missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, value in current.items():
        restored = arrays[key]
        if restored.shape != value.shape:
            raise ValueError(
                f"Checkpoint parameter {key} has shape {restored.shape}, expected {value.shape}."
            )
    model.update(tree_unflatten(arrays), strict=False)


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    run_config: dict,
    model_config: dict,
    model,
    optimizer,
    train_loader,
    step: int,
    total_training_time: float,
    smooth_train_loss: float,
    step_telemetry: dict | None = None,
    total_checkpoint_time: float = 0.0,
    checkpoint_count: int = 0,
) -> float:
    t_checkpoint_start = time.perf_counter()
    paths = _checkpoint_paths(checkpoint_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)

    model_arrays = tree_flatten(model.trainable_parameters(), destination={})
    optimizer_arrays = tree_flatten(optimizer.state, destination={})
    loader_metadata, loader_arrays = serialize_loader_state(train_loader)

    _save_safetensors(paths["model"], model_arrays)
    _save_safetensors(paths["optimizer"], optimizer_arrays)
    _save_npz(paths["loader"], loader_arrays)
    elapsed_before_metadata = time.perf_counter() - t_checkpoint_start
    _write_json(
        paths["metadata"],
        {
            "version": CHECKPOINT_VERSION,
            "run_config": run_config,
            "model_config": model_config,
            "training_state": {
                "step": step,
                "total_training_time": total_training_time,
                "smooth_train_loss": smooth_train_loss,
                "step_telemetry": step_telemetry,
                # Include the current save cost up to metadata emission. The returned
                # checkpoint_seconds includes the metadata write itself.
                "total_checkpoint_time": total_checkpoint_time + elapsed_before_metadata,
                "checkpoint_count": checkpoint_count + 1,
            },
            "loader_state": loader_metadata,
        },
    )
    checkpoint_seconds = time.perf_counter() - t_checkpoint_start
    return checkpoint_seconds


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> dict:
    paths = _checkpoint_paths(checkpoint_dir)
    if not paths["metadata"].exists():
        raise RuntimeError(f"Missing checkpoint metadata: {paths['metadata']}")
    payload = json.loads(paths["metadata"].read_text())
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint version {payload.get('version')}, expected {CHECKPOINT_VERSION}."
        )
    return payload


def restore_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model,
    optimizer,
    train_loader,
) -> dict:
    paths = _checkpoint_paths(checkpoint_dir)
    metadata = load_checkpoint_metadata(checkpoint_dir)

    model_arrays = mx.load(str(paths["model"]))
    optimizer_arrays = mx.load(str(paths["optimizer"]))
    with np.load(paths["loader"]) as loader_arrays:
        loader_arrays_dict = {key: loader_arrays[key] for key in loader_arrays.files}

    _load_trainable_weights(model, model_arrays)
    optimizer.state = tree_unflatten(optimizer_arrays)
    restore_loader_state(train_loader, metadata["loader_state"], loader_arrays_dict)
    mx.eval(model.state, optimizer.state)
    return metadata["training_state"]
