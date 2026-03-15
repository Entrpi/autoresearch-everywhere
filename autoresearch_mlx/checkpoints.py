import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from autoresearch_platform.async_checkpoint import AsyncCheckpointWriter
from autoresearch_platform.checkpoint_policy import (
    CHECKPOINT_SAVE_MODE_ASYNC,
    CHECKPOINT_SAVE_MODE_SYNC,
    CHECKPOINT_SAVE_MODES,
)

from .data import restore_loader_state, serialize_loader_state


CHECKPOINT_VERSION = 2
CHECKPOINT_METADATA = "checkpoint.json"
MODEL_WEIGHTS = "model.safetensors"
OPTIMIZER_STATE = "optimizer.safetensors"
MODEL_WEIGHTS_HOST = "model.npz"
OPTIMIZER_STATE_HOST = "optimizer.npz"
TENSOR_MANIFEST = "tensor_manifest.json"
LOADER_STATE = "loader_state.npz"
CHECKPOINT_MODE_EXACT = "exact"
CHECKPOINT_MODE_WEIGHTS_ONLY = "weights_only"
CHECKPOINT_MODES = (CHECKPOINT_MODE_EXACT, CHECKPOINT_MODE_WEIGHTS_ONLY)
TENSOR_STORAGE_SAFETENSORS = "safetensors"
TENSOR_STORAGE_HOST_NPZ = "host_npz"


def _checkpoint_paths(checkpoint_dir: str | Path) -> dict[str, Path]:
    root = Path(checkpoint_dir)
    return {
        "root": root,
        "metadata": root / CHECKPOINT_METADATA,
        "model": root / MODEL_WEIGHTS,
        "optimizer": root / OPTIMIZER_STATE,
        "model_host": root / MODEL_WEIGHTS_HOST,
        "optimizer_host": root / OPTIMIZER_STATE_HOST,
        "tensor_manifest": root / TENSOR_MANIFEST,
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


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).split(".")[-1]


def _resolve_dtype(dtype_name: str) -> mx.Dtype:
    try:
        return getattr(mx, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype name in checkpoint manifest: {dtype_name!r}") from exc


def _capture_host_array(array: mx.array) -> tuple[np.ndarray, dict[str, str]]:
    original_dtype = _dtype_name(array.dtype)
    storage_array = array
    storage_dtype = original_dtype
    if array.dtype == mx.bfloat16:
        storage_array = mx.view(array, mx.uint16)
        storage_dtype = "uint16"
    host_array = np.from_dlpack(storage_array).copy()
    return host_array, {
        "original_dtype": original_dtype,
        "storage_dtype": storage_dtype,
    }


def _capture_host_arrays(arrays: dict[str, mx.array]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    host_arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict[str, str]] = {}
    for key, value in arrays.items():
        host_arrays[key], manifest[key] = _capture_host_array(value)
    return host_arrays, manifest


def _restore_host_array(array: np.ndarray, spec: dict[str, str]) -> mx.array:
    storage = mx.array(array)
    original_dtype = _resolve_dtype(spec["original_dtype"])
    storage_dtype = _resolve_dtype(spec["storage_dtype"])
    if original_dtype == storage_dtype:
        if storage.dtype != original_dtype:
            storage = storage.astype(original_dtype)
        return storage
    if storage.dtype != storage_dtype:
        storage = storage.astype(storage_dtype)
    return mx.view(storage, original_dtype)


def _load_host_arrays(path: Path, manifest: dict[str, dict[str, str]]) -> dict[str, mx.array]:
    with np.load(path) as payload:
        arrays = {key: payload[key] for key in payload.files}
    if set(arrays) != set(manifest):
        missing = sorted(set(manifest) - set(arrays))
        extra = sorted(set(arrays) - set(manifest))
        raise ValueError(
            f"Checkpoint host-array manifest mismatch. Missing={missing[:5]}, extra={extra[:5]}"
        )
    return {
        key: _restore_host_array(arrays[key], manifest[key])
        for key in manifest
    }


@dataclass
class CheckpointSnapshot:
    checkpoint_dir: str | Path
    checkpoint_mode: str
    checkpoint_save_mode: str
    tensor_storage_format: str
    run_config: dict
    model_config: dict
    training_state: dict
    loader_state: dict | None
    model_arrays: dict[str, np.ndarray]
    optimizer_arrays: dict[str, np.ndarray]
    loader_arrays: dict[str, np.ndarray]
    tensor_manifest: dict[str, dict[str, dict[str, str]]]


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


def _write_host_checkpoint_snapshot(snapshot: CheckpointSnapshot) -> float:
    t_start = time.perf_counter()
    paths = _checkpoint_paths(snapshot.checkpoint_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _save_npz(paths["model_host"], snapshot.model_arrays)
    _save_npz(paths["optimizer_host"], snapshot.optimizer_arrays)
    _save_npz(paths["loader"], snapshot.loader_arrays)
    _safe_unlink(paths["model"])
    _safe_unlink(paths["optimizer"])
    elapsed_before_metadata = time.perf_counter() - t_start
    training_state = dict(snapshot.training_state)
    training_state["total_checkpoint_write_time"] = (
        float(training_state.get("total_checkpoint_write_time", 0.0)) + elapsed_before_metadata
    )
    _write_json(paths["tensor_manifest"], snapshot.tensor_manifest)
    _write_json(
        paths["metadata"],
        {
            "version": CHECKPOINT_VERSION,
            "checkpoint_mode": snapshot.checkpoint_mode,
            "checkpoint_save_mode": snapshot.checkpoint_save_mode,
            "tensor_storage_format": snapshot.tensor_storage_format,
            "run_config": snapshot.run_config,
            "model_config": snapshot.model_config,
            "training_state": training_state,
            "loader_state": snapshot.loader_state,
        },
    )
    return time.perf_counter() - t_start


def make_async_checkpoint_writer() -> AsyncCheckpointWriter[CheckpointSnapshot]:
    return AsyncCheckpointWriter(write_snapshot=_write_host_checkpoint_snapshot)


def capture_async_checkpoint_snapshot(
    checkpoint_dir: str | Path,
    *,
    checkpoint_mode: str = CHECKPOINT_MODE_EXACT,
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
    total_checkpoint_write_time: float = 0.0,
    checkpoint_count: int = 0,
) -> tuple[CheckpointSnapshot, float]:
    if checkpoint_mode != CHECKPOINT_MODE_EXACT:
        raise ValueError("Asynchronous checkpoint capture currently supports only exact checkpoints.")
    t_capture_start = time.perf_counter()
    model_arrays = tree_flatten(model.trainable_parameters(), destination={})
    optimizer_arrays = tree_flatten(optimizer.state, destination={})
    loader_metadata, loader_arrays = serialize_loader_state(train_loader)
    model_host_arrays, model_manifest = _capture_host_arrays(model_arrays)
    optimizer_host_arrays, optimizer_manifest = _capture_host_arrays(optimizer_arrays)
    capture_seconds = time.perf_counter() - t_capture_start
    snapshot = CheckpointSnapshot(
        checkpoint_dir=checkpoint_dir,
        checkpoint_mode=checkpoint_mode,
        checkpoint_save_mode=CHECKPOINT_SAVE_MODE_ASYNC,
        tensor_storage_format=TENSOR_STORAGE_HOST_NPZ,
        run_config=run_config,
        model_config=model_config,
        training_state={
            "step": step,
            "total_training_time": total_training_time,
            "smooth_train_loss": smooth_train_loss,
            "step_telemetry": step_telemetry,
            "total_checkpoint_time": total_checkpoint_time + capture_seconds,
            "total_checkpoint_write_time": total_checkpoint_write_time,
            "checkpoint_count": checkpoint_count + 1,
        },
        loader_state=loader_metadata,
        model_arrays=model_host_arrays,
        optimizer_arrays=optimizer_host_arrays,
        loader_arrays=loader_arrays,
        tensor_manifest={
            "model": model_manifest,
            "optimizer": optimizer_manifest,
        },
    )
    return snapshot, capture_seconds


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    checkpoint_mode: str = CHECKPOINT_MODE_EXACT,
    checkpoint_save_mode: str = CHECKPOINT_SAVE_MODE_SYNC,
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
    total_checkpoint_write_time: float = 0.0,
    checkpoint_count: int = 0,
) -> float:
    if checkpoint_mode not in CHECKPOINT_MODES:
        raise ValueError(f"Unsupported checkpoint_mode {checkpoint_mode!r}. Expected one of {CHECKPOINT_MODES}.")
    if checkpoint_save_mode not in CHECKPOINT_SAVE_MODES:
        raise ValueError(
            f"Unsupported checkpoint_save_mode {checkpoint_save_mode!r}. Expected one of {CHECKPOINT_SAVE_MODES}."
        )
    t_checkpoint_start = time.perf_counter()
    paths = _checkpoint_paths(checkpoint_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)

    model_arrays = tree_flatten(model.trainable_parameters(), destination={})
    _save_safetensors(paths["model"], model_arrays)
    _safe_unlink(paths["model_host"])
    _safe_unlink(paths["tensor_manifest"])
    if checkpoint_mode == CHECKPOINT_MODE_EXACT:
        optimizer_arrays = tree_flatten(optimizer.state, destination={})
        loader_metadata, loader_arrays = serialize_loader_state(train_loader)
        _save_safetensors(paths["optimizer"], optimizer_arrays)
        _save_npz(paths["loader"], loader_arrays)
        _safe_unlink(paths["optimizer_host"])
    else:
        loader_metadata = None
        _safe_unlink(paths["optimizer"])
        _safe_unlink(paths["optimizer_host"])
        _safe_unlink(paths["loader"])
    elapsed_before_metadata = time.perf_counter() - t_checkpoint_start
    _write_json(
        paths["metadata"],
        {
            "version": CHECKPOINT_VERSION,
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_save_mode": checkpoint_save_mode,
            "tensor_storage_format": TENSOR_STORAGE_SAFETENSORS,
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
                "total_checkpoint_write_time": total_checkpoint_write_time + elapsed_before_metadata,
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
    version = int(payload.get("version", -1))
    if version not in {1, CHECKPOINT_VERSION}:
        raise RuntimeError(
            f"Unsupported checkpoint version {payload.get('version')}, expected one of [1, {CHECKPOINT_VERSION}]."
        )
    payload.setdefault("checkpoint_mode", CHECKPOINT_MODE_EXACT)
    payload.setdefault("checkpoint_save_mode", CHECKPOINT_SAVE_MODE_SYNC)
    payload.setdefault("tensor_storage_format", TENSOR_STORAGE_SAFETENSORS)
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
    checkpoint_mode = metadata["checkpoint_mode"]
    tensor_storage_format = metadata["tensor_storage_format"]
    if tensor_storage_format == TENSOR_STORAGE_SAFETENSORS:
        model_arrays = mx.load(str(paths["model"]))
    elif tensor_storage_format == TENSOR_STORAGE_HOST_NPZ:
        tensor_manifest = json.loads(paths["tensor_manifest"].read_text())
        model_arrays = _load_host_arrays(paths["model_host"], tensor_manifest["model"])
    else:
        raise RuntimeError(f"Unsupported checkpoint tensor storage format: {tensor_storage_format!r}")
    _load_trainable_weights(model, model_arrays)
    restored_optimizer_state = checkpoint_mode == CHECKPOINT_MODE_EXACT
    restored_loader_state = checkpoint_mode == CHECKPOINT_MODE_EXACT
    if restored_optimizer_state:
        if tensor_storage_format == TENSOR_STORAGE_SAFETENSORS:
            optimizer_arrays = mx.load(str(paths["optimizer"]))
        else:
            tensor_manifest = json.loads(paths["tensor_manifest"].read_text())
            optimizer_arrays = _load_host_arrays(paths["optimizer_host"], tensor_manifest["optimizer"])
        optimizer.state = tree_unflatten(optimizer_arrays)
    if restored_loader_state:
        with np.load(paths["loader"]) as loader_arrays:
            loader_arrays_dict = {key: loader_arrays[key] for key in loader_arrays.files}
        restore_loader_state(train_loader, metadata["loader_state"], loader_arrays_dict)
    mx.eval(model.state, optimizer.state)
    training_state = dict(metadata["training_state"])
    training_state["loaded_checkpoint_mode"] = checkpoint_mode
    training_state["loaded_checkpoint_save_mode"] = metadata["checkpoint_save_mode"]
    training_state["restored_optimizer_state"] = restored_optimizer_state
    training_state["restored_loader_state"] = restored_loader_state
    return training_state
