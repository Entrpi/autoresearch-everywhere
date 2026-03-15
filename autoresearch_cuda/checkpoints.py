from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autoresearch_platform.async_checkpoint import AsyncCheckpointWriter
from autoresearch_platform.checkpoint_policy import (
    CHECKPOINT_SAVE_MODE_ASYNC,
    CHECKPOINT_SAVE_MODE_SYNC,
    CHECKPOINT_SAVE_MODES,
)


CHECKPOINT_VERSION = 1
CHECKPOINT_BUNDLE = "checkpoint.pt"
CHECKPOINT_METADATA = "checkpoint.json"


@dataclass(frozen=True)
class CheckpointSnapshot:
    checkpoint_dir: str | Path
    bundle_payload: dict[str, Any]
    metadata_payload: dict[str, Any]


def _checkpoint_paths(checkpoint_dir: str | Path) -> dict[str, Path]:
    root = Path(checkpoint_dir)
    return {
        "root": root,
        "bundle": root / CHECKPOINT_BUNDLE,
        "metadata": root / CHECKPOINT_METADATA,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _write_bundle(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def normalize_model_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip torch.compile wrapper prefixes from serialized model weights."""
    prefix = "_orig_mod."
    if not state_dict:
        return state_dict
    if not all(isinstance(key, str) for key in state_dict):
        return state_dict
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _snapshot_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, OrderedDict):
        return OrderedDict((key, _snapshot_to_cpu(item)) for key, item in value.items())
    if isinstance(value, dict):
        return {key: _snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _write_checkpoint_snapshot(snapshot: CheckpointSnapshot) -> float:
    t_start = time.perf_counter()
    paths = _checkpoint_paths(snapshot.checkpoint_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _write_bundle(paths["bundle"], snapshot.bundle_payload)
    _write_json(paths["metadata"], snapshot.metadata_payload)
    return time.perf_counter() - t_start


def make_async_checkpoint_writer() -> AsyncCheckpointWriter[CheckpointSnapshot]:
    return AsyncCheckpointWriter(write_snapshot=_write_checkpoint_snapshot)


def capture_async_checkpoint_snapshot(
    checkpoint_dir: str | Path,
    *,
    run_config: dict[str, Any],
    training_state: dict[str, Any],
    model,
    optimizer,
) -> tuple[CheckpointSnapshot, float]:
    t_capture_start = time.perf_counter()
    bundle_payload = {
        "version": CHECKPOINT_VERSION,
        "checkpoint_save_mode": CHECKPOINT_SAVE_MODE_ASYNC,
        "run_config": copy.deepcopy(run_config),
        "training_state": copy.deepcopy(training_state),
        "model_state_dict": _snapshot_to_cpu(normalize_model_state_dict_keys(model.state_dict())),
        "optimizer_state_dict": _snapshot_to_cpu(optimizer.state_dict()),
    }
    metadata_payload = {
        "version": CHECKPOINT_VERSION,
        "checkpoint_save_mode": CHECKPOINT_SAVE_MODE_ASYNC,
        "run_config": copy.deepcopy(run_config),
        "training_state": copy.deepcopy(training_state),
        "bundle": CHECKPOINT_BUNDLE,
    }
    capture_seconds = time.perf_counter() - t_capture_start
    return (
        CheckpointSnapshot(
            checkpoint_dir=checkpoint_dir,
            bundle_payload=bundle_payload,
            metadata_payload=metadata_payload,
        ),
        capture_seconds,
    )


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    run_config: dict[str, Any],
    training_state: dict[str, Any],
    model,
    optimizer,
    checkpoint_save_mode: str = CHECKPOINT_SAVE_MODE_SYNC,
) -> tuple[Path, float]:
    if checkpoint_save_mode not in CHECKPOINT_SAVE_MODES:
        raise ValueError(
            f"Unsupported checkpoint_save_mode {checkpoint_save_mode!r}. Expected one of {CHECKPOINT_SAVE_MODES}."
        )
    t_start = time.perf_counter()
    paths = _checkpoint_paths(checkpoint_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)

    _write_bundle(
        paths["bundle"],
        {
            "version": CHECKPOINT_VERSION,
            "checkpoint_save_mode": checkpoint_save_mode,
            "run_config": run_config,
            "training_state": training_state,
            "model_state_dict": normalize_model_state_dict_keys(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    _write_json(
        paths["metadata"],
        {
            "version": CHECKPOINT_VERSION,
            "checkpoint_save_mode": checkpoint_save_mode,
            "run_config": run_config,
            "training_state": training_state,
            "bundle": CHECKPOINT_BUNDLE,
        },
    )
    return paths["root"], time.perf_counter() - t_start


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    payload = json.loads(paths["metadata"].read_text())
    payload.setdefault("checkpoint_save_mode", CHECKPOINT_SAVE_MODE_SYNC)
    return payload


def load_training_checkpoint(checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    payload = torch.load(paths["bundle"], map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload["model_state_dict"] = normalize_model_state_dict_keys(payload["model_state_dict"])
        payload.setdefault("checkpoint_save_mode", CHECKPOINT_SAVE_MODE_SYNC)
    return payload


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported CUDA checkpoint version {version!r}; expected {CHECKPOINT_VERSION}."
        )
    required = {
        "run_config",
        "training_state",
        "model_state_dict",
        "optimizer_state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"CUDA checkpoint is missing required keys: {', '.join(missing)}")
