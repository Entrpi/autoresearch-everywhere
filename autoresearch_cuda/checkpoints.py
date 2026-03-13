from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1
CHECKPOINT_BUNDLE = "checkpoint.pt"
CHECKPOINT_METADATA = "checkpoint.json"


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


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    run_config: dict[str, Any],
    training_state: dict[str, Any],
    model,
    optimizer,
) -> Path:
    paths = _checkpoint_paths(checkpoint_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)

    temp_bundle = paths["bundle"].with_suffix(paths["bundle"].suffix + ".tmp")
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "run_config": run_config,
            "training_state": training_state,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        temp_bundle,
    )
    temp_bundle.replace(paths["bundle"])
    _write_json(
        paths["metadata"],
        {
            "version": CHECKPOINT_VERSION,
            "run_config": run_config,
            "training_state": training_state,
            "bundle": CHECKPOINT_BUNDLE,
        },
    )
    return paths["bundle"]


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    return json.loads(paths["metadata"].read_text())


def load_training_checkpoint(checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    return torch.load(paths["bundle"], map_location=map_location)


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
