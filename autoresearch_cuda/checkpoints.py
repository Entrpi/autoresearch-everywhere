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
            "model_state_dict": normalize_model_state_dict_keys(model.state_dict()),
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
    return paths["root"]


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    return json.loads(paths["metadata"].read_text())


def load_training_checkpoint(checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    payload = torch.load(paths["bundle"], map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload["model_state_dict"] = normalize_model_state_dict_keys(payload["model_state_dict"])
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
