from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLATFORM_DEFAULT_CACHE_SCHEMA_VERSION = 1
CACHE_ROOT = Path.home() / ".cache" / "autoresearch" / "platform_defaults"


@dataclass(frozen=True)
class CachedPlatformDefault:
    schema_version: int
    engine: str
    hardware_key: str
    generated_at: str | None
    source_output_dir: str | None
    source_report: str | None
    candidate_default: dict[str, Any]


def platform_default_cache_dir(*, engine_name: str) -> Path:
    return CACHE_ROOT / engine_name


def platform_default_cache_path(*, engine_name: str, hardware_key: str) -> Path:
    return platform_default_cache_dir(engine_name=engine_name) / f"{hardware_key}.json"


def write_platform_default_cache(
    *,
    engine_name: str,
    hardware_key: str,
    candidate_default: dict[str, Any],
    generated_at: str | None,
    source_output_dir: str | None,
    source_report: str | None,
) -> Path:
    path = platform_default_cache_path(engine_name=engine_name, hardware_key=hardware_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PLATFORM_DEFAULT_CACHE_SCHEMA_VERSION,
        "engine": engine_name,
        "hardware_key": hardware_key,
        "generated_at": generated_at,
        "source_output_dir": source_output_dir,
        "source_report": source_report,
        "candidate_default": candidate_default,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_platform_default_cache(*, engine_name: str, hardware_key: str) -> CachedPlatformDefault | None:
    path = platform_default_cache_path(engine_name=engine_name, hardware_key=hardware_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != PLATFORM_DEFAULT_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("engine") != engine_name or payload.get("hardware_key") != hardware_key:
        return None
    candidate_default = payload.get("candidate_default")
    if not isinstance(candidate_default, dict):
        return None
    return CachedPlatformDefault(
        schema_version=PLATFORM_DEFAULT_CACHE_SCHEMA_VERSION,
        engine=engine_name,
        hardware_key=hardware_key,
        generated_at=payload.get("generated_at"),
        source_output_dir=payload.get("source_output_dir"),
        source_report=payload.get("source_report"),
        candidate_default=candidate_default,
    )
