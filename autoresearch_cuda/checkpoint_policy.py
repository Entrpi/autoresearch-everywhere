from __future__ import annotations

from pathlib import Path

from autoresearch_platform.checkpoint_policy import (
    AUTO_CHECKPOINT_INTERVAL_SEC,
    AUTO_CHECKPOINT_INTERVAL_TOKENS,
    AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC,
    AUTO_CHECKPOINT_MIN_TOKEN_BUDGET,
    CheckpointInterval,
    checkpoint_interval_due,
    default_time_budget_checkpoint_interval,
    default_token_budget_checkpoint_interval,
    format_interval_label,
    format_interval_spec,
    format_token_count,
    parse_checkpoint_interval_spec,
)


def default_auto_checkpoint_path(
    preset: str,
    *,
    seq_len: int,
    depth: int,
    total_batch_size: int,
    window_pattern: str,
) -> Path:
    slug = f"{preset}-seq{seq_len}-d{depth}-tb{total_batch_size}-w{window_pattern.lower()}"
    return Path.home() / ".cache" / "autoresearch" / "checkpoints" / "auto" / slug
