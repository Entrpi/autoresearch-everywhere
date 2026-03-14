from __future__ import annotations

from dataclasses import dataclass
import re


AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC = 300.0
AUTO_CHECKPOINT_MIN_TOKEN_BUDGET = 50_000_000
AUTO_CHECKPOINT_INTERVAL_SEC = 300.0
AUTO_CHECKPOINT_INTERVAL_TOKENS = 50_000_000

_TIME_COMPONENT_RE = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?$",
    re.IGNORECASE,
)
_TOKEN_COMPONENT_RE = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)(?P<multiplier>[kmb]?)\s*(?P<unit>tok|token|tokens)$",
    re.IGNORECASE,
)
_TOKEN_MULTIPLIERS = {
    "": 1,
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}
_TIME_UNIT_SECONDS = {
    None: 1.0,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


@dataclass(frozen=True)
class CheckpointInterval:
    seconds: float | None = None
    tokens: int | None = None

    def __post_init__(self) -> None:
        if self.seconds is None and self.tokens is None:
            raise ValueError("CheckpointInterval requires at least one time or token component.")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError("CheckpointInterval.seconds must be positive.")
        if self.tokens is not None and self.tokens <= 0:
            raise ValueError("CheckpointInterval.tokens must be positive.")


@dataclass(frozen=True)
class CheckpointCalibration:
    key: str
    label: str
    checkpoint_mode: str
    max_params_m: float
    checkpoint_cost_sec: float
    resume_ready_penalty_sec: float
    source: str
    resume_ready_source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class HumanIntervalPolicy:
    label: str
    anchor_interval_sec: float
    save_only_overhead_cap_fraction: float
    friendly_intervals_sec: tuple[float, ...]


@dataclass(frozen=True)
class CheckpointIntervalRecommendation:
    interval: CheckpointInterval
    interval_label: str
    save_only_overhead_fraction: float
    failed_shorter_interval_label: str | None

    @property
    def interval_sec(self) -> float | None:
        return self.interval.seconds


@dataclass(frozen=True)
class AutoCheckpointDecision:
    calibration: CheckpointCalibration
    recommendation: CheckpointIntervalRecommendation


DEFAULT_HUMAN_INTERVAL_POLICY = HumanIntervalPolicy(
    label="Generalized human-friendly interval scan anchored at hourly <= 0.1% save-only overhead",
    anchor_interval_sec=3600.0,
    save_only_overhead_cap_fraction=0.001,
    friendly_intervals_sec=(
        60.0,
        120.0,
        300.0,
        600.0,
        900.0,
        1800.0,
        3600.0,
        7200.0,
        14400.0,
        28800.0,
        43200.0,
        86400.0,
    ),
)


def default_time_budget_checkpoint_interval() -> CheckpointInterval:
    return CheckpointInterval(seconds=AUTO_CHECKPOINT_INTERVAL_SEC)


def default_token_budget_checkpoint_interval() -> CheckpointInterval:
    return CheckpointInterval(
        seconds=AUTO_CHECKPOINT_INTERVAL_SEC,
        tokens=AUTO_CHECKPOINT_INTERVAL_TOKENS,
    )


def parse_checkpoint_interval_spec(
    value: CheckpointInterval | dict | str | int | float | None,
) -> CheckpointInterval | None:
    if value is None:
        return None
    if isinstance(value, CheckpointInterval):
        return value
    if isinstance(value, dict):
        seconds = value.get("seconds")
        tokens = value.get("tokens")
        if seconds is None and tokens is None:
            return None
        return CheckpointInterval(
            seconds=float(seconds) if seconds is not None else None,
            tokens=int(tokens) if tokens is not None else None,
        )
    if isinstance(value, (int, float)):
        return CheckpointInterval(seconds=float(value))

    components = [component.strip() for component in str(value).split(",") if component.strip()]
    if not components:
        raise ValueError("Checkpoint interval spec cannot be empty.")

    seconds = None
    tokens = None
    for component in components:
        token_match = _TOKEN_COMPONENT_RE.fullmatch(component)
        if token_match is not None:
            if tokens is not None:
                raise ValueError("Checkpoint interval spec may include at most one token component.")
            multiplier = _TOKEN_MULTIPLIERS[token_match.group("multiplier").lower()]
            tokens = int(float(token_match.group("value")) * multiplier)
            continue

        time_match = _TIME_COMPONENT_RE.fullmatch(component)
        if time_match is not None:
            if seconds is not None:
                raise ValueError("Checkpoint interval spec may include at most one time component.")
            unit = time_match.group("unit")
            seconds = float(time_match.group("value")) * _TIME_UNIT_SECONDS[unit.lower() if unit is not None else None]
            continue

        raise ValueError(
            "Unsupported checkpoint interval component "
            f"{component!r}; use seconds like '300s' or tokens like '50Mtok'."
        )

    return CheckpointInterval(seconds=seconds, tokens=tokens)


def _format_seconds_spec(seconds: float) -> str:
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:g}s"


def _format_token_spec(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}Btok"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}Mtok"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}Ktok"
    return f"{tokens}tok"


def format_interval_spec(interval: CheckpointInterval | dict | str | int | float | None) -> str | None:
    resolved = parse_checkpoint_interval_spec(interval)
    if resolved is None:
        return None
    parts: list[str] = []
    if resolved.seconds is not None:
        parts.append(_format_seconds_spec(resolved.seconds))
    if resolved.tokens is not None:
        parts.append(_format_token_spec(resolved.tokens))
    return ",".join(parts)


def format_token_count(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}B tok"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M tok"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}K tok"
    return f"{tokens:,} tok"


def format_interval_label(interval: CheckpointInterval | dict | str | int | float | None) -> str:
    resolved = parse_checkpoint_interval_spec(interval)
    if resolved is None:
        return "none"

    parts: list[str] = []
    if resolved.seconds is not None:
        seconds = resolved.seconds
        if seconds < 3600.0:
            parts.append(f"{seconds / 60.0:.0f}m" if seconds % 60.0 == 0 else f"{seconds:g}s")
        else:
            parts.append(f"{seconds / 3600.0:.0f}h" if seconds % 3600.0 == 0 else f"{seconds:g}s")
    if resolved.tokens is not None:
        parts.append(format_token_count(resolved.tokens))
    return " or ".join(parts)


def checkpoint_interval_due(
    interval: CheckpointInterval | dict | str | int | float | None,
    *,
    elapsed_seconds: float,
    elapsed_tokens: int,
) -> bool:
    resolved = parse_checkpoint_interval_spec(interval)
    if resolved is None:
        return False
    time_due = resolved.seconds is not None and elapsed_seconds >= resolved.seconds
    token_due = resolved.tokens is not None and elapsed_tokens >= resolved.tokens
    return time_due or token_due


def save_only_overhead_fraction(*, checkpoint_cost_sec: float, interval_sec: float) -> float:
    return checkpoint_cost_sec / interval_sec


def select_checkpoint_calibration(
    num_params_m: float,
    checkpoint_mode: str,
    calibrations: tuple[CheckpointCalibration, ...],
) -> CheckpointCalibration:
    matching = tuple(
        calibration
        for calibration in calibrations
        if calibration.checkpoint_mode == checkpoint_mode
    )
    if not matching:
        raise ValueError(f"No checkpoint calibration registered for checkpoint_mode={checkpoint_mode!r}.")
    for calibration in matching:
        if num_params_m <= calibration.max_params_m:
            return calibration
    return matching[-1]


def recommend_interval_for_cost(
    checkpoint_cost_sec: float,
    policy: HumanIntervalPolicy = DEFAULT_HUMAN_INTERVAL_POLICY,
) -> CheckpointIntervalRecommendation:
    last_fail_label = None
    for interval_sec in policy.friendly_intervals_sec:
        overhead_fraction = save_only_overhead_fraction(
            checkpoint_cost_sec=checkpoint_cost_sec,
            interval_sec=interval_sec,
        )
        if overhead_fraction <= policy.save_only_overhead_cap_fraction:
            return CheckpointIntervalRecommendation(
                interval=CheckpointInterval(seconds=interval_sec),
                interval_label=format_interval_label(interval_sec),
                save_only_overhead_fraction=overhead_fraction,
                failed_shorter_interval_label=last_fail_label,
            )
        last_fail_label = format_interval_label(interval_sec)
    minimum_interval_sec = checkpoint_cost_sec / policy.save_only_overhead_cap_fraction
    return CheckpointIntervalRecommendation(
        interval=CheckpointInterval(seconds=minimum_interval_sec),
        interval_label=f">= {minimum_interval_sec / 60.0:.2f} min",
        save_only_overhead_fraction=policy.save_only_overhead_cap_fraction,
        failed_shorter_interval_label=last_fail_label,
    )


def choose_auto_checkpoint_decision(
    num_params_m: float,
    checkpoint_mode: str,
    *,
    policy: HumanIntervalPolicy = DEFAULT_HUMAN_INTERVAL_POLICY,
    calibrations: tuple[CheckpointCalibration, ...],
) -> AutoCheckpointDecision:
    calibration = select_checkpoint_calibration(
        num_params_m,
        checkpoint_mode,
        calibrations,
    )
    recommendation = recommend_interval_for_cost(calibration.checkpoint_cost_sec, policy)
    return AutoCheckpointDecision(calibration=calibration, recommendation=recommendation)
