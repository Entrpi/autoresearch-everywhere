from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import statistics

from .constants import EVAL_TELEMETRY_PATH


STABLE_RUNG_MIN_COUNT = 3
STABLE_RUNG_MAX_REL_EVAL_SECONDS_MAD = 0.05


@dataclass(frozen=True)
class EvalTelemetryRecord:
    recorded_at: str
    recorded_on: str
    commit: str | None
    preset: str
    hardware_key: str
    policy_version: int | None
    default_shape: bool
    smoke: bool
    benchmark_skip_eval: bool
    calibration_status: str
    canonical_rung: str | None
    canonical_eval_seq_len: int
    canonical_eval_tokens: int
    canonical_eval_batch_size: int
    canonical_eval_slices: int
    canonical_eval_reference_tokens: int | None
    time_budget: float | None
    token_budget: int | None
    time_budget_mode: str
    training_seconds: float
    total_seconds: float
    canonical_eval_seconds: float
    val_bpb: float

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalTelemetryRecord":
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_calibration_eligible(self) -> bool:
        return (
            self.default_shape
            and not self.smoke
            and not self.benchmark_skip_eval
            and self.calibration_status in {"calibrated", "calibrated-limited"}
            and self.canonical_rung in {"cheap", "reference", "full"}
            and self.policy_version is not None
        )


@dataclass(frozen=True)
class EvalTelemetrySummary:
    eligible_count: int
    commit_count: int
    day_count: int
    observed_rungs: tuple[str, ...]
    stable_rungs: tuple[str, ...]
    last_seen_on: str | None
    last_seen_age_days: int | None
    rung_stats: tuple["EvalRungTelemetryStats", ...]


@dataclass(frozen=True)
class EvalRungTelemetryStats:
    rung_key: str
    count: int
    commit_count: int
    day_count: int
    median_eval_seconds: float
    rel_mad_eval_seconds: float

    @property
    def stable_timing(self) -> bool:
        return self.count >= STABLE_RUNG_MIN_COUNT and self.rel_mad_eval_seconds <= STABLE_RUNG_MAX_REL_EVAL_SECONDS_MAD


def append_eval_telemetry(
    record: EvalTelemetryRecord,
    *,
    path: Path = EVAL_TELEMETRY_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def load_eval_telemetry(
    *,
    path: Path = EVAL_TELEMETRY_PATH,
) -> tuple[EvalTelemetryRecord, ...]:
    if not path.exists():
        return ()
    records: list[EvalTelemetryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(EvalTelemetryRecord.from_dict(json.loads(line)))
    return tuple(records)


def summarize_eval_telemetry(
    preset: str,
    *,
    hardware_key: str,
    policy_version: int,
    path: Path = EVAL_TELEMETRY_PATH,
) -> EvalTelemetrySummary:
    matches = [
        record
        for record in load_eval_telemetry(path=path)
        if record.preset == preset
        and record.hardware_key == hardware_key
        and record.policy_version == policy_version
        and record.is_calibration_eligible()
    ]
    if not matches:
        return EvalTelemetrySummary(
            eligible_count=0,
            commit_count=0,
            day_count=0,
            observed_rungs=(),
            stable_rungs=(),
            last_seen_on=None,
            last_seen_age_days=None,
            rung_stats=(),
        )

    observed_rungs = tuple(sorted({str(record.canonical_rung) for record in matches if record.canonical_rung}))
    rung_stats = tuple(
        sorted(
            (
                _summarize_rung(group_key, [record for record in matches if record.canonical_rung == group_key])
                for group_key in observed_rungs
            ),
            key=lambda item: item.rung_key,
        )
    )
    stable_rungs = tuple(sorted(stat.rung_key for stat in rung_stats if stat.stable_timing))
    last_seen_on = max(record.recorded_on for record in matches)
    last_seen_age_days = _age_days(last_seen_on)
    return EvalTelemetrySummary(
        eligible_count=len(matches),
        commit_count=len({record.commit or record.recorded_at for record in matches}),
        day_count=len({record.recorded_on for record in matches}),
        observed_rungs=observed_rungs,
        stable_rungs=stable_rungs,
        last_seen_on=last_seen_on,
        last_seen_age_days=last_seen_age_days,
        rung_stats=rung_stats,
    )


def _age_days(date_label: str) -> int | None:
    try:
        observed = date.fromisoformat(date_label)
    except ValueError:
        return None
    return (date.today() - observed).days


def now_iso() -> tuple[str, str]:
    current = datetime.now().astimezone()
    return current.isoformat(timespec="seconds"), current.date().isoformat()


def _summarize_rung(rung_key: str, records: list[EvalTelemetryRecord]) -> EvalRungTelemetryStats:
    eval_seconds = [record.canonical_eval_seconds for record in records]
    median_eval_seconds = statistics.median(eval_seconds)
    rel_mad_eval_seconds = _relative_mad(eval_seconds, median_eval_seconds)
    return EvalRungTelemetryStats(
        rung_key=rung_key,
        count=len(records),
        commit_count=len({record.commit or record.recorded_at for record in records}),
        day_count=len({record.recorded_on for record in records}),
        median_eval_seconds=median_eval_seconds,
        rel_mad_eval_seconds=rel_mad_eval_seconds,
    )


def _relative_mad(samples: list[float], center: float) -> float:
    if not samples:
        return 0.0
    mad = statistics.median(abs(sample - center) for sample in samples)
    if center == 0.0:
        return 0.0 if mad == 0.0 else float("inf")
    return mad / abs(center)
