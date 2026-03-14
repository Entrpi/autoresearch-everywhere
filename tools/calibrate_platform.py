#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_platform.engines import (  # noqa: E402
    DEFAULT_ENGINE_NAME,
    HardwareFingerprint,
    ProbeResult,
    TrainingEngine,
    available_engines,
    get_engine,
)
from autoresearch_platform.platform_defaults import write_platform_default_cache  # noqa: E402
from autoresearch_platform.curve_projection import (  # noqa: E402
    build_horizon_projection_table,
    build_projection_calibration,
    compare_projected_curves,
    compare_multi_horizon_curves,
    estimate_projected_curve,
    final_training_seconds,
    final_val_bpb,
    load_curve_artifact,
    load_curve_artifacts_from_dir,
    project_curve,
    select_scaling_candidate,
    summarize_longer_horizon_projection,
    suggest_next_horizon,
)

try:  # noqa: E402
    from autoresearch_mlx.eval_policy import DEFAULT_EVAL_HARDWARE_KEY, EVAL_POLICY_VERSION, find_eval_calibration
    from autoresearch_mlx.eval_telemetry import summarize_eval_telemetry

    HAS_MLX_CALIBRATION_SUPPORT = True
except Exception:  # pragma: no cover - exercised on non-MLX hosts
    DEFAULT_EVAL_HARDWARE_KEY = "apple-m5-32gb-10gpu"
    EVAL_POLICY_VERSION = 1
    find_eval_calibration = None
    summarize_eval_telemetry = None
    HAS_MLX_CALIBRATION_SUPPORT = False

M5_REFERENCE_DEFAULT_PRESET = "m5-small"
PLATFORM_CALIBRATION_SCHEMA_VERSION = 6
PLATEAU_FRACTION = 0.9
SHARP_EDGE_DROP_FRACTION = 0.65
PROJECTION_TARGET_SECONDS = 300.0
SCALING_TARGET_SECONDS = 900.0
REPORT_HORIZONS = (60.0, 120.0, PROJECTION_TARGET_SECONDS, SCALING_TARGET_SECONDS)
FRIENDLY_PROGRESS_ENABLED = True

PHASE_FRIENDLY_PREAMBLES = {
    "hardware fingerprint": "I’m identifying your accelerator, runtime, and evaluation semantics so the rest of calibration stays comparable and reproducible.",
    "anchor batch profile": "I’m finding a sensible machine batch shape on one representative preset first, because bad batch sizing can waste a lot of autoresearch time.",
    "coarse envelope": "I’m doing quick coarse checks across the preset ladder to throw out obviously bad fits before the more expensive comparison phases.",
    "projection ranking": "I’m running all candidate preset families long enough to project which one is most likely to win at the real 300-second objective on this hardware.",
    "finalist rerank": "The leading projected families are getting extra time now so we do not lock onto a startup winner that fades at longer horizons.",
    "winner confirmation": "The earlier horizons were not decisive enough, so I’m spending a bit more time on the top candidates before choosing the family.",
    "scaling confirmation": "This deeper run checks whether a larger model is likely to overtake the 300-second winner at a much longer horizon. It is intentionally expensive.",
    "winner batch audit": "Now that the family is chosen, I’m retesting batch shapes on just that family so the final operating point is tuned to the real winner rather than inherited from the anchor preset.",
    "local search": "I’m tuning the winning family’s local shape knobs so the emitted default is a practical operating point, not just a family label.",
    "candidate checkpoint": "I’m minting a checkpoint for the chosen operating point so later eval calibration and research runs can start from a known good state.",
    "eval calibration": "I’m measuring the eval rungs that later reports and promotions rely on, so downstream autoresearch has trustworthy evaluation defaults.",
    "final report": "I’m writing the final human-readable report and machine-readable summary so you can use the result immediately or inspect the details later.",
}

MODE_FAST = "fast"
MODE_FULL = "full"


@dataclass(frozen=True)
class PlatformModeSpec:
    coarse_time_budget: float
    ranking_time_budget: float
    projection_time_budget: float
    finalist_time_budget: float
    finalist_count: int
    batch_audit_time_budget: float
    local_search_time_budget: float
    eval_train_seconds: float
    eval_rungs: tuple[str, ...]


MODE_SPECS = {
    MODE_FAST: PlatformModeSpec(
        coarse_time_budget=1.0,
        ranking_time_budget=5.0,
        projection_time_budget=30.0,
        finalist_time_budget=60.0,
        finalist_count=3,
        batch_audit_time_budget=10.0,
        local_search_time_budget=5.0,
        eval_train_seconds=5.0,
        eval_rungs=("cheap", "reference"),
    ),
    MODE_FULL: PlatformModeSpec(
        coarse_time_budget=60.0,
        ranking_time_budget=60.0,
        projection_time_budget=120.0,
        finalist_time_budget=300.0,
        finalist_count=3,
        batch_audit_time_budget=60.0,
        local_search_time_budget=60.0,
        eval_train_seconds=300.0,
        eval_rungs=("cheap", "reference", "full"),
    ),
}


M5_TRAIN_REFERENCE = {
    "m5-tiny": {"steady_state_tok_per_sec": 103022.3, "peak_vram_mb": 281.8},
    "m5-small": {"steady_state_tok_per_sec": 46033.7, "peak_vram_mb": 1014.2},
    "m5-balanced": {"steady_state_tok_per_sec": 18218.5, "peak_vram_mb": 2772.3},
    "m5-large": {"steady_state_tok_per_sec": 13525.1, "peak_vram_mb": 2664.1},
    "m5-xlarge": {"steady_state_tok_per_sec": 9022.3, "peak_vram_mb": 7440.6},
}


@dataclass(frozen=True)
class RankedProbe:
    probe: ProbeResult
    quality_score: float
    throughput_score: float
    memory_score: float
    eval_overhead_score: float
    telemetry_score: float
    utility_score: float
    on_pareto_front: bool
    frontier_distance: float
    selection_distance: float
    estimated_eval_overhead_fraction: float
    memory_fraction: float | None
    memory_pressure_band: str
    memory_tiebreak_penalty: float
    telemetry_count: int
    stable_rung_count: int
    effective_confidence: str | None
    calibration_status: str | None


def ranked_probe_from_probe(probe: ProbeResult, **overrides) -> RankedProbe:
    payload = {
        "probe": probe,
        "quality_score": 0.0,
        "throughput_score": 0.0,
        "memory_score": 0.0,
        "eval_overhead_score": 0.0,
        "telemetry_score": 0.0,
        "utility_score": 0.0,
        "on_pareto_front": True,
        "frontier_distance": 0.0,
        "selection_distance": 0.0,
        "estimated_eval_overhead_fraction": 0.0,
        "memory_fraction": None,
        "memory_pressure_band": "unknown",
        "memory_tiebreak_penalty": 0.0,
        "telemetry_count": 0,
        "stable_rung_count": 0,
        "effective_confidence": None,
        "calibration_status": None,
    }
    payload.update(overrides)
    return RankedProbe(**payload)


def parse_string_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def current_timestamp_label() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def to_jsonable(value):
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def probe_from_dict(payload: dict) -> ProbeResult:
    allowed = ProbeResult.__dataclass_fields__.keys()
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return ProbeResult(**filtered)


def phase_path(output_dir: Path, phase: str) -> Path:
    return output_dir / f"{phase}.json"


def load_phase_if_matching(output_dir: Path, phase: str, expected_inputs: dict, *, force: bool) -> dict | None:
    if force:
        return None
    payload = read_json(phase_path(output_dir, phase))
    if not isinstance(payload, dict):
        return None
    if payload.get("inputs") != expected_inputs:
        return None
    return payload


def save_phase(output_dir: Path, phase: str, *, inputs: dict, payload: dict) -> dict:
    wrapped = {"inputs": inputs, "payload": payload}
    write_json(phase_path(output_dir, phase), wrapped)
    return wrapped


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1.0:
        return f"{seconds:.1f}s"
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    rounded = int(round(seconds))
    if rounded < 60:
        return f"{rounded}s"
    minutes, secs = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if minutes else f"{hours}h"


def set_friendly_progress(enabled: bool) -> None:
    global FRIENDLY_PROGRESS_ENABLED
    FRIENDLY_PROGRESS_ENABLED = enabled


def calibration_progress(message: str) -> None:
    print(f"[calibrate] {message}", file=sys.stderr, flush=True)


def phase_start(name: str, *, detail: str | None = None, estimate_seconds: float | None = None) -> float:
    if FRIENDLY_PROGRESS_ENABLED:
        headline = f"Starting {name}."
        if estimate_seconds is not None:
            headline += f" Expected time: about {format_duration(estimate_seconds)}."
        calibration_progress(headline)
        preamble = PHASE_FRIENDLY_PREAMBLES.get(name)
        if preamble:
            calibration_progress(preamble)
        if detail:
            calibration_progress(f"Current scope: {detail}.")
    else:
        parts = [f"Starting {name}"]
        if detail:
            parts.append(detail)
        if estimate_seconds is not None:
            parts.append(f"estimate {format_duration(estimate_seconds)}")
        calibration_progress(" | ".join(parts))
    return time.monotonic()


def phase_done(name: str, started_at: float, *, detail: str | None = None) -> None:
    elapsed = time.monotonic() - started_at
    if FRIENDLY_PROGRESS_ENABLED:
        message = f"Finished {name} in {format_duration(elapsed)}."
        if detail:
            message += f" Result: {detail}."
        calibration_progress(message)
    else:
        parts = [f"Finished {name}", f"in {format_duration(elapsed)}"]
        if detail:
            parts.append(detail)
        calibration_progress(" | ".join(parts))


def phase_reused(name: str, *, detail: str | None = None) -> None:
    if FRIENDLY_PROGRESS_ENABLED:
        message = f"Reusing cached {name} results."
        if detail:
            message += f" Cached result: {detail}."
        calibration_progress(message)
    else:
        parts = [f"Reusing {name}"]
        if detail:
            parts.append(detail)
        calibration_progress(" | ".join(parts))


def estimate_calibration_phases(
    *,
    engine: TrainingEngine,
    presets: list[str],
    mode: str,
    batch_profile_time_budget: float,
    coarse_time_budget: float,
    projection_time_budget: float,
    finalist_time_budget: float,
    finalist_count: int,
    batch_audit_time_budget: float,
    local_search_time_budget: float,
    eval_train_seconds: float,
    eval_rungs: list[str],
    scaling_confirmation_enabled: bool,
) -> list[tuple[str, float | None]]:
    estimates: list[tuple[str, float | None]] = [("hardware fingerprint", 1.0)]
    if engine.capabilities.supports_local_search:
        anchor = choose_batch_profile_anchor_preset(engine, presets)
        anchor_seq_len = engine.preset_catalog()[anchor].seq_len
        batch_candidates = len(engine.batch_profile_candidates(anchor, seq_len=anchor_seq_len))
        estimates.append(("anchor batch profile", batch_candidates * batch_profile_time_budget))
    estimates.append(("coarse envelope", len(presets) * coarse_time_budget))
    estimates.append(("projection ranking", len(presets) * projection_time_budget))
    if len(presets) > 1:
        estimates.append(("finalist rerank", min(len(presets), finalist_count) * finalist_time_budget))
        estimates.append(("winner confirmation", 2 * PROJECTION_TARGET_SECONDS))
    if engine.capabilities.supports_local_search:
        estimates.append(("winner batch audit", batch_audit_time_budget * 6))
        estimates.append(("local search", local_search_time_budget * 4))
    if engine.capabilities.supports_checkpoint_mint:
        estimates.append(("candidate checkpoint", eval_train_seconds))
    if engine.capabilities.supports_eval_calibration and engine.capabilities.supports_checkpoint_mint:
        estimates.append(("eval calibration", 30.0 * len(eval_rungs)))
    if scaling_confirmation_enabled:
        estimates.append(("scaling confirmation", 2 * SCALING_TARGET_SECONDS))
    estimates.append(("final report", 2.0))
    return estimates


def announce_calibration_story(
    *,
    engine: TrainingEngine,
    hardware_key: str,
    mode: str,
    output_dir: Path,
    presets: list[str],
    truth_curves_dir: Path | None,
    truth_curve_count: int,
    phase_estimates: list[tuple[str, float | None]],
    scaling_confirmation_enabled: bool,
) -> None:
    if not FRIENDLY_PROGRESS_ENABLED:
        calibration_progress(
            f"Starting {engine.name} calibration on {hardware_key} in {mode} mode."
        )
        calibration_progress(f"Output directory: {output_dir}")
        calibration_progress(
            f"Objective: lowest val_bpb at {int(PROJECTION_TARGET_SECONDS)}s. "
            f"Scaling confirmation is {'enabled' if scaling_confirmation_enabled else 'disabled (opt-in)'}."
        )
        calibration_progress(f"Preset families: {', '.join(presets)}")
        if truth_curves_dir is not None:
            calibration_progress(
                f"Truth curves: {truth_curve_count} loaded from {truth_curves_dir}"
            )
        else:
            calibration_progress("Truth curves: none configured; projections will rely on generic fits.")
        return

    total_estimate = sum(seconds or 0.0 for _, seconds in phase_estimates)
    calibration_progress(
        "This is autoresearch-everywhere's calibration process. It determines sensible training defaults for your hardware so later autoresearch does not waste time on obviously bad batch shapes, model families, or eval settings."
    )
    calibration_progress(
        f"You are calibrating the {engine.name} engine on {hardware_key} in {mode} mode. The main objective is the best validation BPB at {int(PROJECTION_TARGET_SECONDS)} seconds."
    )
    calibration_progress(
        f"Expect roughly {format_duration(total_estimate)} for a fresh run before any cache reuse. Cached reruns are usually much faster."
    )
    calibration_progress(
        f"Output will be written to {output_dir}. There will be a report at the end, plus updates through each phase."
    )
    calibration_progress(f"Preset families under consideration: {', '.join(presets)}.")
    if truth_curves_dir is not None:
        calibration_progress(
            f"Truth-curve support is active from {truth_curves_dir} with {truth_curve_count} loaded curves, so horizon projection will be grounded where possible."
        )
    else:
        calibration_progress(
            "No truth-curve directory is configured, so longer-horizon projection will rely on generic fits instead of hardware-matched truth curves."
        )
    calibration_progress(
        f"Deeper {int(SCALING_TARGET_SECONDS)}-second scaling confirmation is {'enabled' if scaling_confirmation_enabled else 'disabled by default'}."
    )
    calibration_progress("Planned phases:")
    for index, (name, seconds) in enumerate(phase_estimates, start=1):
        duration_text = format_duration(seconds) if seconds is not None else "unknown"
        calibration_progress(f"  {index}. {name} (~{duration_text})")


def run_train_probe(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
    stage: str,
    benchmark_skip_eval: bool,
    checkpoint_path: Path | None = None,
    seq_len: int | None = None,
    window_pattern: str | None = None,
    device_batch_size: int | None = None,
    total_batch_size: int | None = None,
    eval_seq_len: int | None = None,
    eval_tokens: int | None = None,
    eval_batch_size: int | None = None,
    no_checkpoint: bool = True,
) -> ProbeResult:
    return engine.run_train_probe(
        preset=preset,
        time_budget=time_budget,
        logs_dir=logs_dir,
        stage=stage,
        benchmark_skip_eval=benchmark_skip_eval,
        checkpoint_path=checkpoint_path,
        seq_len=seq_len,
        window_pattern=window_pattern,
        device_batch_size=device_batch_size,
        total_batch_size=total_batch_size,
        curve_eval_seconds=None,
        eval_seq_len=eval_seq_len,
        eval_tokens=eval_tokens,
        eval_batch_size=eval_batch_size,
        no_checkpoint=no_checkpoint,
    )


def load_probe_curve_artifacts(rows: list[ProbeResult]) -> list:
    curves = []
    for row in rows:
        if not row.curve_output_path:
            continue
        path = Path(row.curve_output_path)
        if not path.exists():
            continue
        try:
            curves.append(load_curve_artifact(path))
        except Exception:
            continue
    return curves


def run_curve_train_probe(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
    stage: str,
    curve_eval_seconds: tuple[float, ...],
    seq_len: int | None = None,
    window_pattern: str | None = None,
    device_batch_size: int | None = None,
    total_batch_size: int | None = None,
    eval_seq_len: int | None = None,
    eval_tokens: int | None = None,
    eval_batch_size: int | None = None,
) -> ProbeResult:
    return engine.run_train_probe(
        preset=preset,
        time_budget=time_budget,
        logs_dir=logs_dir,
        stage=stage,
        benchmark_skip_eval=False,
        checkpoint_path=None,
        seq_len=seq_len,
        window_pattern=window_pattern,
        device_batch_size=device_batch_size,
        total_batch_size=total_batch_size,
        curve_eval_seconds=curve_eval_seconds,
        eval_seq_len=eval_seq_len,
        eval_tokens=eval_tokens,
        eval_batch_size=eval_batch_size,
        no_checkpoint=True,
    )


def default_output_dir(*, engine_name: str, hardware_key: str) -> Path:
    tag = current_timestamp_label()
    return REPO_ROOT / "results" / "analysis" / f"platform_calibration_{engine_name}_{hardware_key}_{tag}"


def select_presets(engine: TrainingEngine, requested: list[str]) -> list[str]:
    if not requested:
        return list(engine.default_platform_presets())
    selected = [preset for preset in requested if preset in engine.preset_catalog()]
    missing = [preset for preset in requested if preset not in engine.preset_catalog()]
    if missing:
        raise ValueError(f"Unknown presets: {missing}")
    return list(dict.fromkeys(selected))


def preset_index(engine: TrainingEngine, preset: str) -> int:
    return engine.preset_order().index(preset)


def resolve_mode_spec(mode: str) -> PlatformModeSpec:
    try:
        return MODE_SPECS[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown mode: {mode}") from exc


def resolved_budget(value: float | None, *, mode: str, field: str) -> float:
    if value is not None:
        return value
    return getattr(resolve_mode_spec(mode), field)


def resolved_eval_rungs(value: list[str] | None, *, mode: str) -> list[str]:
    if value:
        return value
    return list(resolve_mode_spec(mode).eval_rungs)


def default_local_seq_lens(engine: TrainingEngine, preset: str, *, mode: str) -> list[int]:
    return engine.default_local_seq_lens(preset, mode=mode)


def default_local_window_patterns(engine: TrainingEngine, preset: str, *, mode: str) -> list[str]:
    return engine.default_local_window_patterns(preset, mode=mode)


def estimate_eval_overhead_fraction(row: ProbeResult, ranking_time_budget: float) -> float:
    if row.training_seconds is not None and row.total_seconds is not None:
        return max(0.0, row.total_seconds - row.training_seconds) / max(ranking_time_budget, 1e-9)
    if row.eval_percent is not None:
        return max(0.0, row.eval_percent) / 100.0
    return 0.0


def telemetry_for_preset(preset: str, *, hardware_key: str) -> dict:
    if not HAS_MLX_CALIBRATION_SUPPORT or summarize_eval_telemetry is None:
        return {
            "eligible_count": 0,
            "commit_count": 0,
            "day_count": 0,
            "observed_rungs": [],
            "stable_rungs": [],
            "last_seen_on": None,
            "last_seen_age_days": None,
        }
    summary = summarize_eval_telemetry(preset, hardware_key=hardware_key, policy_version=EVAL_POLICY_VERSION)
    return {
        "eligible_count": summary.eligible_count,
        "commit_count": summary.commit_count,
        "day_count": summary.day_count,
        "observed_rungs": list(summary.observed_rungs),
        "stable_rungs": list(summary.stable_rungs),
        "last_seen_on": summary.last_seen_on,
        "last_seen_age_days": summary.last_seen_age_days,
    }


def _normalize_higher_is_better(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if math.isclose(high, low):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _normalize_lower_is_better(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if math.isclose(high, low):
        return [1.0 for _ in values]
    return [1.0 - ((value - low) / (high - low)) for value in values]


def _pareto_front(flags: list[tuple[float, ...]]) -> list[bool]:
    front = [True] * len(flags)
    epsilon = 1e-9
    for i, score_i in enumerate(flags):
        for j, score_j in enumerate(flags):
            if i == j:
                continue
            dominates = all(
                score_j[k] >= score_i[k] - epsilon
                for k in range(len(score_i))
            ) and any(
                score_j[k] > score_i[k] + epsilon
                for k in range(len(score_i))
            )
            if dominates:
                front[i] = False
                break
    return front


def _memory_pressure_metrics(
    *,
    peak_vram_mb: float | None,
    total_memory_gb: float | None,
    fallback_score: float,
) -> tuple[float, str, float, float]:
    if peak_vram_mb is None:
        return (None, "unknown", fallback_score, 0.1 * (1.0 - fallback_score))
    if total_memory_gb is None or total_memory_gb <= 0:
        return (None, "relative-only", fallback_score, 0.1 * (1.0 - fallback_score))

    total_memory_mb = total_memory_gb * 1024.0
    memory_fraction = peak_vram_mb / max(total_memory_mb, 1e-9)

    if memory_fraction <= 0.50:
        # Comfortable zone: still retain a small tie-break cost as usage rises.
        pressure_band = "comfortable"
        memory_score = 1.0
        t = memory_fraction / 0.50
        tiebreak_penalty = 0.05 * t
    elif memory_fraction <= 0.75:
        # Warm zone: start shaping selection meaningfully but not aggressively.
        pressure_band = "warm"
        t = (memory_fraction - 0.50) / 0.25
        memory_score = 1.0 - (0.30 * t)
        tiebreak_penalty = 0.05 + (0.20 * t)
    elif memory_fraction <= 0.90:
        # Pressured zone: memory should now materially affect the recommendation.
        pressure_band = "pressured"
        t = (memory_fraction - 0.75) / 0.15
        memory_score = 0.70 - (0.50 * t)
        tiebreak_penalty = 0.25 + (0.50 * t)
    else:
        # Critical zone: heavily penalize near-capacity operating points.
        pressure_band = "critical"
        t = min(1.0, (memory_fraction - 0.90) / 0.10)
        memory_score = max(0.0, 0.20 - (0.20 * t))
        tiebreak_penalty = 0.75 + (0.75 * t)

    return (memory_fraction, pressure_band, memory_score, tiebreak_penalty)


def rank_candidate_families(
    rows: list[ProbeResult],
    *,
    engine: TrainingEngine,
    hardware_key: str,
    hardware: HardwareFingerprint,
    ranking_time_budget: float,
) -> list[RankedProbe]:
    candidates = [
        row
        for row in rows
        if row.status == "ok" and row.preset != engine.reference_preset and row.val_bpb is not None
    ]
    if not candidates:
        candidates = [row for row in rows if row.status == "ok" and row.val_bpb is not None]
    if not candidates:
        raise RuntimeError("No successful ranking runs available to choose a candidate family.")

    quality_scores = _normalize_lower_is_better([float(row.val_bpb) for row in candidates])
    throughput_scores = _normalize_higher_is_better([row.steady_state_tok_per_sec or 0.0 for row in candidates])
    relative_memory_scores = _normalize_lower_is_better([row.peak_vram_mb or float("inf") for row in candidates])
    eval_overheads = [estimate_eval_overhead_fraction(row, ranking_time_budget) for row in candidates]
    eval_overhead_scores = _normalize_lower_is_better(eval_overheads)
    memory_metrics = [
        _memory_pressure_metrics(
            peak_vram_mb=candidates[index].peak_vram_mb,
            total_memory_gb=hardware.memory_gb,
            fallback_score=relative_memory_scores[index],
        )
        for index in range(len(candidates))
    ]
    pareto_dimensions = [
        (
            quality_scores[index],
            throughput_scores[index],
            eval_overhead_scores[index],
            memory_metrics[index][2],
        )
        for index in range(len(candidates))
    ]
    on_pareto_front = _pareto_front(pareto_dimensions)
    ranked: list[RankedProbe] = []
    for index, row in enumerate(candidates):
        telemetry = telemetry_for_preset(row.preset, hardware_key=hardware_key)
        stable_rung_count = len(telemetry["stable_rungs"])
        telemetry_score = min(1.0, stable_rung_count / 2.0)
        quality_score = quality_scores[index]
        throughput_score = throughput_scores[index]
        memory_fraction, memory_pressure_band, memory_score, memory_tiebreak_penalty = memory_metrics[index]
        eval_overhead_fraction = eval_overheads[index]
        eval_overhead_score = eval_overhead_scores[index]
        frontier_distance = math.sqrt(
            (
                (1.0 - quality_score) ** 2
                + (1.0 - throughput_score) ** 2
                + (1.0 - eval_overhead_score) ** 2
                + (1.0 - memory_score) ** 2
            )
            / 4.0
        )
        selection_distance = frontier_distance + memory_tiebreak_penalty
        utility_score = 1.0 / (1.0 + selection_distance)
        ranked.append(
            RankedProbe(
                probe=row,
                quality_score=quality_score,
                throughput_score=throughput_score,
                memory_score=memory_score,
                eval_overhead_score=eval_overhead_score,
                telemetry_score=telemetry_score,
                utility_score=utility_score,
                on_pareto_front=on_pareto_front[index],
                frontier_distance=frontier_distance,
                selection_distance=selection_distance,
                estimated_eval_overhead_fraction=eval_overhead_fraction,
                memory_fraction=memory_fraction,
                memory_pressure_band=memory_pressure_band,
                memory_tiebreak_penalty=memory_tiebreak_penalty,
                telemetry_count=telemetry["eligible_count"],
                stable_rung_count=stable_rung_count,
                effective_confidence=row.eval_calibration_effective_confidence,
                calibration_status=row.eval_calibration_status,
            )
        )
    pressure_order = {
        "comfortable": 0,
        "warm": 1,
        "pressured": 2,
        "critical": 3,
        "unknown": 4,
    }
    ranked.sort(
        key=lambda item: (
            item.probe.val_bpb or float("inf"),
            pressure_order.get(item.memory_pressure_band, pressure_order["unknown"]),
            item.memory_tiebreak_penalty,
            -(item.probe.steady_state_tok_per_sec or 0.0),
            item.probe.peak_vram_mb or float("inf"),
            -item.stable_rung_count,
            -item.telemetry_count,
            not item.on_pareto_front,
            item.selection_distance,
        )
    )
    return ranked


def choose_candidate_family(
    rows: list[ProbeResult],
    *,
    engine: TrainingEngine,
    hardware_key: str,
    hardware: HardwareFingerprint,
    ranking_time_budget: float,
) -> RankedProbe:
    return rank_candidate_families(
        rows,
        engine=engine,
        hardware_key=hardware_key,
        hardware=hardware,
        ranking_time_budget=ranking_time_budget,
    )[0]


def ranked_probe_to_dict(item: RankedProbe) -> dict:
    payload = asdict(item.probe)
    payload.update(
        {
            "quality_score": item.quality_score,
            "throughput_score": item.throughput_score,
            "memory_score": item.memory_score,
            "eval_overhead_score": item.eval_overhead_score,
            "telemetry_score": item.telemetry_score,
            "utility_score": item.utility_score,
            "on_pareto_front": item.on_pareto_front,
            "frontier_distance": item.frontier_distance,
            "selection_distance": item.selection_distance,
            "estimated_eval_overhead_fraction": item.estimated_eval_overhead_fraction,
            "memory_fraction": item.memory_fraction,
            "memory_pressure_band": item.memory_pressure_band,
            "memory_tiebreak_penalty": item.memory_tiebreak_penalty,
            "telemetry_count": item.telemetry_count,
            "stable_rung_count": item.stable_rung_count,
            "effective_confidence": item.effective_confidence,
            "calibration_status": item.calibration_status,
        }
    )
    return payload


def projected_row_to_dict(item) -> dict:
    payload = asdict(item.summary)
    payload.update(
        {
            "projected_val_bpb": item.corrected_val_bpb,
            "projection_std": item.projection_std,
            "fit_r2": item.fit_r2,
            "fit_sigma": item.fit_sigma,
            "calibration_horizon_seconds": item.calibration_horizon_seconds,
            "calibration_horizon_tokens": item.calibration_horizon_tokens,
            "calibration_sample_count": item.calibration_sample_count,
            "correction_mean": item.correction_mean,
            "projection_source": item.projection_source,
            "matched_truth_count": item.matched_truth_count,
            "winner_probability": item.winner_probability,
            "enough_signal": item.enough_signal,
            "confidence_reason": item.confidence_reason,
            "truth_anchor_seconds": item.truth_anchor_seconds,
            "truth_anchor_tokens": item.truth_anchor_tokens,
            "extrapolation_ratio": item.extrapolation_ratio,
        }
    )
    return payload


def get_projected_val_bpb(row: dict | None) -> float | None:
    if row is None:
        return None
    return row.get("projected_val_bpb", row.get("projected_final_bpb"))


def decision_enough_signal(decision) -> bool:
    if decision is None:
        return False
    if isinstance(decision, dict):
        return bool(decision.get("enough_signal"))
    return bool(getattr(decision, "enough_signal", False))


def find_horizon_decision(decisions, target_seconds: float):
    for item in decisions or []:
        horizon = item.get("target_seconds") if isinstance(item, dict) else getattr(item, "target_seconds", None)
        if horizon is not None and math.isclose(float(horizon), float(target_seconds)):
            return item
    return None


def rows_for_horizon(rows, target_seconds: float) -> list:
    matched = []
    for item in rows or []:
        horizon = item.get("target_seconds") if isinstance(item, dict) else getattr(item, "target_seconds", None)
        if horizon is not None and math.isclose(float(horizon), float(target_seconds)):
            matched.append(item)
    return matched


def resolve_horizon_control_decision(decisions, *, target_seconds: float):
    target = find_horizon_decision(decisions, target_seconds)
    if target is None:
        return None
    target_preset = target.get("top_preset") if isinstance(target, dict) else getattr(target, "top_preset", None)
    if target_preset is None:
        return target
    disagreements = []
    supporting = 0
    for item in decisions or []:
        horizon = item.get("target_seconds") if isinstance(item, dict) else getattr(item, "target_seconds", None)
        if horizon is None or float(horizon) > float(target_seconds):
            continue
        if not decision_enough_signal(item):
            continue
        item_preset = item.get("top_preset") if isinstance(item, dict) else getattr(item, "top_preset", None)
        if item_preset is None:
            continue
        if item_preset == target_preset:
            supporting += 1
        else:
            disagreements.append(float(horizon))
    enough_signal = decision_enough_signal(target) and not disagreements
    confidence_reason = target.get("confidence_reason") if isinstance(target, dict) else getattr(target, "confidence_reason", None)
    if disagreements:
        disagreement_label = ", ".join(f"{value:.0f}s" for value in disagreements)
        confidence_reason = f"horizon-disagreement:{disagreement_label}"
    elif enough_signal:
        confidence_reason = f"consistent-horizons:{supporting}"
    if isinstance(target, dict):
        resolved = dict(target)
        resolved["enough_signal"] = enough_signal
        resolved["confidence_reason"] = confidence_reason
        return resolved
    return target.__class__(
        target_seconds=target.target_seconds,
        top_preset=target.top_preset,
        top_projected_tokens=target.top_projected_tokens,
        top_winner_probability=target.top_winner_probability,
        top_margin_to_second=target.top_margin_to_second,
        top_margin_snr=target.top_margin_snr,
        enough_signal=enough_signal,
        confidence_reason=confidence_reason,
    )


def load_truth_curves(
    *,
    truth_curves_dir: Path | None,
    engine_name: str,
    hardware_key: str,
    target_seconds: float,
) -> list:
    if truth_curves_dir is None or not truth_curves_dir.exists():
        return []
    return load_curve_artifacts_from_dir(
        truth_curves_dir,
        engine=engine_name,
        hardware_key=hardware_key,
        require_target_seconds=target_seconds,
    )


def resolve_truth_curve_inputs(
    *,
    truth_curves_dir: str | None,
    output_dir: Path,
    engine_name: str,
    hardware_key: str,
    target_seconds: float,
) -> tuple[Path | None, list]:
    search_dirs: list[Path] = []
    if truth_curves_dir:
        search_dirs.append(Path(truth_curves_dir))
    else:
        search_dirs.extend(
            [
                output_dir / "truth_curves",
                output_dir.parent / "truth_curves",
                output_dir.parent / "curve_runs",
                Path.home() / "curve_runs",
                Path.home() / ".cache" / "autoresearch" / "truth_curves",
                REPO_ROOT / "results" / "analysis" / "truth_curves",
            ]
        )
    seen: set[Path] = set()
    for candidate in search_dirs:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        curves = load_truth_curves(
            truth_curves_dir=resolved,
            engine_name=engine_name,
            hardware_key=hardware_key,
            target_seconds=target_seconds,
        )
        if curves:
            return resolved, curves
    return (Path(truth_curves_dir).expanduser() if truth_curves_dir else None), []


def infer_truth_eval_contract(truth_curves: list) -> dict[str, int] | None:
    counts: dict[tuple[int, int, int], int] = {}
    for curve in truth_curves:
        final_eval = curve.final_eval or {}
        seq_len = final_eval.get("eval_seq_len")
        eval_tokens = final_eval.get("eval_tokens")
        batch_size = final_eval.get("eval_batch_size")
        if seq_len is None or eval_tokens is None or batch_size is None:
            continue
        key = (int(seq_len), int(eval_tokens), int(batch_size))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    seq_len, eval_tokens, batch_size = max(counts.items(), key=lambda item: (item[1], item[0][0], item[0][1], item[0][2]))[0]
    return {
        "eval_seq_len": seq_len,
        "eval_tokens": eval_tokens,
        "eval_batch_size": batch_size,
    }


def classify_zones(*, presets: list[str], candidate: RankedProbe, probe_by_preset: dict[str, ProbeResult]) -> dict[str, dict[str, str | float]]:
    zones: dict[str, dict[str, str | float]] = {}
    candidate_probe = candidate.probe
    for preset in presets:
        row = probe_by_preset.get(preset)
        if row is None or row.status != "ok":
            zones[preset] = {"zone": "fail", "reason": "probe failed or did not complete"}
            continue
        if preset == "upstream":
            zones[preset] = {"zone": "reference", "reason": "upstream-style reference preset"}
            continue
        if preset == candidate_probe.preset:
            zones[preset] = {"zone": "recommended", "reason": "best measured frontier candidate among the ranked preset families"}
            continue
        throughput_ratio = (row.steady_state_tok_per_sec or 0.0) / max(candidate_probe.steady_state_tok_per_sec or 1.0, 1e-9)
        memory_ratio = (row.peak_vram_mb or float("inf")) / max(candidate_probe.peak_vram_mb or 1.0, 1e-9)
        quality_delta = (row.val_bpb or 0.0) - (candidate_probe.val_bpb or 0.0)
        lighter_or_faster = throughput_ratio >= 1.0 or memory_ratio <= 0.85
        if lighter_or_faster and quality_delta > 0.0:
            zones[preset] = {
                "zone": "lower",
                "reason": "materially lighter or faster than the candidate, but with worse short-run validation quality",
                "throughput_ratio": throughput_ratio,
                "memory_ratio": memory_ratio,
                "quality_delta": quality_delta,
            }
        else:
            zones[preset] = {
                "zone": "upper",
                "reason": "heavier or slower than the candidate without a strong enough measured short-run advantage to become the default",
                "throughput_ratio": throughput_ratio,
                "memory_ratio": memory_ratio,
                "quality_delta": quality_delta,
            }
    return zones


def choose_batch_profile_anchor_preset(engine: TrainingEngine, presets: list[str]) -> str:
    non_reference = [preset for preset in presets if preset != engine.reference_preset]
    if not non_reference:
        return presets[0]
    if len(non_reference) >= 2:
        return non_reference[1]
    return non_reference[0]


def run_batch_profile(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
) -> dict:
    preset_config = engine.preset_catalog()[preset]
    seen: set[tuple[int, int]] = set()

    def run_candidates(candidates: list[tuple[int, int]]) -> list[ProbeResult]:
        rows: list[ProbeResult] = []
        for device_batch, total_batch in candidates:
            if (device_batch, total_batch) in seen:
                continue
            seen.add((device_batch, total_batch))
            tokens_per_fwdbwd = preset_config.seq_len * device_batch
            if total_batch % tokens_per_fwdbwd != 0:
                continue
            rows.append(
                run_train_probe(
                    engine=engine,
                    preset=preset,
                    time_budget=time_budget,
                    logs_dir=logs_dir,
                    stage="batch-profile",
                    benchmark_skip_eval=True,
                    seq_len=preset_config.seq_len,
                    window_pattern=preset_config.window_pattern,
                    device_batch_size=device_batch,
                    total_batch_size=total_batch,
                    no_checkpoint=True,
                )
            )
        return rows

    coarse_rows = run_candidates(engine.batch_profile_candidates(preset, seq_len=preset_config.seq_len))
    coarse_winner = select_best_batch_profile_row(coarse_rows)
    refinement_rows = run_candidates(
        engine.batch_profile_refinement_candidates(
            preset,
            seq_len=preset_config.seq_len,
            coarse_winner=(coarse_winner.device_batch_size, coarse_winner.total_batch_size),
        )
    )
    all_rows = coarse_rows + refinement_rows
    extension_rows: list[ProbeResult] = []
    while should_extend_batch_profile(all_rows):
        current_winner = select_best_batch_profile_row(all_rows)
        more_candidates = batch_profile_extension_candidates(
            seq_len=preset_config.seq_len,
            winner=current_winner,
        )
        new_rows = run_candidates(more_candidates)
        if not new_rows:
            break
        extension_rows.extend(new_rows)
        all_rows.extend(new_rows)

    winner = select_best_batch_profile_row(all_rows)
    return {
        "coarse_rows": coarse_rows,
        "refinement_rows": refinement_rows,
        "extension_rows": extension_rows,
        "rows": all_rows,
        "winner": winner,
    }


def apply_batch_profile(*, seq_len: int, batch_profile: ProbeResult) -> tuple[int, int, int]:
    device_batch_size = batch_profile.device_batch_size
    tokens_per_fwdbwd = seq_len * device_batch_size
    target_total_tokens = batch_profile.total_batch_size
    grad_accum_steps = max(1, round(target_total_tokens / max(tokens_per_fwdbwd, 1)))
    total_batch_size = tokens_per_fwdbwd * grad_accum_steps
    return device_batch_size, total_batch_size, grad_accum_steps


def batch_profile_overrides_to_dict(
    overrides: dict[str, tuple[int, int]],
    *,
    presets: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    selected_presets = presets if presets is not None else list(overrides)
    return {
        preset: {
            "device_batch_size": overrides[preset][0],
            "total_batch_size": overrides[preset][1],
        }
        for preset in selected_presets
        if preset in overrides
    }


def run_batch_profile_audit(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
    anchor_batch_profile: ProbeResult,
    eval_seq_len: int | None,
    eval_tokens: int | None,
    eval_batch_size: int | None,
    stage: str = "winner-batch-audit",
) -> dict:
    preset_config = engine.preset_catalog()[preset]
    seen: set[tuple[int, int]] = set()
    if time_budget > 5.0:
        early = max(3.0, min(5.0, time_budget / 3.0))
        mid = max(early + 1.0, min(time_budget - 1.0, time_budget * 0.67))
        curve_eval_seconds = tuple(sorted({early, mid, time_budget}))
    else:
        curve_eval_seconds = (time_budget,)

    def run_candidates(candidates: list[tuple[int, int]]) -> list[ProbeResult]:
        rows: list[ProbeResult] = []
        for device_batch, total_batch in candidates:
            if (device_batch, total_batch) in seen:
                continue
            seen.add((device_batch, total_batch))
            tokens_per_fwdbwd = preset_config.seq_len * device_batch
            if total_batch % tokens_per_fwdbwd != 0:
                continue
            rows.append(
                run_curve_train_probe(
                    engine=engine,
                    preset=preset,
                    time_budget=time_budget,
                    logs_dir=logs_dir,
                    stage=stage,
                    curve_eval_seconds=curve_eval_seconds,
                    seq_len=preset_config.seq_len,
                    window_pattern=preset_config.window_pattern,
                    device_batch_size=device_batch,
                    total_batch_size=total_batch,
                    eval_seq_len=eval_seq_len,
                    eval_tokens=eval_tokens,
                    eval_batch_size=eval_batch_size,
                )
            )
        return rows

    anchor_device_batch, anchor_total_batch, _ = apply_batch_profile(
        seq_len=preset_config.seq_len,
        batch_profile=anchor_batch_profile,
    )
    anchor_rows = run_candidates([(anchor_device_batch, anchor_total_batch)])
    coarse_rows = run_candidates(engine.batch_profile_candidates(preset, seq_len=preset_config.seq_len))
    refinement_rows = run_candidates(
        engine.batch_profile_refinement_candidates(
            preset,
            seq_len=preset_config.seq_len,
            coarse_winner=(anchor_device_batch, anchor_total_batch),
        )
    )
    all_rows = anchor_rows + coarse_rows + refinement_rows
    return {
        "anchor_rows": anchor_rows,
        "coarse_rows": coarse_rows,
        "refinement_rows": refinement_rows,
        "rows": all_rows,
        "anchor_batch": {
            "device_batch_size": anchor_device_batch,
            "total_batch_size": anchor_total_batch,
        },
    }


def summarize_batch_audit_rows(
    rows: list[ProbeResult],
    *,
    target_seconds: float,
    calibration=None,
    truth_curves=None,
    artifact_dir: Path | None = None,
) -> list[dict]:
    summarized: list[dict] = []
    for row in rows:
        projection = None
        estimate = None
        exact_truth_finals = []
        if truth_curves:
            for truth_curve in truth_curves:
                if truth_curve.preset != row.preset:
                    continue
                if truth_curve.device_batch_size != row.device_batch_size:
                    continue
                if truth_curve.total_batch_size != row.total_batch_size:
                    continue
                truth_horizon = final_training_seconds(truth_curve)
                truth_final = final_val_bpb(truth_curve)
                if truth_horizon is None or truth_horizon < target_seconds or truth_final is None:
                    continue
                exact_truth_finals.append(float(truth_final))
        if row.curve_output_path:
            curve_path = Path(row.curve_output_path)
            if not curve_path.exists() and artifact_dir is not None:
                resolved = artifact_dir / curve_path.name
                if resolved.exists():
                    curve_path = resolved
            if curve_path.exists():
                try:
                    curve = load_curve_artifact(curve_path)
                    distinct_steps = {point.step for point in curve.curve_points}
                    distinct_tokens = {point.total_tokens for point in curve.curve_points}
                    distinct_times = {round(point.actual_training_seconds, 6) for point in curve.curve_points}
                    meaningful_curve = len(distinct_steps) >= 2 and len(distinct_tokens) >= 2 and len(distinct_times) >= 2
                    if meaningful_curve:
                        projection = project_curve(curve, target_seconds=target_seconds)
                        estimate = estimate_projected_curve(
                            curve,
                            target_seconds=target_seconds,
                            calibration=calibration,
                            truth_curves=truth_curves,
                        )
                except Exception:
                    projection = None
                    estimate = None
        exact_truth_mean = statistics.fmean(exact_truth_finals) if exact_truth_finals else None
        exact_truth_std = (
            max(0.01, statistics.stdev(exact_truth_finals))
            if len(exact_truth_finals) > 1
            else 0.01
            if exact_truth_finals
            else None
        )
        projected_tokens = (
            estimate.summary.projected_tokens
            if estimate is not None
            else projection.projected_tokens
            if projection is not None
            else (row.steady_state_tok_per_sec * target_seconds if row.steady_state_tok_per_sec is not None else None)
        )
        projected_val_bpb = (
            exact_truth_mean
            if exact_truth_mean is not None
            else estimate.corrected_val_bpb
            if estimate is not None
            else projection.projected_val_bpb
            if projection is not None
            else row.val_bpb
        )
        projection_source = (
            "truth-match-exact-batch"
            if exact_truth_mean is not None
            else estimate.projection_source if estimate is not None else None
        )
        projection_std = (
            exact_truth_std
            if exact_truth_std is not None
            else estimate.projection_std if estimate is not None else None
        )
        matched_truth_count = (
            len(exact_truth_finals)
            if exact_truth_finals
            else estimate.matched_truth_count if estimate is not None else 0
        )
        selection_val_bpb = projected_val_bpb
        if projection_source in {None, "generic-projection", "calibrated-extrapolation"} and row.val_bpb is not None:
            selection_val_bpb = row.val_bpb
        projected_optimizer_steps = (
            projected_tokens / row.total_batch_size
            if projected_tokens is not None and row.total_batch_size > 0
            else None
        )
        tokens_per_fwdbwd = row.seq_len * row.device_batch_size
        projected_microsteps = (
            projected_tokens / tokens_per_fwdbwd
            if projected_tokens is not None and tokens_per_fwdbwd > 0
            else None
        )
        summarized.append(
            {
                "preset": row.preset,
                "seq_len": row.seq_len,
                "window_pattern": row.window_pattern,
                "device_batch_size": row.device_batch_size,
                "total_batch_size": row.total_batch_size,
                "grad_accum_steps": row.grad_accum_steps,
                "status": row.status,
                "observed_val_bpb": row.val_bpb,
                "projected_val_bpb": projected_val_bpb,
                "selection_val_bpb": selection_val_bpb,
                "projected_tokens": projected_tokens,
                "projected_optimizer_steps": projected_optimizer_steps,
                "projected_microsteps": projected_microsteps,
                "steady_state_tok_per_sec": row.steady_state_tok_per_sec,
                "optimizer_percent": row.optimizer_percent,
                "accum_percent": row.accum_percent,
                "control_overhead_percent": row.control_overhead_percent,
                "peak_vram_mb": row.peak_vram_mb,
                "curve_points": row.curve_eval_points,
                "projection_method": (
                    estimate.summary.projection_method
                    if estimate is not None
                    else None if projection is None else projection.method
                ),
                "projection_source": projection_source,
                "projection_std": projection_std,
                "matched_truth_count": matched_truth_count,
                "fit_r2": (
                    estimate.fit_r2
                    if estimate is not None
                    else None if projection is None else projection.fit_r2
                ),
                "fit_sigma": (
                    estimate.fit_sigma
                    if estimate is not None
                    else None if projection is None else projection.fit_sigma
                ),
                "stdout_path": row.stdout_path,
                "stderr_path": row.stderr_path,
            }
        )
    return summarized


def select_best_batch_audit_row(
    rows: list[ProbeResult],
    *,
    target_seconds: float,
    calibration=None,
    truth_curves=None,
    artifact_dir: Path | None = None,
) -> tuple[ProbeResult, list[dict], dict]:
    audit_rows = summarize_batch_audit_rows(
        rows,
        target_seconds=target_seconds,
        calibration=calibration,
        truth_curves=truth_curves,
        artifact_dir=artifact_dir,
    )
    keyed = {
        (row.device_batch_size, row.total_batch_size): row
        for row in rows
    }
    projected_rows = [
        row for row in audit_rows
        if row["status"] == "ok" and row["projected_val_bpb"] is not None
    ]
    if projected_rows:
        projected_rows.sort(
            key=lambda row: (
                row["selection_val_bpb"],
                row["observed_val_bpb"] if row["observed_val_bpb"] is not None else float("inf"),
                -(row["projected_optimizer_steps"] or 0.0),
                row["grad_accum_steps"] if row["grad_accum_steps"] is not None else float("inf"),
                row["accum_percent"] if row["accum_percent"] is not None else float("inf"),
                row["control_overhead_percent"] if row["control_overhead_percent"] is not None else float("inf"),
                -(row["steady_state_tok_per_sec"] or 0.0),
                row["total_batch_size"],
            )
        )
        winner = projected_rows[0]
        ordered = projected_rows + [
            row for row in audit_rows
            if row not in projected_rows
        ]
        return keyed[(winner["device_batch_size"], winner["total_batch_size"])], ordered, winner

    observed_rows = [
        row for row in audit_rows
        if row["status"] == "ok" and row["observed_val_bpb"] is not None
    ]
    if observed_rows:
        observed_rows.sort(
            key=lambda row: (
                row["observed_val_bpb"],
                -(row["projected_optimizer_steps"] or 0.0),
                row["grad_accum_steps"] if row["grad_accum_steps"] is not None else float("inf"),
                row["accum_percent"] if row["accum_percent"] is not None else float("inf"),
                row["control_overhead_percent"] if row["control_overhead_percent"] is not None else float("inf"),
                -(row["steady_state_tok_per_sec"] or 0.0),
                row["total_batch_size"],
            )
        )
        winner = observed_rows[0]
        ordered = observed_rows + [
            row for row in audit_rows
            if row not in observed_rows
        ]
        return keyed[(winner["device_batch_size"], winner["total_batch_size"])], ordered, winner

    throughput_winner = select_best_batch_profile_row(rows)
    throughput_row = next(
        row
        for row in audit_rows
        if row["device_batch_size"] == throughput_winner.device_batch_size
        and row["total_batch_size"] == throughput_winner.total_batch_size
    )
    return throughput_winner, audit_rows, throughput_row


def run_local_search(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
    batch_profile: ProbeResult,
    seq_lens: list[int] | None,
    window_patterns: list[str] | None,
    eval_seq_len: int | None,
    eval_tokens: int | None,
    eval_batch_size: int | None,
) -> list[ProbeResult]:
    preset_config = engine.preset_catalog()[preset]
    seq_candidates = seq_lens or [preset_config.seq_len]
    window_candidates = window_patterns or [preset_config.window_pattern]
    rows: list[ProbeResult] = []
    for seq_len in seq_candidates:
        device_batch, total_batch, _ = apply_batch_profile(seq_len=seq_len, batch_profile=batch_profile)
        for window_pattern in window_candidates:
            rows.append(
                run_train_probe(
                    engine=engine,
                    preset=preset,
                    time_budget=time_budget,
                    logs_dir=logs_dir,
                    stage="local-search",
                    benchmark_skip_eval=False,
                    seq_len=seq_len,
                    window_pattern=window_pattern,
                    device_batch_size=device_batch,
                    total_batch_size=total_batch,
                    eval_seq_len=eval_seq_len,
                    eval_tokens=eval_tokens,
                    eval_batch_size=eval_batch_size,
                    no_checkpoint=True,
                )
            )
    return rows


def select_best_batch_profile_row(rows: list[ProbeResult]) -> ProbeResult:
    ok_rows = [row for row in rows if row.status == "ok" and row.steady_state_tok_per_sec is not None]
    if not ok_rows:
        raise RuntimeError("Batch profile produced no successful rows.")
    best_throughput = max(row.steady_state_tok_per_sec or 0.0 for row in ok_rows)
    plateau_floor = best_throughput * PLATEAU_FRACTION
    plateau_rows = [
        row
        for row in ok_rows
        if (row.steady_state_tok_per_sec or 0.0) >= plateau_floor
    ]
    plateau_rows.sort(
        key=lambda row: (
            row.grad_accum_steps if row.grad_accum_steps is not None else float("inf"),
            row.accum_percent if row.accum_percent is not None else float("inf"),
            row.control_overhead_percent if row.control_overhead_percent is not None else float("inf"),
            -(row.steady_state_tok_per_sec or 0.0),
            row.total_batch_size,
            row.peak_vram_mb or float("inf"),
        )
    )
    return plateau_rows[0]


def select_best_local_row(rows: list[ProbeResult]) -> ProbeResult:
    quality_rows = [row for row in rows if row.status == "ok" and row.val_bpb is not None]
    if quality_rows:
        quality_rows.sort(
            key=lambda row: (
                row.val_bpb or float("inf"),
                -(row.steady_state_tok_per_sec or 0.0),
                row.peak_vram_mb or float("inf"),
                row.control_overhead_percent if row.control_overhead_percent is not None else float("inf"),
                row.total_batch_size,
            )
        )
        return quality_rows[0]

    ok_rows = [row for row in rows if row.status == "ok" and row.steady_state_tok_per_sec is not None]
    if not ok_rows:
        raise RuntimeError("Local search produced no successful rows.")
    best_throughput = max(row.steady_state_tok_per_sec or 0.0 for row in ok_rows)
    plateau_floor = best_throughput * 0.99
    plateau_rows = [
        row
        for row in ok_rows
        if (row.steady_state_tok_per_sec or 0.0) >= plateau_floor
    ]
    plateau_rows.sort(
        key=lambda row: (
            row.total_batch_size,
            row.control_overhead_percent if row.control_overhead_percent is not None else float("inf"),
            row.peak_vram_mb or float("inf"),
            -(row.steady_state_tok_per_sec or 0.0),
        )
    )
    return plateau_rows[0]


def should_extend_batch_profile(rows: list[ProbeResult]) -> bool:
    ok_rows = [row for row in rows if row.status == "ok" and row.steady_state_tok_per_sec is not None]
    if len(ok_rows) < 2:
        return False
    best_throughput = max(row.steady_state_tok_per_sec or 0.0 for row in ok_rows)
    max_total_batch = max(row.total_batch_size for row in ok_rows)
    edge_rows = [row for row in ok_rows if row.total_batch_size == max_total_batch]
    edge_best = max((row.steady_state_tok_per_sec or 0.0) for row in edge_rows)
    if edge_best < best_throughput * PLATEAU_FRACTION:
        return False

    lower_rows = [row for row in ok_rows if row.total_batch_size < max_total_batch]
    if not lower_rows:
        return False
    previous_total_batch = max(row.total_batch_size for row in lower_rows)
    previous_best = max(
        (row.steady_state_tok_per_sec or 0.0)
        for row in lower_rows
        if row.total_batch_size == previous_total_batch
    )
    if edge_best < previous_best * SHARP_EDGE_DROP_FRACTION:
        return False
    return True


def batch_profile_extension_candidates(*, seq_len: int, winner: ProbeResult) -> list[tuple[int, int]]:
    device_batch = winner.device_batch_size
    tokens_per_fwdbwd = seq_len * device_batch
    if tokens_per_fwdbwd <= 0:
        return []
    winner_grad_accum = max(1, winner.total_batch_size // tokens_per_fwdbwd)
    extension_grad_accum = {
        winner_grad_accum + 1,
        winner_grad_accum + 2,
        max(1, round(winner_grad_accum * 1.5)),
        winner_grad_accum * 2,
    }
    return sorted(
        {
            (device_batch, tokens_per_fwdbwd * grad_accum)
            for grad_accum in extension_grad_accum
            if winner_grad_accum < grad_accum <= 32
        }
    )


def compare_to_m5_reference(*, preset: str, eval_rows: list[dict], train_probe: ProbeResult) -> dict | None:
    reference = None
    if HAS_MLX_CALIBRATION_SUPPORT and find_eval_calibration is not None:
        reference = find_eval_calibration(preset, hardware_key=DEFAULT_EVAL_HARDWARE_KEY)
    train_reference = M5_TRAIN_REFERENCE.get(preset)
    if reference is None and train_reference is None:
        return None
    by_rung = {row["rung"]: row for row in eval_rows}
    comparison = {
        "reference_preset": preset,
        "reference_hardware_key": DEFAULT_EVAL_HARDWARE_KEY,
        "candidate_family_vs_m5_default": family_relation_to_m5(get_engine(DEFAULT_ENGINE_NAME), preset),
        "rungs": {},
        "train_reference": train_reference,
    }
    if train_reference is not None:
        if train_probe.steady_state_tok_per_sec is not None and train_reference["steady_state_tok_per_sec"] > 0:
            comparison["train_reference"]["local_to_m5_throughput_ratio"] = (
                train_probe.steady_state_tok_per_sec / train_reference["steady_state_tok_per_sec"]
            )
        if train_probe.peak_vram_mb is not None and train_reference["peak_vram_mb"] > 0:
            comparison["train_reference"]["local_to_m5_memory_ratio"] = (
                train_probe.peak_vram_mb / train_reference["peak_vram_mb"]
            )
    if reference is None:
        return comparison
    for rung_key in ("cheap", "reference", "full"):
        if rung_key not in by_rung:
            continue
        ref_rung = reference.rung(rung_key)
        row = by_rung[rung_key]
        comparison["rungs"][rung_key] = {
            "local_eval_seconds": row["eval_seconds"],
            "m5_eval_seconds": ref_rung.eval_seconds,
            "local_to_m5_eval_seconds_ratio": (
                row["eval_seconds"] / ref_rung.eval_seconds if ref_rung.eval_seconds > 0 else None
            ),
            "local_abs_error_vs_full": row.get("abs_error_vs_full"),
            "m5_abs_error_vs_full": ref_rung.abs_error_vs_full,
        }
    return comparison


def family_relation_to_m5(engine: TrainingEngine, preset: str) -> str:
    if preset not in engine.preset_order() or M5_REFERENCE_DEFAULT_PRESET not in engine.preset_order():
        return "not-applicable"
    current = preset_index(engine, preset)
    reference = preset_index(engine, M5_REFERENCE_DEFAULT_PRESET)
    if current == reference:
        return "same-family"
    if current < reference:
        return "smaller-than-m5-default"
    return "larger-than-m5-default"


def compare_to_upstream_reference(
    *,
    candidate: ProbeResult,
    coarse_by_preset: dict[str, ProbeResult],
    zones: dict[str, dict[str, str | float]],
) -> dict:
    upstream = coarse_by_preset.get("upstream")
    payload = {
        "upstream_zone": zones.get("upstream", {}).get("zone", "missing"),
        "upstream_status": upstream.status if upstream is not None else "missing",
    }
    if upstream is None or upstream.status != "ok":
        return payload
    if candidate.steady_state_tok_per_sec and upstream.steady_state_tok_per_sec:
        payload["candidate_to_upstream_throughput_ratio"] = (
            candidate.steady_state_tok_per_sec / upstream.steady_state_tok_per_sec
        )
    if candidate.peak_vram_mb is not None and upstream.peak_vram_mb is not None:
        payload["candidate_to_upstream_memory_ratio"] = candidate.peak_vram_mb / upstream.peak_vram_mb
    return payload


RUNG_SPEC_NAMES = {
    "cheap": "CHEAP_EVAL_RUNG",
    "reference": "REFERENCE_EVAL_RUNG",
    "full": "FULL_EVAL_RUNG",
}


def build_eval_calibration_row(
    *,
    preset: str,
    hardware: HardwareFingerprint,
    eval_payload: dict,
    measured_train_seconds: float,
    confidence: str | None,
    mode: str,
    calibration_signatures: dict[str, str | None],
) -> dict:
    row_payload = {
        "key": f"{preset}_{hardware.hardware_key}",
        "label": f"{preset} eval ladder on {hardware.accelerator_model or hardware.hardware_key}",
        "hardware_key": hardware.hardware_key,
        "preset": preset,
        "seq_len": eval_payload["rows"][0]["seq_len"],
        "batch_size": eval_payload["rows"][0]["batch_size"],
        "source": (
            f"Generated by calibrate.py on {hardware.hardware_key} "
            f"using mode={mode} from a {measured_train_seconds:g}s checkpoint."
        ),
        "policy_version": EVAL_POLICY_VERSION,
        "confidence": confidence or "seed-single-checkpoint",
        "measured_train_seconds": measured_train_seconds,
        "repeat_count": 1,
        "measured_on": date.today().isoformat(),
        "eval_semantics_signature": calibration_signatures["eval_semantics_signature"],
        "runtime_shape_signature": calibration_signatures["runtime_shape_signature"],
        "notes": "Promotion candidate emitted by the one-button platform bring-up tool.",
        "rungs": {},
    }
    for rung in eval_payload["rows"]:
        row_payload["rungs"][rung["rung"]] = {
            "spec": RUNG_SPEC_NAMES[rung["rung"]],
            "eval_seconds": rung["eval_seconds"],
            "val_bpb": rung["val_bpb"],
            "abs_error_vs_full": rung.get("abs_error_vs_full"),
        }
    return row_payload


def build_eval_calibration_snippet(row: dict) -> str:
    lines = [
        "EvalCalibration(",
        f"    key={row['key']!r},",
        f"    label={row['label']!r},",
        f"    hardware_key={row['hardware_key']!r},",
        f"    preset={row['preset']!r},",
        f"    seq_len={row['seq_len']},",
        f"    batch_size={row['batch_size']},",
        f"    source={row['source']!r},",
        f"    policy_version={row['policy_version']},",
        f"    confidence={row['confidence']!r},",
        f"    measured_train_seconds={row['measured_train_seconds']},",
        f"    repeat_count={row['repeat_count']},",
        f"    measured_on={row['measured_on']!r},",
        f"    eval_semantics_signature={row['eval_semantics_signature']!r},",
        f"    runtime_shape_signature={row['runtime_shape_signature']!r},",
        f"    notes={row['notes']!r},",
    ]
    for rung_key in ("cheap", "reference", "full"):
        rung = row["rungs"].get(rung_key)
        if rung is None:
            continue
        lines.extend(
            [
                f"    {rung_key}=_measurement(",
                f"        {rung['spec']},",
                f"        eval_seconds={rung['eval_seconds']},",
                f"        val_bpb={rung['val_bpb']},",
                f"        abs_error_vs_full={rung['abs_error_vs_full']},",
                "    ),",
            ]
        )
    lines.append(")")
    return "\n".join(lines) + "\n"


def build_platform_default_snippet(hardware_key: str, candidate_default: dict) -> str:
    payload = {
        "engine": candidate_default["engine"],
        "backend_family": candidate_default["backend_family"],
        "preset": candidate_default["preset"],
        "seq_len": candidate_default["seq_len"],
        "depth": candidate_default["depth"],
        "window_pattern": candidate_default["window_pattern"],
        "device_batch_size": candidate_default["device_batch_size"],
        "total_batch_size": candidate_default["total_batch_size"],
        "grad_accum_steps": candidate_default["grad_accum_steps"],
        "runtime_shape_signature": candidate_default["runtime_shape_signature"],
    }
    return f"{hardware_key!r}: {json.dumps(payload, indent=2)},\n"


def write_promotion_bundle(
    *,
    output_dir: Path,
    hardware: HardwareFingerprint,
    candidate_default: dict,
    eval_payload: dict,
    measured_train_seconds: float,
    confidence: str | None,
    mode: str,
    calibration_signatures: dict[str, str | None],
) -> dict:
    promotion_dir = output_dir / "promotion"
    promotion_dir.mkdir(parents=True, exist_ok=True)
    eval_row = build_eval_calibration_row(
        preset=candidate_default["preset"],
        hardware=hardware,
        eval_payload=eval_payload,
        measured_train_seconds=measured_train_seconds,
        confidence=confidence,
        mode=mode,
        calibration_signatures=calibration_signatures,
    )
    missing_rungs = [rung for rung in ("cheap", "reference", "full") if rung not in eval_row["rungs"]]
    eval_calibration_promotable = len(missing_rungs) == 0 and all(
        rung["abs_error_vs_full"] is not None
        for rung in eval_row["rungs"].values()
    )
    eval_row["promotable"] = eval_calibration_promotable
    eval_row["missing_rungs"] = missing_rungs

    platform_default_json_path = promotion_dir / "platform_default.json"
    platform_default_pyfrag_path = promotion_dir / "platform_default.pyfrag"
    eval_row_json_path = promotion_dir / "eval_calibration.json"
    eval_row_pyfrag_path = promotion_dir / "eval_calibration.pyfrag"
    summary_path = promotion_dir / "README.md"

    write_json(platform_default_json_path, candidate_default)
    platform_default_pyfrag_path.write_text(
        build_platform_default_snippet(hardware.hardware_key, candidate_default)
    )
    cached_platform_default_path = write_platform_default_cache(
        engine_name=hardware.engine,
        hardware_key=hardware.hardware_key,
        candidate_default=candidate_default,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source_output_dir=str(output_dir),
        source_report=str(output_dir / "report.md"),
    )
    write_json(eval_row_json_path, eval_row)
    if eval_calibration_promotable:
        eval_row_pyfrag_path.write_text(build_eval_calibration_snippet(eval_row))
    else:
        eval_row_pyfrag_path.write_text(
            "# Eval calibration is not yet promotion-ready.\n"
            f"# Missing full audit coverage for: {', '.join(missing_rungs) or 'none'}.\n"
            "# Run calibrate_platform in full mode, or rerun eval-rungs with cheap,reference,full, "
            "before promoting this row into autoresearch_mlx/eval_policy.py.\n"
        )
    summary_path.write_text(
        "\n".join(
            [
                "# Promotion Bundle",
                "",
                "This bundle contains the bring-up tool's promotion-ready artifacts.",
                "",
            f"- Eval semantics signature: `{calibration_signatures['eval_semantics_signature']}`",
            f"- Runtime shape signature: `{calibration_signatures['runtime_shape_signature']}`",
                f"- Platform default JSON: `{platform_default_json_path.name}`",
                f"- Platform default Python fragment: `{platform_default_pyfrag_path.name}`",
                f"- Cached platform default: `{cached_platform_default_path}`",
                f"- Eval calibration JSON: `{eval_row_json_path.name}`",
                f"- Eval calibration Python fragment: `{eval_row_pyfrag_path.name}`",
                f"- Eval calibration promotable now: `{str(eval_calibration_promotable).lower()}`",
            ]
        )
        + "\n"
    )

    return {
        "dir": str(promotion_dir),
        "platform_default_json": str(platform_default_json_path),
        "platform_default_pyfrag": str(platform_default_pyfrag_path),
        "platform_default_cache": str(cached_platform_default_path),
        "platform_default_promotable": True,
        "eval_calibration_json": str(eval_row_json_path),
        "eval_calibration_pyfrag": str(eval_row_pyfrag_path),
        "eval_calibration_promotable": eval_calibration_promotable,
        "eval_calibration_missing_rungs": missing_rungs,
        "readme": str(summary_path),
        "eval_calibration_key": eval_row["key"],
        "confidence": eval_row["confidence"],
        "eval_semantics_signature": calibration_signatures["eval_semantics_signature"],
        "runtime_shape_signature": calibration_signatures["runtime_shape_signature"],
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_cell(row.get(key)) for key, _ in columns) + " |")
    return "\n".join([header, separator, *body])


def format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def horizon_rows_for_report(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        item = dict(row)
        low = item.get("interval_low")
        high = item.get("interval_high")
        item["confidence_interval"] = (
            f"[{low:.6f}, {high:.6f}]"
            if isinstance(low, (int, float)) and isinstance(high, (int, float))
            else ""
        )
        fit_r2 = item.get("fit_r2")
        item["fit_r2_display"] = f"{fit_r2:.3f}" if isinstance(fit_r2, (int, float)) else "n/a"
        fit_sigma = item.get("fit_sigma")
        item["fit_sigma_display"] = f"{fit_sigma:.6f}" if isinstance(fit_sigma, (int, float)) else "n/a"
        correction = item.get("correction_mean")
        item["correction_display"] = f"{correction:.6f}" if isinstance(correction, (int, float)) else "n/a"
        ratio = item.get("extrapolation_ratio")
        item["extrapolation_ratio_display"] = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else "n/a"
        fit_quality_min = item.get("fit_quality_min")
        item["fit_quality_min_display"] = f"{fit_quality_min:.3f}" if isinstance(fit_quality_min, (int, float)) else "n/a"
        effective_damping = item.get("effective_damping")
        item["effective_damping_display"] = (
            f"{effective_damping:.3f}" if isinstance(effective_damping, (int, float)) else "n/a"
        )
        horizon_correction = item.get("horizon_correction")
        item["horizon_correction_display"] = (
            f"{horizon_correction:.6f}" if isinstance(horizon_correction, (int, float)) else "n/a"
        )
        projection_delta = item.get("projection_delta")
        item["projection_delta_display"] = (
            f"{projection_delta:.6f}" if isinstance(projection_delta, (int, float)) else "n/a"
        )
        projection_sigma = item.get("projection_sigma")
        item["projection_sigma_display"] = (
            f"{projection_sigma:.6f}" if isinstance(projection_sigma, (int, float)) else "n/a"
        )
        projection_snr = item.get("projection_snr")
        item["projection_snr_display"] = (
            f"{projection_snr:.3f}" if isinstance(projection_snr, (int, float)) else "n/a"
        )
        formatted.append(item)
    return formatted


def write_report(path: Path, *, payload: dict) -> None:
    fingerprint = payload["hardware_fingerprint"]
    coarse_rows = payload["coarse_envelope"]["rows"]
    ranking_rows = payload["candidate_ranking"]["rows"]
    projection_payload = payload.get("projection_ranking")
    projection_rows = projection_payload["rows"] if isinstance(projection_payload, dict) else None
    projection_horizon_rows = projection_payload["horizon_rows"] if isinstance(projection_payload, dict) else None
    projection_decision_for_report = (
        projection_payload.get("control_decision")
        if isinstance(projection_payload, dict) and isinstance(projection_payload.get("control_decision"), dict)
        else projection_payload.get("decision")
        if isinstance(projection_payload, dict)
        else None
    )
    projection_next_horizon = (
        projection_payload.get("next_horizon")
        if isinstance(projection_payload, dict) and isinstance(projection_payload.get("next_horizon"), dict)
        else None
    )
    finalist_payload = payload.get("finalist_projection")
    finalist_rows = finalist_payload["rows"] if isinstance(finalist_payload, dict) else None
    finalist_horizon_rows = finalist_payload["horizon_rows"] if isinstance(finalist_payload, dict) else None
    finalist_decision_for_report = (
        finalist_payload.get("control_decision")
        if isinstance(finalist_payload, dict) and isinstance(finalist_payload.get("control_decision"), dict)
        else finalist_payload.get("decision")
        if isinstance(finalist_payload, dict)
        else None
    )
    finalist_next_horizon = (
        finalist_payload.get("next_horizon")
        if isinstance(finalist_payload, dict) and isinstance(finalist_payload.get("next_horizon"), dict)
        else None
    )
    confirmation_payload = payload.get("confirmation_projection")
    confirmation_rows = confirmation_payload["rows"] if isinstance(confirmation_payload, dict) else None
    confirmation_horizon_rows = confirmation_payload["horizon_rows"] if isinstance(confirmation_payload, dict) else None
    confirmation_decision_for_report = (
        confirmation_payload.get("control_decision")
        if isinstance(confirmation_payload, dict) and isinstance(confirmation_payload.get("control_decision"), dict)
        else confirmation_payload.get("decision")
        if isinstance(confirmation_payload, dict)
        else None
    )
    confirmation_next_horizon = (
        confirmation_payload.get("next_horizon")
        if isinstance(confirmation_payload, dict) and isinstance(confirmation_payload.get("next_horizon"), dict)
        else None
    )
    scaling_confirmation_payload = payload.get("scaling_confirmation")
    scaling_confirmation_rows = scaling_confirmation_payload["rows"] if isinstance(scaling_confirmation_payload, dict) else None
    scaling_confirmation_horizon_rows = (
        scaling_confirmation_payload["horizon_rows"] if isinstance(scaling_confirmation_payload, dict) else None
    )
    scaling_confirmation_decision_for_report = (
        scaling_confirmation_payload.get("control_decision")
        if isinstance(scaling_confirmation_payload, dict) and isinstance(scaling_confirmation_payload.get("control_decision"), dict)
        else scaling_confirmation_payload.get("decision")
        if isinstance(scaling_confirmation_payload, dict)
        else None
    )
    scaling_confirmation_next_horizon = (
        scaling_confirmation_payload.get("next_horizon")
        if isinstance(scaling_confirmation_payload, dict) and isinstance(scaling_confirmation_payload.get("next_horizon"), dict)
        else None
    )
    winner_batch_audit_payload = payload.get("winner_batch_audit")
    winner_batch_audit_rows = (
        winner_batch_audit_payload.get("audit_rows")
        if isinstance(winner_batch_audit_payload, dict)
        else None
    )
    local_rows = payload["local_search"]["rows"]
    candidate_default = payload["candidate_default"]
    longer_horizon_projection = payload.get("longer_horizon_projection")
    scaling_candidate = payload.get("scaling_candidate")
    scaling_confirmation_enabled = payload.get("scaling_confirmation_enabled", False)
    promotion_bundle = payload["promotion_bundle"]
    eval_rows = payload["eval_calibration"]["rows"]
    zones = payload["zones"]

    coarse_table_rows = []
    for row in coarse_rows:
        row = dict(row)
        zone_info = zones.get(row["preset"], {})
        row["zone"] = zone_info.get("zone", "")
        row["zone_reason"] = zone_info.get("reason", "")
        coarse_table_rows.append(row)

    report = [
        "# Platform Calibration Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Engine: `{fingerprint['engine']}`",
        f"- Backend family: `{fingerprint['backend_family']}`",
        f"- Hardware key: `{fingerprint['hardware_key']}`",
        f"- Eval semantics signature: `{payload['calibration_signatures']['eval_semantics_signature']}`",
        f"- Runtime shape signature: `{payload['calibration_signatures']['runtime_shape_signature']}`",
        "",
        "## Summary",
        "",
        f"- Bring-up mode: `{payload['mode']}`",
        f"- Candidate new default: `{candidate_default['preset']}`",
        f"- Candidate operating point: `seq_len={candidate_default['seq_len']}`, `window_pattern={candidate_default['window_pattern']}`, `device_batch_size={candidate_default['device_batch_size']}`, `total_batch_size={candidate_default['total_batch_size']}`",
        f"- Family relation to M5 default: `{candidate_default['family_relation_to_m5']}`",
        f"- Selection confidence: `{candidate_default['selection_confidence']['eval_calibration_effective_confidence']}`",
        f"- Upstream-style reference zone: `{payload['reference_comparison']['upstream_style']['upstream_zone']}`",
        (
            f"- Winner batch audit: `device_batch_size={winner_batch_audit_payload['winner']['device_batch_size']}`, "
            f"`total_batch_size={winner_batch_audit_payload['winner']['total_batch_size']}`, "
            f"`projected_300s_val_bpb={winner_batch_audit_payload['winner_row']['projected_val_bpb']:.6f}`"
            if isinstance(winner_batch_audit_payload, dict)
            and isinstance(winner_batch_audit_payload.get("winner"), dict)
            and isinstance(winner_batch_audit_payload.get("winner_row"), dict)
            and isinstance(winner_batch_audit_payload["winner_row"].get("projected_val_bpb"), (int, float))
            else "- Winner batch audit: `not-run`"
        ),
        (
            f"- Longer-horizon projected leader at `{int(longer_horizon_projection['scaling_target_seconds'])}s`: "
            f"`{longer_horizon_projection['scaling_winner_preset']}` "
            f"(`projected val_bpb={longer_horizon_projection['scaling_winner_val_bpb']:.6f}`, "
            f"`confidence={longer_horizon_projection['scaling_confidence_label']}`, "
            f"`source={longer_horizon_projection['scaling_projection_source']}`)"
            if isinstance(longer_horizon_projection, dict)
            else "- Longer-horizon projected leader: `none`"
        ),
        (
            f"- Secondary scaling candidate: `{scaling_candidate['candidate_preset']}` "
            f"(gap at `300s`: `{scaling_candidate['candidate_gap_at_target']:.6f}`, "
            f"projected gap at `{int(scaling_candidate['scaling_target_seconds'])}s`: "
            f"`{scaling_candidate['projected_gap_at_scaling_target']:.6f}`)"
            if isinstance(scaling_candidate, dict)
            else "- Secondary scaling candidate: `none`"
        ),
        (
            "- Scaling confirmation: `enabled`"
            if scaling_confirmation_enabled
            else "- Scaling confirmation: `disabled (opt-in)`"
        ),
        "",
        "## Hardware Fingerprint",
        "",
        markdown_table(
            [fingerprint],
            [
                ("engine", "Engine"),
                ("backend_family", "Backend"),
                ("hardware_key", "Hardware key"),
                ("accelerator_vendor", "Vendor"),
                ("accelerator_model", "Accelerator"),
                ("accelerator_architecture", "Architecture"),
                ("accelerator_compute_capability", "Compute capability"),
                ("accelerator_cores", "Accelerator cores"),
                ("memory_gb", "Memory (GB)"),
                ("os_version", "OS"),
                ("runtime_version", "Runtime"),
                ("driver_version", "Driver"),
                ("attention_backend", "Attention backend"),
                ("flash_attention_generation", "Preferred FA gen"),
                ("python_version", "Python"),
            ],
        ),
        "",
        "## Coarse Envelope",
        "",
        markdown_table(
            coarse_table_rows,
            [
                ("preset", "Preset"),
                ("zone", "Zone"),
                ("status", "Status"),
                ("steady_state_tok_per_sec", "Steady tok/s"),
                ("peak_vram_mb", "Peak MB"),
                ("wall_seconds", "Wall sec"),
                ("zone_reason", "Why"),
            ],
        ),
        "",
        "## Candidate Ranking",
        "",
        "These are the raw observed metrics from the same all-family run that feeds the projection stage below. They are useful context, but the actual family decision is made from the projected `300s` objective rather than these shorter-horizon endpoints alone.",
        "",
        markdown_table(
            ranking_rows,
            [
                ("preset", "Preset"),
                ("status", "Status"),
                ("val_bpb", "val_bpb"),
                ("proxy_val_bpb", "proxy_val_bpb"),
                ("steady_state_tok_per_sec", "Steady tok/s"),
                ("peak_vram_mb", "Peak MB"),
                ("memory_fraction", "Mem frac"),
                ("memory_pressure_band", "Mem band"),
                ("canonical_rung", "Canonical rung"),
                ("eval_calibration_status", "Eval status"),
                ("on_pareto_front", "Pareto"),
                ("frontier_distance", "Frontier dist"),
                ("selection_distance", "Selection dist"),
                ("utility_score", "Selection"),
                ("estimated_eval_overhead_fraction", "Eval overhead"),
                ("stable_rung_count", "Stable rungs"),
            ],
        ),
        "",
    ]
    if isinstance(projection_decision_for_report, dict):
        report.extend(
            [
                "### Projection Decision",
                "",
                f"- Winner probability threshold met: `{projection_decision_for_report.get('enough_signal')}`",
                f"- Stability reason: `{projection_decision_for_report.get('stability_reason') or projection_decision_for_report.get('confidence_reason')}`",
                f"- Winner probability: `{projection_decision_for_report.get('top_winner_probability')}`",
                f"- Margin to second: `{projection_decision_for_report.get('top_margin_to_second')}`",
                f"- Margin SNR: `{projection_decision_for_report.get('top_margin_snr')}`",
                "",
            ]
        )
    if isinstance(finalist_decision_for_report, dict):
        report.extend(
            [
                "### Finalist Decision",
                "",
                f"- Winner probability threshold met: `{finalist_decision_for_report.get('enough_signal')}`",
                f"- Stability reason: `{finalist_decision_for_report.get('stability_reason') or finalist_decision_for_report.get('confidence_reason')}`",
                f"- Winner probability: `{finalist_decision_for_report.get('top_winner_probability')}`",
                f"- Margin to second: `{finalist_decision_for_report.get('top_margin_to_second')}`",
                f"- Margin SNR: `{finalist_decision_for_report.get('top_margin_snr')}`",
                "",
            ]
        )
    if isinstance(confirmation_decision_for_report, dict):
        report.extend(
            [
                "### Confirmation Decision",
                "",
                f"- Winner probability threshold met: `{confirmation_decision_for_report.get('enough_signal')}`",
                f"- Stability reason: `{confirmation_decision_for_report.get('stability_reason') or confirmation_decision_for_report.get('confidence_reason')}`",
                f"- Winner probability: `{confirmation_decision_for_report.get('top_winner_probability')}`",
                f"- Margin to second: `{confirmation_decision_for_report.get('top_margin_to_second')}`",
                f"- Margin SNR: `{confirmation_decision_for_report.get('top_margin_snr')}`",
                "",
            ]
        )
    if isinstance(scaling_confirmation_decision_for_report, dict):
        report.extend(
            [
                "### Longer-Horizon Confirmation Decision",
                "",
                f"- Winner probability threshold met: `{scaling_confirmation_decision_for_report.get('enough_signal')}`",
                f"- Stability reason: `{scaling_confirmation_decision_for_report.get('stability_reason') or scaling_confirmation_decision_for_report.get('confidence_reason')}`",
                f"- Winner probability: `{scaling_confirmation_decision_for_report.get('top_winner_probability')}`",
                f"- Margin to second: `{scaling_confirmation_decision_for_report.get('top_margin_to_second')}`",
                f"- Margin SNR: `{scaling_confirmation_decision_for_report.get('top_margin_snr')}`",
                "",
            ]
        )
    if projection_rows:
        report.extend(
            [
                "## Projection Ranking",
                "",
                "All candidate families are run at a longer budget and projected against the `300s` objective using the shared truth-curve model. This establishes the leading family candidates before any deeper rerank or winner-specific batch audit happens.",
                "",
                markdown_table(
                    projection_rows,
                    [
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("curve_points", "Curve points"),
                        ("projected_val_bpb", "Projected 300s val_bpb"),
                        ("correction_mean", "Correction"),
                        ("projection_std", "Projection std"),
                        ("fit_r2", "Fit R²"),
                        ("fit_sigma", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("final_val_bpb", "Observed val_bpb"),
                        ("calibration_horizon_seconds", "Truth horizon"),
                        ("calibration_horizon_tokens", "Truth tokens"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio", "Ratio"),
                        ("projection_source", "Projection source"),
                        ("matched_truth_count", "Truth matches"),
                    ],
                ),
                "",
            ]
        )
    if winner_batch_audit_rows:
        report.extend(
            [
                "## Winner Batch Audit",
                "",
                "After family selection, calibration reruns a short curve-aware batch audit on the selected preset. This is where the operating batch is allowed to move away from the anchor preset's device constant when a different `db/tb` pair projects to a better `300s` outcome once optimizer-step cadence and accumulation cost are visible.",
                "",
                markdown_table(
                    winner_batch_audit_rows,
                    [
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("grad_accum_steps", "Grad accum"),
                        ("projected_val_bpb", "Projected 300s val_bpb"),
                        ("projected_optimizer_steps", "Projected opt steps"),
                        ("projected_microsteps", "Projected microsteps"),
                        ("observed_val_bpb", "Observed val_bpb"),
                        ("accum_percent", "Accum %"),
                        ("control_overhead_percent", "Control %"),
                        ("steady_state_tok_per_sec", "Steady tok/s"),
                        ("peak_vram_mb", "Peak MB"),
                        ("projection_method", "Projection"),
                    ],
                ),
                "",
            ]
        )
    if projection_horizon_rows:
        report.extend(
            [
                "### Projection Horizon Table",
                "",
                "These rows use the same shared projection logic as the standalone `curve_report.py` tool. They show how the current candidate family curves project across multiple horizons, including confidence intervals and whether each horizon is backed by truth curves or only calibrated projection.",
                "",
                markdown_table(
                    horizon_rows_for_report(projection_horizon_rows),
                    [
                        ("target_seconds", "Horizon"),
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("observed_tokens", "Observed tokens"),
                        ("target_tokens", "Target tokens"),
                        ("corrected_val_bpb", "Projected val_bpb"),
                        ("correction_display", "Correction"),
                        ("confidence_interval", "95% CI"),
                        ("fit_r2_display", "Fit R²"),
                        ("fit_sigma_display", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio_display", "Ratio"),
                        ("projection_source", "Source"),
                        ("matched_truth_count", "Truth"),
                        ("confidence_label", "Confidence"),
                    ],
                ),
                "",
            ]
        )
    if isinstance(projection_next_horizon, dict):
        report.extend(
            [
                "### Projection Next Horizon",
                "",
                (
                    f"- Suggested next horizon: `{projection_next_horizon.get('suggested_seconds')}s`"
                    if projection_next_horizon.get("suggested_seconds") is not None
                    else "- Suggested next horizon: `none`"
                ),
                f"- Current observed horizon: `{projection_next_horizon.get('current_observed_seconds')}s`",
                f"- Current observed tokens: `{projection_next_horizon.get('current_observed_tokens')}`",
                f"- Top preset: `{projection_next_horizon.get('top_preset')}`",
                f"- Winner probability: `{projection_next_horizon.get('top_winner_probability')}`",
                f"- Enough signal at target: `{projection_next_horizon.get('target_enough_signal')}`",
                f"- Reason: `{projection_next_horizon.get('suggestion_reason')}`",
                "",
            ]
        )
    if finalist_rows:
        report.extend(
            [
                "## Finalist Ranking",
                "",
                "The top projected candidate families are rerun again at a longer budget and projected to `300s`. This stage resolves whether the first projection leader still wins once more real training time has been observed.",
                "",
                markdown_table(
                    finalist_rows,
                    [
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("curve_points", "Curve points"),
                        ("projected_val_bpb", "Projected 300s val_bpb"),
                        ("correction_mean", "Correction"),
                        ("projection_std", "Projection std"),
                        ("fit_r2", "Fit R²"),
                        ("fit_sigma", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("final_val_bpb", "Observed val_bpb"),
                        ("calibration_horizon_seconds", "Truth horizon"),
                        ("calibration_horizon_tokens", "Truth tokens"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio", "Ratio"),
                        ("projection_source", "Projection source"),
                        ("matched_truth_count", "Truth matches"),
                    ],
                ),
                "",
            ]
        )
    if finalist_horizon_rows:
        report.extend(
            [
                "### Finalist Horizon Table",
                "",
                "These rows use the same shared projection logic as the standalone `curve_report.py` tool, but against the longer finalist curves. This is the deepest calibration view used before the winner is promoted into local search and checkpoint-backed eval calibration.",
                "",
                markdown_table(
                    horizon_rows_for_report(finalist_horizon_rows),
                    [
                        ("target_seconds", "Horizon"),
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("observed_tokens", "Observed tokens"),
                        ("target_tokens", "Target tokens"),
                        ("corrected_val_bpb", "Projected val_bpb"),
                        ("correction_display", "Correction"),
                        ("confidence_interval", "95% CI"),
                        ("fit_r2_display", "Fit R²"),
                        ("fit_sigma_display", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio_display", "Ratio"),
                        ("projection_source", "Source"),
                        ("matched_truth_count", "Truth"),
                        ("confidence_label", "Confidence"),
                    ],
                ),
                "",
            ]
        )
    if isinstance(finalist_next_horizon, dict):
        report.extend(
            [
                "### Finalist Next Horizon",
                "",
                (
                    f"- Suggested next horizon: `{finalist_next_horizon.get('suggested_seconds')}s`"
                    if finalist_next_horizon.get("suggested_seconds") is not None
                    else "- Suggested next horizon: `none`"
                ),
                f"- Current observed horizon: `{finalist_next_horizon.get('current_observed_seconds')}s`",
                f"- Current observed tokens: `{finalist_next_horizon.get('current_observed_tokens')}`",
                f"- Top preset: `{finalist_next_horizon.get('top_preset')}`",
                f"- Winner probability: `{finalist_next_horizon.get('top_winner_probability')}`",
                f"- Enough signal at target: `{finalist_next_horizon.get('target_enough_signal')}`",
                f"- Reason: `{finalist_next_horizon.get('suggestion_reason')}`",
                "",
            ]
        )
    if confirmation_rows:
        report.extend(
            [
                "## Confirmation Ranking",
                "",
                "If the finalist stage still lacks enough signal at the `300s` target, calibration reruns the strongest finalists at the suggested next horizon and uses that result as the deciding family stage before local search.",
                "",
                markdown_table(
                    confirmation_rows,
                    [
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("curve_points", "Curve points"),
                        ("projected_val_bpb", "Projected 300s val_bpb"),
                        ("correction_mean", "Correction"),
                        ("projection_std", "Projection std"),
                        ("fit_r2", "Fit R²"),
                        ("fit_sigma", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("final_val_bpb", "Observed val_bpb"),
                        ("calibration_horizon_seconds", "Truth horizon"),
                        ("calibration_horizon_tokens", "Truth tokens"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio", "Ratio"),
                        ("projection_source", "Projection source"),
                        ("matched_truth_count", "Truth matches"),
                    ],
                ),
                "",
            ]
        )
    if confirmation_horizon_rows:
        report.extend(
            [
                "### Confirmation Horizon Table",
                "",
                markdown_table(
                    horizon_rows_for_report(confirmation_horizon_rows),
                    [
                        ("target_seconds", "Horizon"),
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("observed_tokens", "Observed tokens"),
                        ("target_tokens", "Target tokens"),
                        ("corrected_val_bpb", "Projected val_bpb"),
                        ("correction_display", "Correction"),
                        ("confidence_interval", "95% CI"),
                        ("fit_r2_display", "Fit R²"),
                        ("fit_sigma_display", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio_display", "Ratio"),
                        ("projection_source", "Source"),
                        ("matched_truth_count", "Truth"),
                        ("confidence_label", "Confidence"),
                    ],
                ),
                "",
            ]
        )
    if isinstance(confirmation_next_horizon, dict):
        report.extend(
            [
                "### Confirmation Next Horizon",
                "",
                (
                    f"- Suggested next horizon: `{confirmation_next_horizon.get('suggested_seconds')}s`"
                    if confirmation_next_horizon.get("suggested_seconds") is not None
                    else "- Suggested next horizon: `none`"
                ),
                f"- Current observed horizon: `{confirmation_next_horizon.get('current_observed_seconds')}s`",
                f"- Current observed tokens: `{confirmation_next_horizon.get('current_observed_tokens')}`",
                f"- Top preset: `{confirmation_next_horizon.get('top_preset')}`",
                f"- Winner probability: `{confirmation_next_horizon.get('top_winner_probability')}`",
                f"- Enough signal at target: `{confirmation_next_horizon.get('target_enough_signal')}`",
                f"- Reason: `{confirmation_next_horizon.get('suggestion_reason')}`",
                "",
            ]
        )
    if scaling_confirmation_rows:
        report.extend(
            [
                "## Longer-Horizon Confirmation",
                "",
                "When the longer-horizon summary projects a crossover beyond `300s` but the evidence is still low-confidence, calibration reruns the strict `300s` winner against the projected longer-horizon leader at the deeper target horizon.",
                "",
                markdown_table(
                    scaling_confirmation_rows,
                    [
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("curve_points", "Curve points"),
                        ("projected_val_bpb", "Projected 900s val_bpb"),
                        ("correction_mean", "Correction"),
                        ("projection_std", "Projection std"),
                        ("fit_r2", "Fit R²"),
                        ("fit_sigma", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("final_val_bpb", "Observed val_bpb"),
                        ("calibration_horizon_seconds", "Truth horizon"),
                        ("calibration_horizon_tokens", "Truth tokens"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio", "Ratio"),
                        ("projection_source", "Projection source"),
                        ("matched_truth_count", "Truth matches"),
                    ],
                ),
                "",
            ]
        )
    if scaling_confirmation_horizon_rows:
        report.extend(
            [
                "### Longer-Horizon Confirmation Table",
                "",
                markdown_table(
                    horizon_rows_for_report(scaling_confirmation_horizon_rows),
                    [
                        ("target_seconds", "Horizon"),
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("observed_tokens", "Observed tokens"),
                        ("target_tokens", "Target tokens"),
                        ("corrected_val_bpb", "Projected val_bpb"),
                        ("correction_display", "Correction"),
                        ("confidence_interval", "95% CI"),
                        ("fit_r2_display", "Fit R²"),
                        ("fit_sigma_display", "Fit σ"),
                        ("winner_probability", "Winner p"),
                        ("confidence_reason", "Reason"),
                        ("truth_anchor_seconds", "Anchor sec"),
                        ("truth_anchor_tokens", "Anchor tokens"),
                        ("extrapolation_ratio_display", "Ratio"),
                        ("projection_source", "Source"),
                        ("matched_truth_count", "Truth"),
                        ("confidence_label", "Confidence"),
                    ],
                ),
                "",
            ]
        )
    if isinstance(scaling_confirmation_next_horizon, dict):
        report.extend(
            [
                "### Longer-Horizon Next Horizon",
                "",
                (
                    f"- Suggested next horizon: `{scaling_confirmation_next_horizon.get('suggested_seconds')}s`"
                    if scaling_confirmation_next_horizon.get("suggested_seconds") is not None
                    else "- Suggested next horizon: `none`"
                ),
                f"- Current observed horizon: `{scaling_confirmation_next_horizon.get('current_observed_seconds')}s`",
                f"- Current observed tokens: `{scaling_confirmation_next_horizon.get('current_observed_tokens')}`",
                f"- Top preset: `{scaling_confirmation_next_horizon.get('top_preset')}`",
                f"- Winner probability: `{scaling_confirmation_next_horizon.get('top_winner_probability')}`",
                f"- Enough signal at target: `{scaling_confirmation_next_horizon.get('target_enough_signal')}`",
                f"- Reason: `{scaling_confirmation_next_horizon.get('suggestion_reason')}`",
                "",
            ]
        )
    if isinstance(finalist_payload, dict) and finalist_payload.get("diagnostics"):
        report.extend(
            [
                "### Finalist Stability Diagnostics",
                "",
                "These rows compare the earlier and later finalist curves for the same preset family. They are the calibration-side equivalent of the LR finder drift checks: if short and long horizons disagree too much, the finalist decision should not be trusted yet.",
                "",
                markdown_table(
                    finalist_payload["diagnostics"],
                    [
                        ("preset", "Preset"),
                        ("short_observed_seconds", "Short sec"),
                        ("long_observed_seconds", "Long sec"),
                        ("short_observed_tokens", "Short tokens"),
                        ("long_observed_tokens", "Long tokens"),
                        ("short_projected_val_bpb", "Short proj"),
                        ("long_projected_val_bpb", "Long proj"),
                        ("short_fit_r2", "Short R²"),
                        ("long_fit_r2", "Long R²"),
                        ("fit_quality_min", "Min R²"),
                        ("short_fit_sigma", "Short σ"),
                        ("long_fit_sigma", "Long σ"),
                        ("horizon_alpha", "Alpha"),
                        ("effective_damping", "Damp"),
                        ("horizon_correction", "Corr"),
                        ("projection_delta", "Proj Δ"),
                        ("projection_sigma", "Proj σ"),
                        ("projection_snr", "Proj SNR"),
                        ("stability_gap", "Gap"),
                        ("stability_snr", "SNR"),
                        ("stable_projection", "Stable"),
                        ("stability_reason", "Reason"),
                    ],
                ),
                "",
            ]
        )
    if isinstance(scaling_candidate, dict):
        report.extend(
            [
                "## Scaling Candidate",
                "",
                "The strict winner above remains the only thing that decides the default. This secondary pass looks for a larger near-frontier family whose completed truth curves stay close enough at `300s` that it may be the better long-horizon scaling bet.",
                "",
                markdown_table(
                    [scaling_candidate],
                    [
                        ("strict_winner_preset", "Strict winner"),
                        ("candidate_preset", "Scaling candidate"),
                        ("candidate_device_batch_size", "Candidate device batch"),
                        ("candidate_total_batch_size", "Candidate total batch"),
                        ("strict_winner_val_bpb", "Winner 300s val_bpb"),
                        ("candidate_val_bpb", "Candidate 300s val_bpb"),
                        ("candidate_gap_at_target", "Gap at 300s"),
                        ("strict_projected_val_bpb", "Winner projected longer"),
                        ("candidate_projected_val_bpb", "Candidate projected longer"),
                        ("projected_gap_at_scaling_target", "Projected gap"),
                        ("strict_last_segment_gain", "Winner late gain"),
                        ("candidate_last_segment_gain", "Candidate late gain"),
                        ("source", "Source"),
                        ("rationale", "Rationale"),
                    ],
                ),
                "",
            ]
        )
    if isinstance(longer_horizon_projection, dict):
        report.extend(
            [
                "## Longer-Horizon Projection",
                "",
                "This secondary summary asks a different question from the strict default choice above: if you project beyond the `300s` calibration target, which family appears to lead at the deeper horizon and how trustworthy is that projection?",
                "",
                markdown_table(
                    [longer_horizon_projection],
                    [
                        ("target_winner_preset", "300s winner"),
                        ("target_winner_val_bpb", "300s val_bpb"),
                        ("scaling_winner_preset", "Longer-horizon leader"),
                        ("scaling_winner_val_bpb", "Projected longer val_bpb"),
                        ("scaling_winner_device_batch_size", "Leader device batch"),
                        ("scaling_winner_total_batch_size", "Leader total batch"),
                        ("scaling_gap_at_target", "Leader gap at 300s"),
                        ("target_gap_at_scaling", "300s winner gap at longer horizon"),
                        ("target_winner_rank_at_scaling", "300s winner rank later"),
                        ("scaling_winner_rank_at_target", "Longer leader rank at 300s"),
                        ("scaling_winner_probability", "Winner p"),
                        ("scaling_margin_to_second", "Margin to second"),
                        ("scaling_margin_snr", "Margin SNR"),
                        ("scaling_enough_signal", "Enough signal"),
                        ("scaling_confidence_label", "Confidence"),
                        ("scaling_confidence_reason", "Reason"),
                        ("scaling_projection_source", "Source"),
                        ("scaling_matched_truth_count", "Truth matches"),
                        ("crossover_from_target", "Crossover"),
                    ],
                ),
                "",
            ]
        )
    report.extend(
        [
        "## Local Search",
        "",
        "The local-search winner is chosen by the strict objective first: lowest measured `val_bpb`. Throughput, memory, and control overhead only break ties when local-search rows are too close to separate on validation quality alone.",
        "",
        markdown_table(
            local_rows,
            [
                ("preset", "Preset"),
                ("seq_len", "Seq len"),
                ("window_pattern", "Window"),
                ("device_batch_size", "Device batch"),
                ("total_batch_size", "Total batch"),
                ("grad_accum_steps", "Grad accum"),
                ("steady_state_tok_per_sec", "Steady tok/s"),
                ("optimizer_percent", "Optim %"),
                ("accum_percent", "Accum %"),
                ("control_overhead_percent", "Control %"),
                ("peak_vram_mb", "Peak MB"),
            ],
        ),
        "",
        "## Eval Calibration",
        "",
        markdown_table(
            eval_rows,
            [
                ("rung", "Rung"),
                ("eval_tokens", "Eval tokens"),
                ("eval_seconds", "Eval sec"),
                ("val_bpb", "val_bpb"),
                ("abs_error_vs_full", "Abs error vs full"),
                ("speedup_vs_full", "Speedup vs full"),
            ],
        ),
        "",
        "## Candidate Default Block",
        "",
        "```json",
        json.dumps(candidate_default, indent=2),
        "```",
        "",
        "## Promotion Bundle",
        "",
        "```json",
        json.dumps(promotion_bundle, indent=2),
        "```",
        "",
        "## Reference Comparison",
        "",
        "### M5 Reference",
        "",
        "```json",
        json.dumps(payload["reference_comparison"]["m5"], indent=2),
        "```",
        "",
        "### Upstream-Style Reference",
        "",
        "```json",
        json.dumps(payload["reference_comparison"]["upstream_style"], indent=2),
        "```",
    ]
    )
    path.write_text("\n".join(report) + "\n")


def run_platform_calibration(args) -> dict:
    set_friendly_progress(not args.plain_progress)
    mode = args.mode
    scaling_confirmation_enabled = args.enable_scaling_confirmation
    coarse_time_budget = resolved_budget(args.coarse_time_budget, mode=mode, field="coarse_time_budget")
    ranking_time_budget = resolved_budget(args.ranking_time_budget, mode=mode, field="ranking_time_budget")
    projection_time_budget = resolved_budget(args.projection_time_budget, mode=mode, field="projection_time_budget")
    finalist_time_budget = resolved_budget(args.finalist_time_budget, mode=mode, field="finalist_time_budget")
    batch_audit_time_budget = resolved_budget(args.batch_audit_time_budget, mode=mode, field="batch_audit_time_budget")
    finalist_count = args.finalist_count if args.finalist_count is not None else resolve_mode_spec(mode).finalist_count
    local_search_time_budget = resolved_budget(
        args.local_search_time_budget,
        mode=mode,
        field="local_search_time_budget",
    )
    eval_train_seconds = resolved_budget(args.eval_train_seconds, mode=mode, field="eval_train_seconds")
    eval_rungs = resolved_eval_rungs(args.eval_rungs, mode=mode)
    engine = get_engine(args.engine)
    presets = select_presets(engine, args.presets)
    hardware = engine.detect_hardware_fingerprint()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(engine_name=engine.name, hardware_key=hardware.hardware_key)
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    calibration_signatures = engine.calibration_signatures()
    resolved_truth_curves_dir, truth_curves = resolve_truth_curve_inputs(
        truth_curves_dir=args.truth_curves_dir,
        output_dir=output_dir,
        engine_name=engine.name,
        hardware_key=hardware.hardware_key,
        target_seconds=PROJECTION_TARGET_SECONDS,
    )
    truth_eval_contract = infer_truth_eval_contract(truth_curves)
    projection_calibration = build_projection_calibration(truth_curves, target_seconds=PROJECTION_TARGET_SECONDS)
    phase_estimates = estimate_calibration_phases(
        engine=engine,
        presets=presets,
        mode=mode,
        batch_profile_time_budget=min(5.0, ranking_time_budget),
        coarse_time_budget=coarse_time_budget,
        projection_time_budget=projection_time_budget,
        finalist_time_budget=finalist_time_budget,
        finalist_count=finalist_count,
        batch_audit_time_budget=batch_audit_time_budget,
        local_search_time_budget=local_search_time_budget,
        eval_train_seconds=eval_train_seconds,
        eval_rungs=eval_rungs,
        scaling_confirmation_enabled=scaling_confirmation_enabled,
    )
    announce_calibration_story(
        engine=engine,
        hardware_key=hardware.hardware_key,
        mode=mode,
        output_dir=output_dir,
        presets=presets,
        truth_curves_dir=resolved_truth_curves_dir,
        truth_curve_count=len(truth_curves),
        phase_estimates=phase_estimates,
        scaling_confirmation_enabled=scaling_confirmation_enabled,
    )

    hardware_inputs = {
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "hardware_key": hardware.hardware_key,
    }
    hardware_started = phase_start("hardware fingerprint", detail=hardware.hardware_key)
    hardware_phase = load_phase_if_matching(output_dir, "hardware_fingerprint", hardware_inputs, force=args.force)
    if hardware_phase is None:
        hardware_phase = save_phase(
            output_dir,
            "hardware_fingerprint",
            inputs=hardware_inputs,
            payload=asdict(hardware),
        )
        phase_done("hardware fingerprint", hardware_started)
    else:
        phase_reused("hardware fingerprint", detail=hardware.hardware_key)

    batch_profile_probe: ProbeResult | None = None
    batch_profile_phase: dict | None = None
    batch_profile_time_budget = min(5.0, ranking_time_budget)
    batch_profile_overrides: dict[str, tuple[int, int]] = {}
    if engine.capabilities.supports_local_search:
        batch_profile_anchor = choose_batch_profile_anchor_preset(engine, presets)
        anchor_seq_len = engine.preset_catalog()[batch_profile_anchor].seq_len
        batch_profile_estimate = (
            len(engine.batch_profile_candidates(batch_profile_anchor, seq_len=anchor_seq_len))
            * batch_profile_time_budget
        )
        batch_profile_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "anchor_preset": batch_profile_anchor,
            "time_budget": batch_profile_time_budget,
        }
        batch_profile_started = phase_start(
            "anchor batch profile",
            detail=f"anchor preset {batch_profile_anchor}",
            estimate_seconds=batch_profile_estimate,
        )
        batch_profile_phase = load_phase_if_matching(output_dir, "batch_profile", batch_profile_inputs, force=args.force)
        if batch_profile_phase is None:
            try:
                batch_profile_payload = run_batch_profile(
                    engine=engine,
                    preset=batch_profile_anchor,
                    time_budget=batch_profile_time_budget,
                    logs_dir=logs_dir,
                )
                batch_profile_probe = batch_profile_payload["winner"]
                batch_profile_phase = save_phase(
                    output_dir,
                    "batch_profile",
                    inputs=batch_profile_inputs,
                    payload={
                        "time_budget": batch_profile_time_budget,
                        "anchor_preset": batch_profile_anchor,
                        "coarse_rows": [asdict(row) for row in batch_profile_payload["coarse_rows"]],
                        "refinement_rows": [asdict(row) for row in batch_profile_payload["refinement_rows"]],
                        "extension_rows": [asdict(row) for row in batch_profile_payload["extension_rows"]],
                        "rows": [asdict(row) for row in batch_profile_payload["rows"]],
                        "winner": asdict(batch_profile_probe),
                    },
                )
                phase_done(
                    "anchor batch profile",
                    batch_profile_started,
                    detail=(
                        f"winner db={batch_profile_probe.device_batch_size}, "
                        f"tb={batch_profile_probe.total_batch_size}, "
                        f"accum={batch_profile_probe.grad_accum_steps}"
                    ),
                )
            except Exception:
                batch_profile_phase = None
                batch_profile_probe = None
                calibration_progress("Anchor batch profile failed; continuing without batch overrides.")
        elif batch_profile_phase is not None:
            batch_profile_probe = probe_from_dict(batch_profile_phase["payload"]["winner"])
            phase_reused(
                "anchor batch profile",
                detail=(
                    f"winner db={batch_profile_probe.device_batch_size}, "
                    f"tb={batch_profile_probe.total_batch_size}, "
                    f"accum={batch_profile_probe.grad_accum_steps}"
                ),
            )

        if batch_profile_probe is not None:
            for preset in presets:
                preset_config = engine.preset_catalog()[preset]
                device_batch_size, total_batch_size, _ = apply_batch_profile(
                    seq_len=preset_config.seq_len,
                    batch_profile=batch_profile_probe,
                )
                batch_profile_overrides[preset] = (device_batch_size, total_batch_size)

    coarse_inputs = {
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "mode": mode,
        "presets": presets,
        "time_budget": coarse_time_budget,
        "batch_profile": (
            {
                "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                "device_batch_size": batch_profile_probe.device_batch_size,
                "total_batch_size": batch_profile_probe.total_batch_size,
            }
            if batch_profile_probe is not None and batch_profile_phase is not None
            else None
        ),
    }
    coarse_started = phase_start(
        "coarse envelope",
        detail=f"{len(presets)} presets",
        estimate_seconds=len(presets) * coarse_time_budget,
    )
    coarse_phase = load_phase_if_matching(output_dir, "coarse_envelope", coarse_inputs, force=args.force)
    if coarse_phase is None:
        coarse_rows = [
            run_train_probe(
                engine=engine,
                preset=preset,
                time_budget=coarse_time_budget,
                logs_dir=logs_dir,
                stage="coarse",
                benchmark_skip_eval=True,
                device_batch_size=batch_profile_overrides.get(preset, (None, None))[0],
                total_batch_size=batch_profile_overrides.get(preset, (None, None))[1],
                no_checkpoint=True,
            )
            for preset in presets
        ]
        coarse_phase = save_phase(
            output_dir,
            "coarse_envelope",
            inputs=coarse_inputs,
            payload={"time_budget": coarse_time_budget, "rows": [asdict(row) for row in coarse_rows]},
        )
        successful = sum(1 for row in coarse_rows if row.status == "ok")
        phase_done(
            "coarse envelope",
            coarse_started,
            detail=f"{successful}/{len(coarse_rows)} presets succeeded",
        )
    else:
        phase_reused("coarse envelope", detail=f"{len(presets)} presets")
    coarse_rows = [probe_from_dict(row) for row in coarse_phase["payload"]["rows"]]
    coarse_by_preset = {row.preset: row for row in coarse_rows}

    ranking_presets = [
        row.preset
        for row in coarse_rows
        if row.status == "ok" and row.preset != engine.reference_preset
    ]
    if not ranking_presets:
        ranking_presets = [row.preset for row in coarse_rows if row.status == "ok"]
    if not ranking_presets:
        raise RuntimeError("No successful presets were found during the coarse envelope.")

    ranking_phase: dict | None = None
    ranking_rows: list[ProbeResult] = []
    ranked_candidates: list[RankedProbe] = []
    ranking_winner: RankedProbe | None = None
    projection_phase: dict | None = None
    projection_ranked_metadata: list[RankedProbe] = []
    projection_inputs = {
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "mode": mode,
        "presets": ranking_presets,
        "time_budget": projection_time_budget,
        "target_seconds": PROJECTION_TARGET_SECONDS,
        "hardware_key": hardware.hardware_key,
        "truth_eval_contract": truth_eval_contract,
        "batch_profile": (
            {
                "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                "device_batch_size": batch_profile_probe.device_batch_size,
                "total_batch_size": batch_profile_probe.total_batch_size,
            }
            if batch_profile_probe is not None and batch_profile_phase is not None
            else None
        ),
    }
    projection_started = phase_start(
        "projection ranking",
        detail=f"{len(ranking_presets)} presets projected to {int(PROJECTION_TARGET_SECONDS)}s",
        estimate_seconds=len(ranking_presets) * projection_time_budget,
    )
    projection_phase = load_phase_if_matching(output_dir, "projection_ranking", projection_inputs, force=args.force)
    if projection_phase is None:
        projection_probe_rows = [
            run_curve_train_probe(
                engine=engine,
                preset=preset,
                time_budget=projection_time_budget,
                logs_dir=logs_dir,
                stage="projection",
                curve_eval_seconds=(10.0, projection_time_budget) if projection_time_budget > 10.0 else (projection_time_budget,),
                device_batch_size=batch_profile_overrides.get(preset, (None, None))[0],
                total_batch_size=batch_profile_overrides.get(preset, (None, None))[1],
                eval_seq_len=None if truth_eval_contract is None else truth_eval_contract["eval_seq_len"],
                eval_tokens=None if truth_eval_contract is None else truth_eval_contract["eval_tokens"],
                eval_batch_size=None if truth_eval_contract is None else truth_eval_contract["eval_batch_size"],
            )
            for preset in ranking_presets
        ]
        projection_curves = load_probe_curve_artifacts(projection_probe_rows)
        projected_rows, projection_decision = compare_projected_curves(
            projection_curves,
            target_seconds=PROJECTION_TARGET_SECONDS,
            calibration=projection_calibration,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        projection_horizon_rows, projection_horizon_decisions = build_horizon_projection_table(
            projection_curves,
            horizons_seconds=REPORT_HORIZONS,
            calibration=projection_calibration,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        projection_control_decision = resolve_horizon_control_decision(
            projection_horizon_decisions,
            target_seconds=PROJECTION_TARGET_SECONDS,
        )
        projection_next_horizon = asdict(
            suggest_next_horizon(
                projection_horizon_rows,
                projection_horizon_decisions,
                target_seconds=PROJECTION_TARGET_SECONDS,
                candidate_horizons=REPORT_HORIZONS,
            )
        )
        projection_target_horizon_rows = [asdict(row) for row in rows_for_horizon(projection_horizon_rows, PROJECTION_TARGET_SECONDS)]
        projection_phase_winner = (
            projection_target_horizon_rows[0]
            if projection_target_horizon_rows
            else projected_row_to_dict(projected_rows[0]) if projected_rows else None
        )
        projection_ranked_metadata = rank_candidate_families(
            projection_probe_rows,
            engine=engine,
            hardware_key=hardware.hardware_key,
            hardware=hardware,
            ranking_time_budget=projection_time_budget,
        )
        ranking_phase = save_phase(
            output_dir,
            "candidate_ranking",
            inputs={
                "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
                "mode": mode,
                "presets": ranking_presets,
                "time_budget": projection_time_budget,
                "hardware_key": hardware.hardware_key,
                "batch_profile": (
                    {
                        "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                        "device_batch_size": batch_profile_probe.device_batch_size,
                        "total_batch_size": batch_profile_probe.total_batch_size,
                    }
                    if batch_profile_probe is not None and batch_profile_phase is not None
                    else None
                ),
                "source": "projection-pass",
            },
            payload={
                "time_budget": projection_time_budget,
                "rows": [ranked_probe_to_dict(item) for item in projection_ranked_metadata],
                "winner": ranked_probe_to_dict(projection_ranked_metadata[0]),
            },
        )
        projection_phase = save_phase(
            output_dir,
            "projection_ranking",
            inputs=projection_inputs,
            payload={
                "time_budget": projection_time_budget,
                "target_seconds": PROJECTION_TARGET_SECONDS,
                "probe_rows": [asdict(row) for row in projection_probe_rows],
                "decision": None if projection_decision is None else asdict(projection_decision),
                "control_decision": None if projection_control_decision is None else asdict(projection_control_decision),
                "rows": [projected_row_to_dict(item) for item in projected_rows],
                "horizon_rows": [asdict(row) for row in projection_horizon_rows],
                "horizon_decisions": [asdict(item) for item in projection_horizon_decisions],
                "next_horizon": projection_next_horizon,
                "winner": projection_phase_winner,
            },
        )
        ranking_rows = projection_probe_rows
        ranked_candidates = projection_ranked_metadata
        ranking_winner = projection_ranked_metadata[0]
        projection_winner_name = (
            projection_phase_winner["preset"]
            if isinstance(projection_phase_winner, dict)
            else ranking_winner.probe.preset
        )
        phase_done(
            "projection ranking",
            projection_started,
            detail=f"projected leader {projection_winner_name}",
        )
    else:
        ranking_phase = read_json(phase_path(output_dir, "candidate_ranking"))
        projection_probe_rows = [
            probe_from_dict(row)
            for row in projection_phase["payload"].get("probe_rows", [])
        ]
        if not projection_probe_rows:
            raise RuntimeError("Projection ranking phase is missing probe_rows; rerun with --force to regenerate it.")
        projection_ranked_metadata = rank_candidate_families(
            projection_probe_rows,
            engine=engine,
            hardware_key=hardware.hardware_key,
            hardware=hardware,
            ranking_time_budget=projection_time_budget,
        )
        ranking_rows = projection_probe_rows
        ranked_candidates = projection_ranked_metadata
        ranking_winner = projection_ranked_metadata[0]
        phase_reused(
            "projection ranking",
            detail=f"projected leader {projection_phase['payload'].get('winner', {}).get('preset', ranking_winner.probe.preset)}",
        )
    projection_curves = load_probe_curve_artifacts(projection_probe_rows)
    projection_estimates, projection_decision = compare_projected_curves(
        projection_curves,
        target_seconds=PROJECTION_TARGET_SECONDS,
        calibration=projection_calibration,
        truth_curves=truth_curves,
        winner_probability_threshold=args.winner_probability_threshold,
        projected_margin_threshold=args.projected_margin_threshold,
    )
    projection_horizon_rows, projection_horizon_decisions = build_horizon_projection_table(
        projection_curves,
        horizons_seconds=REPORT_HORIZONS,
        calibration=projection_calibration,
        truth_curves=truth_curves,
        winner_probability_threshold=args.winner_probability_threshold,
        projected_margin_threshold=args.projected_margin_threshold,
    )
    projection_control_decision = resolve_horizon_control_decision(
        projection_horizon_decisions,
        target_seconds=PROJECTION_TARGET_SECONDS,
    )
    projection_next_horizon = asdict(
        suggest_next_horizon(
            projection_horizon_rows,
            projection_horizon_decisions,
            target_seconds=PROJECTION_TARGET_SECONDS,
            candidate_horizons=REPORT_HORIZONS,
        )
    )
    projection_rows = [projected_row_to_dict(item) for item in projection_estimates]
    projection_target_horizon_rows = [asdict(row) for row in rows_for_horizon(projection_horizon_rows, PROJECTION_TARGET_SECONDS)]
    projection_winner = projection_target_horizon_rows[0] if projection_target_horizon_rows else (projection_rows[0] if projection_rows else None)
    projected_presets = [row["preset"] for row in projection_target_horizon_rows] if projection_target_horizon_rows else [row["preset"] for row in projection_rows]
    projection_probe_metadata_by_preset = {item.probe.preset: item for item in projection_ranked_metadata}

    finalist_phase: dict | None = None
    finalist_candidates = []
    finalist_curves = []
    finalist_horizon_rows = []
    finalist_horizon_decisions = []
    confirmation_phase: dict | None = None
    confirmation_curves = []
    confirmation_rows = []
    confirmation_horizon_rows = []
    confirmation_horizon_decisions = []
    scaling_confirmation_phase: dict | None = None
    scaling_confirmation_rows = []
    scaling_confirmation_horizon_rows = []
    scaling_confirmation_horizon_decisions = []
    finalist_presets = projected_presets[: max(1, finalist_count)]
    candidate_family = None
    if (
        projection_control_decision is not None
        and decision_enough_signal(projection_control_decision)
        and projection_winner is not None
    ):
        candidate_family = projection_probe_metadata_by_preset[projection_winner["preset"]]
        calibration_progress(
            f"Projection already has enough signal at {int(PROJECTION_TARGET_SECONDS)}s; "
            f"skipping finalist rerank and using {candidate_family.probe.preset}."
        )
    elif len(finalist_presets) > 1:
        finalist_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "presets": finalist_presets,
            "time_budget": finalist_time_budget,
            "target_seconds": PROJECTION_TARGET_SECONDS,
            "hardware_key": hardware.hardware_key,
            "truth_eval_contract": truth_eval_contract,
            "batch_profile": (
                {
                    "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                    "device_batch_size": batch_profile_probe.device_batch_size,
                    "total_batch_size": batch_profile_probe.total_batch_size,
                }
                if batch_profile_probe is not None and batch_profile_phase is not None
                else None
            ),
            "batch_profile_overrides": batch_profile_overrides_to_dict(
                batch_profile_overrides,
                presets=finalist_presets,
            ),
        }
        finalist_started = phase_start(
            "finalist rerank",
            detail=f"{len(finalist_presets)} presets",
            estimate_seconds=len(finalist_presets) * finalist_time_budget,
        )
        finalist_phase = load_phase_if_matching(output_dir, "finalist_projection", finalist_inputs, force=args.force)
        finalist_diagnostics = []
        if finalist_phase is None:
            finalist_probe_rows = [
                run_curve_train_probe(
                    engine=engine,
                    preset=preset,
                    time_budget=finalist_time_budget,
                    logs_dir=logs_dir,
                    stage="finalist",
                    curve_eval_seconds=(10.0, 30.0, finalist_time_budget) if finalist_time_budget > 30.0 else (10.0, finalist_time_budget),
                    device_batch_size=batch_profile_overrides.get(preset, (None, None))[0],
                    total_batch_size=batch_profile_overrides.get(preset, (None, None))[1],
                    eval_seq_len=None if truth_eval_contract is None else truth_eval_contract["eval_seq_len"],
                    eval_tokens=None if truth_eval_contract is None else truth_eval_contract["eval_tokens"],
                    eval_batch_size=None if truth_eval_contract is None else truth_eval_contract["eval_batch_size"],
                )
                for preset in finalist_presets
            ]
            finalist_curves = load_probe_curve_artifacts(finalist_probe_rows)
            finalist_estimates, finalist_decision, finalist_diagnostics = compare_multi_horizon_curves(
                projection_curves,
                finalist_curves,
                target_seconds=PROJECTION_TARGET_SECONDS,
                calibration=projection_calibration,
                truth_curves=truth_curves,
                winner_probability_threshold=args.winner_probability_threshold,
                projected_margin_threshold=args.projected_margin_threshold,
            )
            finalist_horizon_rows, finalist_horizon_decisions = build_horizon_projection_table(
                finalist_curves,
                horizons_seconds=REPORT_HORIZONS,
                calibration=projection_calibration,
                truth_curves=truth_curves,
                winner_probability_threshold=args.winner_probability_threshold,
                projected_margin_threshold=args.projected_margin_threshold,
            )
            finalist_control_decision = resolve_horizon_control_decision(
                finalist_horizon_decisions,
                target_seconds=PROJECTION_TARGET_SECONDS,
            )
            finalist_next_horizon = asdict(
                suggest_next_horizon(
                    finalist_horizon_rows,
                    finalist_horizon_decisions,
                    target_seconds=PROJECTION_TARGET_SECONDS,
                    candidate_horizons=REPORT_HORIZONS,
                )
            )
            finalist_target_horizon_rows = [asdict(row) for row in rows_for_horizon(finalist_horizon_rows, PROJECTION_TARGET_SECONDS)]
            finalist_phase_winner = (
                finalist_target_horizon_rows[0]
                if finalist_target_horizon_rows
                else projected_row_to_dict(finalist_estimates[0]) if finalist_estimates else None
            )
            finalist_candidates = rank_candidate_families(
                finalist_probe_rows,
                engine=engine,
                hardware_key=hardware.hardware_key,
                hardware=hardware,
                ranking_time_budget=finalist_time_budget,
            )
            finalist_phase = save_phase(
                output_dir,
                "finalist_projection",
                inputs=finalist_inputs,
                payload={
                    "time_budget": finalist_time_budget,
                    "target_seconds": PROJECTION_TARGET_SECONDS,
                    "probe_rows": [asdict(row) for row in finalist_probe_rows],
                    "ranked_rows": [ranked_probe_to_dict(item) for item in finalist_candidates],
                    "decision": None if finalist_decision is None else asdict(finalist_decision),
                    "control_decision": None if finalist_control_decision is None else asdict(finalist_control_decision),
                    "diagnostics": [asdict(item) for item in finalist_diagnostics],
                    "rows": [projected_row_to_dict(item) for item in finalist_estimates],
                    "horizon_rows": [asdict(row) for row in finalist_horizon_rows],
                    "horizon_decisions": [asdict(item) for item in finalist_horizon_decisions],
                    "next_horizon": finalist_next_horizon,
                    "winner": finalist_phase_winner,
                },
            )
            phase_done(
                "finalist rerank",
                finalist_started,
                detail=f"leader {finalist_phase_winner['preset'] if isinstance(finalist_phase_winner, dict) else finalist_presets[0]}",
            )
        else:
            phase_reused(
                "finalist rerank",
                detail=f"leader {finalist_phase['payload'].get('winner', {}).get('preset', finalist_presets[0])}",
            )
        finalist_probe_rows = [
            probe_from_dict(row)
            for row in finalist_phase["payload"].get("probe_rows", [])
        ]
        if not finalist_probe_rows:
            raise RuntimeError("Finalist projection phase is missing probe_rows; rerun with --force to regenerate it.")
        finalist_candidates = rank_candidate_families(
            finalist_probe_rows,
            engine=engine,
            hardware_key=hardware.hardware_key,
            hardware=hardware,
            ranking_time_budget=finalist_time_budget,
        )
        finalist_curves = load_probe_curve_artifacts(finalist_probe_rows)
        finalist_estimates, finalist_decision, finalist_diagnostics = compare_multi_horizon_curves(
            projection_curves,
            finalist_curves,
            target_seconds=PROJECTION_TARGET_SECONDS,
            calibration=projection_calibration,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        finalist_horizon_rows, finalist_horizon_decisions = build_horizon_projection_table(
            finalist_curves,
            horizons_seconds=REPORT_HORIZONS,
            calibration=projection_calibration,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        finalist_control_decision = resolve_horizon_control_decision(
            finalist_horizon_decisions,
            target_seconds=PROJECTION_TARGET_SECONDS,
        )
        finalist_next_horizon = asdict(
            suggest_next_horizon(
                finalist_horizon_rows,
                finalist_horizon_decisions,
                target_seconds=PROJECTION_TARGET_SECONDS,
                candidate_horizons=REPORT_HORIZONS,
            )
        )
        finalist_rows = [projected_row_to_dict(item) for item in finalist_estimates]
        finalist_target_horizon_rows = [asdict(row) for row in rows_for_horizon(finalist_horizon_rows, PROJECTION_TARGET_SECONDS)]
        finalist_probe_metadata_by_preset = {item.probe.preset: item for item in finalist_candidates}
        finalist_winner = finalist_target_horizon_rows[0] if finalist_target_horizon_rows else (finalist_rows[0] if finalist_rows else None)
        confirmation_needed = (
            finalist_control_decision is not None
            and not decision_enough_signal(finalist_control_decision)
            and isinstance(finalist_next_horizon, dict)
            and isinstance(finalist_next_horizon.get("suggested_seconds"), (int, float))
            and finalist_next_horizon.get("suggested_seconds") > finalist_time_budget
            and finalist_next_horizon.get("suggested_seconds") <= PROJECTION_TARGET_SECONDS
        )
        if confirmation_needed:
            confirmation_presets = [
                row["preset"]
                for row in finalist_target_horizon_rows[:2]
                if isinstance(row, dict) and row.get("preset")
            ]
            if len(confirmation_presets) > 1:
                confirmation_time_budget = float(finalist_next_horizon["suggested_seconds"])
                confirmation_started = phase_start(
                    "winner confirmation",
                    detail=f"{' vs '.join(confirmation_presets)}",
                    estimate_seconds=len(confirmation_presets) * confirmation_time_budget,
                )
                confirmation_inputs = {
                    "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
                    "mode": mode,
                    "presets": confirmation_presets,
                    "time_budget": confirmation_time_budget,
                    "target_seconds": PROJECTION_TARGET_SECONDS,
                    "hardware_key": hardware.hardware_key,
                    "truth_eval_contract": truth_eval_contract,
                    "batch_profile": (
                        {
                            "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                            "device_batch_size": batch_profile_probe.device_batch_size,
                            "total_batch_size": batch_profile_probe.total_batch_size,
                        }
                        if batch_profile_probe is not None and batch_profile_phase is not None
                        else None
                    ),
                    "batch_profile_overrides": batch_profile_overrides_to_dict(
                        batch_profile_overrides,
                        presets=confirmation_presets,
                    ),
                }
                confirmation_phase = load_phase_if_matching(
                    output_dir,
                    "confirmation_projection",
                    confirmation_inputs,
                    force=args.force,
                )
                if confirmation_phase is None:
                    confirmation_probe_rows = [
                        run_curve_train_probe(
                            engine=engine,
                            preset=preset,
                            time_budget=confirmation_time_budget,
                            logs_dir=logs_dir,
                            stage="confirmation",
                            curve_eval_seconds=(30.0, 60.0, confirmation_time_budget)
                            if confirmation_time_budget > 60.0
                            else (30.0, confirmation_time_budget),
                            device_batch_size=batch_profile_overrides.get(preset, (None, None))[0],
                            total_batch_size=batch_profile_overrides.get(preset, (None, None))[1],
                            eval_seq_len=None if truth_eval_contract is None else truth_eval_contract["eval_seq_len"],
                            eval_tokens=None if truth_eval_contract is None else truth_eval_contract["eval_tokens"],
                            eval_batch_size=None if truth_eval_contract is None else truth_eval_contract["eval_batch_size"],
                        )
                        for preset in confirmation_presets
                    ]
                    confirmation_curves = load_probe_curve_artifacts(confirmation_probe_rows)
                    confirmation_estimates, confirmation_decision, confirmation_diagnostics = compare_multi_horizon_curves(
                        finalist_curves,
                        confirmation_curves,
                        target_seconds=PROJECTION_TARGET_SECONDS,
                        calibration=projection_calibration,
                        truth_curves=truth_curves,
                        winner_probability_threshold=args.winner_probability_threshold,
                        projected_margin_threshold=args.projected_margin_threshold,
                    )
                    confirmation_horizon_rows, confirmation_horizon_decisions = build_horizon_projection_table(
                        confirmation_curves,
                        horizons_seconds=REPORT_HORIZONS,
                        calibration=projection_calibration,
                        truth_curves=truth_curves,
                        winner_probability_threshold=args.winner_probability_threshold,
                        projected_margin_threshold=args.projected_margin_threshold,
                    )
                    confirmation_control_decision = resolve_horizon_control_decision(
                        confirmation_horizon_decisions,
                        target_seconds=PROJECTION_TARGET_SECONDS,
                    )
                    confirmation_next_horizon = asdict(
                        suggest_next_horizon(
                            confirmation_horizon_rows,
                            confirmation_horizon_decisions,
                            target_seconds=PROJECTION_TARGET_SECONDS,
                            candidate_horizons=REPORT_HORIZONS,
                        )
                    )
                    confirmation_target_horizon_rows = [
                        asdict(row) for row in rows_for_horizon(confirmation_horizon_rows, PROJECTION_TARGET_SECONDS)
                    ]
                    confirmation_phase_winner = (
                        confirmation_target_horizon_rows[0]
                        if confirmation_target_horizon_rows
                        else projected_row_to_dict(confirmation_estimates[0]) if confirmation_estimates else None
                    )
                    confirmation_phase = save_phase(
                        output_dir,
                        "confirmation_projection",
                        inputs=confirmation_inputs,
                        payload={
                            "time_budget": confirmation_time_budget,
                            "target_seconds": PROJECTION_TARGET_SECONDS,
                            "probe_rows": [asdict(row) for row in confirmation_probe_rows],
                            "decision": None if confirmation_decision is None else asdict(confirmation_decision),
                            "control_decision": None if confirmation_control_decision is None else asdict(confirmation_control_decision),
                            "diagnostics": [asdict(item) for item in confirmation_diagnostics],
                            "rows": [projected_row_to_dict(item) for item in confirmation_estimates],
                            "horizon_rows": [asdict(row) for row in confirmation_horizon_rows],
                            "horizon_decisions": [asdict(item) for item in confirmation_horizon_decisions],
                            "next_horizon": confirmation_next_horizon,
                            "winner": confirmation_phase_winner,
                        },
                    )
                    phase_done(
                        "winner confirmation",
                        confirmation_started,
                        detail=f"leader {confirmation_phase_winner['preset'] if isinstance(confirmation_phase_winner, dict) else confirmation_presets[0]}",
                    )
                else:
                    phase_reused(
                        "winner confirmation",
                        detail=f"leader {confirmation_phase['payload'].get('winner', {}).get('preset', confirmation_presets[0])}",
                    )
                confirmation_probe_rows = [
                    probe_from_dict(row)
                    for row in confirmation_phase["payload"].get("probe_rows", [])
                ]
                confirmation_curves = load_probe_curve_artifacts(confirmation_probe_rows)
                confirmation_estimates, confirmation_decision, confirmation_diagnostics = compare_multi_horizon_curves(
                    finalist_curves,
                    confirmation_curves,
                    target_seconds=PROJECTION_TARGET_SECONDS,
                    calibration=projection_calibration,
                    truth_curves=truth_curves,
                    winner_probability_threshold=args.winner_probability_threshold,
                    projected_margin_threshold=args.projected_margin_threshold,
                )
                confirmation_horizon_rows, confirmation_horizon_decisions = build_horizon_projection_table(
                    confirmation_curves,
                    horizons_seconds=REPORT_HORIZONS,
                    calibration=projection_calibration,
                    truth_curves=truth_curves,
                    winner_probability_threshold=args.winner_probability_threshold,
                    projected_margin_threshold=args.projected_margin_threshold,
                )
                confirmation_target_horizon_rows = [
                    asdict(row) for row in rows_for_horizon(confirmation_horizon_rows, PROJECTION_TARGET_SECONDS)
                ]
                confirmation_probe_metadata_by_preset = {
                    probe.preset: probe for probe in confirmation_probe_rows
                }
                confirmation_winner = (
                    confirmation_target_horizon_rows[0]
                    if confirmation_target_horizon_rows
                    else projected_row_to_dict(confirmation_estimates[0]) if confirmation_estimates else None
                )
                confirmation_rows = [projected_row_to_dict(item) for item in confirmation_estimates]
                if isinstance(confirmation_winner, dict):
                    candidate_family = ranked_probe_from_probe(
                        confirmation_probe_metadata_by_preset[confirmation_winner["preset"]]
                    )
        if candidate_family is None and isinstance(finalist_winner, dict):
            candidate_family = finalist_probe_metadata_by_preset[finalist_winner["preset"]]
    if candidate_family is None:
        if projection_winner is None:
            raise RuntimeError("Projection ranking produced no winner.")
        candidate_family = projection_probe_metadata_by_preset[projection_winner["preset"]]
    calibration_progress(
        f"Selected preset family: {candidate_family.probe.preset} "
        f"(current batch db={candidate_family.probe.device_batch_size}, tb={candidate_family.probe.total_batch_size})."
    )

    deepest_projection_curves = confirmation_curves or finalist_curves or projection_curves
    deepest_horizon_rows = confirmation_horizon_rows or finalist_horizon_rows or projection_horizon_rows
    deepest_horizon_decisions = confirmation_horizon_decisions or finalist_horizon_decisions or projection_horizon_decisions

    longer_horizon_projection = summarize_longer_horizon_projection(
        deepest_horizon_rows,
        deepest_horizon_decisions,
        target_seconds=PROJECTION_TARGET_SECONDS,
        scaling_target_seconds=SCALING_TARGET_SECONDS,
    )

    scaling_confirmation_needed = (
        scaling_confirmation_enabled
        and
        longer_horizon_projection is not None
        and longer_horizon_projection.crossover_from_target
        and not longer_horizon_projection.scaling_enough_signal
        and longer_horizon_projection.scaling_winner_preset != longer_horizon_projection.target_winner_preset
        and max(
            (row.observed_seconds for row in deepest_horizon_rows if row.observed_seconds is not None),
            default=0.0,
        )
        < SCALING_TARGET_SECONDS
    )
    if (
        not scaling_confirmation_enabled
        and longer_horizon_projection is not None
        and longer_horizon_projection.crossover_from_target
        and longer_horizon_projection.scaling_winner_preset != longer_horizon_projection.target_winner_preset
    ):
        calibration_progress(
            f"Longer-horizon crossover is projected ({longer_horizon_projection.target_winner_preset} at "
            f"{int(PROJECTION_TARGET_SECONDS)}s, {longer_horizon_projection.scaling_winner_preset} at "
            f"{int(SCALING_TARGET_SECONDS)}s), but deeper scaling confirmation is disabled by default."
        )
    if scaling_confirmation_needed:
        scaling_confirmation_presets = [
            longer_horizon_projection.target_winner_preset,
            longer_horizon_projection.scaling_winner_preset,
        ]
        scaling_confirmation_time_budget = SCALING_TARGET_SECONDS
        scaling_started = phase_start(
            "scaling confirmation",
            detail=f"{' vs '.join(scaling_confirmation_presets)} to {int(SCALING_TARGET_SECONDS)}s",
            estimate_seconds=len(scaling_confirmation_presets) * scaling_confirmation_time_budget,
        )
        scaling_confirmation_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "presets": scaling_confirmation_presets,
            "time_budget": scaling_confirmation_time_budget,
            "target_seconds": SCALING_TARGET_SECONDS,
            "hardware_key": hardware.hardware_key,
            "truth_eval_contract": truth_eval_contract,
            "batch_profile": (
                {
                    "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                    "device_batch_size": batch_profile_probe.device_batch_size,
                    "total_batch_size": batch_profile_probe.total_batch_size,
                }
                if batch_profile_probe is not None and batch_profile_phase is not None
                else None
            ),
            "batch_profile_overrides": batch_profile_overrides_to_dict(
                batch_profile_overrides,
                presets=scaling_confirmation_presets,
            ),
        }
        scaling_confirmation_phase = load_phase_if_matching(
            output_dir,
            "scaling_confirmation",
            scaling_confirmation_inputs,
            force=args.force,
        )
        if scaling_confirmation_phase is None:
            scaling_confirmation_probe_rows = [
                run_curve_train_probe(
                    engine=engine,
                    preset=preset,
                    time_budget=scaling_confirmation_time_budget,
                    logs_dir=logs_dir,
                    stage="scaling",
                    curve_eval_seconds=(300.0, 600.0, scaling_confirmation_time_budget),
                    device_batch_size=batch_profile_overrides.get(preset, (None, None))[0],
                    total_batch_size=batch_profile_overrides.get(preset, (None, None))[1],
                    eval_seq_len=None if truth_eval_contract is None else truth_eval_contract["eval_seq_len"],
                    eval_tokens=None if truth_eval_contract is None else truth_eval_contract["eval_tokens"],
                    eval_batch_size=None if truth_eval_contract is None else truth_eval_contract["eval_batch_size"],
                )
                for preset in scaling_confirmation_presets
            ]
            scaling_confirmation_curves = load_probe_curve_artifacts(scaling_confirmation_probe_rows)
            scaling_confirmation_estimates, scaling_confirmation_decision, scaling_confirmation_diagnostics = compare_multi_horizon_curves(
                deepest_projection_curves,
                scaling_confirmation_curves,
                target_seconds=SCALING_TARGET_SECONDS,
                calibration=None,
                truth_curves=truth_curves,
                winner_probability_threshold=args.winner_probability_threshold,
                projected_margin_threshold=args.projected_margin_threshold,
            )
            scaling_confirmation_horizon_rows, scaling_confirmation_horizon_decisions = build_horizon_projection_table(
                scaling_confirmation_curves,
                horizons_seconds=(PROJECTION_TARGET_SECONDS, SCALING_TARGET_SECONDS),
                calibration=None,
                truth_curves=truth_curves,
                winner_probability_threshold=args.winner_probability_threshold,
                projected_margin_threshold=args.projected_margin_threshold,
            )
            scaling_confirmation_control_decision = resolve_horizon_control_decision(
                scaling_confirmation_horizon_decisions,
                target_seconds=SCALING_TARGET_SECONDS,
            )
            scaling_confirmation_next_horizon = asdict(
                suggest_next_horizon(
                    scaling_confirmation_horizon_rows,
                    scaling_confirmation_horizon_decisions,
                    target_seconds=SCALING_TARGET_SECONDS,
                    candidate_horizons=(PROJECTION_TARGET_SECONDS, SCALING_TARGET_SECONDS),
                )
            )
            scaling_confirmation_target_rows = [
                asdict(row)
                for row in rows_for_horizon(scaling_confirmation_horizon_rows, SCALING_TARGET_SECONDS)
            ]
            scaling_confirmation_phase_winner = (
                scaling_confirmation_target_rows[0]
                if scaling_confirmation_target_rows
                else projected_row_to_dict(scaling_confirmation_estimates[0]) if scaling_confirmation_estimates else None
            )
            scaling_confirmation_phase = save_phase(
                output_dir,
                "scaling_confirmation",
                inputs=scaling_confirmation_inputs,
                payload={
                    "time_budget": scaling_confirmation_time_budget,
                    "target_seconds": SCALING_TARGET_SECONDS,
                    "probe_rows": [asdict(row) for row in scaling_confirmation_probe_rows],
                    "decision": None if scaling_confirmation_decision is None else asdict(scaling_confirmation_decision),
                    "control_decision": None if scaling_confirmation_control_decision is None else asdict(scaling_confirmation_control_decision),
                    "diagnostics": [asdict(item) for item in scaling_confirmation_diagnostics],
                    "rows": [projected_row_to_dict(item) for item in scaling_confirmation_estimates],
                    "horizon_rows": [asdict(row) for row in scaling_confirmation_horizon_rows],
                    "horizon_decisions": [asdict(item) for item in scaling_confirmation_horizon_decisions],
                    "next_horizon": scaling_confirmation_next_horizon,
                    "winner": scaling_confirmation_phase_winner,
                },
            )
            phase_done(
                "scaling confirmation",
                scaling_started,
                detail=(
                    f"leader {scaling_confirmation_phase_winner['preset'] if isinstance(scaling_confirmation_phase_winner, dict) else scaling_confirmation_presets[0]}"
                ),
            )
        else:
            phase_reused(
                "scaling confirmation",
                detail=f"leader {scaling_confirmation_phase['payload'].get('winner', {}).get('preset', scaling_confirmation_presets[0])}",
            )
        scaling_confirmation_probe_rows = [
            probe_from_dict(row)
            for row in scaling_confirmation_phase["payload"].get("probe_rows", [])
        ]
        if not scaling_confirmation_probe_rows:
            raise RuntimeError("Scaling confirmation phase is missing probe_rows; rerun with --force to regenerate it.")
        scaling_confirmation_curves = load_probe_curve_artifacts(scaling_confirmation_probe_rows)
        scaling_confirmation_estimates, scaling_confirmation_decision, scaling_confirmation_diagnostics = compare_multi_horizon_curves(
            deepest_projection_curves,
            scaling_confirmation_curves,
            target_seconds=SCALING_TARGET_SECONDS,
            calibration=None,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        scaling_confirmation_horizon_rows, scaling_confirmation_horizon_decisions = build_horizon_projection_table(
            scaling_confirmation_curves,
            horizons_seconds=(PROJECTION_TARGET_SECONDS, SCALING_TARGET_SECONDS),
            calibration=None,
            truth_curves=truth_curves,
            winner_probability_threshold=args.winner_probability_threshold,
            projected_margin_threshold=args.projected_margin_threshold,
        )
        merged_longer_horizon_rows = (
            rows_for_horizon(deepest_horizon_rows, PROJECTION_TARGET_SECONDS)
            + rows_for_horizon(scaling_confirmation_horizon_rows, SCALING_TARGET_SECONDS)
        )
        merged_longer_horizon_decisions = [
            item
            for item in (
                resolve_horizon_control_decision(
                    deepest_horizon_decisions,
                    target_seconds=PROJECTION_TARGET_SECONDS,
                ),
                resolve_horizon_control_decision(
                    scaling_confirmation_horizon_decisions,
                    target_seconds=SCALING_TARGET_SECONDS,
                ),
            )
            if item is not None
        ]
        longer_horizon_projection = summarize_longer_horizon_projection(
            merged_longer_horizon_rows,
            merged_longer_horizon_decisions,
            target_seconds=PROJECTION_TARGET_SECONDS,
            scaling_target_seconds=SCALING_TARGET_SECONDS,
        )

    scaling_candidate = select_scaling_candidate(
        projection_estimates,
        truth_curves=truth_curves,
        preset_order=presets,
        target_seconds=PROJECTION_TARGET_SECONDS,
        scaling_target_seconds=SCALING_TARGET_SECONDS,
    )

    winner_batch_audit_phase: dict | None = None
    winner_batch_profile = candidate_family.probe
    if engine.capabilities.supports_local_search:
        winner_batch_audit_started = phase_start(
            "winner batch audit",
            detail=f"preset {candidate_family.probe.preset}",
            estimate_seconds=batch_audit_time_budget,
        )
        winner_batch_audit_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "preset": candidate_family.probe.preset,
            "time_budget": batch_audit_time_budget,
            "target_seconds": PROJECTION_TARGET_SECONDS,
            "anchor_batch": {
                "device_batch_size": candidate_family.probe.device_batch_size,
                "total_batch_size": candidate_family.probe.total_batch_size,
            },
            "truth_eval_contract": truth_eval_contract,
        }
        winner_batch_audit_phase = load_phase_if_matching(
            output_dir,
            "winner_batch_audit",
            winner_batch_audit_inputs,
            force=args.force,
        )
        if winner_batch_audit_phase is None:
            batch_audit_payload = run_batch_profile_audit(
                engine=engine,
                preset=candidate_family.probe.preset,
                time_budget=batch_audit_time_budget,
                logs_dir=logs_dir,
                anchor_batch_profile=candidate_family.probe,
                eval_seq_len=None if truth_eval_contract is None else truth_eval_contract["eval_seq_len"],
                eval_tokens=None if truth_eval_contract is None else truth_eval_contract["eval_tokens"],
                eval_batch_size=None if truth_eval_contract is None else truth_eval_contract["eval_batch_size"],
            )
            winner_batch_profile, audit_rows, winner_row = select_best_batch_audit_row(
                batch_audit_payload["rows"],
                target_seconds=PROJECTION_TARGET_SECONDS,
                calibration=projection_calibration,
                truth_curves=truth_curves,
                artifact_dir=logs_dir,
            )
            winner_batch_audit_phase = save_phase(
                output_dir,
                "winner_batch_audit",
                inputs=winner_batch_audit_inputs,
                payload={
                    "time_budget": batch_audit_time_budget,
                    "target_seconds": PROJECTION_TARGET_SECONDS,
                    "anchor_rows": [asdict(row) for row in batch_audit_payload["anchor_rows"]],
                    "coarse_rows": [asdict(row) for row in batch_audit_payload["coarse_rows"]],
                    "refinement_rows": [asdict(row) for row in batch_audit_payload["refinement_rows"]],
                    "rows": [asdict(row) for row in batch_audit_payload["rows"]],
                    "audit_rows": audit_rows,
                    "winner": asdict(winner_batch_profile),
                    "winner_row": winner_row,
                    "anchor_batch": batch_audit_payload["anchor_batch"],
                },
            )
            phase_done(
                "winner batch audit",
                winner_batch_audit_started,
                detail=(
                    f"winner db={winner_batch_profile.device_batch_size}, "
                    f"tb={winner_batch_profile.total_batch_size}, "
                    f"accum={winner_batch_profile.grad_accum_steps}"
                ),
            )
        else:
            winner_batch_profile = probe_from_dict(winner_batch_audit_phase["payload"]["winner"])
            phase_reused(
                "winner batch audit",
                detail=(
                    f"winner db={winner_batch_profile.device_batch_size}, "
                    f"tb={winner_batch_profile.total_batch_size}, "
                    f"accum={winner_batch_profile.grad_accum_steps}"
                ),
            )

    if engine.capabilities.supports_local_search:
        local_seq_lens = args.local_seq_lens or default_local_seq_lens(engine, candidate_family.probe.preset, mode=mode)
        local_window_patterns = args.local_window_patterns or default_local_window_patterns(engine, candidate_family.probe.preset, mode=mode)
        local_variant_count = len(local_seq_lens) * len(local_window_patterns)
        local_started = phase_start(
            "local search",
            detail=f"{local_variant_count} shape variants for {candidate_family.probe.preset}",
            estimate_seconds=local_variant_count * local_search_time_budget,
        )
        local_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "preset": candidate_family.probe.preset,
            "time_budget": local_search_time_budget,
            "batch_profile": {
                "device_batch_size": winner_batch_profile.device_batch_size,
                "total_batch_size": winner_batch_profile.total_batch_size,
            },
            "seq_lens": local_seq_lens,
            "window_patterns": local_window_patterns,
        }
        local_phase = load_phase_if_matching(output_dir, "local_search", local_inputs, force=args.force)
        if local_phase is None:
            local_rows = run_local_search(
                engine=engine,
                preset=candidate_family.probe.preset,
                time_budget=local_search_time_budget,
                logs_dir=logs_dir,
                batch_profile=winner_batch_profile,
                seq_lens=local_seq_lens,
                window_patterns=local_window_patterns,
                eval_seq_len=None if truth_eval_contract is None else truth_eval_contract["eval_seq_len"],
                eval_tokens=None if truth_eval_contract is None else truth_eval_contract["eval_tokens"],
                eval_batch_size=None if truth_eval_contract is None else truth_eval_contract["eval_batch_size"],
            )
            best_local = select_best_local_row(local_rows)
            local_phase = save_phase(
                output_dir,
                "local_search",
                inputs=local_inputs,
                payload={"time_budget": local_search_time_budget, "rows": [asdict(row) for row in local_rows], "winner": asdict(best_local)},
            )
            phase_done(
                "local search",
                local_started,
                detail=(
                    f"winner seq={best_local.seq_len}, wp={best_local.window_pattern}, "
                    f"db={best_local.device_batch_size}, tb={best_local.total_batch_size}"
                ),
            )
        else:
            reused_winner = probe_from_dict(local_phase["payload"]["winner"])
            phase_reused(
                "local search",
                detail=(
                    f"winner seq={reused_winner.seq_len}, wp={reused_winner.window_pattern}, "
                    f"db={reused_winner.device_batch_size}, tb={reused_winner.total_batch_size}"
                ),
            )
        local_rows = [probe_from_dict(row) for row in local_phase["payload"]["rows"]]
        best_local = probe_from_dict(local_phase["payload"]["winner"])
    else:
        local_rows = [winner_batch_profile]
        best_local = winner_batch_profile

    candidate_checkpoint = output_dir / "candidate_checkpoint"
    checkpoint_probe = best_local
    if engine.capabilities.supports_checkpoint_mint:
        candidate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_started = phase_start(
            "candidate checkpoint",
            detail=f"preset {candidate_family.probe.preset}",
            estimate_seconds=eval_train_seconds,
        )
        checkpoint_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "preset": candidate_family.probe.preset,
            "time_budget": eval_train_seconds,
            "seq_len": best_local.seq_len,
            "window_pattern": best_local.window_pattern,
            "device_batch_size": best_local.device_batch_size,
            "total_batch_size": best_local.total_batch_size,
        }
        checkpoint_phase = load_phase_if_matching(output_dir, "candidate_checkpoint", checkpoint_inputs, force=args.force)
        checkpoint_meta = candidate_checkpoint / "checkpoint.json"
        if checkpoint_phase is None or not checkpoint_meta.exists():
            checkpoint_probe = run_train_probe(
                engine=engine,
                preset=candidate_family.probe.preset,
                time_budget=eval_train_seconds,
                logs_dir=logs_dir,
                stage="candidate-checkpoint",
                benchmark_skip_eval=True,
                checkpoint_path=candidate_checkpoint,
                seq_len=best_local.seq_len,
                window_pattern=best_local.window_pattern,
                device_batch_size=best_local.device_batch_size,
                total_batch_size=best_local.total_batch_size,
                no_checkpoint=False,
            )
            if checkpoint_probe.status != "ok":
                raise RuntimeError("Candidate checkpoint run failed; see logs in the output directory.")
            checkpoint_phase = save_phase(
                output_dir,
                "candidate_checkpoint",
                inputs=checkpoint_inputs,
                payload=asdict(checkpoint_probe),
            )
            phase_done(
                "candidate checkpoint",
                checkpoint_started,
                detail=str(candidate_checkpoint),
            )
        else:
            phase_reused("candidate checkpoint", detail=str(candidate_checkpoint))
        checkpoint_probe = probe_from_dict(checkpoint_phase["payload"])

    if engine.capabilities.supports_eval_calibration and engine.capabilities.supports_checkpoint_mint:
        eval_markdown_path = output_dir / "eval_rungs.md"
        eval_started = phase_start(
            "eval calibration",
            detail=f"rungs {', '.join(eval_rungs)}",
        )
        eval_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "preset": candidate_family.probe.preset,
            "checkpoint": str(candidate_checkpoint),
            "hardware_key": hardware.hardware_key,
            "rungs": eval_rungs,
            "budget_seconds": args.eval_budget_seconds,
        }
        eval_phase = load_phase_if_matching(output_dir, "eval_calibration", eval_inputs, force=args.force)
        if eval_phase is None:
            eval_payload = engine.run_eval_calibration(
                preset=candidate_family.probe.preset,
                checkpoint_dir=candidate_checkpoint,
                hardware_key=hardware.hardware_key,
                rungs=eval_rungs,
                budget_seconds=args.eval_budget_seconds,
                markdown_path=eval_markdown_path,
            )
            eval_phase = save_phase(output_dir, "eval_calibration", inputs=eval_inputs, payload=eval_payload)
            phase_done("eval calibration", eval_started, detail=str(eval_markdown_path))
        else:
            phase_reused("eval calibration", detail=str(eval_markdown_path))
        eval_payload = eval_phase["payload"]
    else:
        eval_payload = {
            "mode": "eval-calibration-unsupported",
            "preset": candidate_family.probe.preset,
            "hardware_key": hardware.hardware_key,
            "engine": engine.name,
            "rows": [],
        }

    zone_probe_by_preset = dict(coarse_by_preset)
    zone_probe_by_preset.update({item.probe.preset: item.probe for item in ranked_candidates})
    zones = classify_zones(
        presets=presets,
        candidate=candidate_family,
        probe_by_preset=zone_probe_by_preset,
    )

    candidate_telemetry = telemetry_for_preset(candidate_family.probe.preset, hardware_key=hardware.hardware_key)
    candidate_default = {
        "engine": engine.name,
        "backend_family": engine.backend_family,
        "preset": candidate_family.probe.preset,
        "seq_len": best_local.seq_len,
        "depth": best_local.depth,
        "window_pattern": best_local.window_pattern,
        "device_batch_size": best_local.device_batch_size,
        "total_batch_size": best_local.total_batch_size,
        "grad_accum_steps": best_local.grad_accum_steps,
        "eval_semantics_signature": calibration_signatures["eval_semantics_signature"],
        "runtime_shape_signature": calibration_signatures["runtime_shape_signature"],
        "family_relation_to_m5": (
            family_relation_to_m5(engine, candidate_family.probe.preset)
            if engine.name == DEFAULT_ENGINE_NAME and M5_REFERENCE_DEFAULT_PRESET in engine.preset_order()
            else "not-applicable"
        ),
        "selection_method": "lowest-val_bpb-first-with-memory-and-throughput-tiebreaks",
        "selection_confidence": {
            "eval_calibration_effective_confidence": candidate_family.effective_confidence,
            "eval_calibration_status": candidate_family.calibration_status,
            "telemetry_count": candidate_family.telemetry_count,
            "stable_rung_count": candidate_family.stable_rung_count,
            "telemetry": candidate_telemetry,
        },
        "selection_basis": {
            "family_selection_stage": "finalist-projection" if finalist_phase is not None else "projection",
            "family_selection_val_bpb": candidate_family.probe.val_bpb,
            "family_selection_steady_state_tok_per_sec": candidate_family.probe.steady_state_tok_per_sec,
            "family_selection_projected_300s_val_bpb": (
                get_projected_val_bpb(finalist_phase["payload"]["winner"])
                if finalist_phase is not None
                else get_projected_val_bpb(projection_winner)
            ),
            "family_selection_projected_winner_probability": (
                finalist_phase["payload"]["winner"].get("winner_probability")
                if finalist_phase is not None
                else projection_winner.get("winner_probability")
            ),
            "family_selection_score": candidate_family.utility_score,
            "family_selection_on_pareto_front": candidate_family.on_pareto_front,
            "family_selection_frontier_distance": candidate_family.frontier_distance,
            "family_selection_distance": candidate_family.selection_distance,
            "family_selection_quality_score": candidate_family.quality_score,
            "family_selection_throughput_score": candidate_family.throughput_score,
            "family_selection_memory_score": candidate_family.memory_score,
            "family_selection_memory_fraction": candidate_family.memory_fraction,
            "family_selection_memory_pressure_band": candidate_family.memory_pressure_band,
            "family_selection_memory_tiebreak_penalty": candidate_family.memory_tiebreak_penalty,
            "family_selection_estimated_eval_overhead_fraction": candidate_family.estimated_eval_overhead_fraction,
            "winner_batch_audit_device_batch_size": winner_batch_profile.device_batch_size,
            "winner_batch_audit_total_batch_size": winner_batch_profile.total_batch_size,
            "winner_batch_audit_grad_accum_steps": winner_batch_profile.grad_accum_steps,
            "local_search_steady_state_tok_per_sec": best_local.steady_state_tok_per_sec,
            "local_search_optimizer_percent": best_local.optimizer_percent,
            "local_search_accum_percent": best_local.accum_percent,
            "local_search_control_overhead_percent": best_local.control_overhead_percent,
            "local_search_peak_vram_mb": best_local.peak_vram_mb,
        },
    }
    if engine.capabilities.supports_eval_calibration and engine.capabilities.supports_checkpoint_mint:
        promotion_bundle = write_promotion_bundle(
            output_dir=output_dir,
            hardware=hardware,
            candidate_default=candidate_default,
            eval_payload=eval_payload,
            measured_train_seconds=eval_train_seconds,
            confidence=candidate_family.effective_confidence,
            mode=mode,
            calibration_signatures=calibration_signatures,
        )
    else:
        promotion_bundle = {
            "dir": None,
            "platform_default_promotable": True,
            "eval_calibration_promotable": False,
            "unsupported_reason": f"{engine.name} engine does not support checkpoint-backed eval calibration yet.",
        }

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "mode": mode,
        "scaling_confirmation_enabled": scaling_confirmation_enabled,
        "truth_curves_dir": None if resolved_truth_curves_dir is None else str(resolved_truth_curves_dir),
        "calibration_signatures": calibration_signatures,
        "hardware_fingerprint": asdict(hardware),
        "coarse_envelope": {
            "time_budget": coarse_time_budget,
            "rows": [asdict(row) for row in coarse_rows],
        },
        "candidate_ranking": {
            "time_budget": projection_time_budget,
            "rows": [ranked_probe_to_dict(item) for item in ranked_candidates],
            "winner": ranked_probe_to_dict(ranking_winner),
            "source": "projection-pass",
        },
        "projection_ranking": {
            "time_budget": projection_time_budget,
            "target_seconds": PROJECTION_TARGET_SECONDS,
            "rows": projection_rows,
            "horizon_rows": [asdict(row) for row in projection_horizon_rows],
            "horizon_decisions": [asdict(item) for item in projection_horizon_decisions],
            "next_horizon": projection_next_horizon,
            "winner": projection_winner,
            "decision": projection_decision,
            "control_decision": projection_control_decision,
        },
        "finalist_projection": (
            {
                "time_budget": finalist_time_budget,
                "target_seconds": PROJECTION_TARGET_SECONDS,
                "rows": finalist_phase["payload"]["rows"],
                "horizon_rows": [asdict(row) for row in finalist_horizon_rows],
                "horizon_decisions": [asdict(item) for item in finalist_horizon_decisions],
                "next_horizon": finalist_phase["payload"].get("next_horizon"),
                "winner": finalist_phase["payload"]["winner"],
                "decision": finalist_phase["payload"].get("decision"),
                "control_decision": finalist_phase["payload"].get("control_decision"),
                "diagnostics": finalist_phase["payload"].get("diagnostics", []),
            }
            if finalist_phase is not None
            else None
        ),
        "confirmation_projection": (
            {
                "time_budget": confirmation_phase["payload"]["time_budget"],
                "target_seconds": PROJECTION_TARGET_SECONDS,
                "rows": confirmation_phase["payload"]["rows"],
                "horizon_rows": [asdict(row) for row in confirmation_horizon_rows],
                "horizon_decisions": [asdict(item) for item in confirmation_horizon_decisions],
                "next_horizon": confirmation_phase["payload"].get("next_horizon"),
                "winner": confirmation_phase["payload"]["winner"],
                "decision": confirmation_phase["payload"].get("decision"),
                "control_decision": confirmation_phase["payload"].get("control_decision"),
                "diagnostics": confirmation_phase["payload"].get("diagnostics", []),
            }
            if confirmation_phase is not None
            else None
        ),
        "scaling_confirmation": (
            {
                "time_budget": scaling_confirmation_phase["payload"]["time_budget"],
                "target_seconds": SCALING_TARGET_SECONDS,
                "rows": scaling_confirmation_phase["payload"]["rows"],
                "horizon_rows": scaling_confirmation_phase["payload"]["horizon_rows"],
                "horizon_decisions": scaling_confirmation_phase["payload"]["horizon_decisions"],
                "next_horizon": scaling_confirmation_phase["payload"].get("next_horizon"),
                "winner": scaling_confirmation_phase["payload"]["winner"],
                "decision": scaling_confirmation_phase["payload"].get("decision"),
                "control_decision": scaling_confirmation_phase["payload"].get("control_decision"),
                "diagnostics": scaling_confirmation_phase["payload"].get("diagnostics", []),
            }
            if scaling_confirmation_phase is not None
            else None
        ),
        "batch_profile": (
            {
                "time_budget": batch_profile_time_budget,
                "anchor_preset": batch_profile_phase["payload"]["anchor_preset"],
                "coarse_rows": [
                    asdict(probe_from_dict(row))
                    for row in batch_profile_phase["payload"].get("coarse_rows", batch_profile_phase["payload"]["rows"])
                ],
                "refinement_rows": [
                    asdict(probe_from_dict(row))
                    for row in batch_profile_phase["payload"].get("refinement_rows", [])
                ],
                "extension_rows": [
                    asdict(probe_from_dict(row))
                    for row in batch_profile_phase["payload"].get("extension_rows", [])
                ],
                "rows": [asdict(probe_from_dict(row)) for row in batch_profile_phase["payload"]["rows"]],
                "winner": asdict(batch_profile_probe),
                "overrides": {
                    preset: {
                        "device_batch_size": override[0],
                        "total_batch_size": override[1],
                    }
                    for preset, override in batch_profile_overrides.items()
                },
            }
            if batch_profile_probe is not None
            else None
        ),
        "winner_batch_audit": (
            {
                "time_budget": winner_batch_audit_phase["payload"]["time_budget"],
                "target_seconds": winner_batch_audit_phase["payload"]["target_seconds"],
                "anchor_rows": [
                    asdict(probe_from_dict(row))
                    for row in winner_batch_audit_phase["payload"].get("anchor_rows", [])
                ],
                "coarse_rows": [
                    asdict(probe_from_dict(row))
                    for row in winner_batch_audit_phase["payload"].get("coarse_rows", [])
                ],
                "refinement_rows": [
                    asdict(probe_from_dict(row))
                    for row in winner_batch_audit_phase["payload"].get("refinement_rows", [])
                ],
                "rows": [
                    asdict(probe_from_dict(row))
                    for row in winner_batch_audit_phase["payload"].get("rows", [])
                ],
                "audit_rows": winner_batch_audit_phase["payload"].get("audit_rows", []),
                "winner": asdict(probe_from_dict(winner_batch_audit_phase["payload"]["winner"])),
                "winner_row": winner_batch_audit_phase["payload"].get("winner_row"),
                "anchor_batch": winner_batch_audit_phase["payload"].get("anchor_batch"),
            }
            if winner_batch_audit_phase is not None
            else None
        ),
        "local_search": {
            "time_budget": local_search_time_budget,
            "rows": [asdict(row) for row in local_rows],
            "winner": asdict(best_local),
        },
        "candidate_checkpoint": asdict(checkpoint_probe),
        "candidate_default": candidate_default,
        "longer_horizon_projection": (
            None if longer_horizon_projection is None else asdict(longer_horizon_projection)
        ),
        "scaling_candidate": None if scaling_candidate is None else asdict(scaling_candidate),
        "promotion_bundle": promotion_bundle,
        "zones": zones,
        "eval_calibration": eval_payload,
        "reference_comparison": {
            "m5": compare_to_m5_reference(
                preset=candidate_family.probe.preset,
                eval_rows=eval_payload["rows"],
                train_probe=best_local,
            ) if engine.name == DEFAULT_ENGINE_NAME else None,
            "upstream_style": compare_to_upstream_reference(
                candidate=best_local,
                coarse_by_preset=coarse_by_preset,
                zones=zones,
            ),
        },
    }

    report_path = output_dir / "report.md"
    json_path = output_dir / "report.json"
    report_started = phase_start("final report", detail=str(report_path))
    write_report(report_path, payload=payload)
    json_path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    phase_done(
        "final report",
        report_started,
        detail=(
            f"default {candidate_default['preset']} | "
            f"seq={candidate_default['seq_len']} wp={candidate_default['window_pattern']} "
            f"db={candidate_default['device_batch_size']} tb={candidate_default['total_batch_size']}"
        ),
    )
    calibration_progress(f"JSON summary: {json_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-button platform bring-up calibration for new hardware."
    )
    parser.add_argument(
        "--engine",
        choices=available_engines(),
        default=DEFAULT_ENGINE_NAME,
        help="Training/calibration engine to use for bring-up. MLX is the full-featured default; CUDA is available as a narrower reference engine.",
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_FAST, MODE_FULL),
        default=MODE_FULL,
        help="Bring-up mode. 'fast' favors a quicker first usable default; 'full' expands the search and includes a full eval rung by default.",
    )
    parser.add_argument(
        "--presets",
        type=parse_string_list,
        default=None,
        help="Comma-separated preset list to consider. Defaults come from the selected engine.",
    )
    parser.add_argument(
        "--coarse-time-budget",
        type=float,
        help="Short benchmark-skip-eval budget for the coarse preset envelope. Defaults from --mode.",
    )
    parser.add_argument(
        "--ranking-time-budget",
        type=float,
        help="Comparable training budget with eval enabled for choosing the candidate family. Defaults from --mode.",
    )
    parser.add_argument(
        "--projection-time-budget",
        type=float,
        help="All-family curve-run budget used to project which family is most likely to win at 300s. Defaults from --mode.",
    )
    parser.add_argument(
        "--finalist-time-budget",
        type=float,
        help="Longer curve-run budget for reranking the top projected candidate families when the projection stage still lacks enough signal. Defaults from --mode.",
    )
    parser.add_argument(
        "--finalist-count",
        type=int,
        help="How many top projected candidate families to rerank at the longer finalist budget. Defaults from --mode.",
    )
    parser.add_argument(
        "--batch-audit-time-budget",
        type=float,
        help="Curve-run budget for the winner batch audit that retests `db/tb` on the selected family before local search. Defaults from --mode.",
    )
    parser.add_argument(
        "--truth-curves-dir",
        help="Optional directory of completed truth-curve artifacts used to calibrate and project the 300s winner.",
    )
    parser.add_argument(
        "--enable-scaling-confirmation",
        action="store_true",
        help="Opt in to the deeper 900s scaling-confirmation head-to-head when the longer-horizon projection predicts a crossover but lacks confidence.",
    )
    parser.add_argument(
        "--winner-probability-threshold",
        type=float,
        default=0.9,
        help="Projected winner probability threshold used to stop before the finalist stage.",
    )
    parser.add_argument(
        "--projected-margin-threshold",
        type=float,
        default=0.01,
        help="Minimum projected margin between first and second place used to stop before the finalist stage.",
    )
    parser.add_argument(
        "--local-search-time-budget",
        type=float,
        help="Short benchmark-skip-eval budget for local operating-point search inside the chosen family. Defaults from --mode.",
    )
    parser.add_argument(
        "--eval-train-seconds",
        type=float,
        help="Training budget used to mint the checkpoint for final eval-rung calibration. Defaults from --mode.",
    )
    parser.add_argument(
        "--local-seq-lens",
        type=parse_int_list,
        help="Optional comma-separated seq_len candidates for local search. Defaults to the chosen preset value.",
    )
    parser.add_argument(
        "--local-window-patterns",
        type=parse_string_list,
        help="Optional comma-separated window patterns for local search. Defaults to the chosen preset value.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for logs, JSON, report, and candidate checkpoint. Defaults under results/analysis/.",
    )
    parser.add_argument(
        "--eval-rungs",
        type=parse_string_list,
        help="Comma-separated rung set for final candidate eval calibration. Defaults from --mode.",
    )
    parser.add_argument(
        "--eval-budget-seconds",
        type=lambda value: [float(item.strip()) for item in value.split(",") if item.strip()],
        default=[300.0, 1800.0, 28800.0],
        help="Comma-separated train-budget reference points used to annotate rung overhead in the final eval table.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore reusable phase artifacts in the output directory and recompute all stages.",
    )
    parser.add_argument(
        "--plain-progress",
        action="store_true",
        help="Disable the default friendly progress narration and emit the shorter technical progress messages instead.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_platform_calibration(args)
    print(json.dumps(to_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
