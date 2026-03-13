#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
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
    build_projection_calibration,
    compare_projected_curves,
    load_curve_artifact,
    load_curve_artifacts_from_dir,
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
PLATFORM_CALIBRATION_SCHEMA_VERSION = 5
PLATEAU_FRACTION = 0.99
SHARP_EDGE_DROP_FRACTION = 0.95
PROJECTION_TARGET_SECONDS = 300.0

MODE_FAST = "fast"
MODE_FULL = "full"


@dataclass(frozen=True)
class PlatformModeSpec:
    coarse_time_budget: float
    ranking_time_budget: float
    projection_time_budget: float
    finalist_time_budget: float
    finalist_count: int
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


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


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
            "projected_final_bpb": item.corrected_val_bpb,
            "projection_std": item.projection_std,
            "calibration_horizon_seconds": item.calibration_horizon_seconds,
            "calibration_sample_count": item.calibration_sample_count,
            "correction_mean": item.correction_mean,
            "projection_source": item.projection_source,
            "matched_truth_count": item.matched_truth_count,
            "winner_probability": item.winner_probability,
            "enough_signal": item.enough_signal,
        }
    )
    return payload


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


def run_local_search(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
    batch_profile: ProbeResult,
    seq_lens: list[int] | None,
    window_patterns: list[str] | None,
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
                    benchmark_skip_eval=True,
                    seq_len=seq_len,
                    window_pattern=window_pattern,
                    device_batch_size=device_batch,
                    total_batch_size=total_batch,
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
            -(row.total_batch_size),
            row.control_overhead_percent if row.control_overhead_percent is not None else float("inf"),
            row.peak_vram_mb or float("inf"),
            -(row.steady_state_tok_per_sec or 0.0),
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


def write_report(path: Path, *, payload: dict) -> None:
    fingerprint = payload["hardware_fingerprint"]
    coarse_rows = payload["coarse_envelope"]["rows"]
    ranking_rows = payload["candidate_ranking"]["rows"]
    projection_payload = payload.get("projection_ranking")
    projection_rows = projection_payload["rows"] if isinstance(projection_payload, dict) else None
    finalist_payload = payload.get("finalist_projection")
    finalist_rows = finalist_payload["rows"] if isinstance(finalist_payload, dict) else None
    local_rows = payload["local_search"]["rows"]
    candidate_default = payload["candidate_default"]
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
    if projection_rows:
        report.extend(
            [
                "## Projection Ranking",
                "",
                "All candidate families are run at a longer budget and projected against the `300s` objective using the shared truth-curve model. This is the first real decision stage: if it already has enough signal, calibration stops here; otherwise the top finalists are rerun longer.",
                "",
                markdown_table(
                    projection_rows,
                    [
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("curve_points", "Curve points"),
                        ("projected_final_bpb", "Projected 300s val_bpb"),
                        ("projection_std", "Projection std"),
                        ("winner_probability", "Winner p"),
                        ("final_val_bpb", "Observed val_bpb"),
                        ("calibration_horizon_seconds", "Truth horizon"),
                        ("projection_source", "Projection source"),
                        ("matched_truth_count", "Truth matches"),
                    ],
                ),
                "",
            ]
        )
    if finalist_rows:
        report.extend(
            [
                "## Finalist Ranking",
                "",
                "The top projected candidate families are rerun again at a longer budget and projected to `300s`. This stage is only used when the all-family projection stage still lacks enough signal.",
                "",
                markdown_table(
                    finalist_rows,
                    [
                        ("preset", "Preset"),
                        ("device_batch_size", "Device batch"),
                        ("total_batch_size", "Total batch"),
                        ("curve_points", "Curve points"),
                        ("projected_final_bpb", "Projected 300s val_bpb"),
                        ("projection_std", "Projection std"),
                        ("winner_probability", "Winner p"),
                        ("final_val_bpb", "Observed val_bpb"),
                        ("calibration_horizon_seconds", "Truth horizon"),
                        ("projection_source", "Projection source"),
                        ("matched_truth_count", "Truth matches"),
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
    mode = args.mode
    coarse_time_budget = resolved_budget(args.coarse_time_budget, mode=mode, field="coarse_time_budget")
    ranking_time_budget = resolved_budget(args.ranking_time_budget, mode=mode, field="ranking_time_budget")
    projection_time_budget = resolved_budget(args.projection_time_budget, mode=mode, field="projection_time_budget")
    finalist_time_budget = resolved_budget(args.finalist_time_budget, mode=mode, field="finalist_time_budget")
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
    truth_curves = load_truth_curves(
        truth_curves_dir=Path(args.truth_curves_dir) if args.truth_curves_dir else None,
        engine_name=engine.name,
        hardware_key=hardware.hardware_key,
        target_seconds=PROJECTION_TARGET_SECONDS,
    )
    truth_eval_contract = infer_truth_eval_contract(truth_curves)
    projection_calibration = build_projection_calibration(truth_curves, target_seconds=PROJECTION_TARGET_SECONDS)

    hardware_inputs = {
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "hardware_key": hardware.hardware_key,
    }
    hardware_phase = load_phase_if_matching(output_dir, "hardware_fingerprint", hardware_inputs, force=args.force)
    if hardware_phase is None:
        hardware_phase = save_phase(
            output_dir,
            "hardware_fingerprint",
            inputs=hardware_inputs,
            payload=asdict(hardware),
        )

    batch_profile_probe: ProbeResult | None = None
    batch_profile_phase: dict | None = None
    batch_profile_time_budget = min(5.0, ranking_time_budget)
    batch_profile_overrides: dict[str, tuple[int, int]] = {}
    if engine.capabilities.supports_local_search:
        batch_profile_anchor = choose_batch_profile_anchor_preset(engine, presets)
        batch_profile_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "anchor_preset": batch_profile_anchor,
            "time_budget": batch_profile_time_budget,
        }
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
            except Exception:
                batch_profile_phase = None
                batch_profile_probe = None
        elif batch_profile_phase is not None:
            batch_profile_probe = probe_from_dict(batch_profile_phase["payload"]["winner"])

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
                "rows": [projected_row_to_dict(item) for item in projected_rows],
                "winner": projected_row_to_dict(projected_rows[0]) if projected_rows else None,
            },
        )
        ranking_rows = projection_probe_rows
        ranked_candidates = projection_ranked_metadata
        ranking_winner = projection_ranked_metadata[0]
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
    projection_rows = projection_phase["payload"]["rows"]
    projection_winner = projection_rows[0] if projection_rows else None
    projection_decision = (
        projection_phase["payload"].get("decision")
        if isinstance(projection_phase["payload"].get("decision"), dict)
        else None
    )
    projected_presets = [row["preset"] for row in projection_rows]
    projection_probe_metadata_by_preset = {item.probe.preset: item for item in projection_ranked_metadata}

    finalist_phase: dict | None = None
    finalist_candidates = []
    finalist_presets = projected_presets[: max(1, finalist_count)]
    candidate_family = None
    if projection_decision is not None and bool(projection_decision.get("enough_signal")) and projection_winner is not None:
        candidate_family = projection_probe_metadata_by_preset[projection_winner["preset"]]
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
        }
        finalist_phase = load_phase_if_matching(output_dir, "finalist_projection", finalist_inputs, force=args.force)
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
            finalist_projected_rows, finalist_decision = compare_projected_curves(
                finalist_curves,
                target_seconds=PROJECTION_TARGET_SECONDS,
                calibration=projection_calibration,
                truth_curves=truth_curves,
                winner_probability_threshold=args.winner_probability_threshold,
                projected_margin_threshold=args.projected_margin_threshold,
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
                    "rows": [projected_row_to_dict(item) for item in finalist_projected_rows],
                    "winner": projected_row_to_dict(finalist_projected_rows[0]) if finalist_projected_rows else None,
                },
            )
        finalist_rows = finalist_phase["payload"]["rows"]
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
        finalist_probe_metadata_by_preset = {item.probe.preset: item for item in finalist_candidates}
        finalist_winner = finalist_phase["payload"]["winner"]
        if isinstance(finalist_winner, dict):
            candidate_family = finalist_probe_metadata_by_preset[finalist_winner["preset"]]
    if candidate_family is None:
        if projection_winner is None:
            raise RuntimeError("Projection ranking produced no winner.")
        candidate_family = projection_probe_metadata_by_preset[projection_winner["preset"]]

    if engine.capabilities.supports_local_search:
        local_seq_lens = args.local_seq_lens or default_local_seq_lens(engine, candidate_family.probe.preset, mode=mode)
        local_window_patterns = args.local_window_patterns or default_local_window_patterns(engine, candidate_family.probe.preset, mode=mode)
        local_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "preset": candidate_family.probe.preset,
            "time_budget": local_search_time_budget,
            "batch_profile": {
                "device_batch_size": candidate_family.probe.device_batch_size,
                "total_batch_size": candidate_family.probe.total_batch_size,
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
                batch_profile=candidate_family.probe,
                seq_lens=local_seq_lens,
                window_patterns=local_window_patterns,
            )
            best_local = select_best_local_row(local_rows)
            local_phase = save_phase(
                output_dir,
                "local_search",
                inputs=local_inputs,
                payload={"time_budget": local_search_time_budget, "rows": [asdict(row) for row in local_rows], "winner": asdict(best_local)},
            )
        local_rows = [probe_from_dict(row) for row in local_phase["payload"]["rows"]]
        best_local = probe_from_dict(local_phase["payload"]["winner"])
    else:
        local_rows = [candidate_family.probe]
        best_local = candidate_family.probe

    candidate_checkpoint = output_dir / "candidate_checkpoint"
    checkpoint_probe = best_local
    if engine.capabilities.supports_checkpoint_mint:
        candidate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
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
        checkpoint_probe = probe_from_dict(checkpoint_phase["payload"])

    if engine.capabilities.supports_eval_calibration and engine.capabilities.supports_checkpoint_mint:
        eval_markdown_path = output_dir / "eval_rungs.md"
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
                finalist_phase["payload"]["winner"]["projected_final_bpb"]
                if finalist_phase is not None
                else projection_winner["projected_final_bpb"]
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
            "winner": projection_winner,
            "decision": projection_decision,
        },
        "finalist_projection": (
            {
                "time_budget": finalist_time_budget,
                "target_seconds": PROJECTION_TARGET_SECONDS,
                "rows": finalist_phase["payload"]["rows"],
                "winner": finalist_phase["payload"]["winner"],
                "decision": finalist_phase["payload"].get("decision"),
            }
            if finalist_phase is not None
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
        "local_search": {
            "time_budget": local_search_time_budget,
            "rows": [asdict(row) for row in local_rows],
            "winner": asdict(best_local),
        },
        "candidate_checkpoint": asdict(checkpoint_probe),
        "candidate_default": candidate_default,
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
    write_report(report_path, payload=payload)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
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
        "--truth-curves-dir",
        help="Optional directory of completed truth-curve artifacts used to calibrate and project the 300s winner.",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_platform_calibration(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
