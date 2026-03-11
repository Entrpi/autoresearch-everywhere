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

from autoresearch_mlx.eval_policy import DEFAULT_EVAL_HARDWARE_KEY, EVAL_POLICY_VERSION, find_eval_calibration  # noqa: E402
from autoresearch_mlx.eval_telemetry import summarize_eval_telemetry  # noqa: E402
from autoresearch_platform.engines import (  # noqa: E402
    DEFAULT_ENGINE_NAME,
    HardwareFingerprint,
    ProbeResult,
    TrainingEngine,
    available_engines,
    get_engine,
)
from autoresearch_platform.platform_defaults import write_platform_default_cache  # noqa: E402

M5_REFERENCE_DEFAULT_PRESET = "m5-balanced"
PLATFORM_CALIBRATION_SCHEMA_VERSION = 2

MODE_FAST = "fast"
MODE_FULL = "full"


@dataclass(frozen=True)
class PlatformModeSpec:
    coarse_time_budget: float
    ranking_time_budget: float
    local_search_time_budget: float
    eval_train_seconds: float
    eval_rungs: tuple[str, ...]


MODE_SPECS = {
    MODE_FAST: PlatformModeSpec(
        coarse_time_budget=20.0,
        ranking_time_budget=120.0,
        local_search_time_budget=20.0,
        eval_train_seconds=120.0,
        eval_rungs=("cheap", "reference"),
    ),
    MODE_FULL: PlatformModeSpec(
        coarse_time_budget=60.0,
        ranking_time_budget=300.0,
        local_search_time_budget=60.0,
        eval_train_seconds=300.0,
        eval_rungs=("cheap", "reference", "full"),
    ),
}


M5_TRAIN_REFERENCE = {
    "m5-fast": {"steady_state_tok_per_sec": 70100.0, "peak_vram_mb": 174.5},
    "m5-balanced": {"steady_state_tok_per_sec": 34500.0, "peak_vram_mb": 949.9},
    "m5-large": {"steady_state_tok_per_sec": 14600.0, "peak_vram_mb": 1944.3},
    "m5-xlarge": {"steady_state_tok_per_sec": 7800.0, "peak_vram_mb": 4294.2},
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
        no_checkpoint=no_checkpoint,
    )


def default_output_dir(*, engine_name: str, hardware_key: str) -> Path:
    tag = current_timestamp_label()
    return REPO_ROOT / "results" / "analysis" / f"platform_calibration_{engine_name}_{hardware_key}_{tag}"


def select_presets(engine: TrainingEngine, requested: list[str]) -> list[str]:
    if not requested:
        return list(engine.default_platform_presets())
    allowed = list(engine.preset_order())
    selected = [preset for preset in requested if preset in engine.preset_catalog()]
    missing = [preset for preset in requested if preset not in engine.preset_catalog()]
    if missing:
        raise ValueError(f"Unknown presets: {missing}")
    return selected


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
    ranked.sort(
        key=lambda item: (
            not item.on_pareto_front,
            item.selection_distance,
            item.memory_tiebreak_penalty,
            -item.stable_rung_count,
            -item.telemetry_count,
            item.probe.val_bpb or float("inf"),
            -(item.probe.steady_state_tok_per_sec or 0.0),
            item.probe.peak_vram_mb or float("inf"),
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


def run_local_search(
    *,
    engine: TrainingEngine,
    preset: str,
    time_budget: float,
    logs_dir: Path,
    seq_lens: list[int] | None,
    window_patterns: list[str] | None,
) -> list[ProbeResult]:
    preset_config = engine.preset_catalog()[preset]
    seq_candidates = seq_lens or [preset_config.seq_len]
    window_candidates = window_patterns or [preset_config.window_pattern]
    rows: list[ProbeResult] = []
    for seq_len in seq_candidates:
        for window_pattern in window_candidates:
            for device_batch, total_batch in engine.local_batch_candidates(preset, seq_len=seq_len):
                tokens_per_fwdbwd = seq_len * device_batch
                if total_batch % tokens_per_fwdbwd != 0:
                    continue
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


def select_best_local_row(rows: list[ProbeResult]) -> ProbeResult:
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


def compare_to_m5_reference(*, preset: str, eval_rows: list[dict], train_probe: ProbeResult) -> dict | None:
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
        "## Local Search",
        "",
        "The local-search winner is chosen by a plateau rule, not raw peak throughput alone: keep any point within `1%` of the best measured steady-state tok/s, then prefer the smallest `total_batch_size`, then lower control overhead (`optimizer_percent + accum_percent`), then lower memory.",
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
    path.write_text("\n".join(report) + "\n")


def run_platform_calibration(args) -> dict:
    mode = args.mode
    coarse_time_budget = resolved_budget(args.coarse_time_budget, mode=mode, field="coarse_time_budget")
    ranking_time_budget = resolved_budget(args.ranking_time_budget, mode=mode, field="ranking_time_budget")
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

    coarse_inputs = {
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "mode": mode,
        "presets": presets,
        "time_budget": coarse_time_budget,
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

    ranking_inputs = {
        "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
        "mode": mode,
        "presets": ranking_presets,
        "time_budget": ranking_time_budget,
        "hardware_key": hardware.hardware_key,
    }
    ranking_phase = load_phase_if_matching(output_dir, "candidate_ranking", ranking_inputs, force=args.force)
    if ranking_phase is None:
        ranking_rows = [
            run_train_probe(
                engine=engine,
                preset=preset,
                time_budget=ranking_time_budget,
                logs_dir=logs_dir,
                stage="ranking",
                benchmark_skip_eval=False,
                no_checkpoint=True,
            )
            for preset in ranking_presets
        ]
        ranked_candidates = rank_candidate_families(
            ranking_rows,
            engine=engine,
            hardware_key=hardware.hardware_key,
            hardware=hardware,
            ranking_time_budget=ranking_time_budget,
        )
        ranking_phase = save_phase(
            output_dir,
            "candidate_ranking",
            inputs=ranking_inputs,
            payload={
                "time_budget": ranking_time_budget,
                "rows": [ranked_probe_to_dict(item) for item in ranked_candidates],
                "winner": ranked_probe_to_dict(ranked_candidates[0]),
            },
        )
    ranking_rows = [probe_from_dict(row) for row in ranking_phase["payload"]["rows"]]
    ranked_candidates = [
        RankedProbe(
            probe=probe_from_dict(row),
            quality_score=float(row["quality_score"]),
            throughput_score=float(row["throughput_score"]),
            memory_score=float(row["memory_score"]),
            eval_overhead_score=float(row["eval_overhead_score"]),
            telemetry_score=float(row["telemetry_score"]),
            utility_score=float(row["utility_score"]),
            on_pareto_front=bool(row["on_pareto_front"]),
            frontier_distance=float(row["frontier_distance"]),
            selection_distance=float(row["selection_distance"]),
            estimated_eval_overhead_fraction=float(row["estimated_eval_overhead_fraction"]),
            memory_fraction=float(row["memory_fraction"]) if row.get("memory_fraction") is not None else None,
            memory_pressure_band=str(row["memory_pressure_band"]),
            memory_tiebreak_penalty=float(row["memory_tiebreak_penalty"]),
            telemetry_count=int(row["telemetry_count"]),
            stable_rung_count=int(row["stable_rung_count"]),
            effective_confidence=row.get("effective_confidence"),
            calibration_status=row.get("calibration_status"),
        )
        for row in ranking_phase["payload"]["rows"]
    ]
    candidate_family = ranked_candidates[0]

    if engine.capabilities.supports_local_search:
        local_seq_lens = args.local_seq_lens or default_local_seq_lens(engine, candidate_family.probe.preset, mode=mode)
        local_window_patterns = args.local_window_patterns or default_local_window_patterns(engine, candidate_family.probe.preset, mode=mode)
        local_inputs = {
            "schema_version": PLATFORM_CALIBRATION_SCHEMA_VERSION,
            "mode": mode,
            "preset": candidate_family.probe.preset,
            "time_budget": local_search_time_budget,
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
        "selection_method": "primary-frontier-with-memory-pressure",
        "selection_confidence": {
            "eval_calibration_effective_confidence": candidate_family.effective_confidence,
            "eval_calibration_status": candidate_family.calibration_status,
            "telemetry_count": candidate_family.telemetry_count,
            "stable_rung_count": candidate_family.stable_rung_count,
            "telemetry": candidate_telemetry,
        },
        "selection_basis": {
            "ranking_val_bpb": candidate_family.probe.val_bpb,
            "ranking_steady_state_tok_per_sec": candidate_family.probe.steady_state_tok_per_sec,
            "ranking_selection_score": candidate_family.utility_score,
            "ranking_on_pareto_front": candidate_family.on_pareto_front,
            "ranking_frontier_distance": candidate_family.frontier_distance,
            "ranking_selection_distance": candidate_family.selection_distance,
            "ranking_quality_score": candidate_family.quality_score,
            "ranking_throughput_score": candidate_family.throughput_score,
            "ranking_memory_score": candidate_family.memory_score,
            "ranking_memory_fraction": candidate_family.memory_fraction,
            "ranking_memory_pressure_band": candidate_family.memory_pressure_band,
            "ranking_memory_tiebreak_penalty": candidate_family.memory_tiebreak_penalty,
            "ranking_estimated_eval_overhead_fraction": candidate_family.estimated_eval_overhead_fraction,
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
            "time_budget": ranking_time_budget,
            "rows": [ranked_probe_to_dict(item) for item in ranked_candidates],
            "winner": ranked_probe_to_dict(candidate_family),
        },
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
