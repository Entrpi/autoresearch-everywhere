from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, replace
from typing import Callable

from .lr_profile import DEFAULT_LR_MULTIPLIERS, LrMultipliers, lr_multipliers_to_dict


LR_DISCOVERY_LEVER_GLOBAL = "global"
LR_DISCOVERY_LEVER_MATRIX = "matrix"
LR_DISCOVERY_LEVER_UNEMBEDDING = "unembedding"
LR_DISCOVERY_LEVERS = (
    LR_DISCOVERY_LEVER_GLOBAL,
    LR_DISCOVERY_LEVER_MATRIX,
    LR_DISCOVERY_LEVER_UNEMBEDDING,
)
LR_DISCOVERY_MODE_STANDARD = "standard"
LR_DISCOVERY_MODE_FIND_BOWL = "find-bowl"
LR_DISCOVERY_MODES = (
    LR_DISCOVERY_MODE_STANDARD,
    LR_DISCOVERY_MODE_FIND_BOWL,
)
_LR_DISCOVERY_LEVER_FIELDS = {
    LR_DISCOVERY_LEVER_GLOBAL: "lr_multiplier",
    LR_DISCOVERY_LEVER_MATRIX: "matrix_lr_multiplier",
    LR_DISCOVERY_LEVER_UNEMBEDDING: "unembedding_lr_multiplier",
}


@dataclass(frozen=True)
class AdaptiveLrProbe:
    lr_multiplier: float
    auc: float
    min_bpb: float | None
    last20_bpb: float | None
    honest_bpb: float | None
    points: int
    cycles_completed: int
    history_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    lever_name: str | None = None
    lever_value: float | None = None
    lr_multipliers: dict[str, float] | None = None
    pathology_triggered: bool = False
    pathology_reason: str | None = None


@dataclass(frozen=True)
class AdaptiveLrSweepConfig:
    anchor_multiplier: float = 1.0
    discovery_mode: str = LR_DISCOVERY_MODE_STANDARD
    jump_threshold: float = 2.0
    duplicate_log_tolerance: float = 0.1
    fine_refine_auc_fraction: float = 0.003
    fine_refine_initial_ratio: float = 2 ** (1 / 8)
    fine_refine_rounds: int = 2
    fine_duplicate_log_tolerance: float = 0.03
    near_tie_auc_fraction: float = 0.003
    bowl_auc_fraction: float = 0.01
    bowl_max_extra_probes: int = 4


def normalize_discovery_levers(levers: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not levers:
        return (LR_DISCOVERY_LEVER_GLOBAL,)
    normalized: list[str] = []
    for lever in levers:
        if lever not in _LR_DISCOVERY_LEVER_FIELDS:
            raise ValueError(f"Unknown LR discovery lever: {lever}")
        if lever in normalized:
            continue
        normalized.append(lever)
    if not normalized:
        return (LR_DISCOVERY_LEVER_GLOBAL,)
    return tuple(normalized)


def discovery_lever_field(lever: str) -> str:
    try:
        return _LR_DISCOVERY_LEVER_FIELDS[lever]
    except KeyError as exc:
        raise ValueError(f"Unknown LR discovery lever: {lever}") from exc


def apply_discovery_lever(
    base_multipliers: LrMultipliers,
    *,
    lever: str,
    value: float,
) -> LrMultipliers:
    return replace(
        base_multipliers,
        **{discovery_lever_field(lever): float(value)},
    )


def extrapolate_optimum(
    lr1: float,
    auc1: float,
    lr2: float,
    auc2: float,
) -> tuple[float | None, float]:
    log1 = math.log(lr1)
    log2 = math.log(lr2)
    slope = (auc2 - auc1) / (log2 - log1)
    if slope >= 0:
        return None, 0.0
    target_auc = auc2 * 0.95
    target_log = log2 + (target_auc - auc2) / slope
    steps_away = abs(target_log - log2) / math.log(2)
    if target_log > math.log(sys.float_info.max):
        return None, steps_away
    predicted_lr = math.exp(target_log)
    return predicted_lr, steps_away


def already_tested(multiplier: float, probes: list[AdaptiveLrProbe], *, tol: float) -> bool:
    return any(abs(math.log(probe.lr_multiplier) - math.log(multiplier)) < tol for probe in probes)


def _best_probe(probes: list[AdaptiveLrProbe]) -> AdaptiveLrProbe:
    return min(probes, key=_probe_order_key)


def _ordered_probes(probes: list[AdaptiveLrProbe]) -> list[AdaptiveLrProbe]:
    return sorted(probes, key=_probe_order_key)


def _probe_order_key(probe: AdaptiveLrProbe) -> tuple[int, float]:
    auc = probe.auc
    if not math.isfinite(auc):
        return (1, math.inf)
    return (0, auc)


def _probe_is_pathological(probe: AdaptiveLrProbe) -> bool:
    if probe.pathology_triggered:
        return True
    if not math.isfinite(probe.auc):
        return True
    for value in (probe.min_bpb, probe.last20_bpb, probe.honest_bpb):
        if value is None:
            continue
        if not math.isfinite(value):
            return True
    return False


def _ordered_distinct_probes(
    probes: list[AdaptiveLrProbe],
    *,
    duplicate_log_tolerance: float,
) -> list[AdaptiveLrProbe]:
    distinct: list[AdaptiveLrProbe] = []
    for probe in _ordered_probes(probes):
        if already_tested(probe.lr_multiplier, distinct, tol=duplicate_log_tolerance):
            continue
        distinct.append(probe)
    return distinct


def _relative_auc_gap(best: AdaptiveLrProbe, runner_up: AdaptiveLrProbe) -> float:
    scale = max(abs(best.auc), 1e-12)
    return (runner_up.auc - best.auc) / scale


def _find_best_in_log_order(
    probes: list[AdaptiveLrProbe],
    *,
    duplicate_log_tolerance: float,
) -> tuple[list[AdaptiveLrProbe], AdaptiveLrProbe, int]:
    distinct = _ordered_distinct_probes(
        probes,
        duplicate_log_tolerance=duplicate_log_tolerance,
    )
    if not distinct:
        raise ValueError("At least one probe is required.")
    best = min(distinct, key=_probe_order_key)
    by_lr = sorted(distinct, key=lambda probe: probe.lr_multiplier)
    best_index = next(
        index
        for index, probe in enumerate(by_lr)
        if math.isclose(
            probe.lr_multiplier,
            best.lr_multiplier,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )
    return by_lr, best, best_index


def _bowl_status(
    *,
    probes: list[AdaptiveLrProbe],
    duplicate_log_tolerance: float,
    bowl_auc_fraction: float,
) -> dict:
    by_lr, best, best_index = _find_best_in_log_order(
        probes,
        duplicate_log_tolerance=duplicate_log_tolerance,
    )
    left_candidates = list(reversed(by_lr[:best_index]))
    right_candidates = by_lr[best_index + 1 :]

    nearest_left_probe = left_candidates[0] if left_candidates else None
    nearest_right_probe = right_candidates[0] if right_candidates else None
    nearest_left_gap = _relative_auc_gap(best, nearest_left_probe) if nearest_left_probe is not None else None
    nearest_right_gap = _relative_auc_gap(best, nearest_right_probe) if nearest_right_probe is not None else None

    qualifying_left_probe = next(
        (
            probe
            for probe in left_candidates
            if _relative_auc_gap(best, probe) >= bowl_auc_fraction
        ),
        None,
    )
    qualifying_right_probe = next(
        (
            probe
            for probe in right_candidates
            if _relative_auc_gap(best, probe) >= bowl_auc_fraction
        ),
        None,
    )

    left_probe = qualifying_left_probe or nearest_left_probe
    right_probe = qualifying_right_probe or nearest_right_probe
    left_gap = _relative_auc_gap(best, left_probe) if left_probe is not None else None
    right_gap = _relative_auc_gap(best, right_probe) if right_probe is not None else None
    left_degraded = qualifying_left_probe is not None
    right_degraded = qualifying_right_probe is not None
    return {
        "best": best,
        "best_index": best_index,
        "ordered_by_lr": by_lr,
        "left_probe": left_probe,
        "right_probe": right_probe,
        "left_gap_fraction": left_gap,
        "right_gap_fraction": right_gap,
        "nearest_left_gap_fraction": nearest_left_gap,
        "nearest_right_gap_fraction": nearest_right_gap,
        "left_degraded": left_degraded,
        "right_degraded": right_degraded,
        "bowl_found": bool(left_degraded and right_degraded),
    }


def _near_tie_probes(
    *,
    probes: list[AdaptiveLrProbe],
    auc_fraction: float,
    duplicate_log_tolerance: float,
) -> list[AdaptiveLrProbe]:
    if auc_fraction <= 0:
        return []
    ordered = _ordered_distinct_probes(
        probes,
        duplicate_log_tolerance=duplicate_log_tolerance,
    )
    if len(ordered) < 2:
        return []
    best = ordered[0]
    near_ties = [best]
    for probe in ordered[1:]:
        if _relative_auc_gap(best, probe) <= auc_fraction:
            near_ties.append(probe)
            continue
        break
    return near_ties if len(near_ties) > 1 else []


def should_run_fine_refinement(
    *,
    probes: list[AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig,
) -> bool:
    distinct = _ordered_distinct_probes(
        probes,
        duplicate_log_tolerance=config.fine_duplicate_log_tolerance,
    )
    if len(distinct) < 2:
        return False
    if config.fine_refine_auc_fraction <= 0:
        return False
    return _relative_auc_gap(distinct[0], distinct[1]) > config.fine_refine_auc_fraction


def refine_direction(
    *,
    start_multiplier: float,
    start_auc: float,
    direction: str,
    probes: list[AdaptiveLrProbe],
    run_probe: Callable[[float], AdaptiveLrProbe],
    duplicate_log_tolerance: float,
    stats: dict[str, int] | None = None,
) -> None:
    current = start_multiplier
    best_auc = start_auc
    while True:
        current = current * 2 if direction == "up" else current / 2
        if already_tested(current, probes, tol=duplicate_log_tolerance):
            current = current * 2 if direction == "up" else current / 2
            if already_tested(current, probes, tol=duplicate_log_tolerance):
                return
        probe = run_probe(current)
        if _probe_is_pathological(probe):
            if stats is not None:
                stats["discarded_pathological_probes"] += 1
            return
        probes.append(probe)
        if probe.auc < best_auc:
            best_auc = probe.auc
            continue
        return


def explore_direction(
    *,
    anchor_multiplier: float,
    anchor_auc: float,
    direction: str,
    probes: list[AdaptiveLrProbe],
    run_probe: Callable[[float], AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig,
    stats: dict[str, int] | None = None,
) -> None:
    probe_multiplier = anchor_multiplier * 2 if direction == "up" else anchor_multiplier / 2
    if already_tested(probe_multiplier, probes, tol=config.duplicate_log_tolerance):
        return
    probe = run_probe(probe_multiplier)
    probes.append(probe)
    if probe.auc >= anchor_auc:
        return

    predicted_multiplier, steps_away = extrapolate_optimum(
        anchor_multiplier,
        anchor_auc,
        probe_multiplier,
        probe.auc,
    )
    if predicted_multiplier is not None and steps_away > config.jump_threshold:
        best_direction_probe = probe
        if not already_tested(predicted_multiplier, probes, tol=config.duplicate_log_tolerance):
            jump_probe = run_probe(predicted_multiplier)
            if _probe_is_pathological(jump_probe):
                if stats is not None:
                    stats["discarded_pathological_probes"] += 1
            else:
                probes.append(jump_probe)
                if jump_probe.auc < best_direction_probe.auc:
                    best_direction_probe = jump_probe
        refine_direction(
            start_multiplier=best_direction_probe.lr_multiplier,
            start_auc=best_direction_probe.auc,
            direction=direction,
            probes=probes,
            run_probe=run_probe,
            duplicate_log_tolerance=config.duplicate_log_tolerance,
            stats=stats,
        )
        return

    refine_direction(
        start_multiplier=probe_multiplier,
        start_auc=probe.auc,
        direction=direction,
        probes=probes,
        run_probe=run_probe,
        duplicate_log_tolerance=config.duplicate_log_tolerance,
        stats=stats,
    )


def refine_locally_around_best(
    *,
    probes: list[AdaptiveLrProbe],
    run_probe: Callable[[float], AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig,
    stats: dict[str, int] | None = None,
) -> None:
    if not should_run_fine_refinement(probes=probes, config=config):
        return
    ratio = config.fine_refine_initial_ratio
    for _ in range(config.fine_refine_rounds):
        best = _best_probe(probes)
        candidates = [best.lr_multiplier / ratio, best.lr_multiplier * ratio]
        ran_any = False
        for candidate in candidates:
            if candidate <= 0:
                continue
            if already_tested(candidate, probes, tol=config.fine_duplicate_log_tolerance):
                continue
            probe = run_probe(candidate)
            if _probe_is_pathological(probe):
                if stats is not None:
                    stats["discarded_pathological_probes"] += 1
                continue
            probes.append(probe)
            ran_any = True
        if not ran_any:
            return
        if not should_run_fine_refinement(probes=probes, config=config):
            return
        ratio = math.sqrt(ratio)


def extend_until_bowl(
    *,
    probes: list[AdaptiveLrProbe],
    run_probe: Callable[[float], AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig,
    stats: dict[str, int] | None = None,
) -> dict:
    status = _bowl_status(
        probes=probes,
        duplicate_log_tolerance=config.fine_duplicate_log_tolerance,
        bowl_auc_fraction=config.bowl_auc_fraction,
    )
    extra_probes = 0
    while not status["bowl_found"] and extra_probes < config.bowl_max_extra_probes:
        by_lr = status["ordered_by_lr"]
        candidates: list[float] = []
        if not status["left_degraded"]:
            candidates.append(by_lr[0].lr_multiplier / 2)
        if not status["right_degraded"]:
            candidates.append(by_lr[-1].lr_multiplier * 2)
        ran_any = False
        for candidate in candidates:
            if candidate <= 0:
                continue
            if already_tested(candidate, probes, tol=config.duplicate_log_tolerance):
                continue
            probe = run_probe(candidate)
            extra_probes += 1
            if _probe_is_pathological(probe):
                if stats is not None:
                    stats["discarded_pathological_probes"] += 1
                continue
            probes.append(probe)
            ran_any = True
            if extra_probes >= config.bowl_max_extra_probes:
                break
        if not ran_any:
            break
        status = _bowl_status(
            probes=probes,
            duplicate_log_tolerance=config.fine_duplicate_log_tolerance,
            bowl_auc_fraction=config.bowl_auc_fraction,
        )
    status["extra_probes_used"] = extra_probes
    status["mode"] = config.discovery_mode
    status["required_gap_fraction"] = config.bowl_auc_fraction
    status["max_extra_probes"] = config.bowl_max_extra_probes
    status["status"] = (
        "bowl-confirmed"
        if status["bowl_found"]
        else "budget-exhausted-no-bowl"
        if extra_probes >= config.bowl_max_extra_probes
        else "no-bowl"
    )
    return status


def run_adaptive_lr_sweep(
    *,
    run_probe: Callable[[float], AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig | None = None,
) -> dict:
    if config is None:
        config = AdaptiveLrSweepConfig()
    if config.discovery_mode not in LR_DISCOVERY_MODES:
        raise ValueError(
            f"Unknown LR discovery mode: {config.discovery_mode}. "
            f"Expected one of {LR_DISCOVERY_MODES}."
        )
    probes: list[AdaptiveLrProbe] = []
    stats = {"discarded_pathological_probes": 0}
    anchor = run_probe(config.anchor_multiplier)
    probes.append(anchor)
    explore_direction(
        anchor_multiplier=config.anchor_multiplier,
        anchor_auc=anchor.auc,
        direction="up",
        probes=probes,
        run_probe=run_probe,
        config=config,
        stats=stats,
    )
    explore_direction(
        anchor_multiplier=config.anchor_multiplier,
        anchor_auc=anchor.auc,
        direction="down",
        probes=probes,
        run_probe=run_probe,
        config=config,
        stats=stats,
    )
    refine_locally_around_best(
        probes=probes,
        run_probe=run_probe,
        config=config,
        stats=stats,
    )
    bowl = None
    if config.discovery_mode == LR_DISCOVERY_MODE_FIND_BOWL:
        bowl = extend_until_bowl(
            probes=probes,
            run_probe=run_probe,
            config=config,
            stats=stats,
        )
    ordered = _ordered_probes(probes)
    result = {
        "config": asdict(config),
        "best": asdict(ordered[0]),
        "probes": [asdict(probe) for probe in ordered],
        "total_runs": len(ordered),
    }
    if bowl is not None:
        result["bowl"] = {
            "mode": bowl["mode"],
            "status": bowl["status"],
            "required_gap_fraction": bowl["required_gap_fraction"],
            "max_extra_probes": bowl["max_extra_probes"],
            "extra_probes_used": bowl["extra_probes_used"],
            "bowl_found": bowl["bowl_found"],
            "left_gap_fraction": bowl["left_gap_fraction"],
            "right_gap_fraction": bowl["right_gap_fraction"],
            "nearest_left_gap_fraction": bowl["nearest_left_gap_fraction"],
            "nearest_right_gap_fraction": bowl["nearest_right_gap_fraction"],
            "left_degraded": bowl["left_degraded"],
            "right_degraded": bowl["right_degraded"],
            "left_probe": asdict(bowl["left_probe"]) if bowl["left_probe"] is not None else None,
            "right_probe": asdict(bowl["right_probe"]) if bowl["right_probe"] is not None else None,
        }
    if stats["discarded_pathological_probes"] > 0:
        result["discarded_pathological_probes"] = stats["discarded_pathological_probes"]
    near_tie_probes = _near_tie_probes(
        probes=ordered,
        auc_fraction=config.near_tie_auc_fraction,
        duplicate_log_tolerance=config.fine_duplicate_log_tolerance,
    )
    if near_tie_probes:
        runner_up_probe = near_tie_probes[1]
        lower_multiplier_probe = min(
            (near_tie_probes[0], runner_up_probe),
            key=lambda probe: probe.lr_multiplier,
        )
        has_distinct_longer_horizon_alternative = not math.isclose(
            lower_multiplier_probe.lr_multiplier,
            near_tie_probes[0].lr_multiplier,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        result["near_tie"] = {
            "auc_fraction": config.near_tie_auc_fraction,
            "best_auc_gap_fraction": _relative_auc_gap(near_tie_probes[0], near_tie_probes[1]),
            "candidates": [asdict(probe) for probe in near_tie_probes],
            "runner_up": asdict(runner_up_probe),
            "recommended_for_shorter_runs": asdict(near_tie_probes[0]),
            "recommended_for_longer_runs": (
                asdict(lower_multiplier_probe) if has_distinct_longer_horizon_alternative else None
            ),
            "winner_is_lower_lr": not has_distinct_longer_horizon_alternative,
            "note": (
                "The winner and runner-up are effectively tied by AUC at this probe budget. "
                + (
                    "The best-AUC probe is already the lower-multiplier member of the top pair, "
                    "so keep it as the default pick."
                    if not has_distinct_longer_horizon_alternative
                    else
                    "Prefer the best-AUC candidate for short runs; consider the lower multiplier "
                    "of the top pair for longer trainings."
                )
            ),
        }
    return result


def run_staged_lr_multiplier_sweep(
    *,
    run_probe: Callable[[LrMultipliers, AdaptiveLrProbe | None], AdaptiveLrProbe],
    levers: tuple[str, ...] | list[str] | None = None,
    config: AdaptiveLrSweepConfig | None = None,
    base_multipliers: LrMultipliers = DEFAULT_LR_MULTIPLIERS,
) -> dict:
    if config is None:
        config = AdaptiveLrSweepConfig()
    ordered_levers = normalize_discovery_levers(levers)
    current_multipliers = base_multipliers
    stages: list[dict] = []
    probe_cache: dict[LrMultipliers, AdaptiveLrProbe] = {}

    for lever in ordered_levers:
        stage_config = (
            config
            if lever == LR_DISCOVERY_LEVER_GLOBAL
            else replace(config, anchor_multiplier=1.0)
        )
        stage_base = current_multipliers
        stage_anchor_probe: AdaptiveLrProbe | None = None

        def run_stage_probe(value: float, *, _lever: str = lever, _base: LrMultipliers = stage_base) -> AdaptiveLrProbe:
            nonlocal stage_anchor_probe
            multipliers = apply_discovery_lever(_base, lever=_lever, value=value)
            raw_probe = probe_cache.get(multipliers)
            if raw_probe is None:
                raw_probe = run_probe(multipliers, pathology_baseline=stage_anchor_probe)
                probe_cache[multipliers] = raw_probe
            if (
                stage_anchor_probe is None
                and math.isclose(
                    value,
                    stage_config.anchor_multiplier,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                stage_anchor_probe = raw_probe
            return replace(
                raw_probe,
                lr_multiplier=value,
                lever_name=_lever,
                lever_value=value,
                lr_multipliers=lr_multipliers_to_dict(multipliers),
            )

        sweep = run_adaptive_lr_sweep(
            run_probe=run_stage_probe,
            config=stage_config,
        )
        stages.append(
            {
                "lever": lever,
                "lever_field": discovery_lever_field(lever),
                "base_lr_multipliers": lr_multipliers_to_dict(stage_base),
                "sweep": sweep,
            }
        )
        current_multipliers = apply_discovery_lever(
            current_multipliers,
            lever=lever,
            value=sweep["best"]["lr_multiplier"],
        )

    result = {
        "config": asdict(config),
        "discovery_levers": list(ordered_levers),
        "base_lr_multipliers": lr_multipliers_to_dict(base_multipliers),
        "best_lr_multipliers": lr_multipliers_to_dict(current_multipliers),
        "stages": stages,
        "total_runs": sum(stage["sweep"]["total_runs"] for stage in stages),
    }
    if len(stages) == 1:
        result["sweep"] = stages[0]["sweep"]
    return result
