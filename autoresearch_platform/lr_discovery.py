from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable


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


@dataclass(frozen=True)
class AdaptiveLrSweepConfig:
    anchor_multiplier: float = 1.0
    jump_threshold: float = 2.0
    duplicate_log_tolerance: float = 0.1
    fine_refine_auc_fraction: float = 0.003
    fine_refine_initial_ratio: float = 2 ** (1 / 8)
    fine_refine_rounds: int = 2
    fine_duplicate_log_tolerance: float = 0.03
    near_tie_auc_fraction: float = 0.003


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
    predicted_lr = math.exp(target_log)
    steps_away = abs(target_log - log2) / math.log(2)
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
        )
        return

    refine_direction(
        start_multiplier=probe_multiplier,
        start_auc=probe.auc,
        direction=direction,
        probes=probes,
        run_probe=run_probe,
        duplicate_log_tolerance=config.duplicate_log_tolerance,
    )


def refine_locally_around_best(
    *,
    probes: list[AdaptiveLrProbe],
    run_probe: Callable[[float], AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig,
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
            probes.append(run_probe(candidate))
            ran_any = True
        if not ran_any:
            return
        if not should_run_fine_refinement(probes=probes, config=config):
            return
        ratio = math.sqrt(ratio)


def run_adaptive_lr_sweep(
    *,
    run_probe: Callable[[float], AdaptiveLrProbe],
    config: AdaptiveLrSweepConfig | None = None,
) -> dict:
    if config is None:
        config = AdaptiveLrSweepConfig()
    probes: list[AdaptiveLrProbe] = []
    anchor = run_probe(config.anchor_multiplier)
    probes.append(anchor)
    explore_direction(
        anchor_multiplier=config.anchor_multiplier,
        anchor_auc=anchor.auc,
        direction="up",
        probes=probes,
        run_probe=run_probe,
        config=config,
    )
    explore_direction(
        anchor_multiplier=config.anchor_multiplier,
        anchor_auc=anchor.auc,
        direction="down",
        probes=probes,
        run_probe=run_probe,
        config=config,
    )
    refine_locally_around_best(
        probes=probes,
        run_probe=run_probe,
        config=config,
    )
    ordered = _ordered_probes(probes)
    result = {
        "config": asdict(config),
        "best": asdict(ordered[0]),
        "probes": [asdict(probe) for probe in ordered],
        "total_runs": len(ordered),
    }
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
                    "The best-AUC probe is already the lower-LR member of the top pair, "
                    "so keep it as the default pick."
                    if not has_distinct_longer_horizon_alternative
                    else
                    "Prefer the best-AUC candidate for short runs; consider the lower LR "
                    "of the top pair for longer trainings."
                )
            ),
        }
    return result
