from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from autoresearch_platform.engines import DEFAULT_ENGINE_NAME, get_engine
from autoresearch_platform.entrypoints import detect_default_engine
from autoresearch_platform.lr_discovery import (
    AdaptiveLrProbe,
    AdaptiveLrSweepConfig,
    LR_DISCOVERY_LEVER_GLOBAL,
    LR_DISCOVERY_LEVERS,
    LR_DISCOVERY_MODE_STANDARD,
    LR_DISCOVERY_MODES,
    apply_discovery_lever,
    run_staged_lr_multiplier_sweep,
)
from autoresearch_platform.lr_profile import (
    LrMultipliers,
    lr_multipliers_from_mapping,
    lr_multipliers_to_dict,
)
from autoresearch_platform.streaming_eval import (
    DEFAULT_STREAMING_EVAL_INTERVAL_STEPS,
    STREAMING_EVAL_MODE_CHEAP,
    STREAMING_EVAL_MODES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LR_MULTIPLIER_FLAG_FIELDS = (
    ("lr_multiplier", "--lr-multiplier"),
    ("embedding_lr_multiplier", "--embedding-lr-multiplier"),
    ("unembedding_lr_multiplier", "--unembedding-lr-multiplier"),
    ("matrix_lr_multiplier", "--matrix-lr-multiplier"),
    ("scalar_lr_multiplier", "--scalar-lr-multiplier"),
)
LR_MULTIPLIER_SHORT_LABELS = (
    ("lr_multiplier", "global"),
    ("embedding_lr_multiplier", "embedding"),
    ("unembedding_lr_multiplier", "unembedding"),
    ("matrix_lr_multiplier", "matrix"),
    ("scalar_lr_multiplier", "scalar"),
)
PATHOLOGY_BASELINE_MULTIPLIER = 3.0


def _default_engine() -> str:
    try:
        return detect_default_engine()
    except Exception:
        return DEFAULT_ENGINE_NAME


def _multiplier_label(multiplier: float) -> str:
    return f"{multiplier:.6e}".replace("+", "").replace("-", "n").replace(".", "p")


def _default_output_dir(engine: str, preset: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "results" / "analysis" / f"lr_discovery_{engine}_{preset}_{stamp}"


def _append_lr_multiplier_args(args_list: list[str], multipliers_payload: dict[str, float]) -> None:
    for field, flag in LR_MULTIPLIER_FLAG_FIELDS:
        value = float(multipliers_payload.get(field, 1.0))
        if field == "lr_multiplier" or not math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            args_list.extend([flag, f"{value:.8g}"])


def _lr_multipliers_args(multipliers_payload: dict[str, float]) -> list[str]:
    args_list: list[str] = []
    _append_lr_multiplier_args(args_list, multipliers_payload)
    return args_list


def _format_lr_multipliers(multipliers_payload: dict[str, float]) -> str:
    parts: list[str] = []
    for field, label in LR_MULTIPLIER_SHORT_LABELS:
        value = float(multipliers_payload.get(field, 1.0))
        if field == "lr_multiplier" or not math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            parts.append(f"{label}={value:.8g}")
    return ", ".join(parts)


def _probe_label(multipliers: LrMultipliers) -> str:
    payload = lr_multipliers_to_dict(multipliers)
    parts = [f"global-{_multiplier_label(payload['lr_multiplier'])}"]
    for field, label in LR_MULTIPLIER_SHORT_LABELS[1:]:
        value = payload[field]
        if math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            continue
        parts.append(f"{label}-{_multiplier_label(value)}")
    return "__".join(parts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive LR multiplier discovery using deterministic streaming validation "
            "and AUC-based probe ranking."
        )
    )
    parser.add_argument("--engine", choices=("mlx", "cuda"), default=_default_engine())
    parser.add_argument("--preset", required=True, help="Preset family to probe.")
    parser.add_argument("--time-budget", type=float, default=60.0, help="Per-probe training budget in seconds.")
    parser.add_argument(
        "--anchor-lr-multiplier",
        type=float,
        default=1.0,
        help="Initial multiplier applied to the preset LR ratios.",
    )
    parser.add_argument(
        "--discovery-levers",
        nargs="+",
        choices=LR_DISCOVERY_LEVERS,
        default=[LR_DISCOVERY_LEVER_GLOBAL],
        help=(
            "Ordered LR multiplier axes to discover. Defaults to scalar global discovery only; "
            "add `matrix` and `unembedding` to stage extra sweeps on their grouped LR multipliers."
        ),
    )
    parser.add_argument(
        "--discovery-mode",
        choices=LR_DISCOVERY_MODES,
        default=LR_DISCOVERY_MODE_STANDARD,
        help=(
            "Discovery stop policy. `standard` keeps the current bounded sweep; "
            "`find-bowl` keeps extending outward until the best probe has pronounced degradation on both sides "
            "or the extra-probe cap is reached."
        ),
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=2.0,
        help="Jump if the extrapolated optimum is more than this many 2x steps away.",
    )
    parser.add_argument(
        "--duplicate-log-tolerance",
        type=float,
        default=0.1,
        help="Treat multipliers within this log-distance as duplicates during refinement.",
    )
    parser.add_argument(
        "--fine-refine-auc-fraction",
        type=float,
        default=AdaptiveLrSweepConfig.fine_refine_auc_fraction,
        help=(
            "If the best and runner-up probe AUCs differ by more than this fraction, "
            "run finer local probes around the incumbent."
        ),
    )
    parser.add_argument(
        "--fine-refine-initial-ratio",
        type=float,
        default=AdaptiveLrSweepConfig.fine_refine_initial_ratio,
        help="Initial multiplicative ratio for fine refinement (~1.09x by default).",
    )
    parser.add_argument(
        "--fine-refine-rounds",
        type=int,
        default=AdaptiveLrSweepConfig.fine_refine_rounds,
        help="Maximum number of local fine-refinement rounds once the winner shows enough AUC edge to justify zooming in.",
    )
    parser.add_argument(
        "--fine-duplicate-log-tolerance",
        type=float,
        default=AdaptiveLrSweepConfig.fine_duplicate_log_tolerance,
        help="Tighter duplicate tolerance used for fine local probes around the incumbent.",
    )
    parser.add_argument(
        "--near-tie-auc-fraction",
        type=float,
        default=AdaptiveLrSweepConfig.near_tie_auc_fraction,
        help=(
            "If the top probes are within this fractional AUC gap, report them as a near-tie "
            "and surface the lower LR as the longer-horizon alternative."
        ),
    )
    parser.add_argument(
        "--bowl-auc-fraction",
        type=float,
        default=AdaptiveLrSweepConfig.bowl_auc_fraction,
        help=(
            "Required fractional AUC degradation on both sides of the winner before `find-bowl` "
            "declares that a local bowl has been bracketed."
        ),
    )
    parser.add_argument(
        "--bowl-max-extra-probes",
        type=int,
        default=AdaptiveLrSweepConfig.bowl_max_extra_probes,
        help="Maximum additional outward extension probes per stage in `find-bowl` mode.",
    )
    parser.add_argument("--seq-len", type=int, help="Optional sequence-length override for all probes.")
    parser.add_argument("--window-pattern", type=str, help="Optional window-pattern override for all probes.")
    parser.add_argument("--device-batch-size", type=int, help="Optional device-batch override for all probes.")
    parser.add_argument("--total-batch-size", type=int, help="Optional total-batch override for all probes.")
    parser.add_argument(
        "--streaming-eval-interval-steps",
        type=int,
        default=DEFAULT_STREAMING_EVAL_INTERVAL_STEPS,
        help="Run one deterministic validation batch every N training steps.",
    )
    parser.add_argument(
        "--streaming-eval-mode",
        choices=STREAMING_EVAL_MODES,
        default=STREAMING_EVAL_MODE_CHEAP,
        help="Streaming eval mode for LR probes. Defaults to repeated cheap cycles for comparable AUCs.",
    )
    parser.add_argument(
        "--streaming-eval-tokens",
        type=int,
        help="Streaming-eval cycle size in tokens. Defaults to the backend trainer default.",
    )
    parser.add_argument("--streaming-eval-seq-len", type=int, help="Sequence length for streaming eval.")
    parser.add_argument("--streaming-eval-batch-size", type=int, help="Batch size for streaming eval.")
    parser.add_argument(
        "--complete-streaming-eval-cycle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let each probe run past its nominal budget until the current streaming-eval cycle completes.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory to store probe logs, histories, and summaries.")
    parser.add_argument("--json", action="store_true", help="Print the saved summary JSON at the end.")
    return parser


def _load_probe_from_history(
    *,
    multipliers: LrMultipliers,
    history_path: Path,
    stdout_path: str,
    stderr_path: str,
) -> AdaptiveLrProbe:
    payload = json.loads(history_path.read_text())
    summary = payload.get("summary", {})
    points = payload.get("points", [])
    auc = summary.get("auc")
    if auc is None:
        if len(points) == 1:
            auc = float(points[0]["batch_bpb"])
        else:
            raise RuntimeError(
                f"Streaming eval did not produce a usable AUC in {history_path}. "
                "Increase the probe budget or reduce the eval interval."
            )
    return AdaptiveLrProbe(
        lr_multiplier=float(multipliers.lr_multiplier),
        auc=float(auc),
        min_bpb=summary.get("min_batch_bpb"),
        last20_bpb=summary.get("last20_batch_bpb"),
        honest_bpb=summary.get("honest_bpb"),
        points=int(summary.get("total_points", len(points))),
        cycles_completed=int(summary.get("cycles_completed", 0)),
        history_path=str(history_path),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        lr_multipliers=lr_multipliers_to_dict(multipliers),
        pathology_triggered=bool(payload.get("probe_pathology_triggered", False)),
        pathology_reason=payload.get("probe_pathology_reason"),
    )


def _pathology_thresholds_from_baseline(
    baseline: AdaptiveLrProbe | None,
) -> tuple[float | None, float | None]:
    if baseline is None:
        return None, None
    max_auc = None
    if math.isfinite(baseline.auc):
        max_auc = baseline.auc * PATHOLOGY_BASELINE_MULTIPLIER
    baseline_bpb = next(
        (
            value
            for value in (baseline.honest_bpb, baseline.last20_bpb, baseline.min_bpb)
            if value is not None and math.isfinite(value)
        ),
        None,
    )
    max_bpb = (
        baseline_bpb * PATHOLOGY_BASELINE_MULTIPLIER
        if baseline_bpb is not None
        else None
    )
    return max_auc, max_bpb


def _annotate_probe_pathology(
    probe: AdaptiveLrProbe,
    *,
    baseline: AdaptiveLrProbe | None,
) -> AdaptiveLrProbe:
    max_auc, max_bpb = _pathology_thresholds_from_baseline(baseline)
    reasons: list[str] = []
    if max_auc is not None and math.isfinite(probe.auc) and probe.auc > max_auc:
        reasons.append(f"auc>{max_auc:.6f}")
    representative_bpb = next(
        (
            value
            for value in (probe.honest_bpb, probe.last20_bpb, probe.min_bpb)
            if value is not None and math.isfinite(value)
        ),
        None,
    )
    if max_bpb is not None and representative_bpb is not None and representative_bpb > max_bpb:
        reasons.append(f"bpb>{max_bpb:.6f}")
    if not reasons:
        return probe
    return replace(
        probe,
        pathology_triggered=True,
        pathology_reason=", ".join(reasons),
    )


def _summary_payload(args: argparse.Namespace, output_dir: Path, discovery: dict) -> dict:
    best_lr_multipliers = discovery["best_lr_multipliers"]
    recommended_train_args = [
        "--engine",
        args.engine,
        "--preset",
        args.preset,
    ]
    _append_lr_multiplier_args(recommended_train_args, best_lr_multipliers)
    if args.seq_len is not None:
        recommended_train_args.extend(["--seq-len", str(args.seq_len)])
    if args.window_pattern is not None:
        recommended_train_args.extend(["--window-pattern", args.window_pattern])
    if args.device_batch_size is not None:
        recommended_train_args.extend(["--device-batch-size", str(args.device_batch_size)])
    if args.total_batch_size is not None:
        recommended_train_args.extend(["--total-batch-size", str(args.total_batch_size)])
    payload = {
        "engine": args.engine,
        "preset": args.preset,
        "time_budget": args.time_budget,
        "anchor_lr_multiplier": args.anchor_lr_multiplier,
        "discovery_mode": args.discovery_mode,
        "discovery_levers": discovery["discovery_levers"],
        "seq_len": args.seq_len,
        "window_pattern": args.window_pattern,
        "device_batch_size": args.device_batch_size,
        "total_batch_size": args.total_batch_size,
        "streaming_eval": {
            "interval_steps": args.streaming_eval_interval_steps,
            "mode": args.streaming_eval_mode,
            "tokens": args.streaming_eval_tokens,
            "seq_len": args.streaming_eval_seq_len,
            "batch_size": args.streaming_eval_batch_size,
            "complete_cycle_on_budget": args.complete_streaming_eval_cycle,
        },
        "output_dir": str(output_dir),
        "best_lr_multipliers": best_lr_multipliers,
        "discovery": discovery,
        "recommended_train_args": recommended_train_args,
    }
    if len(discovery["stages"]) == 1:
        payload["sweep"] = discovery["stages"][0]["sweep"]
    final_stage = discovery["stages"][-1]
    near_tie = final_stage["sweep"].get("near_tie")
    if near_tie is not None and near_tie.get("recommended_for_longer_runs") is not None:
        stage_base = lr_multipliers_from_mapping(final_stage["base_lr_multipliers"])
        longer = apply_discovery_lever(
            stage_base,
            lever=final_stage["lever"],
            value=near_tie["recommended_for_longer_runs"]["lr_multiplier"],
        )
        longer_payload = lr_multipliers_to_dict(longer)
        longer_args = [
            "--engine",
            args.engine,
            "--preset",
            args.preset,
        ]
        _append_lr_multiplier_args(longer_args, longer_payload)
        if args.seq_len is not None:
            longer_args.extend(["--seq-len", str(args.seq_len)])
        if args.window_pattern is not None:
            longer_args.extend(["--window-pattern", args.window_pattern])
        if args.device_batch_size is not None:
            longer_args.extend(["--device-batch-size", str(args.device_batch_size)])
        if args.total_batch_size is not None:
            longer_args.extend(["--total-batch-size", str(args.total_batch_size)])
        payload["recommended_longer_horizon_train_args"] = longer_args
        payload["recommended_longer_horizon_lr_multipliers"] = longer_payload
    return payload


def _write_summary_markdown(payload: dict, *, path: Path) -> None:
    discovery = payload["discovery"]
    lines = [
        "# Adaptive LR Sweep",
        "",
        f"- Engine: `{payload['engine']}`",
        f"- Preset: `{payload['preset']}`",
        f"- Probe budget: `{payload['time_budget']}` seconds",
        f"- Discovery levers: `{', '.join(payload['discovery_levers'])}`",
        f"- Final LR multipliers: `{_format_lr_multipliers(payload['best_lr_multipliers'])}`",
        f"- Total probe runs: `{discovery['total_runs']}`",
        f"- Probe histories: `{payload['output_dir']}`",
        "",
    ]
    for index, stage in enumerate(discovery["stages"], start=1):
        sweep = stage["sweep"]
        best = sweep["best"]
        near_tie = sweep.get("near_tie")
        lines.extend(
            [
                f"## Stage {index}: {stage['lever']}",
                "",
                f"- Base LR multipliers: `{_format_lr_multipliers(stage['base_lr_multipliers'])}`",
                f"- Best {stage['lever']} multiplier: `{best['lr_multiplier']:.8g}`",
                f"- Best AUC: `{best['auc']:.6f}`",
                f"- Honest BPB: `{best['honest_bpb']}`",
                f"- Applied LR multipliers: `{_format_lr_multipliers(best.get('lr_multipliers') or stage['base_lr_multipliers'])}`",
                "",
                "| Lever Value | Applied Multipliers | AUC | Min BPB | Last20% | Honest BPB | Points | Cycles |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for probe in sweep["probes"]:
            lines.append(
                "| "
                f"`{probe['lr_multiplier']:.8g}` | "
                f"`{_format_lr_multipliers(probe.get('lr_multipliers') or stage['base_lr_multipliers'])}` | "
                f"`{probe['auc']:.6f}` | "
                f"`{probe['min_bpb']}` | "
                f"`{probe['last20_bpb']}` | "
                f"`{probe['honest_bpb']}` | "
                f"`{probe['points']}` | "
                f"`{probe['cycles_completed']}` |"
            )
        lines.append("")
        discarded_pathological = sweep.get("discarded_pathological_probes")
        if discarded_pathological:
            lines.extend(
                [
                    "### Pathological Probes",
                    "",
                    f"- Discarded pathological probes: `{discarded_pathological}`",
                    "",
                ]
            )
        bowl = sweep.get("bowl")
        if bowl is not None:
            lines.extend(
                [
                    "### Bowl Status",
                    "",
                    f"- Mode: `{bowl['mode']}`",
                    f"- Status: `{bowl['status']}`",
                    f"- Required side degradation: `{bowl['required_gap_fraction']:.6f}`",
                    f"- Left gap (closest qualifying-or-nearest): `{bowl['left_gap_fraction']}`",
                    f"- Right gap (closest qualifying-or-nearest): `{bowl['right_gap_fraction']}`",
                    f"- Nearest-left gap: `{bowl['nearest_left_gap_fraction']}`",
                    f"- Nearest-right gap: `{bowl['nearest_right_gap_fraction']}`",
                    f"- Extra probes used: `{bowl['extra_probes_used']}` / `{bowl['max_extra_probes']}`",
                    "",
                ]
            )
        if near_tie is not None:
            shorter = near_tie["recommended_for_shorter_runs"]
            longer = near_tie["recommended_for_longer_runs"]
            lines.extend(
                [
                    "### Near Tie",
                    "",
                    f"- Threshold: `{near_tie['auc_fraction']:.6f}` fractional AUC gap",
                    f"- Observed top-gap: `{near_tie['best_auc_gap_fraction']:.6f}`",
                    f"- Short-run pick: `{shorter['lr_multiplier']:.8g}`",
                    (
                        f"- Longer-horizon alternative: `{longer['lr_multiplier']:.8g}`"
                        if longer is not None
                        else "- Longer-horizon alternative: none distinct; the default pick is already the lower-multiplier option"
                    ),
                    f"- Note: {near_tie['note']}",
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "## Final Recommended Train Args",
            "",
            "```bash",
            " ".join(payload["recommended_train_args"]),
            "```",
            "",
        ]
    )
    if "recommended_longer_horizon_train_args" in payload:
        lines.extend(
            [
                "## Longer-Horizon Alternative Args",
                "",
                "```bash",
                " ".join(payload["recommended_longer_horizon_train_args"]),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    engine = get_engine(args.engine)
    if args.preset not in engine.preset_catalog():
        raise SystemExit(f"Unknown preset for engine {args.engine}: {args.preset}")

    output_dir = args.output_dir or _default_output_dir(args.engine, args.preset)
    logs_dir = output_dir / "logs"
    histories_dir = output_dir / "histories"
    logs_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    def run_probe(
        multipliers: LrMultipliers,
        *,
        pathology_baseline: AdaptiveLrProbe | None = None,
    ) -> AdaptiveLrProbe:
        label = _probe_label(multipliers)
        history_path = histories_dir / f"{label}.json"
        stage = f"lr-{label}"
        pathology_max_auc, pathology_max_bpb = _pathology_thresholds_from_baseline(pathology_baseline)
        result = engine.run_train_probe(
            preset=args.preset,
            time_budget=args.time_budget,
            logs_dir=logs_dir,
            stage=stage,
            benchmark_skip_eval=True,
            no_checkpoint=True,
            seq_len=args.seq_len,
            window_pattern=args.window_pattern,
            device_batch_size=args.device_batch_size,
            total_batch_size=args.total_batch_size,
            lr_multipliers=multipliers,
            streaming_eval_interval_steps=args.streaming_eval_interval_steps,
            streaming_eval_mode=args.streaming_eval_mode,
            streaming_eval_tokens=args.streaming_eval_tokens,
            streaming_eval_seq_len=args.streaming_eval_seq_len,
            streaming_eval_batch_size=args.streaming_eval_batch_size,
            streaming_eval_history_output=history_path,
            complete_streaming_eval_cycle=args.complete_streaming_eval_cycle,
            probe_pathology_max_auc=pathology_max_auc,
            probe_pathology_max_bpb=pathology_max_bpb,
        )
        if result.returncode != 0:
            tail = result.error_tail or "No stderr tail captured."
            raise SystemExit(
                f"LR probe failed for multipliers {_format_lr_multipliers(lr_multipliers_to_dict(multipliers))}.\n"
                f"stdout: {result.stdout_path}\n"
                f"stderr: {result.stderr_path}\n"
                f"{tail}"
            )
        if not history_path.exists():
            raise SystemExit(
                "LR probe for multipliers "
                f"{_format_lr_multipliers(lr_multipliers_to_dict(multipliers))} "
                f"completed without writing {history_path}."
            )
        probe = _load_probe_from_history(
            multipliers=multipliers,
            history_path=history_path,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
        )
        probe = _annotate_probe_pathology(probe, baseline=pathology_baseline)
        print(
            f"probe multipliers={_format_lr_multipliers(probe.lr_multipliers or lr_multipliers_to_dict(multipliers))} "
            f"auc={probe.auc:.6f} "
            f"min_bpb={'nan' if probe.min_bpb is None else f'{probe.min_bpb:.6f}'} "
            f"honest_bpb={'skipped' if probe.honest_bpb is None else f'{probe.honest_bpb:.6f}'} "
            f"cycles={probe.cycles_completed}"
            + (
                f" pathology={probe.pathology_reason}"
                if probe.pathology_triggered
                else ""
            )
        )
        return probe

    discovery = run_staged_lr_multiplier_sweep(
        run_probe=run_probe,
        levers=args.discovery_levers,
        config=AdaptiveLrSweepConfig(
            anchor_multiplier=args.anchor_lr_multiplier,
            discovery_mode=args.discovery_mode,
            jump_threshold=args.jump_threshold,
            duplicate_log_tolerance=args.duplicate_log_tolerance,
            fine_refine_auc_fraction=args.fine_refine_auc_fraction,
            fine_refine_initial_ratio=args.fine_refine_initial_ratio,
            fine_refine_rounds=args.fine_refine_rounds,
            fine_duplicate_log_tolerance=args.fine_duplicate_log_tolerance,
            near_tie_auc_fraction=args.near_tie_auc_fraction,
            bowl_auc_fraction=args.bowl_auc_fraction,
            bowl_max_extra_probes=args.bowl_max_extra_probes,
        ),
    )
    payload = _summary_payload(args, output_dir, discovery)
    summary_json = output_dir / "lr_sweep_summary.json"
    summary_md = output_dir / "lr_sweep_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_summary_markdown(payload, path=summary_md)

    print("---")
    print(f"best_lr_multipliers: {_format_lr_multipliers(payload['best_lr_multipliers'])}")
    print(f"total_runs: {discovery['total_runs']}")
    for stage in discovery["stages"]:
        sweep = stage["sweep"]
        best = sweep["best"]
        print(f"stage[{stage['lever']}]:")
        print(f"  base={_format_lr_multipliers(stage['base_lr_multipliers'])}")
        print(f"  best_value={best['lr_multiplier']:.8g}")
        print(f"  best_auc={best['auc']:.6f}")
        discarded_pathological = sweep.get("discarded_pathological_probes")
        if discarded_pathological:
            print(f"  discarded_pathological_probes={discarded_pathological}")
        bowl = sweep.get("bowl")
        if bowl is not None:
            print("  bowl:")
            print(
                f"    status={bowl['status']} "
                f"required_gap_fraction={bowl['required_gap_fraction']:.6f} "
                f"left_gap_fraction={bowl['left_gap_fraction']} "
                f"right_gap_fraction={bowl['right_gap_fraction']}"
            )
            print(
                f"    nearest_left_gap_fraction={bowl['nearest_left_gap_fraction']} "
                f"nearest_right_gap_fraction={bowl['nearest_right_gap_fraction']}"
            )
            print(
                f"    extra_probes_used={bowl['extra_probes_used']} "
                f"max_extra_probes={bowl['max_extra_probes']}"
            )
        near_tie = sweep.get("near_tie")
        if near_tie is not None:
            shorter = near_tie["recommended_for_shorter_runs"]
            longer = near_tie["recommended_for_longer_runs"]
            print("  near_tie:")
            print(
                f"    threshold_fraction={near_tie['auc_fraction']:.6f} "
                f"observed_gap_fraction={near_tie['best_auc_gap_fraction']:.6f}"
            )
            print(f"    short_run_pick={shorter['lr_multiplier']:.8g}")
            if longer is not None:
                print(f"    longer_horizon_alternative={longer['lr_multiplier']:.8g}")
            else:
                print("    longer_horizon_alternative=none-distinct")
            print(f"    note={near_tie['note']}")
        print("  ranked_probes:")
        for probe in sweep["probes"]:
            honest_label = "skipped" if probe["honest_bpb"] is None else f"{probe['honest_bpb']:.6f}"
            print(
                f"    value={probe['lr_multiplier']:.8g} "
                f"multipliers={_format_lr_multipliers(probe.get('lr_multipliers') or stage['base_lr_multipliers'])} "
                f"auc={probe['auc']:.6f} "
                f"honest_bpb={honest_label} "
                f"points={probe['points']}"
            )
    print(f"summary_json: {summary_json}")
    print(f"summary_md: {summary_md}")
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
