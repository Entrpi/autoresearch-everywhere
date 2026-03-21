from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from autoresearch_platform.engines import DEFAULT_ENGINE_NAME, get_engine
from autoresearch_platform.entrypoints import detect_default_engine
from autoresearch_platform.lr_discovery import (
    AdaptiveLrProbe,
    AdaptiveLrSweepConfig,
    run_adaptive_lr_sweep,
)
from autoresearch_platform.streaming_eval import (
    DEFAULT_STREAMING_EVAL_INTERVAL_STEPS,
    STREAMING_EVAL_MODE_CHEAP,
    STREAMING_EVAL_MODES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    multiplier: float,
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
        lr_multiplier=multiplier,
        auc=float(auc),
        min_bpb=summary.get("min_batch_bpb"),
        last20_bpb=summary.get("last20_batch_bpb"),
        honest_bpb=summary.get("honest_bpb"),
        points=int(summary.get("total_points", len(points))),
        cycles_completed=int(summary.get("cycles_completed", 0)),
        history_path=str(history_path),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _summary_payload(args: argparse.Namespace, output_dir: Path, sweep: dict) -> dict:
    recommended_train_args = [
        "--engine",
        args.engine,
        "--preset",
        args.preset,
        "--lr-multiplier",
        f"{sweep['best']['lr_multiplier']:.8g}",
    ]
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
        "sweep": sweep,
        "recommended_train_args": recommended_train_args,
    }
    near_tie = sweep.get("near_tie")
    if near_tie is not None and near_tie.get("recommended_for_longer_runs") is not None:
        longer = near_tie["recommended_for_longer_runs"]
        longer_args = [
            "--engine",
            args.engine,
            "--preset",
            args.preset,
            "--lr-multiplier",
            f"{longer['lr_multiplier']:.8g}",
        ]
        if args.seq_len is not None:
            longer_args.extend(["--seq-len", str(args.seq_len)])
        if args.window_pattern is not None:
            longer_args.extend(["--window-pattern", args.window_pattern])
        if args.device_batch_size is not None:
            longer_args.extend(["--device-batch-size", str(args.device_batch_size)])
        if args.total_batch_size is not None:
            longer_args.extend(["--total-batch-size", str(args.total_batch_size)])
        payload["recommended_longer_horizon_train_args"] = longer_args
    return payload


def _write_summary_markdown(payload: dict, *, path: Path) -> None:
    best = payload["sweep"]["best"]
    near_tie = payload["sweep"].get("near_tie")
    lines = [
        "# Adaptive LR Sweep",
        "",
        f"- Engine: `{payload['engine']}`",
        f"- Preset: `{payload['preset']}`",
        f"- Probe budget: `{payload['time_budget']}` seconds",
        f"- Best LR multiplier: `{best['lr_multiplier']:.8g}`",
        f"- Best AUC: `{best['auc']:.6f}`",
        f"- Honest BPB: `{best['honest_bpb']}`",
        f"- Probe histories: `{payload['output_dir']}`",
        "",
    ]
    if near_tie is not None:
        shorter = near_tie["recommended_for_shorter_runs"]
        longer = near_tie["recommended_for_longer_runs"]
        lines.extend(
            [
                "## Near Tie",
                "",
                f"- Threshold: `{near_tie['auc_fraction']:.6f}` fractional AUC gap",
                f"- Observed top-gap: `{near_tie['best_auc_gap_fraction']:.6f}`",
                f"- Short-run pick: `{shorter['lr_multiplier']:.8g}`",
                (
                    f"- Longer-horizon alternative: `{longer['lr_multiplier']:.8g}`"
                    if longer is not None
                    else "- Longer-horizon alternative: none distinct; the default pick is already the lower-LR option"
                ),
                f"- Note: {near_tie['note']}",
                "",
            ]
        )
    lines.extend(
        [
        "## Ranked Probes",
        "",
        "| LR Multiplier | AUC | Min BPB | Last20% | Honest BPB | Points | Cycles |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for probe in payload["sweep"]["probes"]:
        lines.append(
            "| "
            f"`{probe['lr_multiplier']:.8g}` | "
            f"`{probe['auc']:.6f}` | "
            f"`{probe['min_bpb']}` | "
            f"`{probe['last20_bpb']}` | "
            f"`{probe['honest_bpb']}` | "
            f"`{probe['points']}` | "
            f"`{probe['cycles_completed']}` |"
        )
    lines.extend(
        [
            "",
            "## Recommended Train Args",
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

    def run_probe(multiplier: float) -> AdaptiveLrProbe:
        label = _multiplier_label(multiplier)
        history_path = histories_dir / f"{label}.json"
        stage = f"lr-{label}"
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
            lr_multiplier=multiplier,
            streaming_eval_interval_steps=args.streaming_eval_interval_steps,
            streaming_eval_mode=args.streaming_eval_mode,
            streaming_eval_tokens=args.streaming_eval_tokens,
            streaming_eval_seq_len=args.streaming_eval_seq_len,
            streaming_eval_batch_size=args.streaming_eval_batch_size,
            streaming_eval_history_output=history_path,
            complete_streaming_eval_cycle=args.complete_streaming_eval_cycle,
        )
        if result.returncode != 0:
            tail = result.error_tail or "No stderr tail captured."
            raise SystemExit(
                f"LR probe failed for multiplier {multiplier:.8g}.\n"
                f"stdout: {result.stdout_path}\n"
                f"stderr: {result.stderr_path}\n"
                f"{tail}"
            )
        if not history_path.exists():
            raise SystemExit(
                f"LR probe for multiplier {multiplier:.8g} completed without writing {history_path}."
            )
        probe = _load_probe_from_history(
            multiplier=multiplier,
            history_path=history_path,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
        )
        print(
            f"probe lr_multiplier={probe.lr_multiplier:.8g} "
            f"auc={probe.auc:.6f} "
            f"min_bpb={'nan' if probe.min_bpb is None else f'{probe.min_bpb:.6f}'} "
            f"honest_bpb={'skipped' if probe.honest_bpb is None else f'{probe.honest_bpb:.6f}'} "
            f"cycles={probe.cycles_completed}"
        )
        return probe

    sweep = run_adaptive_lr_sweep(
        run_probe=run_probe,
        config=AdaptiveLrSweepConfig(
            anchor_multiplier=args.anchor_lr_multiplier,
            jump_threshold=args.jump_threshold,
            duplicate_log_tolerance=args.duplicate_log_tolerance,
            fine_refine_auc_fraction=args.fine_refine_auc_fraction,
            fine_refine_initial_ratio=args.fine_refine_initial_ratio,
            fine_refine_rounds=args.fine_refine_rounds,
            fine_duplicate_log_tolerance=args.fine_duplicate_log_tolerance,
            near_tie_auc_fraction=args.near_tie_auc_fraction,
        ),
    )
    payload = _summary_payload(args, output_dir, sweep)
    summary_json = output_dir / "lr_sweep_summary.json"
    summary_md = output_dir / "lr_sweep_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_summary_markdown(payload, path=summary_md)

    best = sweep["best"]
    print("---")
    print(f"best_lr_multiplier: {best['lr_multiplier']:.8g}")
    print(f"best_auc: {best['auc']:.6f}")
    near_tie = sweep.get("near_tie")
    if near_tie is not None:
        shorter = near_tie["recommended_for_shorter_runs"]
        longer = near_tie["recommended_for_longer_runs"]
        print("near_tie:")
        print(
            f"  threshold_fraction={near_tie['auc_fraction']:.6f} "
            f"observed_gap_fraction={near_tie['best_auc_gap_fraction']:.6f}"
        )
        print(f"  short_run_pick={shorter['lr_multiplier']:.8g}")
        if longer is not None:
            print(f"  longer_horizon_alternative={longer['lr_multiplier']:.8g}")
        else:
            print("  longer_horizon_alternative=none-distinct")
        print(f"  note={near_tie['note']}")
    print("ranked_probes:")
    for probe in sweep["probes"]:
        honest_label = "skipped" if probe["honest_bpb"] is None else f"{probe['honest_bpb']:.6f}"
        print(
            f"  lr_multiplier={probe['lr_multiplier']:.8g} "
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
