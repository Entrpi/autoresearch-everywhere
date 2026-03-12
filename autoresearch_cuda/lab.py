from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from autoresearch_cuda.config import CUDA_PRESETS
from autoresearch_cuda.lab_trace import (
    CudaKernelLab,
    auto_review_cuda_trace,
    capture_cuda_trace,
    deep_profile_cuda_trace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=os.environ.get("AUTORESEARCH_ENTRYPOINT_PROG", "kernel-lab.py"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-targets", help="List available CUDA trace-target families")

    init_parser = subparsers.add_parser("init", help="Create a CUDA starter workspace for one target family")
    init_parser.add_argument("--target", required=True)
    init_parser.add_argument("--workspace", required=True)

    bench_parser = subparsers.add_parser("bench", help="Run the fixed CUDA workspace bench harness")
    bench_parser.add_argument("--workspace", required=True)
    bench_parser.add_argument("--quick", action="store_true")
    bench_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    verify_parser = subparsers.add_parser("verify", help="Re-run the fixed CUDA workspace harness as verification")
    verify_parser.add_argument("--workspace", required=True)
    verify_parser.add_argument("--quick", action="store_true")
    verify_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    extract_parser = subparsers.add_parser(
        "extract",
        help="Instantiate a CUDA starter workspace from a trace-profile artifact",
    )
    extract_parser.add_argument("--profile", required=True)
    extract_parser.add_argument("--workspace", required=True)
    extract_parser.add_argument("--rank", type=int, default=1)

    capture_parser = subparsers.add_parser("capture", help="Capture a CUDA trainer run under Nsight Systems")
    capture_parser.add_argument("--preset", choices=tuple(CUDA_PRESETS.keys()), default="upstream")
    capture_parser.add_argument("--time-budget", type=float, default=20.0)
    capture_parser.add_argument("--output", required=True, help="Output prefix for the .nsys-rep and metadata sidecar")
    capture_parser.add_argument("--include-eval", action="store_true", help="Include final eval in the traced run")
    capture_parser.add_argument("--seq-len", type=int)
    capture_parser.add_argument("--window-pattern")
    capture_parser.add_argument("--total-batch-size", type=int)
    capture_parser.add_argument("--depth", type=int)
    capture_parser.add_argument("--device-batch-size", type=int)

    trace_profile_parser = subparsers.add_parser(
        "trace-profile",
        help="Summarize a CUDA Nsight trace into ranked kernel target families",
    )
    trace_profile_parser.add_argument("--metadata", required=True, help="Trace metadata sidecar emitted by `capture`")
    trace_profile_parser.add_argument("--output", help="Optional JSON output path for the summarized trace profile")

    deep_profile_parser = subparsers.add_parser(
        "deep-profile",
        help="Run Nsight Compute on the top trace-ranked CUDA family and classify it more deeply",
    )
    deep_profile_parser.add_argument("--trace-profile", required=True)
    deep_profile_parser.add_argument("--rank", type=int, default=1)
    deep_profile_parser.add_argument("--time-budget", type=float, default=5.0)
    deep_profile_parser.add_argument("--output", help="Optional JSON output path for the deeper diagnosis")

    orchestrate_parser = subparsers.add_parser(
        "orchestrate",
        help="Pick the next CUDA target from a trace-profile artifact",
    )
    orchestrate_parser.add_argument("--trace-profile", required=True)
    orchestrate_parser.add_argument("--workspace-root", required=True)
    orchestrate_parser.add_argument("--rank", type=int, default=1)

    evidence_parser = subparsers.add_parser("evidence", help="Summarize persisted CUDA lab evidence for one target")
    evidence_parser.add_argument("--target", required=True)
    evidence_parser.add_argument("--preset")

    promotion_parser = subparsers.add_parser(
        "promotion-check",
        help="Check whether a CUDA target has enough trace evidence to be the next workspace candidate",
    )
    promotion_parser.add_argument("--target", required=True)
    promotion_parser.add_argument("--preset")

    integration_ab_parser = subparsers.add_parser(
        "integration-ab",
        help="Run repeated CUDA trainer A/B using a directly integrated starter workspace target",
    )
    integration_ab_parser.add_argument("--workspace", required=True)
    integration_ab_parser.add_argument("--time-budget", type=float, default=20.0)
    integration_ab_parser.add_argument("--preset", choices=tuple(CUDA_PRESETS.keys()))
    integration_ab_parser.add_argument("--repeats", type=int, default=2)
    integration_ab_parser.add_argument("--benchmark-skip-eval", action="store_true")
    integration_ab_parser.add_argument("--no-checkpoint", action="store_true")

    integration_suite_parser = subparsers.add_parser(
        "integration-suite",
        help="Run CUDA trainer integration A/B across one or more presets",
    )
    integration_suite_parser.add_argument("--workspace", required=True)
    integration_suite_parser.add_argument("--time-budget", type=float, default=20.0)
    integration_suite_parser.add_argument("--preset", choices=tuple(CUDA_PRESETS.keys()))
    integration_suite_parser.add_argument(
        "--presets",
        help="Comma-separated preset list. Defaults to the chosen preset or upstream.",
    )
    integration_suite_parser.add_argument("--repeats", type=int, default=2)
    integration_suite_parser.add_argument("--benchmark-skip-eval", action="store_true")
    integration_suite_parser.add_argument("--no-checkpoint", action="store_true")

    auto_review_parser = subparsers.add_parser(
        "auto-review",
        help="Classify a summarized CUDA trace and record machine-generated review evidence",
    )
    auto_review_parser.add_argument("--trace-profile", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    lab = CudaKernelLab()

    if args.command == "list-targets":
        for target in lab.target_catalog().values():
            note = f" -- {target.notes}" if target.notes else ""
            print(f"{target.key}\t{target.status}\t{target.metric}\t{target.description}{note}")
        return

    if args.command == "init":
        workspace = lab.init_workspace(target=args.target, workspace=Path(args.workspace).expanduser())
        print(workspace)
        return

    if args.command == "bench":
        result = lab.bench_workspace(workspace=Path(args.workspace).expanduser(), quick=args.quick, device=args.device)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "verify":
        result = lab.verify_workspace(
            workspace=Path(args.workspace).expanduser(),
            quick=args.quick,
            device=args.device,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "extract":
        result = lab.extract_from_profile(
            profile_path=Path(args.profile).expanduser(),
            workspace=Path(args.workspace).expanduser(),
            rank=args.rank,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "capture":
        result = capture_cuda_trace(
            preset=args.preset,
            output=Path(args.output).expanduser(),
            time_budget=args.time_budget,
            include_eval=args.include_eval,
            seq_len=args.seq_len,
            window_pattern=args.window_pattern,
            total_batch_size=args.total_batch_size,
            depth=args.depth,
            device_batch_size=args.device_batch_size,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "trace-profile":
        result = lab.trace_profile(metadata_path=Path(args.metadata).expanduser())
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(output)
            return
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "deep-profile":
        result = deep_profile_cuda_trace(
            Path(args.trace_profile).expanduser(),
            rank=args.rank,
            time_budget=args.time_budget,
        )
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(output)
            return
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "orchestrate":
        result = lab.orchestrate_from_profile(
            profile_path=Path(args.trace_profile).expanduser(),
            workspace_root=Path(args.workspace_root).expanduser(),
            rank=args.rank,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "evidence":
        result = lab.summarize_evidence(target=args.target, preset=args.preset)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "promotion-check":
        result = lab.promotion_check(target=args.target, preset=args.preset)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "integration-ab":
        result = lab.run_integration_ab(
            workspace=Path(args.workspace).expanduser(),
            time_budget=args.time_budget,
            preset=args.preset,
            repeats=args.repeats,
            benchmark_skip_eval=args.benchmark_skip_eval,
            no_checkpoint=args.no_checkpoint,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "integration-suite":
        presets = None
        if args.presets:
            presets = tuple(item.strip() for item in args.presets.split(",") if item.strip())
        result = lab.run_integration_suite(
            workspace=Path(args.workspace).expanduser(),
            time_budget=args.time_budget,
            preset=args.preset,
            presets=presets,
            repeats=args.repeats,
            benchmark_skip_eval=args.benchmark_skip_eval,
            no_checkpoint=args.no_checkpoint,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "auto-review":
        result = auto_review_cuda_trace(Path(args.trace_profile).expanduser())
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
