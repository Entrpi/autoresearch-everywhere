from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from autoresearch_mlx.lab_profile import write_profile_result
from autoresearch_mlx.lab_trace import run_capture_bench
from autoresearch_mlx.lab_workspace import MLXKernelLab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=os.environ.get("AUTORESEARCH_ENTRYPOINT_PROG", "kernel-lab.py"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-targets", help="List available MLX lab targets")

    init_parser = subparsers.add_parser("init", help="Create a mutable workspace for one lab target")
    init_parser.add_argument("--target", required=True)
    init_parser.add_argument("--workspace", required=True)

    bench_parser = subparsers.add_parser("bench", help="Run the fixed bench harness against a workspace")
    bench_parser.add_argument("--workspace", required=True)
    bench_parser.add_argument("--quick", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="Re-run the fixed harness as a verification step")
    verify_parser.add_argument("--workspace", required=True)
    verify_parser.add_argument("--quick", action="store_true")

    profile_parser = subparsers.add_parser("profile", help="Rank likely MLX lab targets for a preset")
    profile_parser.add_argument("--preset", required=True)
    profile_parser.add_argument("--top-k", type=int, default=10)
    profile_parser.add_argument("--output")

    extract_parser = subparsers.add_parser("extract", help="Instantiate a workspace from a profile artifact")
    extract_parser.add_argument("--profile", required=True)
    extract_parser.add_argument("--workspace", required=True)
    extract_parser.add_argument("--rank", type=int, default=1)

    orchestrate_parser = subparsers.add_parser("orchestrate", help="Emit the next MLX lab plan from a profile")
    orchestrate_parser.add_argument("--profile", required=True)
    orchestrate_parser.add_argument("--workspace-root", required=True)
    orchestrate_parser.add_argument("--rank", type=int, default=1)

    capture_parser = subparsers.add_parser("capture", help="Capture a workspace run as a Metal trace artifact")
    capture_parser.add_argument("--workspace", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--quick", action="store_true")

    internal_capture = subparsers.add_parser(
        "_capture-bench",
        help=argparse.SUPPRESS,
    )
    internal_capture.add_argument("--workspace", required=True)
    internal_capture.add_argument("--trace", required=True)
    internal_capture.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    lab = MLXKernelLab()

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
        result = lab.bench_workspace(workspace=Path(args.workspace).expanduser(), quick=args.quick)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "verify":
        result = lab.verify_workspace(workspace=Path(args.workspace).expanduser(), quick=args.quick)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "profile":
        result = lab.profile_targets(preset=args.preset, top_k=args.top_k)
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            write_profile_result(result, output)
            print(output)
            return
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

    if args.command == "orchestrate":
        result = lab.orchestrate_from_profile(
            profile_path=Path(args.profile).expanduser(),
            workspace_root=Path(args.workspace_root).expanduser(),
            rank=args.rank,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "capture":
        result = lab.capture_workspace(
            workspace=Path(args.workspace).expanduser(),
            output=Path(args.output).expanduser(),
            quick=args.quick,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command == "_capture-bench":
        payload = run_capture_bench(
            workspace=Path(args.workspace).expanduser(),
            trace_path=Path(args.trace).expanduser(),
            quick=args.quick,
            bench_fn=lab.bench_workspace,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
