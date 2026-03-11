from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from autoresearch_mlx.lab_workspace import MLXKernelLab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernel-lab.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-targets", help="List available MLX lab targets")

    init_parser = subparsers.add_parser("init", help="Create a mutable workspace for one lab target")
    init_parser.add_argument("--target", required=True)
    init_parser.add_argument("--workspace", required=True)

    bench_parser = subparsers.add_parser("bench", help="Run the fixed bench harness against a workspace")
    bench_parser.add_argument("--workspace", required=True)
    bench_parser.add_argument("--quick", action="store_true")
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

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
