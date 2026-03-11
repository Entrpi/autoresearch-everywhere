from __future__ import annotations

import argparse
import os
import platform
import sys

from .labs import available_labs


ENTRYPOINT_TARGETS = {
    "lab": {
        "mlx": ("module", "autoresearch_mlx.lab"),
    },
}


def detect_default_lab() -> str:
    if sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}:
        return "mlx"
    raise SystemExit("Unable to infer a default lab backend. Pass --engine mlx.")


def _parse_entrypoint_args(argv: list[str]) -> tuple[str, list[str], bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", choices=available_labs())
    parser.add_argument("--list-engines", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    engine = args.engine or detect_default_lab()
    return engine, remaining, args.list_engines


def dispatch_lab_entrypoint(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    engine, remaining, list_engines = _parse_entrypoint_args(argv)
    if list_engines:
        print("\n".join(available_labs()))
        raise SystemExit(0)
    try:
        target_kind, target = ENTRYPOINT_TARGETS["lab"][engine]
    except KeyError as exc:
        raise SystemExit(f"Unsupported lab/engine combination: lab:{engine}") from exc
    env = os.environ.copy()
    env["AUTORESEARCH_ENTRYPOINT_PROG"] = "kernel-lab.py"
    if target_kind == "module":
        argv = [sys.executable, "-m", str(target), *remaining]
    else:
        argv = [sys.executable, str(target), *remaining]
    os.execve(sys.executable, argv, env)
