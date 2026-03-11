from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

from .engines import available_engines


REPO_ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT_TARGETS = {
    "train": {
        "mlx": ("module", "autoresearch_mlx.train"),
        "cuda": ("module", "autoresearch_cuda.train"),
    },
    "prepare": {
        "mlx": ("module", "autoresearch_mlx.prepare"),
        "cuda": ("module", "autoresearch_cuda.prepare"),
    },
}


def detect_default_engine() -> str:
    if sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}:
        return "mlx"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        return "cuda"
    raise SystemExit("Unable to infer a default engine. Pass --engine mlx or --engine cuda.")


def _parse_entrypoint_args(argv: list[str]) -> tuple[str, list[str], bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", choices=available_engines())
    parser.add_argument("--list-engines", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    engine = args.engine or detect_default_engine()
    return engine, remaining, args.list_engines


def dispatch_entrypoint(entrypoint: str, argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    engine, remaining, list_engines = _parse_entrypoint_args(argv)
    if list_engines:
        print("\n".join(available_engines()))
        raise SystemExit(0)
    try:
        target_kind, target = ENTRYPOINT_TARGETS[entrypoint][engine]
    except KeyError as exc:
        raise SystemExit(f"Unsupported entrypoint/engine combination: {entrypoint}:{engine}") from exc
    env = os.environ.copy()
    env["AUTORESEARCH_ENTRYPOINT_PROG"] = f"{entrypoint}.py"
    if target_kind == "script":
        argv = [sys.executable, str(target), *remaining]
    elif target_kind == "module":
        argv = [sys.executable, "-m", str(target), *remaining]
    else:
        raise SystemExit(f"Unknown entrypoint target kind: {target_kind}")
    os.execve(sys.executable, argv, env)
