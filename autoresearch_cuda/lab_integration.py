from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


LAB_INTEGRATION_WORKSPACE_ENV = "AUTORESEARCH_CUDA_LAB_INTEGRATION_WORKSPACE"
LAB_INTEGRATION_TARGET_ENV = "AUTORESEARCH_CUDA_LAB_INTEGRATION_TARGET"


SUPPORTED_INTEGRATION_TARGETS = frozenset({"norm"})
_NORM_WEIGHT_CACHE: dict[tuple[str, int | None, str, int], object] = {}


@dataclass(frozen=True)
class IntegrationWorkspace:
    workspace: Path
    target: str
    kernel_module: object


def integration_supported_targets() -> tuple[str, ...]:
    return tuple(sorted(SUPPORTED_INTEGRATION_TARGETS))


def supports_direct_integration(target: str) -> bool:
    return target in SUPPORTED_INTEGRATION_TARGETS


def integration_environment(*, workspace: Path) -> dict[str, str]:
    workspace = workspace.expanduser().resolve()
    metadata = json.loads((workspace / "metadata.json").read_text(encoding="utf-8"))
    target = str(metadata["target"])
    if not supports_direct_integration(target):
        supported = ", ".join(integration_supported_targets())
        raise ValueError(
            f"Target {target!r} does not yet support direct CUDA training integration. "
            f"Supported targets: {supported}"
        )
    return {
        LAB_INTEGRATION_WORKSPACE_ENV: str(workspace),
        LAB_INTEGRATION_TARGET_ENV: target,
    }


@lru_cache(maxsize=1)
def current_integration_workspace() -> IntegrationWorkspace | None:
    workspace_value = os.environ.get(LAB_INTEGRATION_WORKSPACE_ENV)
    if not workspace_value:
        return None
    workspace = Path(workspace_value).expanduser().resolve()
    metadata_path = workspace / "metadata.json"
    kernel_path = workspace / "kernel.py"
    if not metadata_path.exists():
        raise ValueError(f"Integration workspace {workspace} is missing metadata.json")
    if not kernel_path.exists():
        raise ValueError(f"Integration workspace {workspace} is missing kernel.py")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_target = str(metadata["target"])
    env_target = os.environ.get(LAB_INTEGRATION_TARGET_ENV, metadata_target)
    if env_target != metadata_target:
        raise ValueError(
            f"Integration workspace target mismatch: env requested {env_target!r}, "
            f"workspace metadata says {metadata_target!r}"
        )
    if not supports_direct_integration(metadata_target):
        return None
    spec = importlib.util.spec_from_file_location("autoresearch_cuda_lab_integration_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load integration kernel from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "kernel_fn"):
        raise ValueError(f"Integration workspace {workspace} does not define kernel_fn")
    return IntegrationWorkspace(workspace=workspace, target=metadata_target, kernel_module=module)


def _norm_weight_like(x):
    import torch  # type: ignore

    key = (x.device.type, x.device.index, str(x.dtype).split(".")[-1], int(x.shape[-1]))
    cached = _NORM_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = torch.ones((x.shape[-1],), device=x.device, dtype=x.dtype)
        _NORM_WEIGHT_CACHE[key] = cached
    return cached


def maybe_call_integration_target(target: str, *args):
    workspace = current_integration_workspace()
    if workspace is None or workspace.target != target:
        return None
    if target == "norm":
        x = args[0]
        eps = args[1] if len(args) > 1 else 1e-6
        return workspace.kernel_module.kernel_fn(x, _norm_weight_like(x), eps)
    return workspace.kernel_module.kernel_fn(*args)
