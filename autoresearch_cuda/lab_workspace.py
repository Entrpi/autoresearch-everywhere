from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from autoresearch_lab.ledger import append_lab_event
from autoresearch_lab.labs import LabBenchResult, LabExtractResult


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_STARTER_TARGET_KEYS = ("launch_fusion", "norm", "loss_prelude")


LAUNCH_FUSION_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Launch fusion
Mutable file: yes

This starter target represents a launch-bound pointwise region. Replace
`kernel_fn` with a fused Triton/CUDA implementation once the reference path is
working.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


KERNEL_TARGET = "launch_fusion"


def kernel_fn(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return residual + F.silu(x + bias)
'''


NORM_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: RMSNorm
Mutable file: yes

Replace `kernel_fn` with a faster Triton/CUDA implementation once the reference
path is working.
"""

from __future__ import annotations

import torch


KERNEL_TARGET = "norm"


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    scale = torch.rsqrt(torch.mean(x32.square(), dim=-1, keepdim=True) + eps)
    return (x32 * scale * weight.float()).to(dtype=x.dtype)
'''


LOSS_PRELUDE_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Loss prelude
Mutable file: yes

Replace `kernel_fn` with a faster Triton/CUDA implementation once the reference
path is working.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


KERNEL_TARGET = "loss_prelude"


def kernel_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_bytes: torch.Tensor,
    softcap: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits32 = logits.float()
    softcapped = softcap * torch.tanh(logits32 / softcap)
    log_probs = F.log_softmax(softcapped, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    weighted = nll * token_bytes.float()
    return weighted.to(dtype=logits.dtype), weighted.sum(), token_bytes.sum()
'''


@dataclass(frozen=True)
class CudaLabCase:
    shape: tuple[int, ...]
    dtype: str
    aux: dict[str, int] | None = None


@dataclass(frozen=True)
class CudaTargetSpec:
    target: str
    template: str
    tolerance: float
    quick_cases: tuple[CudaLabCase, ...]
    full_cases: tuple[CudaLabCase, ...]
    make_inputs: Callable[[Any, CudaLabCase, Any], tuple[Any, ...]]
    reference: Callable[..., Any]


def _launch_fusion_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, c = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((b, t, c), device=device, dtype=dtype)
    bias = torch.randn((1, 1, c), device=device, dtype=dtype)
    residual = torch.randn((b, t, c), device=device, dtype=dtype)
    return x, bias, residual


def _launch_fusion_reference(torch: Any, x: Any, bias: Any, residual: Any) -> Any:
    return residual + torch.nn.functional.silu(x + bias)


def _norm_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, c = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((b, t, c), device=device, dtype=dtype)
    weight = torch.randn((c,), device=device, dtype=dtype)
    return x, weight


def _norm_reference(torch: Any, x: Any, weight: Any, eps: float = 1e-6) -> Any:
    x32 = x.float()
    scale = torch.rsqrt(torch.mean(x32.square(), dim=-1, keepdim=True) + eps)
    return (x32 * scale * weight.float()).to(dtype=x.dtype)


def _loss_prelude_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    rows, vocab = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    logits = torch.randn((rows, vocab), device=device, dtype=dtype)
    targets = torch.randint(0, vocab, (rows,), device=device, dtype=torch.long)
    token_bytes = torch.randint(1, 5, (rows,), device=device, dtype=torch.int32)
    return logits, targets, token_bytes


def _loss_prelude_reference(
    torch: Any,
    logits: Any,
    targets: Any,
    token_bytes: Any,
    softcap: float = 15.0,
) -> Any:
    logits32 = logits.float()
    softcapped = softcap * torch.tanh(logits32 / softcap)
    log_probs = torch.nn.functional.log_softmax(softcapped, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    weighted = nll * token_bytes.float()
    return weighted.to(dtype=logits.dtype), weighted.sum(), token_bytes.sum()


CUDA_WORKSPACE_TARGET_SPECS: dict[str, CudaTargetSpec] = {
    "launch_fusion": CudaTargetSpec(
        target="launch_fusion",
        template=LAUNCH_FUSION_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 1024), dtype="float32"),
            CudaLabCase(shape=(4, 512, 2048), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(16, 512, 2048), dtype="float16"),
            CudaLabCase(shape=(8, 1024, 4096), dtype="float16"),
        ),
        make_inputs=_launch_fusion_inputs,
        reference=_launch_fusion_reference,
    ),
    "norm": CudaTargetSpec(
        target="norm",
        template=NORM_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 1024), dtype="float32"),
            CudaLabCase(shape=(4, 512, 2048), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(16, 512, 2048), dtype="float16"),
            CudaLabCase(shape=(8, 1024, 4096), dtype="float16"),
        ),
        make_inputs=_norm_inputs,
        reference=_norm_reference,
    ),
    "loss_prelude": CudaTargetSpec(
        target="loss_prelude",
        template=LOSS_PRELUDE_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(512, 2048), dtype="float32"),
            CudaLabCase(shape=(1024, 4096), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(2048, 4096), dtype="float16"),
            CudaLabCase(shape=(1024, 8192), dtype="float16"),
        ),
        make_inputs=_loss_prelude_inputs,
        reference=_loss_prelude_reference,
    ),
}


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _import_torch() -> Any:
    import torch  # type: ignore

    return torch


def _resolve_device(torch: Any, requested_device: str) -> tuple[str, Any]:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")
    return requested_device, torch.device(requested_device)


def _resolve_dtype(torch: Any, dtype_name: str, device: Any) -> Any:
    if device.type == "cpu" and dtype_name in {"float16", "bfloat16"}:
        return torch.float32
    return getattr(torch, dtype_name)


def _synchronize(torch: Any, device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()


def _load_workspace_module(workspace: Path) -> Any:
    kernel_path = workspace / "kernel.py"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Workspace does not contain kernel.py: {kernel_path}")
    spec = importlib.util.spec_from_file_location(f"cuda_lab_{workspace.name}", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import workspace kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_outputs(output: Any) -> tuple[Any, ...]:
    if isinstance(output, tuple):
        return output
    return (output,)


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.element_size() * tensor.numel())


def _io_bytes(inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> int:
    total = 0
    for tensor in inputs:
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += _tensor_bytes(tensor)
    for tensor in outputs:
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += _tensor_bytes(tensor)
    return total


def _max_abs_error(torch: Any, actual: tuple[Any, ...], expected: tuple[Any, ...]) -> float:
    max_error = 0.0
    for got, want in zip(actual, expected, strict=True):
        if hasattr(got, "float") and hasattr(want, "float"):
            diff = (got.float() - want.float()).abs().max().item()
            max_error = max(max_error, float(diff))
    return max_error


def _bench_case(
    torch: Any,
    module: Any,
    spec: CudaTargetSpec,
    case: CudaLabCase,
    *,
    device_name: str,
    device: Any,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    inputs = spec.make_inputs(torch, case, device)
    with torch.no_grad():
        expected = _normalize_outputs(spec.reference(torch, *inputs))
        actual = _normalize_outputs(module.kernel_fn(*inputs))
        max_abs_error = _max_abs_error(torch, actual, expected)
        latencies_ms: list[float] = []
        for _ in range(warmup):
            _ = module.kernel_fn(*inputs)
        _synchronize(torch, device_name)
        for _ in range(iterations):
            start = time.perf_counter()
            actual = _normalize_outputs(module.kernel_fn(*inputs))
            _synchronize(torch, device_name)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        median_latency_ms = float(sorted(latencies_ms)[len(latencies_ms) // 2])
        throughput_gb_s = (_io_bytes(inputs, actual) / 1e9) / (median_latency_ms / 1000.0)
    return {
        "shape": case.shape,
        "dtype": case.dtype,
        "median_latency_ms": median_latency_ms,
        "throughput_gb_s": throughput_gb_s,
        "max_abs_error": max_abs_error,
    }


def _run_workspace_harness(
    *,
    workspace: Path,
    quick: bool,
    requested_device: str,
    event_type: str,
) -> LabBenchResult:
    start = time.perf_counter()
    metadata_path = workspace / "metadata.json"
    metadata: dict[str, Any] = {}
    target = "unknown"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target = str(metadata.get("target") or "unknown")
    try:
        torch = _import_torch()
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="missing-runtime",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": f"PyTorch is unavailable: {exc}"},
        )
    try:
        device_name, device = _resolve_device(torch, requested_device)
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="missing-runtime",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": str(exc)},
        )
    if not metadata_path.exists():
        return LabBenchResult(
            target=target,
            status="invalid-workspace",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": f"Workspace metadata is missing: {metadata_path}"},
        )
    spec = CUDA_WORKSPACE_TARGET_SPECS.get(target)
    if spec is None:
        return LabBenchResult(
            target=target,
            status="unsupported-target",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": f"No CUDA workspace harness exists for target {target!r}."},
        )
    try:
        module = _load_workspace_module(workspace)
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="load-failed",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": str(exc)},
        )
    cases = spec.quick_cases if quick else spec.full_cases
    iterations = 8 if quick else 20
    warmup = 2 if quick else 5
    try:
        case_results = [
            _bench_case(
                torch,
                module,
                spec,
                case,
                device_name=device_name,
                device=device,
                iterations=iterations,
                warmup=warmup,
            )
            for case in cases
        ]
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="execution-failed",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={
                "failure_reason": str(exc),
                "device": device_name,
                "quick": quick,
            },
        )
    max_abs_error = max(result["max_abs_error"] for result in case_results)
    median_throughput = float(sum(result["throughput_gb_s"] for result in case_results) / len(case_results))
    median_latency_ms = float(sum(result["median_latency_ms"] for result in case_results) / len(case_results))
    status = "ok" if max_abs_error <= spec.tolerance else "mismatch"
    result = LabBenchResult(
        target=target,
        status=status,
        metric_name="median_throughput_gb_s",
        metric_value=median_throughput,
        wall_seconds=time.perf_counter() - start,
        details={
            "device": device_name,
            "quick": quick,
            "case_results": case_results,
            "max_abs_error": max_abs_error,
            "tolerance": spec.tolerance,
            "median_latency_ms": median_latency_ms,
        },
    )
    append_lab_event(
        engine="cuda",
        backend_family="cuda",
        target=target,
        workspace=workspace,
        event_type=event_type,
        status=status,
        metric_name=result.metric_name,
        metric_value=result.metric_value,
        details=result.details,
        preset=metadata.get("profile_context", {}).get("preset"),
    )
    return result


def init_cuda_workspace(*, target: str, workspace: Path, profile_context: dict[str, Any] | None = None) -> Path:
    spec = CUDA_WORKSPACE_TARGET_SPECS.get(target)
    if spec is None:
        supported = ", ".join(sorted(CUDA_WORKSPACE_TARGET_SPECS))
        raise ValueError(f"CUDA starter workspace is not available for {target!r}. Supported targets: {supported}")
    workspace = workspace.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    _write_text(workspace / "kernel.py", spec.template)
    _write_json(
        workspace / "metadata.json",
        {
            "engine": "cuda",
            "backend_family": "cuda",
            "target": target,
            "status": "starter-ready",
            "metric": "throughput_gb_s",
            "profile_context": profile_context or {},
        },
    )
    return workspace


def extract_cuda_workspace_from_profile(*, profile_path: Path, workspace: Path, rank: int = 1) -> LabExtractResult:
    start = time.perf_counter()
    payload = json.loads(profile_path.expanduser().read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if rank <= 0 or rank > len(candidates):
        raise ValueError(f"rank must be between 1 and {len(candidates)}")
    candidate = candidates[rank - 1]
    target = str(candidate["target"])
    if target not in CUDA_WORKSPACE_TARGET_SPECS:
        return LabExtractResult(
            engine="cuda",
            target=target,
            workspace=str(workspace.expanduser()),
            status="starter-unavailable",
            wall_seconds=time.perf_counter() - start,
            details={
                "profile_path": str(profile_path.expanduser()),
                "rank": rank,
                "candidate": candidate,
                "supported_targets": sorted(CUDA_WORKSPACE_TARGET_SPECS),
            },
        )
    profile_context = {
        "preset": payload.get("preset"),
        "profile_path": str(profile_path.expanduser()),
        "profile_rank": rank,
        "priority_score": candidate.get("priority_score"),
        "category": candidate.get("category"),
        "rationale": candidate.get("rationale"),
    }
    init_cuda_workspace(target=target, workspace=workspace, profile_context=profile_context)
    return LabExtractResult(
        engine="cuda",
        target=target,
        workspace=str(workspace.expanduser()),
        status="ok",
        wall_seconds=time.perf_counter() - start,
        details={
            "profile_path": str(profile_path.expanduser()),
            "rank": rank,
            "candidate": candidate,
            "profile_context": profile_context,
        },
    )


def bench_cuda_workspace(*, workspace: Path, quick: bool = False, device: str = "auto") -> LabBenchResult:
    return _run_workspace_harness(
        workspace=workspace.expanduser(),
        quick=quick,
        requested_device=device,
        event_type="bench",
    )


def verify_cuda_workspace(*, workspace: Path, quick: bool = False, device: str = "auto") -> LabBenchResult:
    return _run_workspace_harness(
        workspace=workspace.expanduser(),
        quick=quick,
        requested_device=device,
        event_type="verify",
    )
