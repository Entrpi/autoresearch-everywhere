from __future__ import annotations

import importlib.util
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from autoresearch_lab.labs import LabBenchResult, LabCapabilities, LabTarget


@dataclass(frozen=True)
class LabCase:
    shape: tuple[int, ...]
    dtype: str
    aux: dict[str, int] | None = None


@dataclass(frozen=True)
class TargetSpec:
    info: LabTarget
    template: str
    tolerance: float
    quick_cases: tuple[LabCase, ...]
    full_cases: tuple[LabCase, ...]
    make_inputs: Callable[[LabCase], tuple]
    reference: Callable[..., mx.array]
    metric_value: Callable[[LabCase, float], float]


LAYERNORM_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: LayerNorm
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a correct
MLX reference implementation. Later versions can wrap `mx.fast.metal_kernel`
or `@mx.custom_function`.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "layernorm"


def kernel_fn(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    inv = mx.rsqrt(variance + eps)
    y = (x32 - mean) * inv
    return (y * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(x.dtype)
'''


RMSNORM_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: RMSNorm
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a correct
MLX reference implementation. Later versions can wrap `mx.fast.metal_kernel`
or `@mx.custom_function`.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "rmsnorm"


def kernel_fn(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * rms * weight.astype(mx.float32)).astype(x.dtype)
'''


ROTARY_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Rotary embedding
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a correct
MLX reference implementation.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "rotary_embedding"


def kernel_fn(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)
'''


REDUCE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Reduce
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path performs a
row-wise sum reduction over the last dimension.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "reduce"


def kernel_fn(x: mx.array) -> mx.array:
    return mx.sum(x, axis=-1)
'''


SOFTMAX_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Softmax
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a stable
reference implementation over the last dimension.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "softmax"


def kernel_fn(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    shifted = x32 - mx.max(x32, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return (exp / mx.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)
'''


FUSED_MLP_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Fused MLP
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's squared-ReLU MLP block without bias terms.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "fused_mlp"


def kernel_fn(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
    hidden = x @ w1
    hidden = mx.square(mx.maximum(hidden, 0))
    return hidden @ w2
'''


def _dtype(dtype_name: str):
    if dtype_name == "float16":
        return mx.float16
    if dtype_name == "float32":
        return mx.float32
    raise ValueError(f"Unsupported dtype for MLX lab: {dtype_name}")


def _throughput_gb_s(bytes_moved: int, latency_ms: float) -> float:
    return bytes_moved / (latency_ms / 1e3) / 1e9


def _tflops(flops: float, latency_ms: float) -> float:
    return flops / (latency_ms / 1e3) / 1e12


def _rng():
    return np.random.default_rng(42)


def _mx_array(shape: tuple[int, ...], dtype_name: str, scale: float = 1.0) -> mx.array:
    values = _rng().standard_normal(shape, dtype=np.float32) * scale
    return mx.array(values, dtype=_dtype(dtype_name))


def _rmsnorm_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((dim,), case.dtype)


def _rmsnorm_ref(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * rms * weight.astype(mx.float32)).astype(x.dtype)


def _rmsnorm_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = np.dtype(np.float16 if case.dtype == "float16" else np.float32).itemsize
    return _throughput_gb_s((2 * rows * dim + dim) * itemsize, latency_ms)


def _layernorm_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((dim,), case.dtype), _mx_array((dim,), case.dtype)


def _layernorm_ref(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    inv = mx.rsqrt(variance + eps)
    y = (x32 - mean) * inv
    return (y * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(x.dtype)


def _layernorm_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = np.dtype(np.float16 if case.dtype == "float16" else np.float32).itemsize
    return _throughput_gb_s((3 * rows * dim + 2 * dim) * itemsize, latency_ms)


def _rotary_inputs(case: LabCase):
    batch, seq, heads, dim = case.shape
    x = _mx_array((batch, seq, heads, dim), case.dtype)
    half = dim // 2
    cos = _mx_array((1, seq, 1, half), case.dtype)
    sin = _mx_array((1, seq, 1, half), case.dtype)
    return x, cos, sin


def _rotary_ref(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def _rotary_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, heads, dim = case.shape
    half = dim // 2
    itemsize = np.dtype(np.float16 if case.dtype == "float16" else np.float32).itemsize
    bytes_moved = (2 * batch * seq * heads * dim + 2 * seq * half) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _reduce_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype),)


def _reduce_ref(x: mx.array) -> mx.array:
    return mx.sum(x, axis=-1)


def _reduce_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = np.dtype(np.float16 if case.dtype == "float16" else np.float32).itemsize
    return _throughput_gb_s((rows * dim + rows) * itemsize, latency_ms)


def _softmax_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype),)


def _softmax_ref(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    shifted = x32 - mx.max(x32, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return (exp / mx.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def _softmax_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = np.dtype(np.float16 if case.dtype == "float16" else np.float32).itemsize
    return _throughput_gb_s((2 * rows * dim) * itemsize, latency_ms)


def _fused_mlp_inputs(case: LabCase):
    rows, n_embd = case.shape
    hidden = 4 * n_embd
    return (
        _mx_array((rows, n_embd), case.dtype, scale=0.1),
        _mx_array((n_embd, hidden), case.dtype, scale=0.02),
        _mx_array((hidden, n_embd), case.dtype, scale=0.02),
    )


def _fused_mlp_ref(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
    hidden = x @ w1
    hidden = mx.square(mx.maximum(hidden, 0))
    return hidden @ w2


def _fused_mlp_metric(case: LabCase, latency_ms: float) -> float:
    rows, n_embd = case.shape
    hidden = 4 * n_embd
    fc1 = 2.0 * rows * n_embd * hidden
    act = 2.0 * rows * hidden
    fc2 = 2.0 * rows * hidden * n_embd
    return _tflops(fc1 + act + fc2, latency_ms)


class MLXKernelLab:
    name = "mlx"
    backend_family = "mlx"
    capabilities = LabCapabilities(
        supports_workspace_init=True,
        supports_fixed_bench=True,
        supports_profile_extract=False,
    )

    _SPECS: dict[str, TargetSpec] = {
        "rmsnorm": TargetSpec(
            info=LabTarget(
                key="rmsnorm",
                description="RMSNorm forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Good first MLX target: small, common, and replaceable with a custom Metal kernel.",
            ),
            template=RMSNORM_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((128, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((2048, 4096), "float16"),
                LabCase((512, 1024), "float32"),
            ),
            make_inputs=_rmsnorm_inputs,
            reference=_rmsnorm_ref,
            metric_value=_rmsnorm_metric,
        ),
        "layernorm": TargetSpec(
            info=LabTarget(
                key="layernorm",
                description="LayerNorm forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Natural second target after RMSNorm.",
            ),
            template=LAYERNORM_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((128, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((512, 1024), "float32"),
            ),
            make_inputs=_layernorm_inputs,
            reference=_layernorm_ref,
            metric_value=_layernorm_metric,
        ),
        "rotary_embedding": TargetSpec(
            info=LabTarget(
                key="rotary_embedding",
                description="RoPE forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Good fit for a standalone MLX kernel harness.",
            ),
            template=ROTARY_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 256, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float32"),
            ),
            make_inputs=_rotary_inputs,
            reference=_rotary_ref,
            metric_value=_rotary_metric,
        ),
        "reduce": TargetSpec(
            info=LabTarget(
                key="reduce",
                description="Reduction kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Useful low-level primitive once the harness is stable.",
            ),
            template=REDUCE_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 1024), "float16"),
                LabCase((2048, 2048), "float16"),
            ),
            full_cases=(
                LabCase((256, 512), "float16"),
                LabCase((1024, 1024), "float16"),
                LabCase((2048, 2048), "float16"),
                LabCase((1024, 1024), "float32"),
            ),
            make_inputs=_reduce_inputs,
            reference=_reduce_ref,
            metric_value=_reduce_metric,
        ),
        "softmax": TargetSpec(
            info=LabTarget(
                key="softmax",
                description="Softmax forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Numerically sensitive but common enough to justify a dedicated lab.",
            ),
            template=SOFTMAX_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((256, 512), "float16"),
                LabCase((1024, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((1024, 1024), "float32"),
            ),
            make_inputs=_softmax_inputs,
            reference=_softmax_ref,
            metric_value=_softmax_metric,
        ),
        "fused_mlp": TargetSpec(
            info=LabTarget(
                key="fused_mlp",
                description="Fused MLP lab",
                metric="throughput_tflops",
                status="starter-ready",
                notes="Higher-value target once the norm and elementwise labs are in place.",
            ),
            template=FUSED_MLP_TEMPLATE,
            tolerance=5e-2,
            quick_cases=(
                LabCase((256, 512), "float16"),
                LabCase((512, 1024), "float16"),
            ),
            full_cases=(
                LabCase((128, 256), "float16"),
                LabCase((256, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((256, 512), "float32"),
            ),
            make_inputs=_fused_mlp_inputs,
            reference=_fused_mlp_ref,
            metric_value=_fused_mlp_metric,
        ),
        "flash_attention": TargetSpec(
            info=LabTarget(
                key="flash_attention",
                description="Attention kernel lab",
                metric="throughput_tflops",
                status="deferred",
                notes="Do not start here; MLX already has optimized attention primitives and custom backward is a worse first target.",
            ),
            template="",
            tolerance=0.0,
            quick_cases=(),
            full_cases=(),
            make_inputs=lambda case: (),
            reference=lambda: mx.array(0),
            metric_value=lambda case, latency_ms: 0.0,
        ),
    }

    def target_catalog(self) -> dict[str, LabTarget]:
        return {key: spec.info for key, spec in self._SPECS.items()}

    def init_workspace(self, *, target: str, workspace: Path) -> Path:
        spec = self._get_spec(target)
        if spec.info.status != "starter-ready":
            raise ValueError(
                f"Target {target} is {spec.info.status}, not starter-ready. "
                "Use `kernel-lab.py --engine mlx list-targets` to inspect the current catalog."
            )

        workspace.mkdir(parents=True, exist_ok=True)
        metadata = {
            "engine": self.name,
            "target": target,
            "metric": spec.info.metric,
            "status": spec.info.status,
            "template_version": 1,
        }
        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (workspace / "kernel.py").write_text(spec.template, encoding="utf-8")
        (workspace / "README.md").write_text(
            "\n".join(
                [
                    "# MLX Kernel Lab Workspace",
                    "",
                    f"Target: `{target}`",
                    "",
                    "- Edit `kernel.py` only.",
                    "- Benchmark with `uv run kernel-lab.py --engine mlx bench --workspace <this-dir>`.",
                    "- The benchmark harness is fixed and lives in the main repo.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return workspace

    def bench_workspace(self, *, workspace: Path, quick: bool = False) -> LabBenchResult:
        metadata = json.loads((workspace / "metadata.json").read_text(encoding="utf-8"))
        target = metadata["target"]
        spec = self._get_spec(target)
        kernel_mod = self._load_kernel_module(workspace / "kernel.py")
        if not hasattr(kernel_mod, "kernel_fn"):
            raise ValueError("Workspace kernel.py must define kernel_fn")

        cases = spec.quick_cases if quick else spec.full_cases
        start = time.perf_counter()
        max_abs_error = 0.0
        worst_case = None
        metric_values: list[float] = []
        latencies_ms: list[float] = []

        for case in cases:
            inputs = spec.make_inputs(case)
            ref = spec.reference(*inputs)
            out = kernel_mod.kernel_fn(*inputs)
            mx.eval(ref, out)
            ref_np = np.array(ref)
            out_np = np.array(out)
            if not np.isfinite(ref_np).all() or not np.isfinite(out_np).all():
                abs_error = float("inf")
            else:
                abs_error = float(np.max(np.abs(out_np - ref_np)))
            max_abs_error = max(max_abs_error, abs_error)
            if worst_case is None or abs_error >= worst_case["max_abs_error"]:
                worst_case = {
                    "shape": list(case.shape),
                    "dtype": case.dtype,
                    "max_abs_error": abs_error,
                }

            latency_ms = self._bench_case(kernel_mod.kernel_fn, inputs)
            latencies_ms.append(latency_ms)
            metric_values.append(spec.metric_value(case, latency_ms))

        wall_seconds = time.perf_counter() - start
        status = "ok" if max_abs_error <= spec.tolerance else "fail"
        details = {
            "cases": len(cases),
            "max_abs_error": max_abs_error,
            "worst_case": worst_case,
            "median_latency_ms": statistics.median(latencies_ms),
            f"median_{spec.info.metric}": statistics.median(metric_values),
        }
        return LabBenchResult(
            target=target,
            status=status,
            metric_name=spec.info.metric,
            metric_value=statistics.median(metric_values),
            wall_seconds=wall_seconds,
            details=details,
        )

    def _get_spec(self, target: str) -> TargetSpec:
        try:
            return self._SPECS[target]
        except KeyError as exc:
            raise ValueError(f"Unknown MLX lab target: {target}") from exc

    def _load_kernel_module(self, kernel_path: Path):
        spec = importlib.util.spec_from_file_location("autoresearch_mlx_lab_kernel", kernel_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load kernel module from {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def _bench_case(self, kernel_fn, inputs: tuple) -> float:
        for _ in range(3):
            out = kernel_fn(*inputs)
            mx.eval(out)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            out = kernel_fn(*inputs)
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1e3)
        return statistics.median(times)
