from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from autoresearch_lab.labs import LabBenchResult, LabCapabilities, LabTarget


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
    # Starter path: correct MLX implementation. Replace this with a custom
    # Metal kernel or custom autodiff wrapper once the workspace is stable.
    rms = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return x * rms * weight
'''


class MLXKernelLab:
    name = "mlx"
    backend_family = "mlx"
    capabilities = LabCapabilities(
        supports_workspace_init=True,
        supports_fixed_bench=True,
        supports_profile_extract=False,
    )

    _TARGETS = {
        "rmsnorm": LabTarget(
            key="rmsnorm",
            description="RMSNorm forward kernel lab",
            metric="throughput_gb_s",
            status="starter-ready",
            notes="Good first MLX target: small, common, and replaceable with a custom Metal kernel.",
        ),
        "layernorm": LabTarget(
            key="layernorm",
            description="LayerNorm forward kernel lab",
            metric="throughput_gb_s",
            status="planned",
            notes="Natural second target after RMSNorm.",
        ),
        "rotary_embedding": LabTarget(
            key="rotary_embedding",
            description="RoPE forward kernel lab",
            metric="throughput_gb_s",
            status="planned",
            notes="Good fit for a standalone MLX kernel harness.",
        ),
        "reduce": LabTarget(
            key="reduce",
            description="Reduction kernel lab",
            metric="throughput_gb_s",
            status="planned",
            notes="Useful low-level primitive once the harness is stable.",
        ),
        "fused_mlp": LabTarget(
            key="fused_mlp",
            description="Fused MLP lab",
            metric="throughput_tflops",
            status="planned",
            notes="Higher-value target, but more complex than the initial norm kernels.",
        ),
        "flash_attention": LabTarget(
            key="flash_attention",
            description="Attention kernel lab",
            metric="throughput_tflops",
            status="deferred",
            notes="Do not start here; MLX already has optimized attention primitives and custom backward is a worse first target.",
        ),
    }

    def target_catalog(self) -> dict[str, LabTarget]:
        return dict(self._TARGETS)

    def init_workspace(self, *, target: str, workspace: Path) -> Path:
        if target not in self._TARGETS:
            raise ValueError(f"Unknown MLX lab target: {target}")
        target_info = self._TARGETS[target]
        if target_info.status != "starter-ready":
            raise ValueError(
                f"Target {target} is {target_info.status}, not starter-ready. "
                "Use `lab.py --engine mlx list-targets` to inspect the current catalog."
            )

        workspace.mkdir(parents=True, exist_ok=True)
        metadata = {
            "engine": self.name,
            "target": target,
            "metric": target_info.metric,
            "status": target_info.status,
        }
        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (workspace / "kernel.py").write_text(RMSNORM_TEMPLATE, encoding="utf-8")
        (workspace / "README.md").write_text(
            "\n".join(
                [
                    "# MLX Kernel Lab Workspace",
                    "",
                    f"Target: `{target}`",
                    "",
                    "- Edit `kernel.py` only.",
                    "- Benchmark with `uv run lab.py --engine mlx bench --workspace <this-dir>`.",
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
        if target != "rmsnorm":
            raise ValueError(f"Unsupported MLX lab target in workspace: {target}")

        kernel_mod = self._load_kernel_module(workspace / "kernel.py")
        if not hasattr(kernel_mod, "kernel_fn"):
            raise ValueError("Workspace kernel.py must define kernel_fn")

        cases = self._quick_cases() if quick else self._full_cases()
        start = time.perf_counter()
        max_abs_error = 0.0
        worst_case = None
        benchmark_throughputs = []
        benchmark_latencies_ms = []

        for rows, dim, dtype_name in cases:
            x, weight = self._gen_inputs(rows, dim, dtype_name)
            ref = self._rmsnorm_ref(x, weight)
            out = kernel_mod.kernel_fn(x, weight)
            mx.eval(ref, out)
            abs_error = float(np.max(np.abs(np.array(out) - np.array(ref))))
            max_abs_error = max(max_abs_error, abs_error)
            if worst_case is None or abs_error >= worst_case["max_abs_error"]:
                worst_case = {
                    "rows": rows,
                    "dim": dim,
                    "dtype": dtype_name,
                    "max_abs_error": abs_error,
                }

            latency_ms = self._bench_case(kernel_mod.kernel_fn, x, weight)
            benchmark_latencies_ms.append(latency_ms)
            benchmark_throughputs.append(self._throughput_gb_s(rows, dim, dtype_name, latency_ms))

        wall_seconds = time.perf_counter() - start
        status = "ok" if max_abs_error <= 5e-3 else "fail"
        details = {
            "cases": len(cases),
            "max_abs_error": max_abs_error,
            "worst_case": worst_case,
            "median_latency_ms": statistics.median(benchmark_latencies_ms),
            "median_throughput_gb_s": statistics.median(benchmark_throughputs),
        }
        return LabBenchResult(
            target=target,
            status=status,
            metric_name="throughput_gb_s",
            metric_value=statistics.median(benchmark_throughputs),
            wall_seconds=wall_seconds,
            details=details,
        )

    def _load_kernel_module(self, kernel_path: Path):
        spec = importlib.util.spec_from_file_location("autoresearch_mlx_lab_kernel", kernel_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load kernel module from {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def _gen_inputs(self, rows: int, dim: int, dtype_name: str):
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((rows, dim), dtype=np.float32), dtype=self._dtype(dtype_name))
        weight = mx.array(rng.standard_normal((dim,), dtype=np.float32), dtype=self._dtype(dtype_name))
        return x, weight

    def _rmsnorm_ref(self, x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
        rms = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
        return x * rms * weight

    def _bench_case(self, kernel_fn, x: mx.array, weight: mx.array) -> float:
        for _ in range(3):
            out = kernel_fn(x, weight)
            mx.eval(out)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            out = kernel_fn(x, weight)
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1e3)
        return statistics.median(times)

    def _throughput_gb_s(self, rows: int, dim: int, dtype_name: str, latency_ms: float) -> float:
        itemsize = np.dtype(np.float16 if dtype_name == "float16" else np.float32).itemsize
        bytes_moved = (2 * rows * dim + dim) * itemsize
        return bytes_moved / (latency_ms / 1e3) / 1e9

    def _quick_cases(self) -> list[tuple[int, int, str]]:
        return [
            (512, 1024, "float16"),
            (1024, 2048, "float16"),
        ]

    def _full_cases(self) -> list[tuple[int, int, str]]:
        return [
            (128, 512, "float16"),
            (512, 1024, "float16"),
            (1024, 2048, "float16"),
            (2048, 4096, "float16"),
            (512, 1024, "float32"),
        ]

    def _dtype(self, dtype_name: str):
        if dtype_name == "float16":
            return mx.float16
        if dtype_name == "float32":
            return mx.float32
        raise ValueError(f"Unsupported dtype for MLX lab: {dtype_name}")

