from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from autoresearch_lab.labs import LabBenchResult, LabTraceResult


TRACE_SCHEMA_VERSION = 1


def _metadata_path_for(trace_path: Path) -> Path:
    return trace_path.with_suffix(".metadata.json")


def capture_workspace_trace(*, workspace: Path, output: Path, quick: bool = False) -> LabTraceResult:
    trace_path = output.expanduser()
    if trace_path.suffix != ".gputrace":
        trace_path = trace_path.with_suffix(".gputrace")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MLX_METAL_DEBUG"] = "1"
    env["MTL_CAPTURE_ENABLED"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_mlx.lab",
            "_capture-bench",
            "--workspace",
            str(workspace.expanduser()),
            "--trace",
            str(trace_path),
            *(["--quick"] if quick else []),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    metadata_path = _metadata_path_for(trace_path)
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return LabTraceResult(
        engine="mlx",
        backend_family="mlx",
        target=payload["target"],
        workspace=str(workspace.expanduser()),
        trace_path=str(trace_path),
        metadata_path=str(metadata_path),
        status=payload["status"],
        wall_seconds=time.perf_counter() - start,
        details={
            "quick": quick,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            **payload["details"],
        },
    )


def run_capture_bench(*, workspace: Path, trace_path: Path, quick: bool, bench_fn) -> dict:
    workspace = workspace.expanduser()
    trace_path = trace_path.expanduser()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    mx.metal.start_capture(str(trace_path))
    try:
        bench_result: LabBenchResult = bench_fn(workspace=workspace, quick=quick)
    finally:
        mx.metal.stop_capture()
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "target": bench_result.target,
        "status": bench_result.status,
        "metric_name": bench_result.metric_name,
        "metric_value": bench_result.metric_value,
        "wall_seconds": time.perf_counter() - start,
        "details": {
            "bench_wall_seconds": bench_result.wall_seconds,
            "bench_details": bench_result.details,
            "device_info": mx.metal.device_info(),
            "mlx_metal_debug": bool(os.environ.get("MLX_METAL_DEBUG")),
            "mtl_capture_enabled": bool(os.environ.get("MTL_CAPTURE_ENABLED")),
            "quick": quick,
        },
    }
