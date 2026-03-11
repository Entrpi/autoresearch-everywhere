from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from autoresearch_lab.ledger import append_lab_event
from autoresearch_lab.labs import LabBenchResult, LabTraceResult


TRACE_SCHEMA_VERSION = 1


def _metadata_path_for(trace_path: Path) -> Path:
    return trace_path.with_suffix(".metadata.json")


def load_trace_metadata(metadata_path: Path) -> dict:
    payload = json.loads(metadata_path.expanduser().read_text(encoding="utf-8"))
    if payload.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"Trace metadata {metadata_path} has unsupported schema version "
            f"{payload.get('trace_schema_version')!r}."
        )
    return payload


def summarize_trace_metadata(metadata_path: Path) -> dict[str, object]:
    payload = load_trace_metadata(metadata_path)
    details = payload.get("details", {})
    bench_details = details.get("bench_details", {})
    return {
        "trace_metadata_path": str(metadata_path.expanduser()),
        "trace_status": payload.get("status"),
        "trace_target": payload.get("target"),
        "trace_metric_name": payload.get("metric_name"),
        "trace_metric_value": payload.get("metric_value"),
        "trace_wall_seconds": payload.get("wall_seconds"),
        "bench_wall_seconds": details.get("bench_wall_seconds"),
        "bench_max_abs_error": bench_details.get("max_abs_error"),
        "bench_median_latency_ms": bench_details.get("median_latency_ms"),
        "bench_cases": bench_details.get("cases"),
        "device_info": details.get("device_info"),
        "quick_capture": details.get("quick"),
    }


def record_trace_review(
    *,
    workspace: Path,
    metadata_path: Path,
    relevance: str,
    notes: str | None = None,
) -> dict[str, object]:
    if relevance not in {"none", "low", "medium", "high"}:
        raise ValueError("relevance must be one of: none, low, medium, high")
    trace_summary = summarize_trace_metadata(metadata_path)
    target = str(trace_summary["trace_target"])
    relevance_score = {
        "none": 0.0,
        "low": 0.25,
        "medium": 0.6,
        "high": 1.0,
    }[relevance]
    append_lab_event(
        engine="mlx",
        backend_family="mlx",
        target=target,
        workspace=workspace,
        event_type="trace-review",
        status="ok",
        metric_name="trace_relevance_score",
        metric_value=relevance_score,
        details={
            "trace_metadata_path": str(metadata_path.expanduser()),
            "relevance": relevance,
            "notes": notes,
        },
    )
    return {
        "engine": "mlx",
        "backend_family": "mlx",
        "target": target,
        "workspace": str(workspace.expanduser()),
        "status": "ok",
        "relevance": relevance,
        "relevance_score": relevance_score,
        "trace_metadata_path": str(metadata_path.expanduser()),
        "notes": notes,
        "trace_summary": trace_summary,
    }


def capture_workspace_trace(*, workspace: Path, output: Path, quick: bool = False) -> LabTraceResult:
    trace_path = output.expanduser()
    if trace_path.suffix != ".gputrace":
        trace_path = trace_path.with_suffix(".gputrace")
    if trace_path.exists():
        if trace_path.is_dir():
            shutil.rmtree(trace_path)
        else:
            trace_path.unlink()
    metadata_path = _metadata_path_for(trace_path)
    if metadata_path.exists():
        metadata_path.unlink()
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
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result = LabTraceResult(
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
    append_lab_event(
        engine="mlx",
        backend_family="mlx",
        target=payload["target"],
        workspace=workspace,
        event_type="capture",
        status=payload["status"],
        metric_name=payload.get("metric_name"),
        metric_value=payload.get("metric_value"),
        details={
            "trace_path": str(trace_path),
            "trace_metadata_path": str(metadata_path),
            "quick": quick,
            "trace_wall_seconds": result.wall_seconds,
        },
    )
    return result


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
