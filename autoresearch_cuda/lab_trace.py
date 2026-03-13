from __future__ import annotations

import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from autoresearch_cuda.config import CUDA_PRESETS
from autoresearch_cuda.lab_integration import (
    integration_environment,
    integration_supported_targets,
    supports_direct_integration,
)
from autoresearch_cuda.lab_workspace import (
    CUDA_STARTER_TARGET_KEYS,
    bench_cuda_workspace,
    extract_cuda_workspace_from_profile,
    init_cuda_workspace,
    verify_cuda_workspace,
)
from autoresearch_cuda.runtime import detect_cuda_runtime_profile, query_nvidia_driver_version
from autoresearch_lab.ledger import append_lab_event
from autoresearch_lab.labs import (
    LabEvidenceResult,
    LabAutoTraceReviewResult,
    LabCapabilities,
    LabDeepProfileResult,
    LabIntegrationABResult,
    LabIntegrationSuiteResult,
    LabOrchestrationPlan,
    LabPromotionCheck,
    LabTarget,
    LabTraceProfileCandidate,
    LabTraceProfileResult,
    LabTraceResult,
)
from autoresearch_lab.ledger import summarize_lab_evidence
from autoresearch_platform.summary import parse_summary


TRACE_SCHEMA_VERSION = 1
CUDA_NCU_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_TRACE_TARGETS: dict[str, LabTarget] = {
    "flash_attention": LabTarget(
        key="flash_attention",
        description="Flash-attention or fused attention-core kernels",
        metric="time_share_pct",
        status="trace-ready",
        notes="Prioritize when FMHA/flash kernels dominate real traces.",
    ),
    "attention_prelude": LabTarget(
        key="attention_prelude",
        description="Attention setup, masking, softmax-adjacent, and Q/K/V staging work",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists for Q/K/V staging and norm-prelude work, with a first optional Triton gate-application implementation inside attention staging.",
    ),
    "value_embed_gate": LabTarget(
        key="value_embed_gate",
        description="Value-embedding gating work inside attention staging",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists for the value-embed gate seam, with a first optional Triton pointwise gate-application kernel.",
    ),
    "rope_qk_fused": LabTarget(
        key="rope_qk_fused",
        description="RoPE application and Q/K normalization work between attention staging and the attention core",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists for fused RoPE + Q/K RMSNorm, with a first optional Triton row-wise implementation.",
    ),
    "norm": LabTarget(
        key="norm",
        description="RMSNorm / LayerNorm family kernels",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists, with a first optional Triton RMSNorm implementation.",
    ),
    "fused_mlp": LabTarget(
        key="fused_mlp",
        description="MLP / activation / feed-forward path kernels",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists, with a first optional Triton pointwise squared-ReLU activation implementation.",
    ),
    "loss_prelude": LabTarget(
        key="loss_prelude",
        description="Softmax / logits / cross-entropy-side kernels",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists, with a first optional Triton row-wise cross-entropy-prelude implementation.",
    ),
    "logits_softcap": LabTarget(
        key="logits_softcap",
        description="Final logits softcap and related pointwise output shaping",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists, with a first optional Triton pointwise softcap kernel.",
    ),
    "optimizer_update": LabTarget(
        key="optimizer_update",
        description="Optimizer and parameter-update kernels",
        metric="time_share_pct",
        status="trace-ready",
    ),
    "matmul_epilogue": LabTarget(
        key="matmul_epilogue",
        description="GEMM-adjacent epilogue and projection kernels",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists, with a first optional Triton matmul+bias implementation.",
    ),
    "data_movement": LabTarget(
        key="data_movement",
        description="Copy, cast, transpose, and data-movement kernels",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fixed workspace harness exists, with a first optional Triton copy/reshape implementation.",
    ),
    "launch_fusion": LabTarget(
        key="launch_fusion",
        description="Launch-bound regions where fusion or batching may matter more than a single kernel rewrite",
        metric="time_share_pct",
        status="starter-ready",
        notes="Starter-ready fused pointwise workspace exists, with a first optional Triton residual-add kernel; use traces to validate that launch pressure is real.",
    ),
}

_TARGET_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "flash_attention",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"flash",
                r"fmha",
                r"fused.*attn",
                r"attn_(fwd|bwd)",
                r"attention_fwd",
            )
        ),
    ),
    (
        "data_movement",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"copy",
                r"cast",
                r"memcpy",
                r"memset",
                r"transpose",
                r"permute",
                r"contiguous",
            )
        ),
    ),
    (
        "launch_fusion",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"distribution_elementwise",
                r"distribution_nullary",
                r"uniform_kernel",
                r"normal_kernel",
                r"fillfunctor",
                r"vectorized_elementwise_kernel",
                r"elementwise_kernel",
                r"gpu_kernel_impl",
                r"gpu_kernel_impl_nocast",
                r"arange_cuda_out",
                r"sin_kernel_cuda",
                r"cos_kernel_cuda",
                r"reciprocal_kernel_cuda",
                r"pow_tensor_tensor_kernel",
                r"philox",
            )
        ),
    ),
    (
        "value_embed_gate",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\bgate\b",
                r"sigmoid",
                r"value",
                r"embed",
            )
        ),
    ),
    (
        "rope_qk_fused",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"rotary",
                r"\brope\b",
                r"\bqk\b",
            )
        ),
    ),
    (
        "attention_prelude",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"softmax",
                r"mask",
                r"transpose",
                r"permute",
                r"attention",
            )
        ),
    ),
    (
        "norm",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"rms",
                r"layernorm",
                r"\bnorm\b",
                r"\bln\b",
            )
        ),
    ),
    (
        "fused_mlp",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\bmlp\b",
                r"\bffn\b",
                r"gelu",
                r"relu",
                r"silu",
                r"swiglu",
                r"gated",
            )
        ),
    ),
    (
        "logits_softcap",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"softcap",
                r"tanh",
                r"logits",
            )
        ),
    ),
    (
        "loss_prelude",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"cross.?entropy",
                r"xentropy",
                r"softmax",
                r"loss",
            )
        ),
    ),
    (
        "optimizer_update",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"adam",
                r"muon",
                r"optimizer",
                r"update",
                r"\bsgd\b",
            )
        ),
    ),
    (
        "matmul_epilogue",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"cutlass",
                r"cublas",
                r"gemm",
                r"matmul",
                r"\bmma\b",
                r"wmma",
                r"epilogue",
            )
        ),
    ),
)

_LAUNCH_API_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"cudalaunch",
        r"culaunchkernel",
        r"graphlaunch",
    )
)
_SYNC_API_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"synchronize",
        r"waitevent",
        r"streamwait",
    )
)
_COPY_API_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"memcpy",
        r"memset",
        r"memcpypeer",
    )
)


def _metadata_path_for(output_prefix: Path) -> Path:
    return output_prefix.expanduser().with_suffix(".metadata.json")


def _trace_path_for(output_prefix: Path) -> Path:
    prefix = output_prefix.expanduser()
    if prefix.suffix == ".nsys-rep":
        return prefix
    return prefix.with_suffix(".nsys-rep")


def _default_capture_command(
    *,
    preset: str,
    time_budget: float,
    include_eval: bool,
    seq_len: int | None,
    window_pattern: str | None,
    total_batch_size: int | None,
    depth: int | None,
    device_batch_size: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--engine",
        "cuda",
        "--preset",
        preset,
        "--time-budget",
        str(time_budget),
    ]
    if not include_eval:
        command.append("--benchmark-skip-eval")
    if seq_len is not None:
        command.extend(["--seq-len", str(seq_len)])
    if window_pattern is not None:
        command.extend(["--window-pattern", window_pattern])
    if total_batch_size is not None:
        command.extend(["--total-batch-size", str(total_batch_size)])
    if depth is not None:
        command.extend(["--depth", str(depth)])
    if device_batch_size is not None:
        command.extend(["--device-batch-size", str(device_batch_size)])
    return command


def _cuda_runtime_ready() -> tuple[bool, str | None]:
    if find_spec("torch") is None:
        return False, "PyTorch is not installed in this environment."
    try:
        import torch  # type: ignore
    except Exception as exc:
        return False, f"Unable to import torch: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available in this environment."
    return True, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _override_or_append_flag(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag in updated:
        idx = updated.index(flag)
        if idx + 1 < len(updated):
            updated[idx + 1] = value
            return updated
    updated.extend([flag, value])
    return updated


def _tool_version(tool: str) -> str | None:
    path = shutil.which(tool)
    if not path:
        return None
    for flag in ("--version", "-v"):
        proc = _safe_run([path, flag])
        if proc.returncode == 0:
            output = proc.stdout.strip() or proc.stderr.strip()
            if output:
                return output.splitlines()[0]
    return None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    header_index = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "," in line and ("time" in lowered or "name" in lowered or "operation" in lowered):
            header_index = idx
            break
    if header_index is None:
        return []
    reader = csv.DictReader(lines[header_index:])
    rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {str(key).strip(): (value.strip() if isinstance(value, str) else "") for key, value in row.items()}
        if any(value for value in normalized.values()):
            rows.append(normalized)
    return rows


def _load_ncu_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    header_index = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "," not in line:
            continue
        if (
            "metric name" in lowered
            or "metric value" in lowered
            or "launch__kernel_name" in lowered
            or "sm__throughput.avg.pct_of_peak_sustained_elapsed" in lowered
        ):
            header_index = idx
            break
    if header_index is None:
        return []
    reader = csv.DictReader(lines[header_index:])
    rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {str(key).strip(): (value.strip() if isinstance(value, str) else "") for key, value in row.items()}
        if any(value for value in normalized.values()):
            rows.append(normalized)
    return rows


def _coerce_float(row: dict[str, str], *candidate_keys: str) -> float | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidate_keys:
        value = lowered.get(candidate.lower())
        if value is None:
            continue
        cleaned = value.replace("%", "").replace(",", "").strip().strip('"')
        if not cleaned:
            continue
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def _coerce_int(row: dict[str, str], *candidate_keys: str) -> int | None:
    value = _coerce_float(row, *candidate_keys)
    if value is None:
        return None
    return int(value)


def _summarize_ncu_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    # Newer Nsight Compute CSV exports can be "wide", with the requested metrics
    # present as columns on each kernel row instead of a long Metric Name/Value table.
    wide_values: dict[str, list[float]] = {}
    for metric in CUDA_NCU_METRICS:
        values = [value for row in rows if (value := _coerce_float(row, metric)) is not None]
        if values:
            wide_values[metric] = values
    if wide_values:
        return {
            metric: float(statistics.median(values))
            for metric, values in wide_values.items()
            if values
        }

    values_by_metric: dict[str, list[float]] = {}
    for row in rows:
        metric_name = row.get("Metric Name") or row.get("Metric") or row.get("Name")
        if not metric_name:
            continue
        metric_value = _coerce_float(row, "Metric Value", "Value", "Avg")
        if metric_value is None:
            continue
        values_by_metric.setdefault(metric_name, []).append(metric_value)
    return {
        metric: float(statistics.median(values))
        for metric, values in values_by_metric.items()
        if values
    }


def _classify_ncu_metrics(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    sm_pct = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    dram_pct = metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    occ_pct = metrics.get("smsp__warps_active.avg.pct_of_peak_sustained_active")
    if occ_pct is not None and occ_pct < 35.0:
        return "under-occupied", 0.7
    if dram_pct is not None and dram_pct >= 70.0 and (sm_pct is None or dram_pct - sm_pct >= 10.0):
        return "bandwidth-bound", 0.8
    if sm_pct is not None and sm_pct >= 70.0 and (dram_pct is None or sm_pct - dram_pct >= 10.0):
        return "compute-bound", 0.8
    if sm_pct is not None or dram_pct is not None or occ_pct is not None:
        return "mixed", 0.45
    return None, None


def _match_target_family(name: str) -> str:
    for target, patterns in _TARGET_PATTERNS:
        if any(pattern.search(name) for pattern in patterns):
            return target
    return "matmul_epilogue"


def _classify_api_bucket(name: str) -> str:
    if any(pattern.search(name) for pattern in _LAUNCH_API_PATTERNS):
        return "launch"
    if any(pattern.search(name) for pattern in _SYNC_API_PATTERNS):
        return "sync"
    if any(pattern.search(name) for pattern in _COPY_API_PATTERNS):
        return "copy"
    return "other"


def load_trace_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.expanduser().read_text(encoding="utf-8"))
    if payload.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"Trace metadata {metadata_path} has unsupported schema version {payload.get('trace_schema_version')!r}."
        )
    return payload


def capture_cuda_trace(
    *,
    preset: str,
    output: Path,
    time_budget: float,
    include_eval: bool = False,
    seq_len: int | None = None,
    window_pattern: str | None = None,
    total_batch_size: int | None = None,
    depth: int | None = None,
    device_batch_size: int | None = None,
) -> LabTraceResult:
    output_prefix = output.expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = _trace_path_for(output_prefix)
    metadata_path = _metadata_path_for(output_prefix)
    nsys_path = shutil.which("nsys")
    ncu_path = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    train_command = _default_capture_command(
        preset=preset,
        time_budget=time_budget,
        include_eval=include_eval,
        seq_len=seq_len,
        window_pattern=window_pattern,
        total_batch_size=total_batch_size,
        depth=depth,
        device_batch_size=device_batch_size,
    )
    details: dict[str, Any] = {
        "preset": preset,
        "time_budget": time_budget,
        "include_eval": include_eval,
        "train_command": train_command,
        "tool_paths": {
            "nsys": nsys_path,
            "ncu": ncu_path,
        },
        "tool_versions": {
            "nsys": _tool_version("nsys"),
            "ncu": _tool_version("ncu") or _tool_version("nv-nsight-cu-cli"),
        },
        "driver_version": query_nvidia_driver_version(),
        "report_artifacts": {},
    }
    start = time.perf_counter()
    status = "ok"
    if nsys_path is None:
        status = "missing-tool"
        details["failure_reason"] = "Nsight Systems CLI (`nsys`) is not installed or not on PATH."
    else:
        profile_command = [
            nsys_path,
            "profile",
            "--force-overwrite",
            "true",
            "--sample",
            "none",
            "--trace",
            "cuda,nvtx,osrt",
            "--output",
            str(output_prefix),
            *train_command,
        ]
        details["profile_command"] = profile_command
        proc = _safe_run(profile_command)
        stdout_path = output_prefix.with_suffix(".capture.stdout.log")
        stderr_path = output_prefix.with_suffix(".capture.stderr.log")
        _write_text(stdout_path, proc.stdout)
        _write_text(stderr_path, proc.stderr)
        details["capture_stdout_path"] = str(stdout_path)
        details["capture_stderr_path"] = str(stderr_path)
        details["capture_returncode"] = proc.returncode
        if proc.returncode != 0:
            status = "capture-failed"
            details["failure_reason"] = "nsys profile returned a non-zero exit status."
        else:
            report_artifacts: dict[str, dict[str, Any]] = {}
            for report_name in ("cuda_gpu_kern_sum", "cuda_api_sum", "osrt_sum"):
                report_command = [
                    nsys_path,
                    "stats",
                    "--report",
                    report_name,
                    "--format",
                    "csv",
                    str(trace_path),
                ]
                report_proc = _safe_run(report_command)
                report_csv = output_prefix.with_suffix(f".{report_name}.csv")
                report_stderr = output_prefix.with_suffix(f".{report_name}.stderr.log")
                _write_text(report_csv, report_proc.stdout)
                _write_text(report_stderr, report_proc.stderr)
                report_artifacts[report_name] = {
                    "command": report_command,
                    "status": "ok" if report_proc.returncode == 0 else "failed",
                    "returncode": report_proc.returncode,
                    "csv_path": str(report_csv),
                    "stderr_path": str(report_stderr),
                }
            details["report_artifacts"] = report_artifacts
    payload = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "engine": "cuda",
        "backend_family": "cuda",
        "target": "end_to_end_trace",
        "status": status,
        "wall_seconds": time.perf_counter() - start,
        "trace_path": str(trace_path),
        "details": details,
    }
    _write_json(metadata_path, payload)
    return LabTraceResult(
        engine="cuda",
        backend_family="cuda",
        target="end_to_end_trace",
        workspace=str(output_prefix.parent),
        trace_path=str(trace_path),
        metadata_path=str(metadata_path),
        status=status,
        wall_seconds=payload["wall_seconds"],
        details=details,
    )


def _build_trace_candidates(metadata: dict[str, Any]) -> tuple[list[LabTraceProfileCandidate], dict[str, Any]]:
    details = metadata.get("details", {})
    report_artifacts = details.get("report_artifacts", {})
    kernel_rows = _load_csv_rows(Path(report_artifacts.get("cuda_gpu_kern_sum", {}).get("csv_path", "")))
    api_rows = _load_csv_rows(Path(report_artifacts.get("cuda_api_sum", {}).get("csv_path", "")))
    family_stats: dict[str, dict[str, Any]] = {}
    total_kernel_ns = 0.0
    short_kernel_avg_ns: list[float] = []
    for row in kernel_rows:
        name = row.get("Name") or row.get("Kernel Name") or row.get("Operation") or row.get("Kernel") or "unknown"
        total_ns = _coerce_float(row, "Total Time (ns)", "Total Time", "Total Time(ns)", "Sum")
        if total_ns is None:
            continue
        avg_ns = _coerce_float(row, "Avg (ns)", "Average", "Avg") or 0.0
        instances = _coerce_int(row, "Instances", "Calls") or 0
        total_kernel_ns += total_ns
        short_kernel_avg_ns.extend([avg_ns] if avg_ns else [])
        family = _match_target_family(name)
        bucket = family_stats.setdefault(
            family,
            {"total_ns": 0.0, "instances": 0, "names": [], "avg_ns_samples": []},
        )
        bucket["total_ns"] += total_ns
        bucket["instances"] += instances
        bucket["names"].append(name)
        if avg_ns:
            bucket["avg_ns_samples"].append(avg_ns)
    api_totals = {"launch": 0.0, "sync": 0.0, "copy": 0.0, "other": 0.0}
    total_api_ns = 0.0
    for row in api_rows:
        name = row.get("Name") or row.get("Operation") or row.get("API") or "unknown"
        total_ns = _coerce_float(row, "Total Time (ns)", "Total Time", "Total Time(ns)", "Sum")
        if total_ns is None:
            continue
        total_api_ns += total_ns
        api_totals[_classify_api_bucket(name)] += total_ns
    median_kernel_ns = float(statistics.median(short_kernel_avg_ns)) if short_kernel_avg_ns else None
    launch_pct = (api_totals["launch"] / total_api_ns * 100.0) if total_api_ns else 0.0
    sync_pct = (api_totals["sync"] / total_api_ns * 100.0) if total_api_ns else 0.0
    copy_pct = (api_totals["copy"] / total_api_ns * 100.0) if total_api_ns else 0.0
    if total_kernel_ns <= 0:
        dominant_issue = "missing-profiler-summary"
    elif launch_pct >= 25.0 and median_kernel_ns is not None and median_kernel_ns <= 200_000:
        dominant_issue = "launch-bound"
    elif sync_pct >= 20.0:
        dominant_issue = "sync-bound"
    elif copy_pct >= 20.0:
        dominant_issue = "copy-bound"
    elif total_kernel_ns >= total_api_ns * 2.0:
        dominant_issue = "kernel-dominated"
    else:
        dominant_issue = "mixed"
    candidates: list[LabTraceProfileCandidate] = []
    for family, stats in family_stats.items():
        time_share_pct = (stats["total_ns"] / total_kernel_ns * 100.0) if total_kernel_ns else None
        priority_score = float(time_share_pct or 0.0)
        if dominant_issue == "launch-bound" and family in {"launch_fusion", "attention_prelude", "loss_prelude"}:
            priority_score += 15.0
        if dominant_issue == "copy-bound" and family == "data_movement":
            priority_score += 10.0
        if dominant_issue == "kernel-dominated" and family in {"flash_attention", "fused_mlp", "matmul_epilogue", "norm", "rope_qk_fused"}:
            priority_score += 8.0
        if family == "matmul_epilogue" and (time_share_pct or 0.0) < 15.0:
            priority_score -= 5.0
        sample_names = tuple(dict.fromkeys(stats["names"]))[:5]
        candidates.append(
            LabTraceProfileCandidate(
                target=family,
                rank=0,
                priority_score=priority_score,
                time_share_pct=time_share_pct,
                category=family,
                status="trace-ranked",
                rationale=f"{family} accounted for {time_share_pct:.1f}% of observed CUDA kernel time" if time_share_pct is not None else f"{family} was detected in CUDA trace data",
                details={
                    "total_kernel_time_ns": stats["total_ns"],
                    "instances": stats["instances"],
                    "sample_kernel_names": sample_names,
                    "median_kernel_avg_ns": (
                        float(statistics.median(stats["avg_ns_samples"])) if stats["avg_ns_samples"] else None
                    ),
                },
            )
        )
    candidates.sort(key=lambda item: item.priority_score, reverse=True)
    ranked_candidates = [replace(candidate, rank=index) for index, candidate in enumerate(candidates, start=1)]
    return ranked_candidates, {
        "dominant_issue": dominant_issue,
        "total_kernel_time_ns": total_kernel_ns,
        "total_api_time_ns": total_api_ns,
        "launch_api_time_pct": launch_pct,
        "sync_api_time_pct": sync_pct,
        "copy_api_time_pct": copy_pct,
        "median_kernel_avg_ns": median_kernel_ns,
        "kernel_row_count": len(kernel_rows),
        "api_row_count": len(api_rows),
    }


def trace_profile_cuda(metadata_path: Path) -> LabTraceProfileResult:
    start = time.perf_counter()
    metadata = load_trace_metadata(metadata_path)
    details = metadata.get("details", {})
    preset = str(details.get("preset") or "upstream")
    if metadata.get("status") != "ok":
        return LabTraceProfileResult(
            engine="cuda",
            backend_family="cuda",
            preset=preset,
            status="capture-unavailable",
            trace_path=metadata.get("trace_path"),
            wall_seconds=time.perf_counter() - start,
            dominant_issue="capture-unavailable",
            candidates=(),
            details={
                "trace_metadata_path": str(metadata_path.expanduser()),
                "capture_status": metadata.get("status"),
                "capture_failure_reason": details.get("failure_reason"),
            },
        )
    candidates, summary = _build_trace_candidates(metadata)
    status = "ok" if candidates else "missing-profiler-summary"
    profile = LabTraceProfileResult(
        engine="cuda",
        backend_family="cuda",
        preset=preset,
        status=status,
        trace_path=metadata.get("trace_path"),
        wall_seconds=time.perf_counter() - start,
        dominant_issue=summary.get("dominant_issue"),
        candidates=tuple(candidates),
        details={
            "trace_metadata_path": str(metadata_path.expanduser()),
            "driver_version": details.get("driver_version"),
            "tool_versions": details.get("tool_versions"),
            "summary": summary,
        },
    )
    for candidate in candidates[: min(5, len(candidates))]:
        append_lab_event(
            engine="cuda",
            backend_family="cuda",
            target=candidate.target,
            workspace=metadata_path.expanduser().parent,
            event_type="trace-profile",
            status=profile.status,
            metric_name="priority_score",
            metric_value=candidate.priority_score,
            preset=preset,
            details={
                "trace_metadata_path": str(metadata_path.expanduser()),
                "dominant_issue": profile.dominant_issue,
                "rank": candidate.rank,
                "priority_score": candidate.priority_score,
                "time_share_pct": candidate.time_share_pct,
                "sample_kernel_names": candidate.details.get("sample_kernel_names"),
            },
        )
    return profile


def deep_profile_cuda_trace(
    trace_profile_path: Path,
    *,
    rank: int = 1,
    time_budget: float = 5.0,
) -> LabDeepProfileResult:
    start = time.perf_counter()
    payload = json.loads(trace_profile_path.expanduser().read_text(encoding="utf-8"))
    preset = str(payload.get("preset") or "upstream")
    candidates = payload.get("candidates", [])
    if not candidates:
        return LabDeepProfileResult(
            engine="cuda",
            backend_family="cuda",
            preset=preset,
            target="unknown",
            status="insufficient-trace-data",
            wall_seconds=time.perf_counter() - start,
            diagnosis=None,
            confidence=None,
            details={
                "failure_reason": "Trace profile has no candidates.",
                "trace_profile_path": str(trace_profile_path.expanduser()),
            },
        )
    if rank <= 0 or rank > len(candidates):
        raise ValueError(f"rank must be between 1 and {len(candidates)}")
    candidate = candidates[rank - 1]
    target = str(candidate.get("target") or "unknown")
    trace_metadata_path = payload.get("details", {}).get("trace_metadata_path")
    if not trace_metadata_path:
        return LabDeepProfileResult(
            engine="cuda",
            backend_family="cuda",
            preset=preset,
            target=target,
            status="insufficient-trace-data",
            wall_seconds=time.perf_counter() - start,
            diagnosis=None,
            confidence=None,
            details={
                "failure_reason": "Trace profile did not record a trace metadata path.",
                "trace_profile_path": str(trace_profile_path.expanduser()),
            },
        )
    metadata = load_trace_metadata(Path(trace_metadata_path))
    details = metadata.get("details", {})
    train_command = list(details.get("train_command") or [])
    if not train_command:
        return LabDeepProfileResult(
            engine="cuda",
            backend_family="cuda",
            preset=preset,
            target=target,
            status="insufficient-trace-data",
            wall_seconds=time.perf_counter() - start,
            diagnosis=None,
            confidence=None,
            details={
                "failure_reason": "Trace metadata did not record the original train command.",
                "trace_profile_path": str(trace_profile_path.expanduser()),
                "trace_metadata_path": trace_metadata_path,
            },
        )
    train_command = _override_or_append_flag(train_command, "--time-budget", str(time_budget))
    ncu_path = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    sample_kernel_names = tuple(candidate.get("details", {}).get("sample_kernel_names") or ())
    kernel_selector = None
    if sample_kernel_names:
        kernel_selector = f"regex:.*{re.escape(str(sample_kernel_names[0]))}.*"
    elif target:
        kernel_selector = f"regex:.*{re.escape(target)}.*"
    if ncu_path is None:
        return LabDeepProfileResult(
            engine="cuda",
            backend_family="cuda",
            preset=preset,
            target=target,
            status="missing-tool",
            wall_seconds=time.perf_counter() - start,
            diagnosis=None,
            confidence=None,
            details={
                "failure_reason": "Nsight Compute CLI (`ncu` or `nv-nsight-cu-cli`) is not installed or not on PATH.",
                "trace_profile_path": str(trace_profile_path.expanduser()),
                "trace_metadata_path": trace_metadata_path,
                "kernel_selector": kernel_selector,
            },
        )
    output_prefix = trace_profile_path.expanduser().with_suffix("")
    csv_path = output_prefix.parent / f"{output_prefix.name}.rank{rank}.ncu.csv"
    stderr_path = output_prefix.parent / f"{output_prefix.name}.rank{rank}.ncu.stderr.log"
    ncu_command = [
        ncu_path,
        "--csv",
        "--page",
        "raw",
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--metrics",
        ",".join(CUDA_NCU_METRICS),
        "--log-file",
        str(csv_path),
    ]
    if kernel_selector:
        ncu_command.extend(["--kernel-name", kernel_selector])
    ncu_command.extend(train_command)
    proc = _safe_run(ncu_command)
    _write_text(stderr_path, proc.stderr)
    rows = _load_ncu_metric_rows(csv_path)
    metrics = _summarize_ncu_metrics(rows)
    diagnosis, confidence = _classify_ncu_metrics(metrics)
    status = "ok" if proc.returncode == 0 and metrics else "insufficient-metrics"
    if proc.returncode != 0:
        status = "profile-failed"
    if status == "ok":
        append_lab_event(
            engine="cuda",
            backend_family="cuda",
            target=target,
            workspace=trace_profile_path.expanduser().parent,
            event_type="deep-profile",
            status=status,
            metric_name="diagnosis_confidence",
            metric_value=confidence,
            preset=preset,
            details={
                "trace_profile_path": str(trace_profile_path.expanduser()),
                "trace_metadata_path": trace_metadata_path,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "kernel_selector": kernel_selector,
                "sample_kernel_names": sample_kernel_names,
                "metrics": metrics,
                "ncu_csv_path": str(csv_path),
                "ncu_stderr_path": str(stderr_path),
            },
        )
    return LabDeepProfileResult(
        engine="cuda",
        backend_family="cuda",
        preset=preset,
        target=target,
        status=status,
        wall_seconds=time.perf_counter() - start,
        diagnosis=diagnosis,
        confidence=confidence,
        details={
            "trace_profile_path": str(trace_profile_path.expanduser()),
            "trace_metadata_path": trace_metadata_path,
            "kernel_selector": kernel_selector,
            "sample_kernel_names": sample_kernel_names,
            "train_command": train_command,
            "ncu_command": ncu_command,
            "ncu_returncode": proc.returncode,
            "ncu_csv_path": str(csv_path),
            "ncu_stderr_path": str(stderr_path),
            "metrics": metrics,
            "review_notes": (
                "Deep CUDA diagnosis uses Nsight Compute to classify the top trace-ranked family as compute-bound, bandwidth-bound, under-occupied, or mixed."
            ),
        },
    )


def auto_review_cuda_trace(trace_profile_path: Path) -> LabAutoTraceReviewResult:
    start = time.perf_counter()
    payload = json.loads(trace_profile_path.expanduser().read_text(encoding="utf-8"))
    preset = str(payload.get("preset") or "upstream")
    candidates = payload.get("candidates", [])
    dominant_issue = payload.get("dominant_issue")
    details = payload.get("details", {})
    summary = details.get("summary", {})
    status = str(payload.get("status") or "unknown")
    confidence: float | None = None
    if status == "ok":
        if dominant_issue in {"launch-bound", "sync-bound", "copy-bound"}:
            confidence = 0.7
        elif dominant_issue == "kernel-dominated":
            confidence = 0.55
        elif dominant_issue == "mixed":
            confidence = 0.4
        else:
            confidence = 0.25
        status = "ok"
    else:
        status = "insufficient-trace-data"
    recommended_targets = [
        {
            "target": candidate.get("target"),
            "priority_score": candidate.get("priority_score"),
            "time_share_pct": candidate.get("time_share_pct"),
        }
        for candidate in candidates[:3]
    ]
    result = LabAutoTraceReviewResult(
        engine="cuda",
        backend_family="cuda",
        preset=preset,
        status=status,
        dominant_issue=dominant_issue,
        confidence=confidence,
        details={
            "trace_profile_path": str(trace_profile_path.expanduser()),
            "recommended_targets": recommended_targets,
            "summary": summary,
            "review_notes": (
                "Automated CUDA trace review is intended as the default first pass; escalate to Nsight GUI only when the summary stays ambiguous."
            ),
        },
    )
    for candidate in recommended_targets:
        rank = next(
            (
                index
                for index, raw_candidate in enumerate(candidates[:3], start=1)
                if raw_candidate.get("target") == candidate["target"]
            ),
            None,
        )
        time_share_pct = candidate.get("time_share_pct") or 0.0
        if rank == 1 and time_share_pct >= 20.0:
            relevance = "high"
        elif rank is not None and rank <= 2 and time_share_pct >= 10.0:
            relevance = "medium"
        else:
            relevance = "low"
        append_lab_event(
            engine="cuda",
            backend_family="cuda",
            target=str(candidate["target"]),
            workspace=trace_profile_path.expanduser().parent,
            event_type="auto-review",
            status=result.status,
            metric_name="time_share_pct",
            metric_value=candidate.get("time_share_pct"),
            preset=preset,
            details={
                "trace_profile_path": str(trace_profile_path.expanduser()),
                "dominant_issue": dominant_issue,
                "confidence": confidence,
                "priority_score": candidate.get("priority_score"),
                "time_share_pct": time_share_pct,
                "rank": rank,
                "relevance": relevance,
            },
        )
    return result


class CudaKernelLab:
    name = "cuda"
    backend_family = "cuda"
    capabilities = LabCapabilities(
        supports_workspace_init=True,
        supports_fixed_bench=True,
        supports_profile=False,
        supports_extract=True,
        supports_orchestrate=True,
        supports_verify=True,
        supports_capture=True,
        supports_trace_profile=True,
        supports_auto_trace_review=True,
        supports_deep_trace_profile=True,
    )

    def target_catalog(self) -> dict[str, LabTarget]:
        return {
            key: replace(
                target,
                status="starter-ready" if key in CUDA_STARTER_TARGET_KEYS else target.status,
            )
            for key, target in CUDA_TRACE_TARGETS.items()
        }

    def init_workspace(self, *, target: str, workspace: Path) -> Path:
        return init_cuda_workspace(target=target, workspace=workspace)

    def bench_workspace(self, *, workspace: Path, quick: bool = False, device: str = "auto"):
        return bench_cuda_workspace(workspace=workspace, quick=quick, device=device)

    def verify_workspace(self, *, workspace: Path, quick: bool = False, device: str = "auto"):
        return verify_cuda_workspace(workspace=workspace, quick=quick, device=device)

    def profile_targets(self, *, preset: str, top_k: int = 10):
        raise NotImplementedError("CUDA heuristic target profiling is not implemented yet; start from a real trace.")

    def extract_from_profile(self, *, profile_path: Path, workspace: Path, rank: int = 1):
        return extract_cuda_workspace_from_profile(profile_path=profile_path, workspace=workspace, rank=rank)

    def orchestrate_from_profile(
        self,
        *,
        profile_path: Path,
        workspace_root: Path,
        rank: int = 1,
        trace_metadata_path: Path | None = None,
    ) -> LabOrchestrationPlan:
        payload = json.loads(profile_path.expanduser().read_text(encoding="utf-8"))
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Trace profile has no candidates to orchestrate.")
        preset = str(payload.get("preset") or "upstream")
        rescored_candidates = []
        for raw_candidate in candidates:
            target = str(raw_candidate["target"])
            summary = summarize_lab_evidence(engine="cuda", backend_family="cuda", target=target, preset=preset)
            adjusted_priority = float(raw_candidate.get("priority_score") or 0.0) + float(summary.evidence_bonus)
            rescored_candidates.append(
                {
                    **raw_candidate,
                    "adjusted_priority_score": adjusted_priority,
                    "evidence_summary": asdict(summary),
                }
            )
        rescored_candidates.sort(
            key=lambda item: (float(item["adjusted_priority_score"]), float(item.get("priority_score") or 0.0)),
            reverse=True,
        )
        if rank <= 0 or rank > len(rescored_candidates):
            raise ValueError(f"rank must be between 1 and {len(rescored_candidates)}")
        candidate = rescored_candidates[rank - 1]
        target = str(candidate["target"])
        summary = summarize_lab_evidence(engine="cuda", backend_family="cuda", target=target, preset=preset)
        status = "trace-prioritized"
        commands: list[str] = [
            f"uv run kernel-lab.py --engine cuda evidence --target {target} --preset {preset}",
        ]
        if summary.auto_review_ok_count == 0:
            commands.append(
                f"uv run kernel-lab.py --engine cuda auto-review --trace-profile {profile_path.expanduser()}"
            )
        if summary.promotion_status == "trace-deprioritized":
            status = "trace-deprioritized"
            commands.append("# this target currently looks low-value; prioritize a different CUDA target first")
        else:
            if target in CUDA_STARTER_TARGET_KEYS:
                if summary.deep_profile_ok_count == 0 and target in {"norm", "fused_mlp", "matmul_epilogue", "rope_qk_fused", "flash_attention"}:
                    commands.append(
                        f"uv run kernel-lab.py --engine cuda deep-profile --trace-profile {profile_path.expanduser()} --rank {rank}"
                    )
                commands.extend(
                    [
                        f"uv run kernel-lab.py --engine cuda extract --profile {profile_path.expanduser()} --workspace {(workspace_root.expanduser() / target)} --rank {rank}",
                        f"uv run kernel-lab.py --engine cuda bench --workspace {(workspace_root.expanduser() / target)} --device cuda --quick",
                        f"uv run kernel-lab.py --engine cuda verify --workspace {(workspace_root.expanduser() / target)} --device cuda --quick",
                    ]
                )
            else:
                commands.extend(
                    [
                        "# this target is trace-backed, but no starter workspace exists yet",
                        "# use it to decide the next CUDA/Triton workspace family to add",
                    ]
                )
        return LabOrchestrationPlan(
            engine="cuda",
            target=target,
            workspace=str(workspace_root.expanduser() / target),
            status=status,
            commands=tuple(commands),
            details={
                "preset": preset,
                "rank": rank,
                "candidate": candidate,
                "rescored_candidates": rescored_candidates[: min(5, len(rescored_candidates))],
                "trace_profile_path": str(profile_path.expanduser()),
                "trace_metadata_path": str(trace_metadata_path.expanduser()) if trace_metadata_path else None,
                "evidence_summary": asdict(summary),
                "notes": (
                    "CUDA orchestration stays trace-first. Starter workspaces currently exist for a narrow set of families."
                ),
            },
        )

    def capture_workspace(self, *, workspace: Path, output: Path, quick: bool = False):
        raise NotImplementedError("CUDA trace capture is run against the trainer, not a workspace. Use `capture` from autoresearch_cuda.lab`.")

    def trace_profile(self, *, metadata_path: Path) -> LabTraceProfileResult:
        return trace_profile_cuda(metadata_path)

    def auto_review_trace(self, *, trace_profile_path: Path) -> LabAutoTraceReviewResult:
        return auto_review_cuda_trace(trace_profile_path)

    def summarize_evidence(self, *, target: str, preset: str | None = None):
        summary = summarize_lab_evidence(
            engine="cuda",
            backend_family="cuda",
            target=target,
            preset=preset,
        )
        return LabEvidenceResult(
            engine="cuda",
            backend_family="cuda",
            target=target,
            preset=preset,
            status=summary.promotion_status,
            details=asdict(summary),
        )

    def promotion_check(self, *, target: str, preset: str | None = None, workspace: Path | None = None):
        summary = summarize_lab_evidence(
            engine="cuda",
            backend_family="cuda",
            target=target,
            preset=preset,
        )
        if summary.promotion_status == "trace-deprioritized":
            status = "trace-deprioritized"
            commands = (
                "# this target is currently deprioritized by automated CUDA trace evidence",
                "# capture a new trace on a different preset only if you believe the current trace is unrepresentative",
            )
        elif target in CUDA_STARTER_TARGET_KEYS and summary.deep_profile_ok_count == 0 and summary.auto_review_ok_count > 0:
            status = "needs-deep-profile"
            commands = (
                "# this target is trace-backed, but deeper kernel diagnosis has not been recorded yet",
                "# run `uv run kernel-lab.py --engine cuda deep-profile --trace-profile <profile.json> --rank <n>` before promoting it further",
            )
        elif target in CUDA_STARTER_TARGET_KEYS and bool(summary.details.get("last_deep_profile_is_weak")) and summary.verify_ok_count > 0:
            status = "needs-manual-cuda-review"
            commands = (
                "# the deeper CUDA diagnosis for this target is still weak or mixed",
                "# inspect the Nsight Systems / Nsight Compute artifacts manually before promoting it toward trainer integration",
            )
        elif target in CUDA_STARTER_TARGET_KEYS and supports_direct_integration(target) and summary.verify_ok_count > 0:
            status = "ready-for-cuda-integration"
            commands = (
                "# this starter workspace has both trace-backed relevance and a passing fixed-harness verify run",
                "# the next step is a real CUDA trainer integration path for this target family",
                f"uv run kernel-lab.py --engine cuda integration-suite --workspace <workspace> --preset {preset or 'upstream'} --time-budget 20 --repeats 2 --benchmark-skip-eval --no-checkpoint",
            )
        elif summary.auto_review_ok_count > 0 or summary.trace_profile_ok_count > 0:
            if target in CUDA_STARTER_TARGET_KEYS:
                status = "ready-for-cuda-workspace"
                commands = (
                    "# this target is trace-backed and has a starter CUDA workspace available now",
                    "# start with `extract`, then `bench`, then `verify` before attempting trainer integration",
                )
            else:
                status = "ready-for-cuda-workspace-family"
                commands = (
                    "# this target is trace-backed and is the right place to start for a future CUDA/Triton workspace family",
                    "# keep collecting traces and use `evidence` to confirm that it stays important",
                )
        elif summary.capture_ok_count > 0:
            status = "needs-trace-profile"
            commands = (
                "# capture exists, but no structured trace summary was recorded",
                "# run `uv run kernel-lab.py --engine cuda trace-profile --metadata <trace.metadata.json> --output <profile.json>` and then `auto-review`",
            )
        else:
            status = "needs-capture"
            commands = (
                "# no CUDA trace evidence exists yet for this target",
                "# start with `uv run kernel-lab.py --engine cuda capture --preset upstream --time-budget 20 --output /tmp/cuda-trace`",
            )
        return LabPromotionCheck(
            engine="cuda",
            backend_family="cuda",
            target=target,
            preset=preset,
            workspace=None,
            status=status,
            commands=commands,
            details={
                "evidence_summary": asdict(summary),
                "notes": (
                    "CUDA promotion is currently split: starter targets can move into fixed workspaces now, while broader families remain trace-backed planning targets."
                ),
            },
        )

    def run_integration_ab(
        self,
        *,
        workspace: Path,
        time_budget: float,
        preset: str | None = None,
        repeats: int = 2,
        benchmark_skip_eval: bool = True,
        no_checkpoint: bool = True,
    ) -> LabIntegrationABResult:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        workspace = workspace.expanduser().resolve()
        metadata = json.loads((workspace / "metadata.json").read_text(encoding="utf-8"))
        target = str(metadata["target"])
        resolved_preset = preset or "upstream"
        if not supports_direct_integration(target):
            return LabIntegrationABResult(
                engine="cuda",
                backend_family="cuda",
                target=target,
                preset=resolved_preset,
                workspace=str(workspace),
                status="unsupported-target",
                wall_seconds=0.0,
                details={
                    "failure_reason": f"Target {target!r} does not yet support direct CUDA trainer integration.",
                    "supported_targets": integration_supported_targets(),
                },
            )
        runtime_ok, failure_reason = _cuda_runtime_ready()
        if not runtime_ok:
            append_lab_event(
                engine="cuda",
                backend_family="cuda",
                target=target,
                workspace=workspace,
                event_type="integration-ab",
                status="missing-runtime",
                metric_name="steady_state_tok_per_sec_delta",
                metric_value=None,
                preset=resolved_preset,
                details={
                    "preset": resolved_preset,
                    "time_budget": time_budget,
                    "repeats": repeats,
                    "failure_reason": failure_reason,
                },
            )
            return LabIntegrationABResult(
                engine="cuda",
                backend_family="cuda",
                target=target,
                preset=resolved_preset,
                workspace=str(workspace),
                status="missing-runtime",
                wall_seconds=0.0,
                details={
                    "failure_reason": failure_reason,
                    "repeats": repeats,
                    "benchmark_skip_eval": benchmark_skip_eval,
                    "no_checkpoint": no_checkpoint,
                },
            )

        run_root = workspace / "integration-ab" / time.strftime("%Y%m%d-%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        warmup_budget = min(2.0, time_budget)

        def build_cmd(run_time_budget: float) -> list[str]:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "train.py"),
                "--engine",
                "cuda",
                "--preset",
                resolved_preset,
                "--time-budget",
                str(run_time_budget),
            ]
            if benchmark_skip_eval:
                cmd.append("--benchmark-skip-eval")
            return cmd

        warmup_cmd = build_cmd(warmup_budget)
        measured_cmd = build_cmd(time_budget)
        baseline_env = os.environ.copy()
        candidate_env = os.environ.copy()
        candidate_env.update(integration_environment(workspace=workspace))

        start = time.perf_counter()
        baseline_warmup = self._run_train_command(
            warmup_cmd,
            run_root / "baseline-warmup.stdout.log",
            run_root / "baseline-warmup.stderr.log",
            env=baseline_env,
        )
        candidate_warmup = self._run_train_command(
            warmup_cmd,
            run_root / "candidate-warmup.stdout.log",
            run_root / "candidate-warmup.stderr.log",
            env=candidate_env,
        )
        measured_runs: list[dict[str, object]] = []
        baseline_results: list[subprocess.CompletedProcess[str]] = []
        candidate_results: list[subprocess.CompletedProcess[str]] = []
        for repeat_index in range(repeats):
            order = ("baseline", "candidate") if repeat_index % 2 == 0 else ("candidate", "baseline")
            for leg in order:
                stdout_path = run_root / f"round-{repeat_index + 1:02d}-{leg}.stdout.log"
                stderr_path = run_root / f"round-{repeat_index + 1:02d}-{leg}.stderr.log"
                env = baseline_env if leg == "baseline" else candidate_env
                completed = self._run_train_command(
                    measured_cmd,
                    stdout_path,
                    stderr_path,
                    env=env,
                )
                summary = parse_summary(completed.stdout) if completed.returncode == 0 else {}
                measured_runs.append(
                    {
                        "round": repeat_index + 1,
                        "leg": leg,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "returncode": completed.returncode,
                        "summary": summary,
                    }
                )
                if leg == "baseline":
                    baseline_results.append(completed)
                else:
                    candidate_results.append(completed)
        wall_seconds = time.perf_counter() - start

        baseline_warmup_summary = parse_summary(baseline_warmup.stdout) if baseline_warmup.returncode == 0 else {}
        candidate_warmup_summary = parse_summary(candidate_warmup.stdout) if candidate_warmup.returncode == 0 else {}
        baseline_summaries = [parse_summary(result.stdout) for result in baseline_results if result.returncode == 0]
        candidate_summaries = [parse_summary(result.stdout) for result in candidate_results if result.returncode == 0]
        baseline_summary = self._aggregate_summaries(baseline_summaries)
        candidate_summary = self._aggregate_summaries(candidate_summaries)
        status = (
            "ok"
            if all(
                result.returncode == 0
                for result in (baseline_warmup, candidate_warmup, *baseline_results, *candidate_results)
            )
            else "error"
        )
        delta = self._summary_delta(baseline_summary, candidate_summary)
        delta_relative_pct = self._summary_delta_relative_pct(baseline_summary, candidate_summary)
        details = {
            "warmup_budget": warmup_budget,
            "repeats": repeats,
            "measured_pair_count": repeats * 2,
            "warmup": {
                "baseline_returncode": baseline_warmup.returncode,
                "candidate_returncode": candidate_warmup.returncode,
                "baseline": baseline_warmup_summary,
                "candidate": candidate_warmup_summary,
            },
            "benchmark_skip_eval": benchmark_skip_eval,
            "no_checkpoint": no_checkpoint,
            "measured_runs": measured_runs,
            "measured_order": [f"{entry['round']}:{entry['leg']}" for entry in measured_runs],
            "baseline_run_returncodes": [result.returncode for result in baseline_results],
            "candidate_run_returncodes": [result.returncode for result in candidate_results],
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "baseline_runs": baseline_summaries,
            "candidate_runs": candidate_summaries,
            "delta": delta,
            "delta_relative_pct": delta_relative_pct,
        }
        append_lab_event(
            engine="cuda",
            backend_family="cuda",
            target=target,
            workspace=workspace,
            event_type="integration-ab",
            status=status,
            metric_name="steady_state_tok_per_sec_delta",
            metric_value=delta.get("steady_state_tok_per_sec"),
            preset=resolved_preset,
            details={
                "preset": resolved_preset,
                "time_budget": time_budget,
                "warmup_budget": warmup_budget,
                "repeats": repeats,
                "measured_pair_count": repeats * 2,
                "benchmark_skip_eval": benchmark_skip_eval,
                "no_checkpoint": no_checkpoint,
                "measured_order": details["measured_order"],
                "baseline_run_returncodes": details["baseline_run_returncodes"],
                "candidate_run_returncodes": details["candidate_run_returncodes"],
                "warmup_baseline_returncode": baseline_warmup.returncode,
                "warmup_candidate_returncode": candidate_warmup.returncode,
                "delta": delta,
                "delta_relative_pct": delta_relative_pct,
            },
        )
        return LabIntegrationABResult(
            engine="cuda",
            backend_family="cuda",
            target=target,
            preset=resolved_preset,
            workspace=str(workspace),
            status=status,
            wall_seconds=wall_seconds,
            details=details,
        )

    def run_integration_suite(
        self,
        *,
        workspace: Path,
        time_budget: float,
        preset: str | None = None,
        presets: tuple[str, ...] | None = None,
        repeats: int = 2,
        benchmark_skip_eval: bool = True,
        no_checkpoint: bool = True,
    ) -> LabIntegrationSuiteResult:
        workspace = workspace.expanduser().resolve()
        resolved_preset = preset or "upstream"
        selected_presets = tuple(dict.fromkeys(presets or (resolved_preset,)))
        start = time.perf_counter()
        runs = []
        for suite_preset in selected_presets:
            result = self.run_integration_ab(
                workspace=workspace,
                time_budget=time_budget,
                preset=suite_preset,
                repeats=repeats,
                benchmark_skip_eval=benchmark_skip_eval,
                no_checkpoint=no_checkpoint,
            )
            runs.append(result)
        wall_seconds = time.perf_counter() - start
        target = str(json.loads((workspace / "metadata.json").read_text(encoding="utf-8"))["target"])
        overall_summary = summarize_lab_evidence(
            engine="cuda",
            backend_family="cuda",
            target=target,
            preset=None,
        )
        status = "ok" if all(result.status == "ok" for result in runs) else (runs[-1].status if runs else "error")
        return LabIntegrationSuiteResult(
            engine="cuda",
            backend_family="cuda",
            target=target,
            presets=selected_presets,
            workspace=str(workspace),
            status=status,
            wall_seconds=wall_seconds,
            details={
                "selected_presets": selected_presets,
                "repeats": repeats,
                "benchmark_skip_eval": benchmark_skip_eval,
                "no_checkpoint": no_checkpoint,
                "runs": [asdict(result) for result in runs],
                "overall_evidence": asdict(overall_summary),
            },
        )

    def _run_train_command(
        self,
        cmd: list[str],
        stdout_path: Path,
        stderr_path: Path,
        *,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return completed

    def _summary_delta(
        self,
        baseline_summary: dict[str, str | float | int],
        candidate_summary: dict[str, str | float | int],
    ) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for key in ("steady_state_tok_per_sec", "peak_vram_mb", "val_bpb"):
            baseline = baseline_summary.get(key)
            candidate = candidate_summary.get(key)
            if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
                deltas[key] = float(candidate) - float(baseline)
        return deltas

    def _summary_delta_relative_pct(
        self,
        baseline_summary: dict[str, str | float | int],
        candidate_summary: dict[str, str | float | int],
    ) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for key in ("steady_state_tok_per_sec", "peak_vram_mb", "val_bpb"):
            baseline = baseline_summary.get(key)
            candidate = candidate_summary.get(key)
            if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)) and float(baseline) != 0.0:
                deltas[key] = ((float(candidate) - float(baseline)) / float(baseline)) * 100.0
        return deltas

    def _aggregate_summaries(
        self,
        summaries: list[dict[str, str | float | int]],
    ) -> dict[str, str | float]:
        if not summaries:
            return {}
        aggregated: dict[str, str | float] = {}
        keys = set().union(*(summary.keys() for summary in summaries))
        for key in sorted(keys):
            values = [summary.get(key) for summary in summaries]
            numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
            if len(numeric_values) == len(values) and numeric_values:
                aggregated[key] = float(statistics.median(numeric_values))
                continue
            first = values[0]
            if all(value == first for value in values):
                aggregated[key] = first  # type: ignore[assignment]
        return aggregated
