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
from pathlib import Path
from typing import Any

from autoresearch_cuda.config import CUDA_PRESETS
from autoresearch_cuda.runtime import detect_cuda_runtime_profile, query_nvidia_driver_version
from autoresearch_lab.ledger import append_lab_event
from autoresearch_lab.labs import (
    LabEvidenceResult,
    LabAutoTraceReviewResult,
    LabCapabilities,
    LabOrchestrationPlan,
    LabPromotionCheck,
    LabTarget,
    LabTraceProfileCandidate,
    LabTraceProfileResult,
    LabTraceResult,
)
from autoresearch_lab.ledger import summarize_lab_evidence


TRACE_SCHEMA_VERSION = 1
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
        status="trace-ready",
        notes="Good target family when attention glue outruns the core attention kernel.",
    ),
    "norm": LabTarget(
        key="norm",
        description="RMSNorm / LayerNorm family kernels",
        metric="time_share_pct",
        status="trace-ready",
    ),
    "fused_mlp": LabTarget(
        key="fused_mlp",
        description="MLP / activation / feed-forward path kernels",
        metric="time_share_pct",
        status="trace-ready",
    ),
    "loss_prelude": LabTarget(
        key="loss_prelude",
        description="Softmax / logits / cross-entropy-side kernels",
        metric="time_share_pct",
        status="trace-ready",
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
        status="trace-ready",
    ),
    "data_movement": LabTarget(
        key="data_movement",
        description="Copy, cast, transpose, and data-movement kernels",
        metric="time_share_pct",
        status="trace-ready",
    ),
    "launch_fusion": LabTarget(
        key="launch_fusion",
        description="Launch-bound regions where fusion or batching may matter more than a single kernel rewrite",
        metric="time_share_pct",
        status="trace-ready",
        notes="This is a workflow target family, not one kernel workspace.",
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
        "attention_prelude",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"softmax",
                r"mask",
                r"rotary",
                r"\brope\b",
                r"\bqk\b",
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
        "loss_prelude",
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"cross.?entropy",
                r"xentropy",
                r"logits",
                r"softcap",
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


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
        if dominant_issue == "kernel-dominated" and family in {"flash_attention", "fused_mlp", "matmul_epilogue", "norm"}:
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
        supports_workspace_init=False,
        supports_fixed_bench=False,
        supports_profile=False,
        supports_extract=False,
        supports_orchestrate=True,
        supports_verify=False,
        supports_capture=True,
        supports_trace_profile=True,
        supports_auto_trace_review=True,
    )

    def target_catalog(self) -> dict[str, LabTarget]:
        return CUDA_TRACE_TARGETS

    def init_workspace(self, *, target: str, workspace: Path) -> Path:
        raise NotImplementedError("CUDA kernel workspaces are not implemented yet; use capture/trace-profile/auto-review first.")

    def bench_workspace(self, *, workspace: Path, quick: bool = False):
        raise NotImplementedError("CUDA fixed bench workspaces are not implemented yet.")

    def verify_workspace(self, *, workspace: Path, quick: bool = False):
        raise NotImplementedError("CUDA fixed bench workspaces are not implemented yet.")

    def profile_targets(self, *, preset: str, top_k: int = 10):
        raise NotImplementedError("CUDA heuristic target profiling is not implemented yet; start from a real trace.")

    def extract_from_profile(self, *, profile_path: Path, workspace: Path, rank: int = 1):
        raise NotImplementedError("CUDA workspace extraction is not implemented yet.")

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
            commands.extend(
                [
                    "# this target is trace-backed and is a good candidate for future CUDA/Triton workspace work",
                    "# once CUDA workspaces exist, start from this target family before chasing lower-ranked kernels",
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
                "notes": "CUDA orchestration is trace-first today; no fixed workspace implementation exists yet.",
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
        elif summary.auto_review_ok_count > 0 or summary.trace_profile_ok_count > 0:
            status = "ready-for-cuda-workspace"
            commands = (
                "# this target is trace-backed and is the right place to start once CUDA/Triton workspaces are added",
                "# until then, keep collecting traces and use `evidence` to confirm that it stays important",
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
                "notes": "CUDA promotion currently stops at trace-backed prioritization because Triton/CUDA workspaces are not implemented yet.",
            },
        )

    def run_integration_ab(self, *, workspace: Path, time_budget: float, preset: str | None = None, repeats: int = 2, benchmark_skip_eval: bool = True, no_checkpoint: bool = True):
        raise NotImplementedError("CUDA lab integration A/B is not implemented yet.")

    def run_integration_suite(self, *, workspace: Path, time_budget: float, preset: str | None = None, presets: tuple[str, ...] | None = None, repeats: int = 2, benchmark_skip_eval: bool = True, no_checkpoint: bool = True):
        raise NotImplementedError("CUDA lab integration A/B is not implemented yet.")
