from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEDGER_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAB_LEDGER_PATH = REPO_ROOT / "results" / "kernel_lab" / "ledger.jsonl"


@dataclass(frozen=True)
class LabEvidenceSummary:
    engine: str
    backend_family: str
    target: str
    preset: str | None
    total_events: int
    verify_ok_count: int
    capture_ok_count: int
    trace_review_ok_count: int
    integration_ab_ok_count: int
    last_event_at: str | None
    last_verify_at: str | None
    last_capture_at: str | None
    last_trace_review_at: str | None
    last_integration_ab_at: str | None
    last_workspace: str | None
    last_trace_metadata_path: str | None
    unique_workspaces: int
    promotion_status: str
    evidence_bonus: float
    details: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return None


def _read_workspace_context(workspace: Path) -> dict[str, Any]:
    metadata_path = workspace / "metadata.json"
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile_context = payload.get("profile_context", {})
    return {
        "target": payload.get("target"),
        "metric": payload.get("metric"),
        "workspace_status": payload.get("status"),
        "preset": profile_context.get("preset"),
        "profile_path": profile_context.get("profile_path"),
        "profile_rank": profile_context.get("profile_rank"),
        "priority_score": profile_context.get("priority_score"),
        "category": profile_context.get("category"),
        "rationale": profile_context.get("rationale"),
    }


def append_lab_event(
    *,
    engine: str,
    backend_family: str,
    target: str,
    workspace: Path,
    event_type: str,
    status: str,
    details: dict[str, Any],
    preset: str | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    ledger_path: Path = DEFAULT_LAB_LEDGER_PATH,
) -> None:
    workspace = workspace.expanduser()
    context = _read_workspace_context(workspace)
    entry = {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "git_commit": _current_git_commit(),
        "engine": engine,
        "backend_family": backend_family,
        "target": target,
        "workspace": str(workspace),
        "event_type": event_type,
        "status": status,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "preset": preset if preset is not None else context.get("preset"),
        "context": context,
        "details": details,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def load_lab_events(
    *,
    ledger_path: Path = DEFAULT_LAB_LEDGER_PATH,
    engine: str | None = None,
    backend_family: str | None = None,
    target: str | None = None,
    preset: str | None = None,
) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
            continue
        if engine is not None and payload.get("engine") != engine:
            continue
        if backend_family is not None and payload.get("backend_family") != backend_family:
            continue
        if target is not None and payload.get("target") != target:
            continue
        if preset is not None and payload.get("preset") != preset:
            continue
        events.append(payload)
    return events


def summarize_lab_evidence(
    *,
    engine: str,
    backend_family: str,
    target: str,
    preset: str | None,
    ledger_path: Path = DEFAULT_LAB_LEDGER_PATH,
) -> LabEvidenceSummary:
    events = load_lab_events(
        ledger_path=ledger_path,
        engine=engine,
        backend_family=backend_family,
        target=target,
        preset=preset,
    )
    events.sort(key=lambda item: item.get("created_at", ""))

    verify_ok = [event for event in events if event.get("event_type") == "verify" and event.get("status") == "ok"]
    capture_ok = [event for event in events if event.get("event_type") == "capture" and event.get("status") == "ok"]
    trace_review_ok = [
        event for event in events if event.get("event_type") == "trace-review" and event.get("status") == "ok"
    ]
    integration_ab_ok = [
        event for event in events if event.get("event_type") == "integration-ab" and event.get("status") == "ok"
    ]
    last_trace_review = trace_review_ok[-1] if trace_review_ok else None
    trace_relevance = last_trace_review.get("details", {}).get("relevance") if last_trace_review else None
    last_integration_ab = integration_ab_ok[-1] if integration_ab_ok else None
    integration_deltas = [
        event.get("details", {}).get("delta", {}).get("steady_state_tok_per_sec")
        for event in integration_ab_ok
        if event.get("details", {}).get("delta", {}).get("steady_state_tok_per_sec") is not None
    ]
    positive_integration_count = sum(1 for value in integration_deltas if value > 0)
    negative_integration_count = sum(1 for value in integration_deltas if value < 0)
    zero_integration_count = sum(1 for value in integration_deltas if value == 0)
    integration_delta = integration_deltas[-1] if integration_deltas else None
    median_integration_delta = (
        float(statistics.median(integration_deltas))
        if integration_deltas
        else None
    )

    if trace_relevance == "none":
        promotion_status = "trace-deprioritized"
        evidence_bonus = -0.75
    elif trace_relevance == "low":
        promotion_status = "trace-deprioritized"
        evidence_bonus = -0.35
    elif integration_ab_ok:
        if positive_integration_count and negative_integration_count:
            promotion_status = "integration-mixed"
            evidence_bonus = 0.2
        elif integration_delta is not None and integration_delta > 0:
            promotion_status = "integration-validated"
            evidence_bonus = 1.5
        elif integration_delta is not None and integration_delta < 0:
            promotion_status = "integration-regressed"
            evidence_bonus = -0.5
        else:
            promotion_status = "integration-tested"
            evidence_bonus = 1.1
    elif verify_ok and capture_ok:
        promotion_status = "ready-for-integration-test"
        evidence_bonus = 1.2 if trace_relevance == "high" else 1.0
    elif capture_ok:
        promotion_status = "trace-backed"
        evidence_bonus = 0.65 if trace_relevance == "high" else 0.5
    elif verify_ok:
        promotion_status = "verified-only"
        evidence_bonus = 0.25
    else:
        promotion_status = "no-evidence"
        evidence_bonus = 0.0

    last_capture = capture_ok[-1] if capture_ok else None
    return LabEvidenceSummary(
        engine=engine,
        backend_family=backend_family,
        target=target,
        preset=preset,
        total_events=len(events),
        verify_ok_count=len(verify_ok),
        capture_ok_count=len(capture_ok),
        trace_review_ok_count=len(trace_review_ok),
        integration_ab_ok_count=len(integration_ab_ok),
        last_event_at=events[-1]["created_at"] if events else None,
        last_verify_at=verify_ok[-1]["created_at"] if verify_ok else None,
        last_capture_at=last_capture["created_at"] if last_capture else None,
        last_trace_review_at=last_trace_review["created_at"] if last_trace_review else None,
        last_integration_ab_at=last_integration_ab["created_at"] if last_integration_ab else None,
        last_workspace=events[-1]["workspace"] if events else None,
        last_trace_metadata_path=(
            last_capture.get("details", {}).get("trace_metadata_path") if last_capture else None
        ),
        unique_workspaces=len({event["workspace"] for event in events}),
        promotion_status=promotion_status,
        evidence_bonus=evidence_bonus,
        details={
            "last_metric_name": events[-1].get("metric_name") if events else None,
            "last_metric_value": events[-1].get("metric_value") if events else None,
            "last_trace_relevance": trace_relevance,
            "last_integration_delta_steady_state_tok_per_sec": integration_delta,
            "integration_ab_positive_count": positive_integration_count,
            "integration_ab_negative_count": negative_integration_count,
            "integration_ab_zero_count": zero_integration_count,
            "integration_ab_median_delta_steady_state_tok_per_sec": median_integration_delta,
            "recommended_next_step": (
                "reprioritize"
                if promotion_status == "trace-deprioritized"
                else "promote"
                if promotion_status == "integration-validated"
                else "stabilize-integration"
                if promotion_status == "integration-mixed"
                else "review-integration"
                if promotion_status in {"integration-tested", "integration-regressed"}
                else "integration-test"
                if promotion_status == "ready-for-integration-test"
                else "verify"
                if promotion_status == "trace-backed"
                else "capture"
                if promotion_status == "verified-only"
                else "init"
            ),
        },
    )
