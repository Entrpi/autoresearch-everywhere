from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from autoresearch_lab.labs import (
    LabExtractResult,
    LabOrchestrationPlan,
    LabProfileCandidate,
    LabProfileResult,
)
from autoresearch_mlx.lab_trace import summarize_trace_metadata
from autoresearch_mlx.model import has_ve
from autoresearch_mlx.train import PRESETS, build_model_config


PROFILE_SCHEMA_VERSION = 1


def _current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _expand_window_pattern(pattern: str, n_layer: int) -> tuple[str, ...]:
    chars = tuple(pattern[i % len(pattern)].upper() for i in range(n_layer))
    if not chars:
        return chars
    return chars[:-1] + ("L",)


def _seq_factor(seq_len: int) -> float:
    if seq_len <= 256:
        return 1.0
    if seq_len <= 512:
        return 1.5
    if seq_len <= 1024:
        return 2.0
    return 2.5


def _candidate_payload(
    *,
    target: str,
    category: str,
    priority_score: float,
    status: str,
    rationale: str,
    rank: int,
    details: dict[str, object],
) -> LabProfileCandidate:
    return LabProfileCandidate(
        target=target,
        rank=rank,
        priority_score=round(priority_score, 3),
        category=category,
        status=status,
        rationale=rationale,
        details=details,
    )


def profile_mlx_targets(*, target_catalog: dict[str, object], preset: str, top_k: int = 10) -> LabProfileResult:
    if preset not in PRESETS:
        raise ValueError(f"Unknown MLX preset: {preset}")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    start = time.perf_counter()
    preset_config = PRESETS[preset]
    model_config = build_model_config(
        preset_config.depth,
        32768,
        sequence_len=preset_config.seq_len,
        window_pattern=preset_config.window_pattern,
    )

    seq_factor = _seq_factor(preset_config.seq_len)
    model_factor = model_config.n_embd / 256.0
    pattern = _expand_window_pattern(preset_config.window_pattern, model_config.n_layer)
    short_layers = sum(1 for char in pattern if char == "S")
    ve_layers = sum(1 for layer_idx in range(model_config.n_layer) if has_ve(layer_idx, model_config.n_layer))

    profile_details = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "git_commit": _current_git_commit(),
        "preset_description": preset_config.description,
        "seq_len": preset_config.seq_len,
        "window_pattern": preset_config.window_pattern,
        "expanded_window_pattern": "".join(pattern),
        "depth": model_config.n_layer,
        "n_embd": model_config.n_embd,
        "n_head": model_config.n_head,
        "n_kv_head": model_config.n_kv_head,
        "head_dim": model_config.n_embd // model_config.n_head,
        "vocab_size": model_config.vocab_size,
        "short_layers": short_layers,
        "value_embed_layers": ve_layers,
        "sequence_factor": seq_factor,
        "model_factor": model_factor,
    }

    scored: list[tuple[str, str, float, str, dict[str, object]]] = []

    def add(
        target: str,
        category: str,
        priority_score: float,
        rationale: str,
        details: dict[str, object],
    ) -> None:
        info = target_catalog[target]
        scored.append((target, category, priority_score, rationale, {"status": info.status, **details}))

    add(
        "fused_mlp",
        "mlp",
        5.0 + 0.60 * model_config.n_layer + 0.50 * seq_factor + 0.35 * model_factor,
        f"MLP work repeats in all {model_config.n_layer} layers and grows with model width ({model_config.n_embd}).",
        {"repeat_sites": model_config.n_layer, "path_scope": "all_layers"},
    )
    add(
        "block_prelude",
        "attention_block",
        4.8 + 0.55 * model_config.n_layer + 0.75 * seq_factor + 0.25 * short_layers,
        "Composed block setup covers residual blend, RMSNorm, and attention staging before SDPA.",
        {"repeat_sites": model_config.n_layer, "short_layers": short_layers, "path_scope": "block_setup"},
    )
    add(
        "attention_prelude",
        "attention_setup",
        4.4 + 0.45 * model_config.n_layer + 0.75 * seq_factor + 0.35 * short_layers,
        "Attention setup repeats every layer and becomes more important with longer context.",
        {"repeat_sites": model_config.n_layer, "short_layers": short_layers, "path_scope": "attention_setup"},
    )
    add(
        "cross_entropy_full",
        "loss_path",
        4.2 + 0.90 * seq_factor + 0.00005 * model_config.vocab_size,
        "Loss-side work is paid on every training step and directly tracks the large logits path.",
        {"path_scope": "loss_path", "vocab_size": model_config.vocab_size},
    )
    add(
        "rope_qk_fused",
        "attention_setup",
        3.6 + 0.40 * model_config.n_layer + 0.70 * seq_factor,
        "RoPE plus Q/K RMSNorm is a repeated attention-side subpath before SDPA.",
        {"repeat_sites": model_config.n_layer, "path_scope": "attention_setup"},
    )
    add(
        "residual_rmsnorm",
        "residual_norm",
        3.8 + 0.40 * model_config.n_layer + 0.40 * model_factor,
        "Residual blend plus RMSNorm is paid once per block before major compute.",
        {"repeat_sites": model_config.n_layer, "path_scope": "residual_norm"},
    )
    if ve_layers:
        add(
            "value_embed_gate",
            "value_embed",
            2.5 + 0.70 * ve_layers + 0.30 * seq_factor,
            f"Value-embed gating only appears in {ve_layers} layer(s), but it is a real attention-side branch.",
            {"repeat_sites": ve_layers, "path_scope": "value_embed"},
        )
        add(
            "ve_lookup_reshape",
            "value_embed",
            1.8 + 0.50 * ve_layers + 0.40 * seq_factor,
            f"Value-embed lookup/reshape feeds the same {ve_layers} value-embed layer(s).",
            {"repeat_sites": ve_layers, "path_scope": "value_embed"},
        )
    add(
        "qk_rmsnorm",
        "attention_setup",
        2.8 + 0.30 * model_config.n_layer + 0.50 * seq_factor,
        "Q/K RMSNorm is narrower than the fused RoPE target, but still repeats at every attention site.",
        {"repeat_sites": model_config.n_layer, "path_scope": "attention_setup"},
    )
    add(
        "logits_softcap",
        "loss_path",
        2.6 + 0.70 * seq_factor,
        "The logits softcap is tiny compared with the full loss path, but it is still on every step.",
        {"path_scope": "loss_path"},
    )
    add(
        "activation_pointwise",
        "mlp",
        2.5 + 0.25 * model_config.n_layer + 0.20 * model_factor,
        "Activation work repeats every layer, but is narrower than the full fused MLP target.",
        {"repeat_sites": model_config.n_layer, "path_scope": "mlp"},
    )
    add(
        "cross_entropy_prelude",
        "loss_path",
        2.2 + 0.50 * seq_factor,
        "The byte-aware reduction is part of the loss path, but narrower than the full composed loss target.",
        {"path_scope": "loss_path"},
    )
    add(
        "proj_head_reshape",
        "attention_support",
        2.0 + 0.25 * model_config.n_layer + 0.50 * seq_factor,
        "Attention output reshape is support-path work repeated once per block.",
        {"repeat_sites": model_config.n_layer, "path_scope": "attention_support"},
    )
    add(
        "attention_mask_local",
        "attention_support",
        (1.5 + 0.40 * short_layers + 0.50 * seq_factor) if short_layers else (0.5 + 0.25 * seq_factor),
        "Local mask construction matters more when short-window layers are present.",
        {"repeat_sites": short_layers, "path_scope": "attention_support"},
    )
    add(
        "rotary_embedding",
        "attention_support",
        1.6 + 0.20 * model_config.n_layer + 0.40 * seq_factor,
        "Standalone RoPE is a useful stepping stone, but the fused Q/K target is usually a better first destination.",
        {"repeat_sites": model_config.n_layer, "path_scope": "attention_support"},
    )
    add(
        "residual_blend",
        "residual_norm",
        1.4 + 0.20 * model_config.n_layer + 0.20 * model_factor,
        "Residual blend is repeated everywhere, but a better first target is the fused residual+RMSNorm path.",
        {"repeat_sites": model_config.n_layer, "path_scope": "residual_norm"},
    )
    add(
        "layernorm",
        "norm",
        1.2 + 0.15 * model_config.n_layer,
        "LayerNorm is useful as a lab target, but the current model path leans more heavily on RMSNorm.",
        {"repeat_sites": model_config.n_layer, "path_scope": "norm"},
    )
    add(
        "rmsnorm",
        "norm",
        1.3 + 0.15 * model_config.n_layer,
        "RMSNorm is common in the model path and remains a good low-risk benchmark target.",
        {"repeat_sites": model_config.n_layer, "path_scope": "norm"},
    )
    add(
        "rmsnorm_backward",
        "backward_norm",
        1.7 + 0.20 * model_config.n_layer,
        "Backward norm work is training-relevant, but still narrower than fused forward-path block targets.",
        {"repeat_sites": model_config.n_layer, "path_scope": "backward_norm"},
    )
    add(
        "layernorm_backward",
        "backward_norm",
        1.5 + 0.20 * model_config.n_layer,
        "LayerNorm backward is a useful harness stress test, but it is less central than RMSNorm in this model.",
        {"repeat_sites": model_config.n_layer, "path_scope": "backward_norm"},
    )
    add(
        "softmax",
        "primitive",
        0.9 + 0.20 * seq_factor,
        "Softmax is broadly useful, but the current model path already leans on optimized MLX attention primitives.",
        {"path_scope": "primitive"},
    )
    add(
        "reduce",
        "primitive",
        0.8 + 0.10 * seq_factor,
        "Reduction primitives are reusable, but further from an immediate end-to-end model-path win.",
        {"path_scope": "primitive"},
    )

    scored.sort(key=lambda item: (-item[2], item[0]))
    candidates = tuple(
        _candidate_payload(
            target=target,
            category=category,
            priority_score=priority_score,
            status=details["status"],
            rationale=rationale,
            rank=idx,
            details={
                key: value
                for key, value in {
                    **profile_details,
                    **details,
                }.items()
                if key != "status"
            },
        )
        for idx, (target, category, priority_score, rationale, details) in enumerate(scored[:top_k], start=1)
    )

    return LabProfileResult(
        engine="mlx",
        backend_family="mlx",
        preset=preset,
        status="ok",
        wall_seconds=time.perf_counter() - start,
        candidates=candidates,
        details={
            **profile_details,
            "top_k": top_k,
            "candidate_count": len(candidates),
        },
    )


def _load_profile_payload(profile_path: Path) -> dict:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("engine") != "mlx":
        raise ValueError(f"Profile {profile_path} is not an MLX lab profile.")
    return payload


def _pick_candidate(payload: dict, rank: int) -> dict:
    candidates = payload.get("candidates", [])
    if rank <= 0:
        raise ValueError("rank must be positive.")
    if rank > len(candidates):
        raise ValueError(f"Requested rank {rank}, but profile only contains {len(candidates)} candidates.")
    return candidates[rank - 1]


def extract_from_profile(
    *,
    init_workspace,
    profile_path: Path,
    workspace: Path,
    rank: int = 1,
) -> LabExtractResult:
    start = time.perf_counter()
    payload = _load_profile_payload(profile_path)
    candidate = _pick_candidate(payload, rank)
    target = candidate["target"]
    init_workspace(target=target, workspace=workspace)

    metadata_path = workspace / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["profile_context"] = {
        "profile_path": str(profile_path),
        "profile_rank": rank,
        "profile_target": target,
        "priority_score": candidate["priority_score"],
        "category": candidate["category"],
        "rationale": candidate["rationale"],
        "preset": payload["preset"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (workspace / "profile-context.json").write_text(
        json.dumps(
            {
                "profile": payload,
                "selected_candidate": candidate,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return LabExtractResult(
        engine="mlx",
        target=target,
        workspace=str(workspace),
        status="ok",
        wall_seconds=time.perf_counter() - start,
        details={
            "profile_path": str(profile_path),
            "profile_rank": rank,
            "preset": payload["preset"],
            "priority_score": candidate["priority_score"],
            "category": candidate["category"],
        },
    )


def orchestrate_from_profile(
    *,
    profile_path: Path,
    workspace_root: Path,
    rank: int = 1,
    trace_metadata_path: Path | None = None,
) -> LabOrchestrationPlan:
    payload = _load_profile_payload(profile_path)
    candidate = _pick_candidate(payload, rank)
    target = candidate["target"]
    workspace = workspace_root / f"{payload['preset']}-{target}"
    trace_summary: dict[str, object] | None = None
    trace_commands: list[str] = []
    status = "ok"
    if trace_metadata_path is not None:
        trace_summary = summarize_trace_metadata(trace_metadata_path)
        if trace_summary["trace_target"] != target:
            raise ValueError(
                f"Trace metadata target {trace_summary['trace_target']!r} does not match "
                f"profile target {target!r}."
            )
        trace_metadata = Path(str(trace_summary["trace_metadata_path"]))
        trace_path = Path(str(trace_metadata).removesuffix(".metadata.json") + ".gputrace")
        trace_commands.extend(
            [
                f"# inspect {trace_path} in Xcode Metal Debugger before editing",
                f"# review {trace_metadata_path} for bench and device context",
            ]
        )
        status = "trace-backed"
    commands = tuple(
        [
            f"uv run kernel-lab.py --engine mlx init --target {target} --workspace {workspace}",
            *trace_commands,
            f"# edit {workspace / 'kernel.py'}",
            f"uv run kernel-lab.py --engine mlx verify --workspace {workspace} --quick",
            f"uv run kernel-lab.py --engine mlx bench --workspace {workspace}",
            f"uv run kernel-lab.py --engine mlx capture --workspace {workspace} --output {workspace / (target + '.gputrace')} --quick",
        ]
    )
    return LabOrchestrationPlan(
        engine="mlx",
        target=target,
        workspace=str(workspace),
        status=status,
        commands=commands,
        details={
            "profile_path": str(profile_path),
            "profile_rank": rank,
            "preset": payload["preset"],
            "priority_score": candidate["priority_score"],
            "category": candidate["category"],
            "rationale": candidate["rationale"],
            "trace_summary": trace_summary,
        },
    )


def write_profile_result(result: LabProfileResult, output: Path) -> None:
    output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
