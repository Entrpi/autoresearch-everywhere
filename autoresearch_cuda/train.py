"""
CUDA training implementation for autoresearch.
Top-level callers should prefer `uv run train.py --engine cuda`.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import json
import gc
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

from autoresearch_cuda.config import CUDA_PRESETS, resolve_run_preset
from autoresearch_platform.lr_profile import (
    LR_MULTIPLIER_ARG_FIELDS,
    LrMultipliers,
    ResolvedLrProfile,
    apply_lr_multipliers,
    lr_multipliers_from_mapping,
    lr_multipliers_to_dict,
    lr_profile_from_mapping,
    lr_profile_to_dict,
    resolve_effective_lr_profile,
)
from autoresearch_platform.streaming_eval import (
    DEFAULT_STREAMING_EVAL_INTERVAL_STEPS,
    DEFAULT_STREAMING_EVAL_TOKENS,
    STREAMING_EVAL_MODES,
    STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH,
    streaming_mode_uses_cheap,
    streaming_mode_uses_subref,
    StreamingEvalConfig,
    StreamingSupercycleAccumulator,
    StreamingEvalTracker,
)

DEFAULT_CUDA_STREAMING_SEQ_LEN = 2048
DEFAULT_CUDA_STREAMING_BATCH_SIZE = 32
DEFAULT_CUDA_SUBREF_ONE_SIXTH_TOKENS = 262144


def build_parser():
    parser = argparse.ArgumentParser(
        prog=os.environ.get("AUTORESEARCH_ENTRYPOINT_PROG"),
        description="Autoresearch CUDA training script.",
    )
    parser.add_argument("--preset", choices=tuple(CUDA_PRESETS.keys()), default="upstream")
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Directory containing a previously saved CUDA training checkpoint.",
    )
    parser.add_argument("--time-budget", type=float, help="Training time budget in seconds.")
    parser.add_argument("--token-budget", type=int, help="Training token budget.")
    parser.add_argument("--seq-len", type=int, help="Sequence length override.")
    parser.add_argument("--window-pattern", type=str, help="Window pattern override.")
    parser.add_argument("--total-batch-size", type=int, help="Total batch size override.")
    parser.add_argument("--depth", type=int, help="Transformer depth override.")
    parser.add_argument("--device-batch-size", type=int, help="Per-device batch size override.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short end-to-end sanity check with smaller defaults.",
    )
    parser.add_argument(
        "--benchmark-skip-eval",
        action="store_true",
        help="Skip the final validation eval and only report training throughput.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Directory to save a resumable training checkpoint. If omitted, eligible runs auto-select a checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=str,
        help="Save a checkpoint every N training seconds or tokens, for example '300', '5m', '50Mtok', or '300s,50Mtok'. If no path is provided, an automatic checkpoint directory is used.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writing, including the automatic defaults for longer and token-budgeted runs.",
    )
    parser.add_argument(
        "--checkpoint-save-mode",
        choices=("sync", "async"),
        help="Checkpoint write path. 'sync' writes on the training thread; 'async' captures an exact CPU snapshot and writes it in the background.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate a resumed checkpoint with the configured eval settings.",
    )
    parser.add_argument(
        "--eval-seq-len",
        type=int,
        help="Sequence length for eval-only validation.",
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        help="Token budget for eval-only validation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Batch size for eval-only validation.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for short probes and debugging runs.",
    )
    parser.add_argument(
        "--curve-eval-seconds",
        type=str,
        help="Comma-separated training-time checkpoints in seconds for periodic validation evals.",
    )
    parser.add_argument(
        "--curve-output",
        type=str,
        help="Optional JSON path to write periodic validation curve data to.",
    )
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        help="Multiply all preset LR groups by this scalar while preserving their ratios.",
    )
    parser.add_argument(
        "--embedding-lr-multiplier",
        type=float,
        help="Additional multiplier for embedding-side LR groups after the global LR multiplier.",
    )
    parser.add_argument(
        "--unembedding-lr-multiplier",
        type=float,
        help="Additional multiplier for the lm_head / unembedding LR after the global LR multiplier.",
    )
    parser.add_argument(
        "--matrix-lr-multiplier",
        type=float,
        help="Additional multiplier for Muon matrix LR groups after the global LR multiplier.",
    )
    parser.add_argument(
        "--scalar-lr-multiplier",
        type=float,
        help="Additional multiplier for scalar LR groups after the global LR multiplier.",
    )
    parser.add_argument(
        "--streaming-eval-interval-steps",
        type=int,
        help=f"Run one deterministic validation batch every N training steps. Default for discovery flows is {DEFAULT_STREAMING_EVAL_INTERVAL_STEPS}.",
    )
    parser.add_argument(
        "--streaming-eval-mode",
        choices=STREAMING_EVAL_MODES,
        help="Streaming eval mode. 'subref-one-sixth' is the default online path; 'cheap' favors repeated comparable cycles; 'both' enables both streams.",
    )
    parser.add_argument(
        "--streaming-eval-tokens",
        type=int,
        help=f"Token budget represented by one full streaming-eval cycle. Defaults to the cheap/subref-one-sixth budget ({DEFAULT_CUDA_SUBREF_ONE_SIXTH_TOKENS}).",
    )
    parser.add_argument(
        "--streaming-eval-seq-len",
        type=int,
        help=f"Sequence length for deterministic streaming validation batches. Defaults to the canonical streaming seq_len ({DEFAULT_CUDA_STREAMING_SEQ_LEN}).",
    )
    parser.add_argument(
        "--streaming-eval-batch-size",
        type=int,
        help=f"Batch size for deterministic streaming validation batches. Defaults to the canonical streaming batch size ({DEFAULT_CUDA_STREAMING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--streaming-eval-history-output",
        type=str,
        help="Optional JSON path to write deterministic streaming validation history.",
    )
    parser.add_argument(
        "--complete-streaming-eval-cycle",
        action="store_true",
        help="If a run stops mid streaming-eval cycle, continue until the current cycle boundary so the final honest metric is complete.",
    )
    parser.add_argument("--probe-pathology-max-auc", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--probe-pathology-max-bpb", type=float, help=argparse.SUPPRESS)
    return parser


ARGS = build_parser().parse_args()
if ARGS.time_budget is not None and ARGS.token_budget is not None:
    raise SystemExit("--time-budget and --token-budget are mutually exclusive.")


def _lr_multipliers_from_args(args) -> LrMultipliers:
    return LrMultipliers(
        lr_multiplier=1.0 if args.lr_multiplier is None else args.lr_multiplier,
        embedding_lr_multiplier=(
            1.0 if args.embedding_lr_multiplier is None else args.embedding_lr_multiplier
        ),
        unembedding_lr_multiplier=(
            1.0 if args.unembedding_lr_multiplier is None else args.unembedding_lr_multiplier
        ),
        matrix_lr_multiplier=1.0 if args.matrix_lr_multiplier is None else args.matrix_lr_multiplier,
        scalar_lr_multiplier=1.0 if args.scalar_lr_multiplier is None else args.scalar_lr_multiplier,
    )


def _parse_curve_eval_seconds(value: str | None) -> list[float]:
    if not value:
        return []
    checkpoints: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        checkpoint = float(item)
        if checkpoint <= 0:
            raise ValueError("--curve-eval-seconds values must be positive.")
        checkpoints.append(checkpoint)
    return sorted(dict.fromkeys(checkpoints))


CURVE_EVAL_SECONDS = _parse_curve_eval_seconds(ARGS.curve_eval_seconds)


def _resolve_streaming_eval_config(args, *, seq_len: int, device_batch_size: int, resume_run_config: dict | None = None):
    saved = resume_run_config or {}
    interval_steps = (
        args.streaming_eval_interval_steps
        if args.streaming_eval_interval_steps is not None
        else saved.get("streaming_eval_interval_steps")
    )
    if interval_steps is None:
        return None
    if interval_steps <= 0:
        raise ValueError("--streaming-eval-interval-steps must be positive.")
    mode = (
        args.streaming_eval_mode
        if args.streaming_eval_mode is not None
        else saved.get("streaming_eval_mode")
        or STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH
    )
    if mode not in STREAMING_EVAL_MODES:
        raise ValueError(f"Unsupported --streaming-eval-mode: {mode!r}")
    eval_tokens = (
        args.streaming_eval_tokens
        if args.streaming_eval_tokens is not None
        else saved.get("streaming_eval_tokens")
        or DEFAULT_CUDA_SUBREF_ONE_SIXTH_TOKENS
    )
    if eval_tokens <= 0:
        raise ValueError("--streaming-eval-tokens must be positive.")
    streaming_seq_len = (
        args.streaming_eval_seq_len
        if args.streaming_eval_seq_len is not None
        else saved.get("streaming_eval_seq_len")
        or DEFAULT_CUDA_STREAMING_SEQ_LEN
    )
    streaming_batch_size = (
        args.streaming_eval_batch_size
        if args.streaming_eval_batch_size is not None
        else saved.get("streaming_eval_batch_size")
        or DEFAULT_CUDA_STREAMING_BATCH_SIZE
    )
    complete_cycle_on_budget = bool(
        args.complete_streaming_eval_cycle
        or saved.get("complete_streaming_eval_cycle", False)
    )
    return StreamingEvalConfig(
        interval_steps=interval_steps,
        eval_tokens=eval_tokens,
        batch_size=streaming_batch_size,
        seq_len=streaming_seq_len,
        mode=mode,
        complete_cycle_on_budget=complete_cycle_on_budget,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoresearch_cuda.lab_integration import maybe_call_integration_target
from autoresearch_cuda.runtime import detect_cuda_runtime_profile
from autoresearch_cuda.prepare import (
    EVAL_TOKENS as CUDA_EVAL_TOKENS,
    Tokenizer,
    make_dataloader,
    evaluate_bpb,
    evaluate_bpb_batch,
    evaluate_bpb_batch_stats,
    evaluate_bpb_configured,
    get_token_bytes,
)
from autoresearch_cuda.eval_policy import (
    CUDA_REFERENCE_RUNG,
    CUDA_SUBREF_ONE_SIXTH_RUNG,
    EVAL_POLICY_VERSION,
    choose_runtime_eval,
    current_eval_semantics_signature,
    current_runtime_shape_signature,
    find_calibration,
    has_calibration_for_preset,
)
from autoresearch_cuda.checkpoint_policy import (
    AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC,
    AUTO_CHECKPOINT_MIN_TOKEN_BUDGET,
    auto_checkpoint_due,
    choose_auto_checkpoint_plan,
    default_auto_checkpoint_path,
    format_interval_label,
    format_interval_spec,
    parse_checkpoint_interval_spec,
)
from autoresearch_cuda.checkpoints import (
    CHECKPOINT_SAVE_MODE_ASYNC,
    CHECKPOINT_SAVE_MODE_SYNC,
    CHECKPOINT_SAVE_MODES,
    capture_async_checkpoint_snapshot,
    load_checkpoint_metadata,
    load_training_checkpoint,
    make_async_checkpoint_writer,
    save_training_checkpoint,
    validate_checkpoint_payload,
)
from autoresearch_platform.step_telemetry import (
    OBS_INPUT_PIPELINE,
    PHASE_FORWARD_BACKWARD,
    PHASE_LOADER,
    PHASE_OPTIMIZER,
    StepTelemetry,
    StepTiming,
    summarize_step_telemetry,
)


SUBREF_ONE_SIXTHS_PER_REFERENCE, _SUBREF_REFERENCE_REMAINDER = divmod(
    CUDA_REFERENCE_RUNG.eval_tokens,
    CUDA_SUBREF_ONE_SIXTH_RUNG.eval_tokens,
)
if _SUBREF_REFERENCE_REMAINDER != 0:
    raise SystemExit("CUDA subref-one-sixth must divide the reference rung exactly.")
def _resolved_run_config_from_checkpoint(checkpoint_dir: str):
    metadata = load_checkpoint_metadata(checkpoint_dir)
    bundle = load_training_checkpoint(checkpoint_dir, map_location="cpu")
    validate_checkpoint_payload(bundle)
    if metadata.get("version") != bundle.get("version"):
        raise ValueError(
            f"Checkpoint metadata/version mismatch: metadata={metadata.get('version')!r}, "
            f"bundle={bundle.get('version')!r}."
        )
    run_config = bundle["run_config"]
    training_state = bundle["training_state"]
    return metadata, bundle, run_config, training_state


def _resolve_checkpoint_settings(
    args,
    *,
    num_params_m: float,
    checkpoint_mode: str,
    preset: str,
    time_budget: float | None,
    token_budget: int | None,
    seq_len: int,
    depth: int,
    total_batch_size: int,
    window_pattern: str,
    resume_run_config: dict | None = None,
):
    checkpoint_save_mode = (
        args.checkpoint_save_mode
        or (resume_run_config or {}).get("checkpoint_save_mode")
        or CHECKPOINT_SAVE_MODE_SYNC
    )
    if checkpoint_save_mode not in CHECKPOINT_SAVE_MODES:
        raise ValueError(
            f"Unsupported checkpoint_save_mode {checkpoint_save_mode!r}. Expected one of {CHECKPOINT_SAVE_MODES}."
        )
    if args.no_checkpoint:
        return None, None, checkpoint_save_mode, "disabled via --no-checkpoint", False
    auto_path = default_auto_checkpoint_path(
        preset,
        seq_len=seq_len,
        depth=depth,
        total_batch_size=total_batch_size,
        window_pattern=window_pattern,
        checkpoint_save_mode=checkpoint_save_mode,
    )
    if args.checkpoint_interval is not None:
        interval = parse_checkpoint_interval_spec(args.checkpoint_interval)
        checkpoint_path = args.checkpoint_path or args.resume_from or str(auto_path)
        path_source = (
            "existing path"
            if args.checkpoint_path is not None
            else "resume path"
            if args.resume_from
            else "auto path"
        )
        reason = (
            f"using explicit checkpoint interval {format_interval_label(interval)} "
            f"with {path_source}; checkpoint_save_mode={checkpoint_save_mode}"
        )
        return checkpoint_path, interval, checkpoint_save_mode, reason, False
    if args.resume_from:
        checkpoint_path = args.checkpoint_path or (resume_run_config or {}).get("checkpoint_path") or args.resume_from
        checkpoint_interval = parse_checkpoint_interval_spec((resume_run_config or {}).get("checkpoint_interval"))
        checkpoint_interval_is_auto = bool((resume_run_config or {}).get("checkpoint_interval_is_auto", False))
        if checkpoint_interval is not None:
            reason = (
                f"reusing resume checkpoint path and interval "
                f"({format_interval_label(checkpoint_interval)}); checkpoint_save_mode={checkpoint_save_mode}"
            )
            return checkpoint_path, checkpoint_interval, checkpoint_save_mode, reason, checkpoint_interval_is_auto
        return checkpoint_path, None, checkpoint_save_mode, (
            f"reusing resume checkpoint path; checkpoint_save_mode={checkpoint_save_mode}"
        ), False
    auto_plan = choose_auto_checkpoint_plan(
        num_params_m=num_params_m,
        checkpoint_mode=checkpoint_mode,
        checkpoint_save_mode=checkpoint_save_mode,
        time_budget=time_budget,
        token_budget=token_budget,
    )
    if auto_plan is not None:
        checkpoint_path = args.checkpoint_path or str(auto_path)
        checkpoint_interval = auto_plan.interval
        if auto_plan.trigger_mode == "time":
            if auto_plan.calibration is not None and auto_plan.recommendation is not None:
                resume_suffix = (
                    f"; measured resume-ready penalty {auto_plan.calibration.resume_ready_penalty_sec:.3f}s"
                    if auto_plan.calibration.resume_ready_source
                    else ""
                )
                reason = (
                    f"auto-enabled for time_budget>{AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC:.0f}s using "
                    f"{auto_plan.calibration.label}; selected {auto_plan.recommendation.interval_label} "
                    f"at {auto_plan.recommendation.save_only_overhead_fraction * 100.0:.4f}% save-only overhead; "
                    f"checkpoint_save_mode={checkpoint_save_mode}{resume_suffix}"
                )
            else:
                reason = (
                    f"auto-enabled for time_budget>{AUTO_CHECKPOINT_MIN_TIME_BUDGET_SEC:.0f}s "
                    f"with fallback {auto_plan.interval_label} interval; "
                    f"checkpoint_save_mode={checkpoint_save_mode}"
                )
        else:
            if auto_plan.calibration is not None and auto_plan.recommendation is not None:
                resume_suffix = (
                    f"; measured resume-ready penalty {auto_plan.calibration.resume_ready_penalty_sec:.3f}s"
                    if auto_plan.calibration.resume_ready_source
                    else ""
                )
                reason = (
                    f"auto-enabled for token_budget>={AUTO_CHECKPOINT_MIN_TOKEN_BUDGET:,} using "
                    f"{auto_plan.calibration.label}; selected earlier-of {auto_plan.interval_label} "
                    f"at {auto_plan.recommendation.save_only_overhead_fraction * 100.0:.4f}% save-only overhead; "
                    f"checkpoint_save_mode={checkpoint_save_mode}{resume_suffix}"
                )
            else:
                reason = (
                    f"auto-enabled for token_budget>={AUTO_CHECKPOINT_MIN_TOKEN_BUDGET:,} "
                    f"with earlier-of {auto_plan.interval_label} interval; "
                    f"checkpoint_save_mode={checkpoint_save_mode}"
                )
        return checkpoint_path, checkpoint_interval, checkpoint_save_mode, reason, True
    if args.checkpoint_path is not None:
        return args.checkpoint_path, None, checkpoint_save_mode, (
            f"using explicit checkpoint path without periodic autosaves; checkpoint_save_mode={checkpoint_save_mode}"
        ), False
    return None, None, checkpoint_save_mode, None, False


def _resolve_run_preset_with_resume(args):
    if not args.resume_from:
        resolved = resolve_run_preset(
            args.preset,
            time_budget=args.time_budget,
            seq_len=args.seq_len,
            window_pattern=args.window_pattern,
            total_batch_size=args.total_batch_size,
            depth=args.depth,
            device_batch_size=args.device_batch_size,
        )
        if args.smoke:
            smoke_seq_len = min(resolved.seq_len, 256)
            smoke_depth = min(resolved.depth, 2)
            smoke_device_batch_size = min(resolved.device_batch_size, 2)
            resolved = replace(
                resolved,
                time_budget=1.0,
                seq_len=smoke_seq_len,
                window_pattern="L",
                total_batch_size=smoke_seq_len * smoke_device_batch_size,
                depth=smoke_depth,
                device_batch_size=smoke_device_batch_size,
            )
            resolved = replace(
                resolved,
                time_budget=args.time_budget if args.time_budget is not None else resolved.time_budget,
                seq_len=args.seq_len if args.seq_len is not None else resolved.seq_len,
                window_pattern=args.window_pattern if args.window_pattern is not None else resolved.window_pattern,
                total_batch_size=args.total_batch_size if args.total_batch_size is not None else resolved.total_batch_size,
                depth=args.depth if args.depth is not None else resolved.depth,
                device_batch_size=(
                    args.device_batch_size
                    if args.device_batch_size is not None
                    else resolved.device_batch_size
                ),
            )
        return resolved, None, None

    metadata, bundle, run_config, training_state = _resolved_run_config_from_checkpoint(args.resume_from)
    checkpoint_preset = run_config["preset"]
    if args.preset != checkpoint_preset:
        raise ValueError(
            f"--resume-from expects preset {checkpoint_preset!r}; received {args.preset!r}."
        )
    if args.smoke:
        raise ValueError("--resume-from cannot be combined with --smoke.")

    disallowed_overrides: list[str] = []
    if args.seq_len is not None and args.seq_len != run_config["seq_len"]:
        disallowed_overrides.append("--seq-len")
    if args.window_pattern is not None and args.window_pattern != run_config["window_pattern"]:
        disallowed_overrides.append("--window-pattern")
    if args.total_batch_size is not None and args.total_batch_size != run_config["total_batch_size"]:
        disallowed_overrides.append("--total-batch-size")
    if args.depth is not None and args.depth != run_config["depth"]:
        disallowed_overrides.append("--depth")
    if args.device_batch_size is not None and args.device_batch_size != run_config["device_batch_size"]:
        disallowed_overrides.append("--device-batch-size")
    saved_lr_multipliers = lr_multipliers_from_mapping(run_config)
    for field in LR_MULTIPLIER_ARG_FIELDS:
        arg_value = getattr(args, field)
        if arg_value is None:
            continue
        if arg_value != getattr(saved_lr_multipliers, field):
            disallowed_overrides.append(f"--{field.replace('_', '-')}")
    if disallowed_overrides:
        raise ValueError(
            "Resume does not allow shape-changing overrides: "
            + ", ".join(disallowed_overrides)
        )

    resolved = resolve_run_preset(
        checkpoint_preset,
        time_budget=args.time_budget if args.time_budget is not None else run_config["time_budget"],
        seq_len=run_config["seq_len"],
        window_pattern=run_config["window_pattern"],
        total_batch_size=run_config["total_batch_size"],
        depth=run_config["depth"],
        device_batch_size=run_config["device_batch_size"],
    )
    return resolved, bundle, training_state


RUN_PRESET, RESUME_BUNDLE, RESUME_TRAINING_STATE = _resolve_run_preset_with_resume(ARGS)

if ARGS.eval_only and ARGS.benchmark_skip_eval:
    raise ValueError("--eval-only cannot be combined with --benchmark-skip-eval.")
if ARGS.eval_only and RESUME_BUNDLE is None:
    raise ValueError("--eval-only requires --resume-from.")

try:
    from kernels import get_kernel
except Exception:
    get_kernel = None

cap = torch.cuda.get_device_capability()
cuda_runtime = detect_cuda_runtime_profile(cap, device_name=torch.cuda.get_device_name())


def _resolve_flash_attention_interface():
    installed_candidates = [
        "flash_attn.flash_attn_interface",
        "hopper.flash_attn_interface",
    ]
    for module_name in installed_candidates:
        try:
            module = __import__(module_name, fromlist=["flash_attn_func"])
            if hasattr(module, "flash_attn_func"):
                return module, f"installed:{module_name}"
        except Exception:
            continue
    if get_kernel is not None:
        try:
            module = get_kernel(cuda_runtime.selected_flash_attention_repo).flash_attn_interface
            if hasattr(module, "flash_attn_func"):
                return module, f"kernels:{cuda_runtime.selected_flash_attention_repo}"
        except Exception:
            pass
    return None, "torch-sdpa"


FLASH_ATTN_INTERFACE, RESOLVED_ATTENTION_BACKEND = _resolve_flash_attention_interface()


def _runtime_eval_plan():
    if ARGS.smoke:
        return {
            "rung": "smoke",
            "status": "smoke",
            "effective_confidence": None,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": MAX_SEQ_LEN,
            "eval_tokens": MAX_SEQ_LEN * DEVICE_BATCH_SIZE,
            "batch_size": DEVICE_BATCH_SIZE,
            "reason": "smoke override",
        }
    if ARGS.eval_only:
        return {
            "rung": "manual",
            "status": "manual",
            "effective_confidence": None,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": ARGS.eval_seq_len or MAX_SEQ_LEN,
            "eval_tokens": ARGS.eval_tokens or CUDA_EVAL_TOKENS,
            "batch_size": ARGS.eval_batch_size or DEVICE_BATCH_SIZE,
            "reason": "eval-only override",
        }
    if ARGS.eval_seq_len is not None or ARGS.eval_tokens is not None or ARGS.eval_batch_size is not None:
        return {
            "rung": "manual",
            "status": "manual",
            "effective_confidence": None,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": ARGS.eval_seq_len or MAX_SEQ_LEN,
            "eval_tokens": ARGS.eval_tokens or CUDA_EVAL_TOKENS,
            "batch_size": ARGS.eval_batch_size or DEVICE_BATCH_SIZE,
            "reason": "manual eval override",
        }

    calibration = find_calibration(preset=ARGS.preset, hardware_key=hardware_key)
    if calibration is None:
        status = "missing-calibration" if has_calibration_for_preset(preset=ARGS.preset) else "hardware-unmatched"
        return {
            "rung": "default",
            "status": status,
            "effective_confidence": None,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": MAX_SEQ_LEN,
            "eval_tokens": CUDA_EVAL_TOKENS,
            "batch_size": DEVICE_BATCH_SIZE,
            "reason": status,
        }

    if calibration.policy_version != EVAL_POLICY_VERSION:
        return {
            "rung": "default",
            "status": "stale-policy",
            "effective_confidence": calibration.confidence,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": MAX_SEQ_LEN,
            "eval_tokens": CUDA_EVAL_TOKENS,
            "batch_size": DEVICE_BATCH_SIZE,
            "reason": "stale policy version",
        }

    if calibration.eval_semantics_signature != current_eval_semantics_signature() or calibration.runtime_shape_signature != current_runtime_shape_signature():
        return {
            "rung": "default",
            "status": "signature-mismatch",
            "effective_confidence": calibration.confidence,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": MAX_SEQ_LEN,
            "eval_tokens": CUDA_EVAL_TOKENS,
            "batch_size": DEVICE_BATCH_SIZE,
            "reason": "signature mismatch",
        }

    if (
        calibration.seq_len != MAX_SEQ_LEN
        or calibration.depth != DEPTH
        or calibration.window_pattern != WINDOW_PATTERN
    ):
        return {
            "rung": "default",
            "status": "shape-fallback",
            "effective_confidence": calibration.confidence,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": MAX_SEQ_LEN,
            "eval_tokens": CUDA_EVAL_TOKENS,
            "batch_size": DEVICE_BATCH_SIZE,
            "reason": "shape fallback",
        }

    decision = choose_runtime_eval(calibration=calibration, time_budget=TIME_BUDGET)
    return {
        "rung": decision.rung_key,
        "status": "calibrated" if decision.limited_by is None else "calibrated-limited",
        "effective_confidence": decision.effective_confidence,
        "freshness": decision.freshness,
        "policy_version": EVAL_POLICY_VERSION,
        "limited_by": decision.limited_by,
        "seq_len": decision.rung.spec.seq_len,
        "eval_tokens": decision.rung.spec.eval_tokens,
        "batch_size": decision.rung.spec.batch_size,
        "reason": f"{decision.rung_key} rung from {decision.calibration.label}",
    }


def _build_local_causal_mask(seq_len: int, window_size: tuple[int, int], device: torch.device):
    left_window, _ = window_size
    if left_window < 0 or left_window >= seq_len:
        return None
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    return (distance >= 0) & (distance < left_window)


def attention_func(q, k, v, *, causal: bool, window_size: tuple[int, int]):
    if FLASH_ATTN_INTERFACE is not None:
        return FLASH_ATTN_INTERFACE.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    if q.size(1) != k.size(1):
        repeat_factor = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
    attn_mask = _build_local_causal_mask(q.size(2), window_size, q.device) if causal else None
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal and attn_mask is None)
    return y.permute(0, 2, 1, 3)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    overridden = maybe_call_integration_target("norm", x)
    if overridden is not None:
        return overridden
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        ve_gate_weight = None
        if self.ve_gate is not None:
            ve_gate_weight = self.ve_gate.weight.t()
        overridden_qkv = maybe_call_integration_target(
            "attention_prelude",
            x,
            self.c_q.weight.t(),
            self.c_k.weight.t(),
            self.c_v.weight.t(),
            self.n_head,
            self.head_dim,
            ve,
            ve_gate_weight,
            self.ve_gate_channels,
        )
        if overridden_qkv is not None:
            q, k, v = overridden_qkv
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            if ve is not None:
                ve = ve.view(B, T, self.n_kv_head, self.head_dim)
                overridden_v = maybe_call_integration_target(
                    "value_embed_gate",
                    x[..., :self.ve_gate_channels],
                    v,
                    ve,
                    self.ve_gate.weight.t(),
                )
                if overridden_v is not None:
                    v = overridden_v
                else:
                    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                    v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        overridden_qk = maybe_call_integration_target("rope_qk_fused", q, k, cos, sin)
        if overridden_qk is not None:
            q, k = overridden_qk
        else:
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

        y = attention_func(q, k, v, causal=True, window_size=window_size)
        overridden_y = maybe_call_integration_target("data_movement", y)
        if overridden_y is not None:
            y = overridden_y
        else:
            y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        overridden = maybe_call_integration_target("fused_mlp", x, self.c_fc.weight.t(), self.c_proj.weight.t())
        if overridden is not None:
            return overridden
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        attn_update = self.attn(norm(x), ve, cos_sin, window_size)
        overridden_attn = maybe_call_integration_target("launch_fusion", x, attn_update)
        x = overridden_attn if overridden_attn is not None else x + attn_update
        mlp_update = self.mlp(norm(x))
        overridden_mlp = maybe_call_integration_target("launch_fusion", x, mlp_update)
        x = overridden_mlp if overridden_mlp is not None else x + mlp_update
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, lr_profile: ResolvedLrProfile, *, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=lr_profile.lm_head_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=lr_profile.embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=lr_profile.value_embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=lr_profile.resid_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=lr_profile.x0_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=lr_profile.matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        flat_x = x.view(-1, x.size(-1))
        overridden_logits = maybe_call_integration_target("matmul_epilogue", flat_x, self.lm_head.weight.t())
        if overridden_logits is not None:
            logits = overridden_logits.view(B, T, -1)
        else:
            logits = self.lm_head(x)
        logits = logits.float()
        overridden_softcap = maybe_call_integration_target("logits_softcap", logits, softcap)
        if overridden_softcap is not None:
            logits = overridden_softcap

        if targets is not None:
            flat_targets = targets.view(-1)
            valid_mask = flat_targets.ne(-1)
            safe_targets = flat_targets.masked_fill(~valid_mask, 0)
            token_bytes = valid_mask.to(dtype=torch.int32)
            overridden = maybe_call_integration_target(
                "loss_prelude",
                logits.view(-1, logits.size(-1)),
                safe_targets,
                token_bytes,
                softcap,
            )
            if overridden is not None:
                weighted, weighted_sum, token_count = overridden
                if overridden_softcap is None:
                    logits = softcap * torch.tanh(logits / softcap)
                if reduction == 'none':
                    loss = weighted.view_as(targets)
                elif reduction == 'sum':
                    loss = weighted_sum
                else:
                    loss = weighted_sum / token_count.clamp_min(1).float()
            else:
                if overridden_softcap is None:
                    logits = softcap * torch.tanh(logits / softcap)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                       ignore_index=-1, reduction=reduction)
            return loss
        if overridden_softcap is None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def _adamw_step_fused_impl(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def _muon_step_fused_impl(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                         momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


if ARGS.no_compile:
    adamw_step_fused = _adamw_step_fused_impl
    muon_step_fused = _muon_step_fused_impl
else:
    adamw_step_fused = torch.compile(_adamw_step_fused_impl, dynamic=False, fullgraph=True)
    muon_step_fused = torch.compile(_muon_step_fused_impl, dynamic=False, fullgraph=True)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
MAX_SEQ_LEN = RUN_PRESET.seq_len
TIME_BUDGET = RUN_PRESET.time_budget
TOKEN_BUDGET = (
    int(ARGS.token_budget)
    if ARGS.token_budget is not None
    else int(RESUME_BUNDLE["run_config"].get("token_budget"))
    if RESUME_BUNDLE is not None and RESUME_BUNDLE.get("run_config", {}).get("token_budget") is not None
    else None
)
ASPECT_RATIO = RUN_PRESET.aspect_ratio
HEAD_DIM = RUN_PRESET.head_dim
WINDOW_PATTERN = RUN_PRESET.window_pattern

# Optimization
TOTAL_BATCH_SIZE = RUN_PRESET.total_batch_size
WEIGHT_DECAY = RUN_PRESET.weight_decay
ADAM_BETAS = RUN_PRESET.adam_betas
WARMUP_RATIO = RUN_PRESET.warmup_ratio
WARMDOWN_RATIO = RUN_PRESET.warmdown_ratio
FINAL_LR_FRAC = RUN_PRESET.final_lr_frac

# Model size
DEPTH = RUN_PRESET.depth
DEVICE_BATCH_SIZE = RUN_PRESET.device_batch_size
RESUME_RUN_CONFIG = RESUME_BUNDLE.get("run_config") if RESUME_BUNDLE is not None else {}
BASE_LR_PROFILE = lr_profile_from_mapping(
    RESUME_RUN_CONFIG if RESUME_RUN_CONFIG else {"lr_profile": lr_profile_to_dict(RUN_PRESET.lr_profile)},
    default_profile=RUN_PRESET.lr_profile,
)
LR_MULTIPLIERS = (
    _lr_multipliers_from_args(ARGS)
    if any(getattr(ARGS, field) is not None for field in LR_MULTIPLIER_ARG_FIELDS)
    else lr_multipliers_from_mapping(RESUME_RUN_CONFIG)
)
TUNED_LR_PROFILE = apply_lr_multipliers(BASE_LR_PROFILE, LR_MULTIPLIERS)
STREAMING_EVAL_CONFIG = _resolve_streaming_eval_config(
    ARGS,
    seq_len=MAX_SEQ_LEN,
    device_batch_size=DEVICE_BATCH_SIZE,
    resume_run_config=RESUME_RUN_CONFIG,
)
STREAMING_EVAL_HISTORY_OUTPUT = (
    ARGS.streaming_eval_history_output
    if ARGS.streaming_eval_history_output is not None
    else RESUME_RUN_CONFIG.get("streaming_eval_history_output")
)
CHECKPOINT_PATH = None
CHECKPOINT_INTERVAL = None
CHECKPOINT_SAVE_MODE = None
CHECKPOINT_DECISION_REASON = None


def active_budget_mode() -> str:
    return "tokens" if TOKEN_BUDGET is not None else "seconds"


def budget_progress(*, total_training_time: float, total_tokens: int) -> float:
    if TOKEN_BUDGET is not None:
        return min(total_tokens / max(TOKEN_BUDGET, 1), 1.0)
    return min(total_training_time / max(TIME_BUDGET, 1e-9), 1.0)


def nominal_budget_satisfied(*, total_training_time: float, total_tokens: int) -> bool:
    if TOKEN_BUDGET is not None:
        return total_tokens >= TOKEN_BUDGET
    return total_training_time >= TIME_BUDGET


def budget_satisfied(*, total_training_time: float, total_tokens: int) -> bool:
    if not nominal_budget_satisfied(total_training_time=total_training_time, total_tokens=total_tokens):
        return False
    if streaming_eval_tracker is not None and streaming_eval_tracker.requires_cycle_completion():
        return False
    if subref_streaming_tracker is not None and subref_streaming_tracker.requires_cycle_completion():
        return False
    return True


def budget_remaining(*, total_training_time: float, total_tokens: int) -> str:
    if TOKEN_BUDGET is not None:
        remaining_tokens = max(0, TOKEN_BUDGET - total_tokens)
        if remaining_tokens >= 1_000_000:
            return f"{remaining_tokens / 1_000_000:.1f}M tok"
        if remaining_tokens >= 1_000:
            return f"{remaining_tokens / 1_000:.1f}K tok"
        return f"{remaining_tokens} tok"
    return f"{max(0.0, TIME_BUDGET - total_training_time):.0f}s"

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")
device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
memory_gb = device_props.total_memory / (1024**3)
hardware_key = f"nvidia-{cuda_runtime.reference_family.family_key}-{int(round(memory_gb))}gb"
print(f"CUDA runtime: {cuda_runtime.architecture_name}")
print(f"CUDA reference family: {cuda_runtime.reference_family.family_name}")
print(f"Preferred attention backend: {cuda_runtime.selected_attention_backend}")
print(f"Resolved attention backend: {RESOLVED_ATTENTION_BACKEND}")
if RESUME_TRAINING_STATE is not None and RESUME_TRAINING_STATE.get("hardware_key") not in (None, hardware_key):
    print(
        "Resume hardware note: "
        f"checkpoint was created on {RESUME_TRAINING_STATE.get('hardware_key')}, "
        f"current hardware is {hardware_key}"
    )
print(f"Preferred flash-attention repo: {cuda_runtime.selected_flash_attention_repo}")
RUNTIME_EVAL_PLAN = _runtime_eval_plan()
print(
    "Runtime eval: "
    f"status={RUNTIME_EVAL_PLAN['status']}, "
    f"rung={RUNTIME_EVAL_PLAN['rung']}, "
    f"seq_len={RUNTIME_EVAL_PLAN['seq_len']}, "
    f"eval_tokens={RUNTIME_EVAL_PLAN['eval_tokens']}, "
    f"batch_size={RUNTIME_EVAL_PLAN['batch_size']}"
)

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
RESOLVED_LR_PROFILE = resolve_effective_lr_profile(
    TUNED_LR_PROFILE,
    model_dim=config.n_embd,
)
print(f"Model config: {asdict(config)}")
print(f"LR profile: {lr_profile_to_dict(BASE_LR_PROFILE)}")
print(f"LR multipliers: {lr_multipliers_to_dict(LR_MULTIPLIERS)}")
print(f"Tuned LR profile: {lr_profile_to_dict(TUNED_LR_PROFILE)}")
print(
    "Resolved LR profile: "
    f"lm_head={RESOLVED_LR_PROFILE.lm_head_lr:.6f}, "
    f"embedding={RESOLVED_LR_PROFILE.embedding_lr:.6f}, "
    f"value_embedding={RESOLVED_LR_PROFILE.value_embedding_lr:.6f}, "
    f"resid={RESOLVED_LR_PROFILE.resid_lr:.6f}, "
    f"x0={RESOLVED_LR_PROFILE.x0_lr:.6f}, "
    f"matrix={RESOLVED_LR_PROFILE.matrix_lr:.6f}, "
    f"dmodel_scale={RESOLVED_LR_PROFILE.dmodel_lr_scale:.6f}"
)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
(
    CHECKPOINT_PATH,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_SAVE_MODE,
    CHECKPOINT_DECISION_REASON,
    CHECKPOINT_INTERVAL_IS_AUTO,
) = _resolve_checkpoint_settings(
    ARGS,
    num_params_m=num_params / 1e6,
    checkpoint_mode="exact",
    preset=ARGS.preset,
    time_budget=TIME_BUDGET,
    token_budget=TOKEN_BUDGET,
    seq_len=MAX_SEQ_LEN,
    depth=DEPTH,
    total_batch_size=TOTAL_BATCH_SIZE,
    window_pattern=WINDOW_PATTERN,
    resume_run_config=RESUME_BUNDLE.get("run_config") if RESUME_BUNDLE is not None else None,
)
ARGS.checkpoint_path = CHECKPOINT_PATH
ARGS.checkpoint_interval = format_interval_spec(CHECKPOINT_INTERVAL)
ARGS.checkpoint_save_mode = CHECKPOINT_SAVE_MODE
ARGS.checkpoint_interval_is_auto = CHECKPOINT_INTERVAL_IS_AUTO

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = None
train_loader = None
x = y = None
epoch = 1
loader_advance_steps = 0
checkpoint_saved_to = None
t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = float(RESUME_TRAINING_STATE["total_training_time"]) if RESUME_TRAINING_STATE is not None else 0.0
steady_state_training_time = (
    float(RESUME_TRAINING_STATE.get("steady_state_training_time", 0.0))
    if RESUME_TRAINING_STATE is not None
    else 0.0
)
step = int(RESUME_TRAINING_STATE["step"]) if RESUME_TRAINING_STATE is not None else 0
total_tokens = int(RESUME_TRAINING_STATE["total_tokens"]) if RESUME_TRAINING_STATE is not None else 0
smooth_train_loss = (
    float(RESUME_TRAINING_STATE.get("smooth_train_loss", 0.0))
    if RESUME_TRAINING_STATE is not None
    else 0.0
)
resumed_checkpoint_seconds = (
    float(RESUME_TRAINING_STATE.get("total_checkpoint_time", 0.0))
    if RESUME_TRAINING_STATE is not None
    else 0.0
)
resumed_checkpoint_write_seconds = (
    float(RESUME_TRAINING_STATE.get("total_checkpoint_write_time", resumed_checkpoint_seconds))
    if RESUME_TRAINING_STATE is not None
    else 0.0
)
resumed_checkpoint_count = (
    int(RESUME_TRAINING_STATE.get("checkpoint_count", 0))
    if RESUME_TRAINING_STATE is not None
    else 0
)
step_telemetry = StepTelemetry.from_dict(
    RESUME_TRAINING_STATE.get("step_telemetry")
    if RESUME_TRAINING_STATE is not None
    else None
)
checkpoint_seconds = 0.0
checkpoint_write_seconds = 0.0
checkpoint_count = 0
steady_state_step_count = (
    int(RESUME_TRAINING_STATE.get("steady_state_step_count", 0))
    if RESUME_TRAINING_STATE is not None
    else 0
)
timing_count_starts_after_step = 10 if RESUME_TRAINING_STATE is None else step
curve_points: list[dict] = []
curve_eval_total_seconds = 0.0
curve_eval_index = 0
last_train_loss = None
last_smoothed_train_loss = None
while curve_eval_index < len(CURVE_EVAL_SECONDS) and CURVE_EVAL_SECONDS[curve_eval_index] <= total_training_time:
    curve_eval_index += 1
last_checkpoint_training_time = total_training_time
last_checkpoint_tokens = total_tokens
async_checkpoint_writer = None
streaming_eval_seconds = 0.0
subref_streaming_eval_seconds = 0.0
cheap_stream_enabled = STREAMING_EVAL_CONFIG is not None and streaming_mode_uses_cheap(STREAMING_EVAL_CONFIG.mode)
subref_stream_enabled = STREAMING_EVAL_CONFIG is not None and streaming_mode_uses_subref(STREAMING_EVAL_CONFIG.mode)
probe_pathology_triggered = False
probe_pathology_reason = None
streaming_eval_tracker = (
    StreamingEvalTracker(STREAMING_EVAL_CONFIG)
    if cheap_stream_enabled
    else None
)
streaming_eval_token_bytes = (
    get_token_bytes(device="cuda")
    if cheap_stream_enabled or subref_stream_enabled
    else None
)
streaming_eval_loader = (
    make_dataloader(
        tokenizer,
        STREAMING_EVAL_CONFIG.batch_size,
        STREAMING_EVAL_CONFIG.seq_len,
        "val",
    )
    if cheap_stream_enabled
    else None
)
subref_streaming_loader = (
    make_dataloader(
        tokenizer,
        CUDA_SUBREF_ONE_SIXTH_RUNG.batch_size,
        CUDA_SUBREF_ONE_SIXTH_RUNG.seq_len,
        "val",
    )
    if subref_stream_enabled
    else None
)
subref_streaming_tracker = (
    StreamingEvalTracker(
        StreamingEvalConfig(
            interval_steps=STREAMING_EVAL_CONFIG.interval_steps,
            eval_tokens=CUDA_SUBREF_ONE_SIXTH_RUNG.eval_tokens,
            batch_size=CUDA_SUBREF_ONE_SIXTH_RUNG.batch_size,
            seq_len=CUDA_SUBREF_ONE_SIXTH_RUNG.seq_len,
            mode=STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH,
            complete_cycle_on_budget=False,
        )
    )
    if subref_stream_enabled and STREAMING_EVAL_CONFIG is not None
    else None
)
derived_reference_streaming = (
    StreamingSupercycleAccumulator(subcycles_per_supercycle=SUBREF_ONE_SIXTHS_PER_REFERENCE)
    if subref_stream_enabled
    else None
)


def maybe_mark_probe_pathology(metric_name: str, value: float | None, limit: float | None):
    global probe_pathology_triggered, probe_pathology_reason
    if probe_pathology_triggered or limit is None or value is None:
        return
    if not math.isfinite(value):
        return
    if value <= limit:
        return
    probe_pathology_triggered = True
    probe_pathology_reason = f"{metric_name}={value:.6f}>{limit:.6f}"
    print(f"\nprobe_pathology_stop: {probe_pathology_reason}")


def maybe_save_checkpoint(*, force: bool = False):
    global checkpoint_saved_to, last_checkpoint_training_time, last_checkpoint_tokens, checkpoint_seconds, checkpoint_write_seconds, checkpoint_count
    if CHECKPOINT_PATH is None or ARGS.no_checkpoint or optimizer is None:
        return
    if async_checkpoint_writer is not None:
        async_checkpoint_writer.check_health()
    if not force:
        if CHECKPOINT_INTERVAL is None:
            return
        if not auto_checkpoint_due(
            CHECKPOINT_INTERVAL,
            total_elapsed_seconds=total_training_time,
            elapsed_seconds=total_training_time - last_checkpoint_training_time,
            elapsed_tokens=total_tokens - last_checkpoint_tokens,
            checkpoints_completed=resumed_checkpoint_count + checkpoint_count,
            interval_is_auto=CHECKPOINT_INTERVAL_IS_AUTO,
        ):
            return
    elif (
        checkpoint_saved_to is not None
        and total_training_time == last_checkpoint_training_time
        and total_tokens == last_checkpoint_tokens
    ):
        return

    run_config = {
        "preset": ARGS.preset,
        "smoke": ARGS.smoke,
        "time_budget": RUN_PRESET.time_budget,
        "token_budget": TOKEN_BUDGET,
        "seq_len": MAX_SEQ_LEN,
        "depth": DEPTH,
        "window_pattern": WINDOW_PATTERN,
        "device_batch_size": DEVICE_BATCH_SIZE,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "grad_accum_steps": grad_accum_steps,
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_interval": format_interval_spec(CHECKPOINT_INTERVAL),
        "checkpoint_interval_is_auto": CHECKPOINT_INTERVAL_IS_AUTO,
        "checkpoint_save_mode": CHECKPOINT_SAVE_MODE,
        "lr_multiplier": LR_MULTIPLIERS.lr_multiplier,
        "embedding_lr_multiplier": LR_MULTIPLIERS.embedding_lr_multiplier,
        "unembedding_lr_multiplier": LR_MULTIPLIERS.unembedding_lr_multiplier,
        "matrix_lr_multiplier": LR_MULTIPLIERS.matrix_lr_multiplier,
        "scalar_lr_multiplier": LR_MULTIPLIERS.scalar_lr_multiplier,
        "lr_profile": lr_profile_to_dict(BASE_LR_PROFILE),
        "lr_multipliers": lr_multipliers_to_dict(LR_MULTIPLIERS),
        "streaming_eval_interval_steps": (
            STREAMING_EVAL_CONFIG.interval_steps if STREAMING_EVAL_CONFIG is not None else None
        ),
        "streaming_eval_mode": (
            STREAMING_EVAL_CONFIG.mode if STREAMING_EVAL_CONFIG is not None else None
        ),
        "streaming_eval_tokens": (
            STREAMING_EVAL_CONFIG.eval_tokens if STREAMING_EVAL_CONFIG is not None else None
        ),
        "streaming_eval_seq_len": (
            STREAMING_EVAL_CONFIG.seq_len if STREAMING_EVAL_CONFIG is not None else None
        ),
        "streaming_eval_batch_size": (
            STREAMING_EVAL_CONFIG.batch_size if STREAMING_EVAL_CONFIG is not None else None
        ),
        "streaming_eval_history_output": STREAMING_EVAL_HISTORY_OUTPUT,
        "complete_streaming_eval_cycle": (
            STREAMING_EVAL_CONFIG.complete_cycle_on_budget if STREAMING_EVAL_CONFIG is not None else False
        ),
        "probe_pathology_max_auc": ARGS.probe_pathology_max_auc,
        "probe_pathology_max_bpb": ARGS.probe_pathology_max_bpb,
    }
    training_state_base = {
        "step": step,
        "total_training_time": total_training_time,
        "steady_state_training_time": steady_state_training_time,
        "steady_state_step_count": steady_state_step_count,
        "total_tokens": total_tokens,
        "smooth_train_loss": smooth_train_loss,
        "step_telemetry": step_telemetry.to_dict(),
        "epoch": epoch,
        "hardware_key": hardware_key,
        "accelerator_architecture": cuda_runtime.reference_family.family_key,
        "accelerator_compute_capability": f"{cap[0]}.{cap[1]}",
        "resolved_attention_backend": RESOLVED_ATTENTION_BACKEND,
    }
    if async_checkpoint_writer is None:
        training_state = {
            **training_state_base,
            "total_checkpoint_time": resumed_checkpoint_seconds + checkpoint_seconds,
            "total_checkpoint_write_time": resumed_checkpoint_write_seconds + checkpoint_write_seconds,
            "checkpoint_count": resumed_checkpoint_count + checkpoint_count + 1,
        }
        checkpoint_saved_to, elapsed = save_training_checkpoint(
            CHECKPOINT_PATH,
            run_config=run_config,
            training_state=training_state,
            model=model,
            optimizer=optimizer,
            checkpoint_save_mode=CHECKPOINT_SAVE_MODE,
        )
        checkpoint_seconds += elapsed
        checkpoint_write_seconds += elapsed
        checkpoint_count += 1
    else:
        training_state = {
            **training_state_base,
            "total_checkpoint_time": resumed_checkpoint_seconds + checkpoint_seconds,
            "total_checkpoint_write_time": resumed_checkpoint_write_seconds + checkpoint_write_seconds,
            "checkpoint_count": resumed_checkpoint_count + checkpoint_count + 1,
        }
        snapshot, capture_seconds = capture_async_checkpoint_snapshot(
            CHECKPOINT_PATH,
            run_config=run_config,
            training_state=training_state,
            model=model,
            optimizer=optimizer,
        )
        checkpoint_seconds += capture_seconds
        async_checkpoint_writer.submit(snapshot)
        checkpoint_write_seconds = async_checkpoint_writer.completed_write_seconds
        checkpoint_count = async_checkpoint_writer.completed_count
        checkpoint_saved_to = Path(CHECKPOINT_PATH)
    interval_label = (
        f" ({format_interval_label(CHECKPOINT_INTERVAL)} interval)"
        if CHECKPOINT_INTERVAL is not None
        else ""
    )
    last_checkpoint_training_time = total_training_time
    last_checkpoint_tokens = total_tokens
    if async_checkpoint_writer is None:
        print(f"\nCheckpoint saved to {CHECKPOINT_PATH} at step {step}{interval_label}.")
    else:
        print(f"\nCheckpoint queued to {CHECKPOINT_PATH} at step {step}{interval_label}.")
    return checkpoint_saved_to


def maybe_run_streaming_eval():
    global streaming_eval_loader, subref_streaming_loader, streaming_eval_seconds, subref_streaming_eval_seconds
    if STREAMING_EVAL_CONFIG is None:
        return
    if not STREAMING_EVAL_CONFIG.interval_steps or step <= 0 or step % STREAMING_EVAL_CONFIG.interval_steps != 0:
        return
    model.eval()
    if streaming_eval_tracker is not None and streaming_eval_loader is not None:
        torch.cuda.synchronize()
        t_streaming_eval_start = time.perf_counter()
        x_val, y_val, _ = next(streaming_eval_loader)
        batch_bpb, batch_nats, batch_bytes = evaluate_bpb_batch_stats(
            model,
            x_val,
            y_val,
            token_bytes=streaming_eval_token_bytes,
        )
        torch.cuda.synchronize()
        streaming_eval_seconds += time.perf_counter() - t_streaming_eval_start
        point = streaming_eval_tracker.record(
            step=step,
            total_training_time=total_training_time,
            total_tokens=total_tokens,
            batch_bpb=batch_bpb,
            batch_nats=batch_nats,
            batch_bytes=batch_bytes,
        )
        if point.cycle_completed:
            summary = streaming_eval_tracker.summary()
            auc_label = "skipped" if summary.auc is None else f"{summary.auc:.6f}"
            honest_label = "skipped" if point.honest_bpb is None else f"{point.honest_bpb:.6f}"
            print(
                f"\nstreaming_eval: step={point.step} cycle={point.cycle_index} "
                f"honest_bpb={honest_label} auc={auc_label}"
            )
            maybe_mark_probe_pathology("streaming_honest_bpb", point.honest_bpb, ARGS.probe_pathology_max_bpb)
            maybe_mark_probe_pathology("streaming_auc", summary.auc, ARGS.probe_pathology_max_auc)
            streaming_eval_loader = make_dataloader(
                tokenizer,
                STREAMING_EVAL_CONFIG.batch_size,
                STREAMING_EVAL_CONFIG.seq_len,
                "val",
            )
        maybe_mark_probe_pathology("streaming_batch_bpb", point.batch_bpb, ARGS.probe_pathology_max_bpb)
    if subref_streaming_tracker is not None and subref_streaming_loader is not None:
        torch.cuda.synchronize()
        t_subref_eval_start = time.perf_counter()
        x_subref, y_subref, _ = next(subref_streaming_loader)
        subref_bpb, subref_nats, subref_bytes = evaluate_bpb_batch_stats(
            model,
            x_subref,
            y_subref,
            token_bytes=streaming_eval_token_bytes,
        )
        torch.cuda.synchronize()
        subref_streaming_eval_seconds += time.perf_counter() - t_subref_eval_start
        subref_point = subref_streaming_tracker.record(
            step=step,
            total_training_time=total_training_time,
            total_tokens=total_tokens,
            batch_bpb=subref_bpb,
            batch_nats=subref_nats,
            batch_bytes=subref_bytes,
        )
        if subref_point.cycle_completed:
            honest_label = "skipped" if subref_point.honest_bpb is None else f"{subref_point.honest_bpb:.6f}"
            print(
                f"\nsubref_one_sixth_eval: step={subref_point.step} cycle={subref_point.cycle_index} "
                f"honest_bpb={honest_label}"
            )
            maybe_mark_probe_pathology("subref_honest_bpb", subref_point.honest_bpb, ARGS.probe_pathology_max_bpb)
            if (
                derived_reference_streaming is not None
                and derived_reference_streaming.record_completed_subcycle(
                    cycle_nats=subref_point.cycle_nats,
                    cycle_bytes=subref_point.cycle_bytes,
                    honest_bpb=subref_point.honest_bpb,
                )
            ):
                reference_label = (
                    "skipped"
                    if derived_reference_streaming.honest_supercycle_bpb is None
                    else f"{derived_reference_streaming.honest_supercycle_bpb:.6f}"
                )
                print(
                    f"\nreference_streaming_eval: cycle={derived_reference_streaming.supercycles_completed} "
                    f"honest_bpb={reference_label}"
                )
                maybe_mark_probe_pathology(
                    "reference_honest_bpb",
                    derived_reference_streaming.honest_supercycle_bpb,
                    ARGS.probe_pathology_max_bpb,
                )
                subref_streaming_loader = make_dataloader(
                    tokenizer,
                    CUDA_SUBREF_ONE_SIXTH_RUNG.batch_size,
                    CUDA_SUBREF_ONE_SIXTH_RUNG.seq_len,
                    "val",
                )
        maybe_mark_probe_pathology("subref_batch_bpb", subref_point.batch_bpb, ARGS.probe_pathology_max_bpb)
    model.train()

resumed_from = None
if RESUME_BUNDLE is not None:
    model.load_state_dict(RESUME_BUNDLE["model_state_dict"])
if not ARGS.no_compile:
    model = torch.compile(model, dynamic=False)

if ARGS.eval_only:
    if RESUME_BUNDLE is not None:
        resumed_from = ARGS.resume_from
        epoch = int(RESUME_TRAINING_STATE["epoch"])
    if TOKEN_BUDGET is not None:
        print(f"Token budget: {TOKEN_BUDGET}")
        print(f"Reference time budget: {TIME_BUDGET}s")
    else:
        print(f"Time budget: {TIME_BUDGET}s")
    print(f"Smoke mode: {str(ARGS.smoke).lower()}")
    print(f"LR multiplier: {LR_MULTIPLIERS.lr_multiplier:.6f}")
    print(f"Embedding LR multiplier: {LR_MULTIPLIERS.embedding_lr_multiplier:.6f}")
    print(f"Unembedding LR multiplier: {LR_MULTIPLIERS.unembedding_lr_multiplier:.6f}")
    print(f"Matrix LR multiplier: {LR_MULTIPLIERS.matrix_lr_multiplier:.6f}")
    print(f"Scalar LR multiplier: {LR_MULTIPLIERS.scalar_lr_multiplier:.6f}")
    if CHECKPOINT_PATH is not None:
        print(f"Checkpoint path: {CHECKPOINT_PATH}")
    if CHECKPOINT_INTERVAL is not None:
        print(f"Checkpoint interval: {format_interval_label(CHECKPOINT_INTERVAL)}")
    print(f"Checkpoint save mode: {CHECKPOINT_SAVE_MODE}")
    if CHECKPOINT_DECISION_REASON is not None:
        print(f"Checkpoint policy: {CHECKPOINT_DECISION_REASON}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Compile enabled: {str(not ARGS.no_compile).lower()}")
    if STREAMING_EVAL_CONFIG is not None:
        print(
            "Streaming eval: "
            f"every={STREAMING_EVAL_CONFIG.interval_steps} steps, "
            f"mode={STREAMING_EVAL_CONFIG.mode}, "
            f"tokens={STREAMING_EVAL_CONFIG.eval_tokens}, "
            f"seq_len={STREAMING_EVAL_CONFIG.seq_len}, "
            f"batch_size={STREAMING_EVAL_CONFIG.batch_size}, "
            f"cycle_batches={STREAMING_EVAL_CONFIG.cycle_batches}, "
            f"complete_cycle_on_budget={STREAMING_EVAL_CONFIG.complete_cycle_on_budget}"
        )
    else:
        print("Streaming eval: disabled")
    print(
        "Eval only: "
        f"seq_len={ARGS.eval_seq_len or MAX_SEQ_LEN}, "
        f"eval_tokens={ARGS.eval_tokens or CUDA_EVAL_TOKENS}, "
        f"batch_size={ARGS.eval_batch_size or DEVICE_BATCH_SIZE}"
    )
    if resumed_from is not None:
        print(
            "Resume checkpoint: "
            f"{resumed_from} (step={RESUME_TRAINING_STATE['step']}, "
            f"training_seconds={RESUME_TRAINING_STATE['total_training_time']:.1f}, "
            f"checkpoint_save_mode={RESUME_BUNDLE.get('checkpoint_save_mode', CHECKPOINT_SAVE_MODE_SYNC)})"
        )
else:
    optimizer = model.setup_optimizer(
        lr_profile=RESOLVED_LR_PROFILE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    if RESUME_BUNDLE is not None:
        optimizer.load_state_dict(RESUME_BUNDLE["optimizer_state_dict"])
        resumed_from = ARGS.resume_from
        resume_step = int(RESUME_TRAINING_STATE["step"])
        resume_epoch = int(RESUME_TRAINING_STATE["epoch"])
        loader_advance_steps = resume_step * grad_accum_steps
        if loader_advance_steps:
            for _ in range(loader_advance_steps):
                x, y, epoch = next(train_loader)
        if epoch != resume_epoch:
            raise RuntimeError(
                f"Deterministic loader replay drifted: resumed epoch {epoch}, expected {resume_epoch}."
            )

    if TOKEN_BUDGET is not None:
        print(f"Token budget: {TOKEN_BUDGET}")
        print(f"Reference time budget: {TIME_BUDGET}s")
    else:
        print(f"Time budget: {TIME_BUDGET}s")
    print(f"Smoke mode: {str(ARGS.smoke).lower()}")
    print(f"LR multiplier: {LR_MULTIPLIERS.lr_multiplier:.6f}")
    print(f"Embedding LR multiplier: {LR_MULTIPLIERS.embedding_lr_multiplier:.6f}")
    print(f"Unembedding LR multiplier: {LR_MULTIPLIERS.unembedding_lr_multiplier:.6f}")
    print(f"Matrix LR multiplier: {LR_MULTIPLIERS.matrix_lr_multiplier:.6f}")
    print(f"Scalar LR multiplier: {LR_MULTIPLIERS.scalar_lr_multiplier:.6f}")
    if CHECKPOINT_PATH is not None:
        print(f"Checkpoint path: {CHECKPOINT_PATH}")
    if CHECKPOINT_INTERVAL is not None:
        print(f"Checkpoint interval: {format_interval_label(CHECKPOINT_INTERVAL)}")
    print(f"Checkpoint save mode: {CHECKPOINT_SAVE_MODE}")
    if CHECKPOINT_DECISION_REASON is not None:
        print(f"Checkpoint policy: {CHECKPOINT_DECISION_REASON}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Compile enabled: {str(not ARGS.no_compile).lower()}")
    if STREAMING_EVAL_CONFIG is not None:
        print(
            "Streaming eval: "
            f"every={STREAMING_EVAL_CONFIG.interval_steps} steps, "
            f"tokens={STREAMING_EVAL_CONFIG.eval_tokens}, "
            f"seq_len={STREAMING_EVAL_CONFIG.seq_len}, "
            f"batch_size={STREAMING_EVAL_CONFIG.batch_size}, "
            f"cycle_batches={STREAMING_EVAL_CONFIG.cycle_batches}, "
            f"complete_cycle_on_budget={STREAMING_EVAL_CONFIG.complete_cycle_on_budget}"
        )
    else:
        print("Streaming eval: disabled")
    async_checkpoint_writer = (
        make_async_checkpoint_writer()
        if CHECKPOINT_PATH is not None and CHECKPOINT_SAVE_MODE == CHECKPOINT_SAVE_MODE_ASYNC
        else None
    )
    if resumed_from is not None:
        print(
            "Resume checkpoint: "
            f"{resumed_from} (step={RESUME_TRAINING_STATE['step']}, "
            f"training_seconds={RESUME_TRAINING_STATE['total_training_time']:.1f}, "
            f"total_tokens={RESUME_TRAINING_STATE['total_tokens']}, "
            f"loader_batches={loader_advance_steps}, "
            f"checkpoint_save_mode={RESUME_BUNDLE.get('checkpoint_save_mode', CHECKPOINT_SAVE_MODE_SYNC)})"
        )

# Schedules follow the active stop budget: time when using --time-budget,
# tokens when using --token-budget.

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


def run_validation_eval(model, tokenizer):
    if ARGS.eval_only or ARGS.eval_seq_len is not None or ARGS.eval_tokens is not None or ARGS.eval_batch_size is not None:
        eval_plan = {
            "rung": "manual",
            "status": "manual",
            "effective_confidence": None,
            "freshness": None,
            "policy_version": EVAL_POLICY_VERSION,
            "limited_by": None,
            "seq_len": ARGS.eval_seq_len or MAX_SEQ_LEN,
            "eval_tokens": ARGS.eval_tokens or CUDA_EVAL_TOKENS,
            "batch_size": ARGS.eval_batch_size or DEVICE_BATCH_SIZE,
        }
    else:
        eval_plan = RUNTIME_EVAL_PLAN

    eval_seq_len = eval_plan["seq_len"]
    eval_tokens = eval_plan["eval_tokens"]
    eval_batch_size = eval_plan["batch_size"]

    model.eval()
    with autocast_ctx:
        # Warm one step so compile/setup does not dominate the measured rung time.
        warmup_tokens = max(eval_batch_size * eval_seq_len, eval_seq_len)
        evaluate_bpb_configured(
            model,
            tokenizer,
            eval_batch_size,
            seq_len=eval_seq_len,
            eval_tokens=warmup_tokens,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        val_bpb = evaluate_bpb_configured(
            model,
            tokenizer,
            eval_batch_size,
            seq_len=eval_seq_len,
            eval_tokens=eval_tokens,
        )
        torch.cuda.synchronize()
        eval_seconds = time.perf_counter() - t0
    return {
        "val_bpb": val_bpb,
        "eval_seconds": eval_seconds,
        "eval_seq_len": eval_seq_len,
        "eval_tokens": eval_tokens,
        "eval_batch_size": eval_batch_size,
        "canonical_rung": eval_plan["rung"],
        "eval_calibration_status": eval_plan["status"],
        "eval_calibration_effective_confidence": eval_plan["effective_confidence"],
        "eval_calibration_freshness": eval_plan["freshness"],
        "eval_calibration_limited_by": eval_plan["limited_by"],
        "eval_policy_version": eval_plan["policy_version"],
    }


def write_curve_output(path: str | None, payload: dict) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def maybe_run_curve_eval(
    *,
    model,
    tokenizer,
    step: int,
    total_training_time: float,
    total_tokens: int,
    curve_points: list[dict],
    curve_eval_total_seconds: float,
    curve_eval_index: int,
):
    while curve_eval_index < len(CURVE_EVAL_SECONDS) and total_training_time >= CURVE_EVAL_SECONDS[curve_eval_index]:
        target_training_seconds = CURVE_EVAL_SECONDS[curve_eval_index]
        eval_result = run_validation_eval(model, tokenizer)
        curve_eval_total_seconds += eval_result["eval_seconds"]
        curve_points.append(
            {
                "target_training_seconds": target_training_seconds,
                "actual_training_seconds": total_training_time,
                "step": step,
                "total_tokens": total_tokens,
                **eval_result,
            }
        )
        print(
            "curve_eval:      "
            f"target={target_training_seconds:.1f}s "
            f"actual={total_training_time:.1f}s "
            f"val_bpb={eval_result['val_bpb']:.6f} "
            f"eval_seconds={eval_result['eval_seconds']:.1f}"
        )
        model.train()
        curve_eval_index += 1
    return curve_eval_total_seconds, curve_eval_index

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

if not ARGS.eval_only:
    if budget_satisfied(total_training_time=total_training_time, total_tokens=total_tokens):
        if TOKEN_BUDGET is not None:
            print(
                f"Training budget already satisfied at resume point "
                f"({total_tokens} >= {TOKEN_BUDGET}); skipping training loop."
            )
        else:
            print(
                f"Training budget already satisfied at resume point "
                f"({total_training_time:.1f}s >= {TIME_BUDGET:.1f}s); skipping training loop."
            )

    while not budget_satisfied(total_training_time=total_training_time, total_tokens=total_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_timing = StepTiming()
        input_pipeline_cpu_seconds = 0.0
        forward_backward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _micro_step in range(grad_accum_steps):
            forward_backward_start = torch.cuda.Event(enable_timing=True)
            forward_backward_end = torch.cuda.Event(enable_timing=True)
            forward_backward_start.record()
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            forward_backward_end.record()
            forward_backward_events.append((forward_backward_start, forward_backward_end))
            t_loader_start = time.perf_counter()
            x, y, epoch = next(train_loader)
            input_pipeline_cpu_seconds += time.perf_counter() - t_loader_start

        progress = budget_progress(total_training_time=total_training_time, total_tokens=total_tokens)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        optimizer_start.record()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        optimizer_end.record()

        train_loss_f = train_loss.item()
        if train_loss_f > 100:
            print("FAIL")
            exit(1)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_timing.add_phase_seconds(
            PHASE_FORWARD_BACKWARD,
            sum(start.elapsed_time(end) for start, end in forward_backward_events) / 1000.0,
        )
        step_timing.add_phase_seconds(
            PHASE_OPTIMIZER,
            optimizer_start.elapsed_time(optimizer_end) / 1000.0,
        )
        step_timing.add_observed_seconds(OBS_INPUT_PIPELINE, input_pipeline_cpu_seconds)
        step_timing.total_seconds = t1 - t0
        dt = step_timing.total_seconds

        total_training_time += dt
        include_in_steady = step > timing_count_starts_after_step
        if include_in_steady:
            steady_state_training_time += dt
            steady_state_step_count += 1
        step_telemetry.record_step(step_timing, include_in_steady=include_in_steady)

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        last_train_loss = train_loss_f
        last_smoothed_train_loss = debiased_smooth_loss
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
        compute_share = 100.0 * step_timing.sum_phase_seconds((PHASE_FORWARD_BACKWARD, PHASE_OPTIMIZER)) / dt if dt > 0 else 0.0
        remaining = budget_remaining(total_training_time=total_training_time, total_tokens=total_tokens)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | util: {compute_share:.1f}% | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining}    ", end="", flush=True)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        total_tokens = step * TOTAL_BATCH_SIZE
        maybe_run_streaming_eval()
        curve_eval_total_seconds, curve_eval_index = maybe_run_curve_eval(
            model=model,
            tokenizer=tokenizer,
            step=step,
            total_training_time=total_training_time,
            total_tokens=total_tokens,
            curve_points=curve_points,
            curve_eval_total_seconds=curve_eval_total_seconds,
            curve_eval_index=curve_eval_index,
        )
        maybe_save_checkpoint()
        if probe_pathology_triggered:
            break

    print()  # newline after \r training log
    maybe_save_checkpoint(force=True)
    if async_checkpoint_writer is not None:
        async_checkpoint_writer.wait_until_idle()
        checkpoint_write_seconds = async_checkpoint_writer.completed_write_seconds
        checkpoint_count = async_checkpoint_writer.completed_count
        async_checkpoint_writer.close()

# Final eval
val_bpb = None
eval_seconds = None
eval_seq_len = None
eval_tokens = None
eval_batch_size = None
canonical_rung = None
eval_calibration_status = None
eval_calibration_effective_confidence = None
eval_calibration_freshness = None
eval_calibration_limited_by = None
eval_policy_version = None
if ARGS.eval_only or not ARGS.benchmark_skip_eval:
    eval_result = run_validation_eval(model, tokenizer)
    val_bpb = eval_result["val_bpb"]
    eval_seconds = eval_result["eval_seconds"]
    eval_seq_len = eval_result["eval_seq_len"]
    eval_tokens = eval_result["eval_tokens"]
    eval_batch_size = eval_result["eval_batch_size"]
    canonical_rung = eval_result["canonical_rung"]
    eval_calibration_status = eval_result["eval_calibration_status"]
    eval_calibration_effective_confidence = eval_result["eval_calibration_effective_confidence"]
    eval_calibration_freshness = eval_result["eval_calibration_freshness"]
    eval_calibration_limited_by = eval_result["eval_calibration_limited_by"]
    eval_policy_version = eval_result["eval_policy_version"]

curve_payload = None
if CURVE_EVAL_SECONDS:
    curve_payload = {
        "engine": "cuda",
        "preset": ARGS.preset,
        "hardware_key": hardware_key,
        "accelerator_architecture": cuda_runtime.reference_family.family_key,
        "accelerator_compute_capability": f"{cap[0]}.{cap[1]}",
        "resolved_attention_backend": RESOLVED_ATTENTION_BACKEND,
        "time_budget": TIME_BUDGET,
        "token_budget": TOKEN_BUDGET,
        "sequence_len": MAX_SEQ_LEN,
        "window_pattern": WINDOW_PATTERN,
        "depth": DEPTH,
        "device_batch_size": DEVICE_BATCH_SIZE,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "grad_accum_steps": grad_accum_steps,
        "curve_eval_seconds": CURVE_EVAL_SECONDS,
        "curve_eval_total_seconds": curve_eval_total_seconds,
        "curve_points": curve_points,
        "final_eval": {
            "val_bpb": val_bpb,
            "eval_seconds": eval_seconds,
            "eval_seq_len": eval_seq_len,
            "eval_tokens": eval_tokens,
            "eval_batch_size": eval_batch_size,
            "canonical_rung": canonical_rung,
        } if val_bpb is not None else None,
    }
    write_curve_output(ARGS.curve_output, curve_payload)

# Final summary
t_end = time.time()
effective_steady_time = steady_state_training_time if steady_state_training_time > 0 else total_training_time
effective_steady_steps = steady_state_step_count if steady_state_step_count > 0 else step
session_total_seconds = t_end - t_start
streaming_eval_summary = streaming_eval_tracker.summary() if streaming_eval_tracker is not None else None
subref_streaming_eval_summary = (
    subref_streaming_tracker.summary() if subref_streaming_tracker is not None else None
)
if STREAMING_EVAL_HISTORY_OUTPUT is not None and (
    streaming_eval_tracker is not None or subref_streaming_tracker is not None
):
    history_output_path = Path(STREAMING_EVAL_HISTORY_OUTPUT)
    history_output_path.parent.mkdir(parents=True, exist_ok=True)
    primary_tracker = (
        streaming_eval_tracker
        if streaming_eval_tracker is not None
        else subref_streaming_tracker
    )
    history_payload = primary_tracker.to_dict()
    history_payload["streaming_eval_mode"] = STREAMING_EVAL_CONFIG.mode if STREAMING_EVAL_CONFIG is not None else None
    if (
        streaming_eval_summary is not None
        and subref_streaming_eval_summary is not None
        and subref_streaming_tracker is not None
    ):
        history_payload["subref_one_sixth"] = subref_streaming_tracker.to_dict()
    if derived_reference_streaming is not None:
        history_payload["reference_from_subref"] = {
            "subcycle_key": CUDA_SUBREF_ONE_SIXTH_RUNG.key,
            "supercycle_key": CUDA_REFERENCE_RUNG.key,
            **derived_reference_streaming.to_dict(),
        }
    history_payload["probe_pathology_triggered"] = probe_pathology_triggered
    history_payload["probe_pathology_reason"] = probe_pathology_reason
    history_output_path.write_text(json.dumps(history_payload, indent=2) + "\n")
session_checkpoint_count = checkpoint_count
cumulative_checkpoint_seconds = resumed_checkpoint_seconds + checkpoint_seconds
cumulative_checkpoint_write_seconds = resumed_checkpoint_write_seconds + checkpoint_write_seconds
cumulative_checkpoint_count = resumed_checkpoint_count + checkpoint_count
session_eval_seconds = (
    curve_eval_total_seconds + (eval_seconds or 0.0) + streaming_eval_seconds + subref_streaming_eval_seconds
)
checkpoint_percent = 100.0 * checkpoint_seconds / session_total_seconds if session_total_seconds > 0 else 0.0
checkpoint_write_percent = (
    100.0 * checkpoint_write_seconds / session_total_seconds if session_total_seconds > 0 else 0.0
)
eval_percent = 100.0 * session_eval_seconds / session_total_seconds if session_total_seconds > 0 else 0.0
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * effective_steady_steps
    / effective_steady_time
    / H100_BF16_PEAK_FLOPS
    if effective_steady_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
steady_state_tok_per_sec = (
    TOTAL_BATCH_SIZE * effective_steady_steps / effective_steady_time
    if effective_steady_time > 0
    else 0.0
)
telemetry_summary = summarize_step_telemetry(
    step_telemetry,
    num_flops_per_token=num_flops_per_token,
    total_batch_size=TOTAL_BATCH_SIZE,
    compute_phases=(PHASE_FORWARD_BACKWARD, PHASE_OPTIMIZER),
    peak_flop_utilization_percent=steady_state_mfu,
)

print("---")
if val_bpb is not None:
    print(f"val_bpb:          {val_bpb:.6f}")
if eval_seconds is not None:
    print(f"eval_seconds:     {eval_seconds:.1f}")
if eval_seq_len is not None:
    print(f"eval_seq_len:     {eval_seq_len}")
if eval_tokens is not None:
    print(f"eval_tokens:      {eval_tokens}")
if eval_batch_size is not None:
    print(f"eval_batch_size:  {eval_batch_size}")
if canonical_rung is not None:
    print(f"canonical_rung:   {canonical_rung}")
if eval_calibration_status is not None:
    print(f"eval_calibration_status: {eval_calibration_status}")
if eval_calibration_effective_confidence is not None:
    print(f"eval_calibration_effective_confidence: {eval_calibration_effective_confidence}")
if eval_calibration_freshness is not None:
    print(f"eval_calibration_freshness: {eval_calibration_freshness}")
if eval_calibration_limited_by is not None:
    print(f"eval_calibration_limited_by: {eval_calibration_limited_by}")
if eval_policy_version is not None:
    print(f"eval_policy_version: {eval_policy_version}")
print(f"eval_only:        {str(ARGS.eval_only).lower()}")
if last_train_loss is not None:
    print(f"last_train_loss:  {last_train_loss:.6f}")
if last_smoothed_train_loss is not None:
    print(f"smoothed_train_loss: {last_smoothed_train_loss:.6f}")
if TOKEN_BUDGET is not None:
    print(f"token_budget:     {TOKEN_BUDGET}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {session_total_seconds:.1f}")
print(f"lr_multiplier:   {LR_MULTIPLIERS.lr_multiplier:.6f}")
print(f"embedding_lr_multiplier: {LR_MULTIPLIERS.embedding_lr_multiplier:.6f}")
print(f"unembedding_lr_multiplier: {LR_MULTIPLIERS.unembedding_lr_multiplier:.6f}")
print(f"matrix_lr_multiplier: {LR_MULTIPLIERS.matrix_lr_multiplier:.6f}")
print(f"scalar_lr_multiplier: {LR_MULTIPLIERS.scalar_lr_multiplier:.6f}")
print(f"base_embedding_lr: {BASE_LR_PROFILE.embedding_lr:.6f}")
print(f"base_unembedding_lr: {BASE_LR_PROFILE.unembedding_lr:.6f}")
print(f"base_matrix_lr: {BASE_LR_PROFILE.matrix_lr:.6f}")
print(f"base_scalar_lr: {BASE_LR_PROFILE.scalar_lr:.6f}")
print(f"resolved_lm_head_lr: {RESOLVED_LR_PROFILE.lm_head_lr:.6f}")
print(f"resolved_embedding_lr: {RESOLVED_LR_PROFILE.embedding_lr:.6f}")
print(f"resolved_value_embedding_lr: {RESOLVED_LR_PROFILE.value_embedding_lr:.6f}")
print(f"resolved_resid_lr: {RESOLVED_LR_PROFILE.resid_lr:.6f}")
print(f"resolved_x0_lr: {RESOLVED_LR_PROFILE.x0_lr:.6f}")
print(f"resolved_matrix_lr: {RESOLVED_LR_PROFILE.matrix_lr:.6f}")
print(f"probe_pathology_triggered: {str(probe_pathology_triggered).lower()}")
print(f"probe_pathology_reason: {probe_pathology_reason or 'none'}")
if STREAMING_EVAL_CONFIG is None:
    print("streaming_eval_points: skipped")
    print("streaming_eval_cycles_completed: skipped")
    print("streaming_eval_cycle_batches: skipped")
    print("streaming_eval_total_seconds: skipped")
    print("streaming_val_auc: skipped")
    print("honest_val_bpb: skipped")
    print("subref_one_sixth_eval_points: skipped")
    print("subref_one_sixth_eval_cycles_completed: skipped")
    print("honest_subref_one_sixth_bpb: skipped")
    print("reference_streaming_eval_cycles_completed: skipped")
    print("honest_reference_bpb: skipped")
else:
    print(f"streaming_eval_mode: {STREAMING_EVAL_CONFIG.mode}")
    if streaming_eval_summary is None:
        print("streaming_eval_points: skipped")
        print("streaming_eval_cycles_completed: skipped")
        print("streaming_eval_cycle_batches: skipped")
    else:
        print(f"streaming_eval_points: {streaming_eval_summary.total_points}")
        print(f"streaming_eval_cycles_completed: {streaming_eval_summary.cycles_completed}")
        print(f"streaming_eval_cycle_batches: {streaming_eval_summary.cycle_batches}")
    print(f"streaming_eval_total_seconds: {streaming_eval_seconds + subref_streaming_eval_seconds:.1f}")
    if streaming_eval_summary is None or streaming_eval_summary.auc is None:
        print("streaming_val_auc: skipped")
    else:
        print(f"streaming_val_auc: {streaming_eval_summary.auc:.6f}")
    if streaming_eval_summary is None or streaming_eval_summary.honest_bpb is None:
        print("honest_val_bpb: skipped")
    else:
        print(f"honest_val_bpb: {streaming_eval_summary.honest_bpb:.6f}")
    if subref_streaming_eval_summary is None:
        print("subref_one_sixth_eval_points: skipped")
        print("subref_one_sixth_eval_cycles_completed: skipped")
        print("honest_subref_one_sixth_bpb: skipped")
        print("reference_streaming_eval_cycles_completed: skipped")
        print("honest_reference_bpb: skipped")
    else:
        print(f"subref_one_sixth_eval_points: {subref_streaming_eval_summary.total_points}")
        print(f"subref_one_sixth_eval_cycles_completed: {subref_streaming_eval_summary.cycles_completed}")
        if derived_reference_streaming is None or derived_reference_streaming.honest_subcycle_bpb is None:
            print("honest_subref_one_sixth_bpb: skipped")
        else:
            print(f"honest_subref_one_sixth_bpb: {derived_reference_streaming.honest_subcycle_bpb:.6f}")
        if derived_reference_streaming is None:
            print("reference_streaming_eval_cycles_completed: skipped")
            print("honest_reference_bpb: skipped")
        else:
            print(
                "reference_streaming_eval_cycles_completed: "
                f"{derived_reference_streaming.supercycles_completed}"
            )
            if derived_reference_streaming.honest_supercycle_bpb is None:
                print("honest_reference_bpb: skipped")
            else:
                print(f"honest_reference_bpb: {derived_reference_streaming.honest_supercycle_bpb:.6f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"peak_flop_utilization_percent: {steady_state_mfu:.2f}")
print(f"compute_share_percent: {telemetry_summary.compute_share_percent:.2f}")
print(f"input_pipeline_percent: {telemetry_summary.observed_percent(OBS_INPUT_PIPELINE):.2f}")
print(f"train_tflops:     {telemetry_summary.train_tflops:.3f}")
print(f"optimizer_percent: {telemetry_summary.phase_percent(PHASE_OPTIMIZER):.2f}")
print(f"phase_optimizer_percent: {telemetry_summary.phase_percent(PHASE_OPTIMIZER):.2f}")
print(f"forward_backward_percent: {telemetry_summary.phase_percent(PHASE_FORWARD_BACKWARD):.2f}")
print(f"phase_forward_backward_percent: {telemetry_summary.phase_percent(PHASE_FORWARD_BACKWARD):.2f}")
print(f"other_step_percent: {telemetry_summary.other_step_percent:.2f}")
print(f"checkpoint_percent: {checkpoint_percent:.2f}")
print(f"checkpoint_write_percent: {checkpoint_write_percent:.2f}")
print(f"eval_percent:     {eval_percent:.2f}")
print(f"checkpoint_count: {session_checkpoint_count}")
print(f"util_window_steps: {telemetry_summary.window_steps}")
print(f"util_window:      cumulative {telemetry_summary.window_label}")
print(f"steady_state_tok_per_sec: {steady_state_tok_per_sec:.1f}")
print(f"cumulative_checkpoint_seconds: {cumulative_checkpoint_seconds:.3f}")
print(f"cumulative_checkpoint_write_seconds: {cumulative_checkpoint_write_seconds:.3f}")
print(f"cumulative_checkpoint_count: {cumulative_checkpoint_count}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
if CURVE_EVAL_SECONDS:
    print(f"curve_eval_points: {len(curve_points)}")
    print(f"curve_eval_total_seconds: {curve_eval_total_seconds:.1f}")
    if ARGS.curve_output:
        print(f"curve_output:     {ARGS.curve_output}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"hardware_key:     {hardware_key}")
print(f"accelerator_architecture: {cuda_runtime.reference_family.family_key}")
print(f"accelerator_compute_capability: {cap[0]}.{cap[1]}")
print(f"cuda_reference_family: {cuda_runtime.reference_family.family_key}")
if cuda_runtime.preferred_flash_attention_generation is not None:
    print(f"preferred_flash_attention_generation: {cuda_runtime.preferred_flash_attention_generation}")
print(f"preferred_flash_attention_repo: {cuda_runtime.selected_flash_attention_repo}")
print(f"resolved_attention_backend: {RESOLVED_ATTENTION_BACKEND}")
if checkpoint_saved_to is not None:
    print(f"checkpoint_path:  {checkpoint_saved_to}")
if CHECKPOINT_INTERVAL is not None:
    print(f"checkpoint_interval: {format_interval_spec(CHECKPOINT_INTERVAL)}")
print(f"checkpoint_save_mode: {CHECKPOINT_SAVE_MODE}")
if resumed_from is not None:
    print(f"resumed_from:     {resumed_from}")
