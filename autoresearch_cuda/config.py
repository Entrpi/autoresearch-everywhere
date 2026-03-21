from __future__ import annotations

from dataclasses import dataclass, replace

from autoresearch_platform.lr_profile import DEFAULT_LR_PROFILE, LrProfile


CUDA_DEFAULT_SEQ_LEN = 2048
CUDA_DEFAULT_TIME_BUDGET = 300.0


@dataclass(frozen=True)
class CudaRunPreset:
    description: str
    time_budget: float
    seq_len: int
    aspect_ratio: int
    head_dim: int
    window_pattern: str
    total_batch_size: int
    weight_decay: float
    adam_betas: tuple[float, float]
    warmup_ratio: float
    warmdown_ratio: float
    final_lr_frac: float
    depth: int
    device_batch_size: int
    lr_profile: LrProfile = DEFAULT_LR_PROFILE


CUDA_PRESETS: dict[str, CudaRunPreset] = {
    "m5-tiny": CudaRunPreset(
        description="CUDA starter preset matching the MLX tiny model scale",
        time_budget=CUDA_DEFAULT_TIME_BUDGET,
        seq_len=256,
        aspect_ratio=64,
        head_dim=128,
        window_pattern="L",
        total_batch_size=65_536,
        weight_decay=0.2,
        adam_betas=(0.8, 0.95),
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        final_lr_frac=0.0,
        depth=2,
        device_batch_size=64,
    ),
    "m5-small": CudaRunPreset(
        description="CUDA default starter preset matching the MLX small model scale",
        time_budget=CUDA_DEFAULT_TIME_BUDGET,
        seq_len=512,
        aspect_ratio=64,
        head_dim=128,
        window_pattern="L",
        total_batch_size=65_536,
        weight_decay=0.2,
        adam_betas=(0.8, 0.95),
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        final_lr_frac=0.0,
        depth=4,
        device_batch_size=32,
    ),
    "m5-balanced": CudaRunPreset(
        description="CUDA balanced preset matching the MLX validation-centered model scale",
        time_budget=CUDA_DEFAULT_TIME_BUDGET,
        seq_len=1024,
        aspect_ratio=64,
        head_dim=128,
        window_pattern="SSSSL",
        total_batch_size=65_536,
        weight_decay=0.2,
        adam_betas=(0.8, 0.95),
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        final_lr_frac=0.0,
        depth=6,
        device_batch_size=16,
    ),
    "m5-large": CudaRunPreset(
        description="CUDA bridge preset matching the MLX large model scale",
        time_budget=CUDA_DEFAULT_TIME_BUDGET,
        seq_len=512,
        aspect_ratio=64,
        head_dim=128,
        window_pattern="SSSSL",
        total_batch_size=65_536,
        weight_decay=0.2,
        adam_betas=(0.8, 0.95),
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        final_lr_frac=0.0,
        depth=8,
        device_batch_size=32,
    ),
    "m5-xlarge": CudaRunPreset(
        description="CUDA largest practical local preset matching the MLX xlarge model scale",
        time_budget=CUDA_DEFAULT_TIME_BUDGET,
        seq_len=2048,
        aspect_ratio=64,
        head_dim=128,
        window_pattern="L",
        total_batch_size=65_536,
        weight_decay=0.2,
        adam_betas=(0.8, 0.95),
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        final_lr_frac=0.0,
        depth=8,
        device_batch_size=8,
    ),
    "upstream": CudaRunPreset(
        description="Upstream CUDA reference preset",
        time_budget=CUDA_DEFAULT_TIME_BUDGET,
        seq_len=CUDA_DEFAULT_SEQ_LEN,
        aspect_ratio=64,
        head_dim=128,
        window_pattern="SSSL",
        total_batch_size=2**19,
        weight_decay=0.2,
        adam_betas=(0.8, 0.95),
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        final_lr_frac=0.0,
        depth=8,
        device_batch_size=128,
    ),
}


def resolve_run_preset(
    preset: str,
    *,
    time_budget: float | None = None,
    seq_len: int | None = None,
    window_pattern: str | None = None,
    total_batch_size: int | None = None,
    depth: int | None = None,
    device_batch_size: int | None = None,
) -> CudaRunPreset:
    try:
        base = CUDA_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown CUDA preset: {preset}") from exc
    return replace(
        base,
        time_budget=time_budget if time_budget is not None else base.time_budget,
        seq_len=seq_len if seq_len is not None else base.seq_len,
        window_pattern=window_pattern if window_pattern is not None else base.window_pattern,
        total_batch_size=total_batch_size if total_batch_size is not None else base.total_batch_size,
        depth=depth if depth is not None else base.depth,
        device_batch_size=device_batch_size if device_batch_size is not None else base.device_batch_size,
    )
