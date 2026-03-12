from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from autoresearch_lab.ledger import append_lab_event
from autoresearch_lab.labs import LabBenchResult, LabExtractResult


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_STARTER_TARGET_KEYS = (
    "launch_fusion",
    "norm",
    "loss_prelude",
    "data_movement",
    "matmul_epilogue",
    "attention_prelude",
    "rope_qk_fused",
    "fused_mlp",
)


LAUNCH_FUSION_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Launch fusion
Mutable file: yes

This starter target represents a launch-bound residual-add region. It ships
with an optional Triton implementation and falls back to the reference path
when Triton or CUDA is unavailable.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - workspace fallback path
    triton = None
    tl = None

KERNEL_TARGET = "launch_fusion"
WORKSPACE_IMPL = "triton-optional"

if triton is not None:

    @triton.jit
    def _launch_fusion_kernel(
        residual_ptr,
        update_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        residual = tl.load(residual_ptr + offsets, mask=mask)
        update = tl.load(update_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, residual + update, mask=mask)


def _launch_fusion_triton(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    residual_flat = residual.contiguous().view(-1)
    update_flat = update.contiguous().view(-1)
    output_flat = torch.empty_like(residual_flat)
    n_elements = output_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _launch_fusion_kernel[grid](
        residual_flat,
        update_flat,
        output_flat,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return output_flat.view_as(residual)


def kernel_fn(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if (
        triton is not None
        and residual.is_cuda
        and update.is_cuda
        and residual.is_contiguous()
        and update.is_contiguous()
    ):
        return _launch_fusion_triton(residual, update)
    return residual + update
'''


NORM_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: RMSNorm
Mutable file: yes

This starter target ships with an optional Triton row-wise RMSNorm
implementation and falls back to the reference path when Triton or CUDA is
unavailable.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - workspace fallback path
    triton = None
    tl = None


KERNEL_TARGET = "norm"
WORKSPACE_IMPL = "triton-optional"

if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        stride_row,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_idx * stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        y = x * inv_rms * weight
        tl.store(output_ptr + row_idx * stride_row + offsets, y, mask=mask)


def _rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_contig = x.contiguous()
    rows = x_contig.numel() // x_contig.shape[-1]
    n_cols = x_contig.shape[-1]
    x_2d = x_contig.view(rows, n_cols)
    output = torch.empty_like(x_2d)
    block_size = min(4096, triton.next_power_of_2(n_cols))
    num_warps = 4 if block_size <= 1024 else 8
    _rmsnorm_kernel[(rows,)](
        x_2d,
        weight.contiguous(),
        output,
        x_2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output.view_as(x_contig)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if triton is not None and x.is_cuda and weight.is_cuda and x.is_contiguous() and weight.is_contiguous():
        return _rmsnorm_triton(x, weight, eps=eps)
    x32 = x.float()
    scale = torch.rsqrt(torch.mean(x32.square(), dim=-1, keepdim=True) + eps)
    return (x32 * scale * weight.float()).to(dtype=x.dtype)
'''


LOSS_PRELUDE_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Loss prelude
Mutable file: yes

Replace `kernel_fn` with a faster Triton/CUDA implementation once the reference
path is working.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


KERNEL_TARGET = "loss_prelude"


def kernel_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_bytes: torch.Tensor,
    softcap: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits32 = logits.float()
    softcapped = softcap * torch.tanh(logits32 / softcap)
    log_probs = F.log_softmax(softcapped, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    weighted = nll * token_bytes.float()
    return weighted.to(dtype=logits.dtype), weighted.sum(), token_bytes.sum()
'''


DATA_MOVEMENT_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Data movement
Mutable file: yes

This starter target ships with an optional Triton copy/reshape implementation
and falls back to the reference path when Triton or CUDA is unavailable. The
starter target models the attention-output reshape from `[B, T, H, D]` to
`[B, T, H*D]`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - workspace fallback path
    triton = None
    tl = None


KERNEL_TARGET = "data_movement"
WORKSPACE_IMPL = "triton-optional"

if triton is not None:

    @triton.jit
    def _data_movement_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, values, mask=mask)


def _data_movement_triton(x: torch.Tensor) -> torch.Tensor:
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    output_flat = output.view(-1)
    n_elements = output_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _data_movement_kernel[grid](
        x_contig.view(-1),
        output_flat,
        n_elements,
        BLOCK_SIZE=1024,
    )
    bsz, seqlen, _, _ = x_contig.shape
    return output.view(bsz, seqlen, -1)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    if triton is not None and x.is_cuda and x.is_contiguous():
        return _data_movement_triton(x)
    bsz, seqlen, _, _ = x.shape
    return x.contiguous().view(bsz, seqlen, -1)
'''


MATMUL_EPILOGUE_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Matmul epilogue
Mutable file: yes

This starter target ships with an optional Triton matmul+bias implementation
and falls back to the reference path when Triton or CUDA is unavailable.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - workspace fallback path
    triton = None
    tl = None


KERNEL_TARGET = "matmul_epilogue"
WORKSPACE_IMPL = "triton-optional"

if triton is not None:

    @triton.jit
    def _matmul_epilogue_kernel(
        a_ptr,
        b_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + offs_k
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
            b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            accumulator += tl.dot(a, b)
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


def _matmul_epilogue_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    bias_contig = bias.contiguous()
    m, k = x_contig.shape
    k_w, n = weight_contig.shape
    if k != k_w:
        raise ValueError(f"matmul_epilogue expects matching K dimensions, got {k} and {k_w}")
    output = torch.empty((m, n), device=x_contig.device, dtype=x_contig.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _matmul_epilogue_kernel[grid](
        x_contig,
        weight_contig,
        bias_contig,
        output,
        m,
        n,
        k,
        x_contig.stride(0),
        x_contig.stride(1),
        weight_contig.stride(0),
        weight_contig.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
    )
    return output


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if triton is not None and x.is_cuda and weight.is_cuda and bias.is_cuda and x.dim() == 2 and weight.dim() == 2:
        return _matmul_epilogue_triton(x, weight, bias)
    return torch.matmul(x, weight) + bias
'''


ATTENTION_PRELUDE_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Attention prelude
Mutable file: yes

Replace `kernel_fn` with a faster Triton/CUDA implementation once the reference
path is working.
"""

from __future__ import annotations

import torch


KERNEL_TARGET = "attention_prelude"


def kernel_fn(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    n_head: int,
    head_dim: int,
    ve: torch.Tensor | None = None,
    ve_gate_weight: torch.Tensor | None = None,
    ve_gate_channels: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, _ = x.shape
    q = torch.matmul(x, wq).view(bsz, seqlen, n_head, head_dim)
    k = torch.matmul(x, wk).view(bsz, seqlen, n_head, head_dim)
    v = torch.matmul(x, wv).view(bsz, seqlen, n_head, head_dim)
    if ve is not None and ve_gate_weight is not None:
        ve_view = ve.view(bsz, seqlen, n_head, head_dim)
        gate = 2 * torch.sigmoid(torch.matmul(x[..., :ve_gate_channels], ve_gate_weight))
        v = v + gate.unsqueeze(-1) * ve_view
    return q, k, v
'''


ROPE_QK_FUSED_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: RoPE + Q/K RMSNorm
Mutable file: yes

Replace `kernel_fn` with a faster Triton/CUDA implementation once the reference
path is working.
"""

from __future__ import annotations

import torch


KERNEL_TARGET = "rope_qk_fused"


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    scale = torch.rsqrt(torch.mean(x32.square(), dim=-1, keepdim=True) + eps)
    return (x32 * scale).to(dtype=x.dtype)


def kernel_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = _apply_rotary(q, cos, sin)
    k = _apply_rotary(k, cos, sin)
    return _rms_norm(q, eps=eps), _rms_norm(k, eps=eps)
'''


FUSED_MLP_TEMPLATE = '''"""
Autoresearch CUDA kernel lab workspace.

Target: Fused MLP
Mutable file: yes

Replace `kernel_fn` with a faster Triton/CUDA implementation once the reference
path is working.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


KERNEL_TARGET = "fused_mlp"


def kernel_fn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    hidden = torch.matmul(x, w1)
    hidden = F.relu(hidden).square()
    return torch.matmul(hidden, w2)
'''


@dataclass(frozen=True)
class CudaLabCase:
    shape: tuple[int, ...]
    dtype: str
    aux: dict[str, int] | None = None


@dataclass(frozen=True)
class CudaTargetSpec:
    target: str
    template: str
    tolerance: float
    quick_cases: tuple[CudaLabCase, ...]
    full_cases: tuple[CudaLabCase, ...]
    make_inputs: Callable[[Any, CudaLabCase, Any], tuple[Any, ...]]
    reference: Callable[..., Any]


def _launch_fusion_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, c = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    residual = torch.randn((b, t, c), device=device, dtype=dtype)
    update = torch.randn((b, t, c), device=device, dtype=dtype)
    return residual, update


def _launch_fusion_reference(torch: Any, residual: Any, update: Any) -> Any:
    return residual + update


def _norm_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, c = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((b, t, c), device=device, dtype=dtype)
    weight = torch.randn((c,), device=device, dtype=dtype)
    return x, weight


def _norm_reference(torch: Any, x: Any, weight: Any, eps: float = 1e-6) -> Any:
    x32 = x.float()
    scale = torch.rsqrt(torch.mean(x32.square(), dim=-1, keepdim=True) + eps)
    return (x32 * scale * weight.float()).to(dtype=x.dtype)


def _loss_prelude_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    rows, vocab = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    logits = torch.randn((rows, vocab), device=device, dtype=dtype)
    targets = torch.randint(0, vocab, (rows,), device=device, dtype=torch.long)
    token_bytes = torch.randint(1, 5, (rows,), device=device, dtype=torch.int32)
    return logits, targets, token_bytes


def _loss_prelude_reference(
    torch: Any,
    logits: Any,
    targets: Any,
    token_bytes: Any,
    softcap: float = 15.0,
) -> Any:
    logits32 = logits.float()
    softcapped = softcap * torch.tanh(logits32 / softcap)
    log_probs = torch.nn.functional.log_softmax(softcapped, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    weighted = nll * token_bytes.float()
    return weighted.to(dtype=logits.dtype), weighted.sum(), token_bytes.sum()


def _data_movement_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, h, d = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((b, t, h, d), device=device, dtype=dtype)
    return (x,)


def _data_movement_reference(torch: Any, x: Any) -> Any:
    bsz, seqlen, _, _ = x.shape
    return x.contiguous().view(bsz, seqlen, -1)


def _matmul_epilogue_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    m, k, n = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = torch.randn((k, n), device=device, dtype=dtype)
    bias = torch.randn((n,), device=device, dtype=dtype)
    return x, weight, bias


def _matmul_epilogue_reference(torch: Any, x: Any, weight: Any, bias: Any) -> Any:
    return torch.matmul(x, weight) + bias


def _attention_prelude_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, c = case.shape
    aux = case.aux or {}
    n_head = int(aux.get("n_head", 8))
    head_dim = c // n_head
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((b, t, c), device=device, dtype=dtype)
    wq = torch.randn((c, n_head * head_dim), device=device, dtype=dtype)
    wk = torch.randn((c, n_head * head_dim), device=device, dtype=dtype)
    wv = torch.randn((c, n_head * head_dim), device=device, dtype=dtype)
    ve = torch.randn((b, t, n_head * head_dim), device=device, dtype=dtype)
    ve_gate_channels = int(aux.get("ve_gate_channels", 32))
    ve_gate_weight = torch.randn((ve_gate_channels, n_head), device=device, dtype=dtype)
    return x, wq, wk, wv, n_head, head_dim, ve, ve_gate_weight, ve_gate_channels


def _attention_prelude_reference(
    torch: Any,
    x: Any,
    wq: Any,
    wk: Any,
    wv: Any,
    n_head: int,
    head_dim: int,
    ve: Any | None = None,
    ve_gate_weight: Any | None = None,
    ve_gate_channels: int = 32,
) -> Any:
    bsz, seqlen, _ = x.shape
    q = torch.matmul(x, wq).view(bsz, seqlen, n_head, head_dim)
    k = torch.matmul(x, wk).view(bsz, seqlen, n_head, head_dim)
    v = torch.matmul(x, wv).view(bsz, seqlen, n_head, head_dim)
    if ve is not None and ve_gate_weight is not None:
        ve_view = ve.view(bsz, seqlen, n_head, head_dim)
        gate = 2 * torch.sigmoid(torch.matmul(x[..., :ve_gate_channels], ve_gate_weight))
        v = v + gate.unsqueeze(-1) * ve_view
    return q, k, v


def _rope_qk_fused_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, h, d = case.shape
    dtype = _resolve_dtype(torch, case.dtype, device)
    q = torch.randn((b, t, h, d), device=device, dtype=dtype)
    k = torch.randn((b, t, h, d), device=device, dtype=dtype)
    cos = torch.randn((1, t, 1, d // 2), device=device, dtype=dtype)
    sin = torch.randn((1, t, 1, d // 2), device=device, dtype=dtype)
    return q, k, cos, sin


def _rope_qk_fused_reference(
    torch: Any,
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    eps: float = 1e-6,
) -> Any:
    def _apply_rotary(x: Any) -> Any:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1)

    def _rms_norm(x: Any) -> Any:
        x32 = x.float()
        scale = torch.rsqrt(torch.mean(x32.square(), dim=-1, keepdim=True) + eps)
        return (x32 * scale).to(dtype=x.dtype)

    return _rms_norm(_apply_rotary(q)), _rms_norm(_apply_rotary(k))


def _fused_mlp_inputs(torch: Any, case: CudaLabCase, device: Any) -> tuple[Any, ...]:
    b, t, c = case.shape
    aux = case.aux or {}
    hidden = int(aux.get("hidden_dim", 4 * c))
    dtype = _resolve_dtype(torch, case.dtype, device)
    x = torch.randn((b, t, c), device=device, dtype=dtype)
    w1 = torch.randn((c, hidden), device=device, dtype=dtype)
    w2 = torch.randn((hidden, c), device=device, dtype=dtype)
    return x, w1, w2


def _fused_mlp_reference(torch: Any, x: Any, w1: Any, w2: Any) -> Any:
    hidden = torch.matmul(x, w1)
    hidden = torch.nn.functional.relu(hidden).square()
    return torch.matmul(hidden, w2)


CUDA_WORKSPACE_TARGET_SPECS: dict[str, CudaTargetSpec] = {
    "launch_fusion": CudaTargetSpec(
        target="launch_fusion",
        template=LAUNCH_FUSION_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 1024), dtype="float32"),
            CudaLabCase(shape=(4, 512, 2048), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(16, 512, 2048), dtype="float16"),
            CudaLabCase(shape=(8, 1024, 4096), dtype="float16"),
        ),
        make_inputs=_launch_fusion_inputs,
        reference=_launch_fusion_reference,
    ),
    "norm": CudaTargetSpec(
        target="norm",
        template=NORM_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 1024), dtype="float32"),
            CudaLabCase(shape=(4, 512, 2048), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(16, 512, 2048), dtype="float16"),
            CudaLabCase(shape=(8, 1024, 4096), dtype="float16"),
        ),
        make_inputs=_norm_inputs,
        reference=_norm_reference,
    ),
    "loss_prelude": CudaTargetSpec(
        target="loss_prelude",
        template=LOSS_PRELUDE_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(512, 2048), dtype="float32"),
            CudaLabCase(shape=(1024, 4096), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(2048, 4096), dtype="float16"),
            CudaLabCase(shape=(1024, 8192), dtype="float16"),
        ),
        make_inputs=_loss_prelude_inputs,
        reference=_loss_prelude_reference,
    ),
    "data_movement": CudaTargetSpec(
        target="data_movement",
        template=DATA_MOVEMENT_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(32, 128, 8, 64), dtype="float32"),
            CudaLabCase(shape=(16, 256, 8, 128), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(32, 256, 8, 128), dtype="float16"),
            CudaLabCase(shape=(16, 512, 16, 128), dtype="float16"),
        ),
        make_inputs=_data_movement_inputs,
        reference=_data_movement_reference,
    ),
    "matmul_epilogue": CudaTargetSpec(
        target="matmul_epilogue",
        template=MATMUL_EPILOGUE_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(512, 1024, 2048), dtype="float32"),
            CudaLabCase(shape=(1024, 1024, 2048), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(1024, 2048, 4096), dtype="float16"),
            CudaLabCase(shape=(2048, 2048, 4096), dtype="float16"),
        ),
        make_inputs=_matmul_epilogue_inputs,
        reference=_matmul_epilogue_reference,
    ),
    "attention_prelude": CudaTargetSpec(
        target="attention_prelude",
        template=ATTENTION_PRELUDE_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 1024), dtype="float32", aux={"n_head": 8}),
            CudaLabCase(shape=(4, 512, 2048), dtype="float32", aux={"n_head": 8}),
        ),
        full_cases=(
            CudaLabCase(shape=(8, 512, 2048), dtype="float16", aux={"n_head": 8}),
            CudaLabCase(shape=(4, 1024, 4096), dtype="float16", aux={"n_head": 8}),
        ),
        make_inputs=_attention_prelude_inputs,
        reference=_attention_prelude_reference,
    ),
    "rope_qk_fused": CudaTargetSpec(
        target="rope_qk_fused",
        template=ROPE_QK_FUSED_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 8, 128), dtype="float32"),
            CudaLabCase(shape=(4, 512, 8, 128), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(8, 512, 8, 128), dtype="float16"),
            CudaLabCase(shape=(4, 1024, 16, 128), dtype="float16"),
        ),
        make_inputs=_rope_qk_fused_inputs,
        reference=_rope_qk_fused_reference,
    ),
    "fused_mlp": CudaTargetSpec(
        target="fused_mlp",
        template=FUSED_MLP_TEMPLATE,
        tolerance=1e-5,
        quick_cases=(
            CudaLabCase(shape=(8, 256, 1024), dtype="float32"),
            CudaLabCase(shape=(4, 512, 2048), dtype="float32"),
        ),
        full_cases=(
            CudaLabCase(shape=(8, 512, 2048), dtype="float16"),
            CudaLabCase(shape=(4, 1024, 4096), dtype="float16"),
        ),
        make_inputs=_fused_mlp_inputs,
        reference=_fused_mlp_reference,
    ),
}


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _import_torch() -> Any:
    import torch  # type: ignore

    return torch


def _resolve_device(torch: Any, requested_device: str) -> tuple[str, Any]:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")
    return requested_device, torch.device(requested_device)


def _resolve_dtype(torch: Any, dtype_name: str, device: Any) -> Any:
    if device.type == "cpu" and dtype_name in {"float16", "bfloat16"}:
        return torch.float32
    return getattr(torch, dtype_name)


def _synchronize(torch: Any, device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()


def _load_workspace_module(workspace: Path) -> Any:
    kernel_path = workspace / "kernel.py"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Workspace does not contain kernel.py: {kernel_path}")
    spec = importlib.util.spec_from_file_location(f"cuda_lab_{workspace.name}", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import workspace kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_outputs(output: Any) -> tuple[Any, ...]:
    if isinstance(output, tuple):
        return output
    return (output,)


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.element_size() * tensor.numel())


def _io_bytes(inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> int:
    total = 0
    for tensor in inputs:
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += _tensor_bytes(tensor)
    for tensor in outputs:
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += _tensor_bytes(tensor)
    return total


def _max_abs_error(torch: Any, actual: tuple[Any, ...], expected: tuple[Any, ...]) -> float:
    max_error = 0.0
    for got, want in zip(actual, expected, strict=True):
        if hasattr(got, "float") and hasattr(want, "float"):
            diff = (got.float() - want.float()).abs().max().item()
            max_error = max(max_error, float(diff))
    return max_error


def _bench_case(
    torch: Any,
    module: Any,
    spec: CudaTargetSpec,
    case: CudaLabCase,
    *,
    device_name: str,
    device: Any,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    inputs = spec.make_inputs(torch, case, device)
    with torch.no_grad():
        expected = _normalize_outputs(spec.reference(torch, *inputs))
        actual = _normalize_outputs(module.kernel_fn(*inputs))
        max_abs_error = _max_abs_error(torch, actual, expected)
        latencies_ms: list[float] = []
        for _ in range(warmup):
            _ = module.kernel_fn(*inputs)
        _synchronize(torch, device_name)
        for _ in range(iterations):
            start = time.perf_counter()
            actual = _normalize_outputs(module.kernel_fn(*inputs))
            _synchronize(torch, device_name)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        median_latency_ms = float(sorted(latencies_ms)[len(latencies_ms) // 2])
        throughput_gb_s = (_io_bytes(inputs, actual) / 1e9) / (median_latency_ms / 1000.0)
    return {
        "shape": case.shape,
        "dtype": case.dtype,
        "median_latency_ms": median_latency_ms,
        "throughput_gb_s": throughput_gb_s,
        "max_abs_error": max_abs_error,
    }


def _run_workspace_harness(
    *,
    workspace: Path,
    quick: bool,
    requested_device: str,
    event_type: str,
) -> LabBenchResult:
    start = time.perf_counter()
    metadata_path = workspace / "metadata.json"
    metadata: dict[str, Any] = {}
    target = "unknown"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target = str(metadata.get("target") or "unknown")
    try:
        torch = _import_torch()
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="missing-runtime",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": f"PyTorch is unavailable: {exc}"},
        )
    try:
        device_name, device = _resolve_device(torch, requested_device)
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="missing-runtime",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": str(exc)},
        )
    if not metadata_path.exists():
        return LabBenchResult(
            target=target,
            status="invalid-workspace",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": f"Workspace metadata is missing: {metadata_path}"},
        )
    spec = CUDA_WORKSPACE_TARGET_SPECS.get(target)
    if spec is None:
        return LabBenchResult(
            target=target,
            status="unsupported-target",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": f"No CUDA workspace harness exists for target {target!r}."},
        )
    try:
        module = _load_workspace_module(workspace)
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="load-failed",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={"failure_reason": str(exc)},
        )
    cases = spec.quick_cases if quick else spec.full_cases
    iterations = 8 if quick else 20
    warmup = 2 if quick else 5
    try:
        case_results = [
            _bench_case(
                torch,
                module,
                spec,
                case,
                device_name=device_name,
                device=device,
                iterations=iterations,
                warmup=warmup,
            )
            for case in cases
        ]
    except Exception as exc:
        return LabBenchResult(
            target=target,
            status="execution-failed",
            metric_name="median_throughput_gb_s",
            metric_value=None,
            wall_seconds=time.perf_counter() - start,
            details={
                "failure_reason": str(exc),
                "device": device_name,
                "quick": quick,
            },
        )
    max_abs_error = max(result["max_abs_error"] for result in case_results)
    median_throughput = float(sum(result["throughput_gb_s"] for result in case_results) / len(case_results))
    median_latency_ms = float(sum(result["median_latency_ms"] for result in case_results) / len(case_results))
    status = "ok" if max_abs_error <= spec.tolerance else "mismatch"
    result = LabBenchResult(
        target=target,
        status=status,
        metric_name="median_throughput_gb_s",
        metric_value=median_throughput,
        wall_seconds=time.perf_counter() - start,
        details={
            "device": device_name,
            "quick": quick,
            "case_results": case_results,
            "max_abs_error": max_abs_error,
            "tolerance": spec.tolerance,
            "median_latency_ms": median_latency_ms,
            "workspace_impl": getattr(module, "WORKSPACE_IMPL", "reference"),
        },
    )
    append_lab_event(
        engine="cuda",
        backend_family="cuda",
        target=target,
        workspace=workspace,
        event_type=event_type,
        status=status,
        metric_name=result.metric_name,
        metric_value=result.metric_value,
        details=result.details,
        preset=metadata.get("profile_context", {}).get("preset"),
    )
    return result


def init_cuda_workspace(*, target: str, workspace: Path, profile_context: dict[str, Any] | None = None) -> Path:
    spec = CUDA_WORKSPACE_TARGET_SPECS.get(target)
    if spec is None:
        supported = ", ".join(sorted(CUDA_WORKSPACE_TARGET_SPECS))
        raise ValueError(f"CUDA starter workspace is not available for {target!r}. Supported targets: {supported}")
    workspace = workspace.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    _write_text(workspace / "kernel.py", spec.template)
    _write_json(
        workspace / "metadata.json",
        {
            "engine": "cuda",
            "backend_family": "cuda",
            "target": target,
            "status": "starter-ready",
            "metric": "throughput_gb_s",
            "workspace_impl": "triton-optional" if target in {"launch_fusion", "norm"} else "reference",
            "profile_context": profile_context or {},
        },
    )
    return workspace


def extract_cuda_workspace_from_profile(*, profile_path: Path, workspace: Path, rank: int = 1) -> LabExtractResult:
    start = time.perf_counter()
    payload = json.loads(profile_path.expanduser().read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if rank <= 0 or rank > len(candidates):
        raise ValueError(f"rank must be between 1 and {len(candidates)}")
    candidate = candidates[rank - 1]
    target = str(candidate["target"])
    if target not in CUDA_WORKSPACE_TARGET_SPECS:
        return LabExtractResult(
            engine="cuda",
            target=target,
            workspace=str(workspace.expanduser()),
            status="starter-unavailable",
            wall_seconds=time.perf_counter() - start,
            details={
                "profile_path": str(profile_path.expanduser()),
                "rank": rank,
                "candidate": candidate,
                "supported_targets": sorted(CUDA_WORKSPACE_TARGET_SPECS),
            },
        )
    profile_context = {
        "preset": payload.get("preset"),
        "profile_path": str(profile_path.expanduser()),
        "profile_rank": rank,
        "priority_score": candidate.get("priority_score"),
        "category": candidate.get("category"),
        "rationale": candidate.get("rationale"),
    }
    init_cuda_workspace(target=target, workspace=workspace, profile_context=profile_context)
    return LabExtractResult(
        engine="cuda",
        target=target,
        workspace=str(workspace.expanduser()),
        status="ok",
        wall_seconds=time.perf_counter() - start,
        details={
            "profile_path": str(profile_path.expanduser()),
            "rank": rank,
            "candidate": candidate,
            "profile_context": profile_context,
        },
    )


def bench_cuda_workspace(*, workspace: Path, quick: bool = False, device: str = "auto") -> LabBenchResult:
    return _run_workspace_harness(
        workspace=workspace.expanduser(),
        quick=quick,
        requested_device=device,
        event_type="bench",
    )


def verify_cuda_workspace(*, workspace: Path, quick: bool = False, device: str = "auto") -> LabBenchResult:
    return _run_workspace_harness(
        workspace=workspace.expanduser(),
        quick=quick,
        requested_device=device,
        event_type="verify",
    )
