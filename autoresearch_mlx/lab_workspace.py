from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from autoresearch_lab.ledger import append_lab_event, summarize_lab_evidence
from autoresearch_lab.labs import (
    LabBenchResult,
    LabCapabilities,
    LabEvidenceResult,
    LabExtractResult,
    LabIntegrationABResult,
    LabOrchestrationPlan,
    LabPromotionCheck,
    LabProfileResult,
    LabTraceResult,
    LabTarget,
)
from autoresearch_mlx.calibration_signature import (
    current_eval_semantics_signature,
    current_runtime_shape_signature,
)
from autoresearch_mlx.eval_policy import detect_current_hardware_key
from autoresearch_mlx.lab_integration import integration_environment, supports_direct_integration
from autoresearch_mlx.lab_profile import (
    extract_from_profile,
    orchestrate_from_profile,
    profile_mlx_targets,
)
from autoresearch_mlx.lab_trace import capture_workspace_trace
from autoresearch_platform.platform_defaults import load_platform_default_cache
from tools.calibrate_eval_policy import parse_summary


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LabCase:
    shape: tuple[int, ...]
    dtype: str
    aux: dict[str, int] | None = None


@dataclass(frozen=True)
class TargetSpec:
    info: LabTarget
    template: str
    tolerance: float
    quick_cases: tuple[LabCase, ...]
    full_cases: tuple[LabCase, ...]
    make_inputs: Callable[[LabCase], tuple]
    reference: Callable[..., mx.array]
    metric_value: Callable[[LabCase, float], float]


LAYERNORM_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: LayerNorm
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a correct
MLX reference implementation. Later versions can wrap `mx.fast.metal_kernel`
or `@mx.custom_function`.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "layernorm"


def kernel_fn(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    inv = mx.rsqrt(variance + eps)
    y = (x32 - mean) * inv
    return (y * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(x.dtype)
'''


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
    x32 = x.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * rms * weight.astype(mx.float32)).astype(x.dtype)
'''


ROTARY_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Rotary embedding
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a correct
MLX reference implementation.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "rotary_embedding"


def kernel_fn(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)
'''


REDUCE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Reduce
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path performs a
row-wise sum reduction over the last dimension.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "reduce"


def kernel_fn(x: mx.array) -> mx.array:
    return mx.sum(x, axis=-1)
'''


SOFTMAX_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Softmax
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is a stable
reference implementation over the last dimension.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "softmax"


def kernel_fn(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    shifted = x32 - mx.max(x32, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return (exp / mx.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)
'''


RESIDUAL_BLEND_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Residual blend
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's per-layer residual blend before each block.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "residual_blend"


def kernel_fn(x: mx.array, x0: mx.array, resid_lambda: float, x0_lambda: float) -> mx.array:
    return resid_lambda * x + x0_lambda * x0
'''


RESIDUAL_RMSNORM_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Residual blend + RMSNorm
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's residual blend followed by RMSNorm.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "residual_rmsnorm"


def kernel_fn(
    x: mx.array,
    x0: mx.array,
    resid_lambda: float,
    x0_lambda: float,
    eps: float = 1e-6,
) -> mx.array:
    blended = resid_lambda * x + x0_lambda * x0
    x32 = blended.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(blended.dtype)
'''


QK_RMSNORM_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Q/K RMSNorm
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's post-RoPE RMSNorm over Q and K.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "qk_rmsnorm"


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(x.dtype)


def kernel_fn(q: mx.array, k: mx.array, eps: float = 1e-6) -> tuple[mx.array, mx.array]:
    return _rms_norm(q, eps=eps), _rms_norm(k, eps=eps)
'''


ROPE_QK_FUSED_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: RoPE + Q/K RMSNorm
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's apply_rotary_emb -> RMSNorm path for Q and K.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "rope_qk_fused"


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(x.dtype)


def kernel_fn(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    q = _apply_rotary(q, cos, sin)
    k = _apply_rotary(k, cos, sin)
    return _rms_norm(q, eps=eps), _rms_norm(k, eps=eps)
'''


LOGITS_SOFTCAP_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Logits softcap
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's final tanh-based logits softcap.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "logits_softcap"


def kernel_fn(logits: mx.array, softcap: float = 15.0) -> mx.array:
    x32 = logits.astype(mx.float32)
    return (softcap * mx.tanh(x32 / softcap)).astype(logits.dtype)
'''


ACTIVATION_POINTWISE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Activation pointwise
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's squared-ReLU activation in the MLP.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "activation_pointwise"


def kernel_fn(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    return mx.square(mx.maximum(x32, 0)).astype(x.dtype)
'''


VALUE_EMBED_GATE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Value embed gate
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's value-embed gate and add path in attention.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "value_embed_gate"


def kernel_fn(x_gate: mx.array, gate_weight: mx.array, v: mx.array, ve: mx.array) -> mx.array:
    gate = 2.0 * mx.sigmoid(x_gate @ gate_weight)
    return v + gate[..., None] * ve
'''


VE_LOOKUP_RESHAPE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Value embed lookup + reshape
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's value-embed lookup followed by reshape into KV heads.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "ve_lookup_reshape"


def kernel_fn(value_embed_table: mx.array, idx: mx.array, n_kv_head: int, head_dim: int) -> mx.array:
    batch_size, seq_len = idx.shape
    ve = value_embed_table[idx]
    return ve.reshape(batch_size, seq_len, n_kv_head, head_dim)
'''


ATTENTION_MASK_LOCAL_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Local attention mask
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's local-causal attention mask construction for one window size.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "attention_mask_local"


def kernel_fn(rows: mx.array, cols: mx.array, window_size: int) -> mx.array:
    causal = cols <= rows
    local = cols >= (rows - window_size + 1)
    allowed = causal & local
    mask = (~allowed).astype(mx.float32) * mx.finfo(mx.float32).min
    return mask[None, None, :, :]
'''


PROJ_HEAD_RESHAPE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Projection-head reshape
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's attention output transpose and reshape before the output projection.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "proj_head_reshape"


def kernel_fn(y: mx.array) -> mx.array:
    batch_size, heads, seq_len, head_dim = y.shape
    return y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, heads * head_dim)
'''


CROSS_ENTROPY_SOFTCAP_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Loss logits cast + softcap
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's final logits cast to float32 followed by tanh softcap.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "loss_logits_cast_softcap"


def kernel_fn(logits_bf16: mx.array, softcap: float = 15.0) -> mx.array:
    logits = logits_bf16.astype(mx.float32)
    return softcap * mx.tanh(logits / softcap)
'''


CROSS_ENTROPY_PRELUDE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Cross-entropy prelude
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's byte-aware masked reduction around flattened cross-entropy loss output.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "cross_entropy_prelude"


def kernel_fn(loss_flat: mx.array, target_ids: mx.array, token_bytes: mx.array) -> tuple[mx.array, mx.array]:
    nbytes = token_bytes[target_ids]
    mask = nbytes > 0
    return mx.sum(loss_flat * mask.astype(loss_flat.dtype)), mx.sum(nbytes.astype(mx.int64))
'''


CROSS_ENTROPY_FULL_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Full cross-entropy loss path
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's final logits cast + softcap + cross-entropy + byte-aware masked
reduction.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


KERNEL_TARGET = "cross_entropy_full"


def kernel_fn(
    logits_bf16: mx.array,
    target_ids: mx.array,
    token_bytes: mx.array,
    softcap: float = 15.0,
) -> tuple[mx.array, mx.array]:
    logits = logits_bf16.astype(mx.float32)
    logits = softcap * mx.tanh(logits / softcap)
    losses = nn.losses.cross_entropy(logits, target_ids, reduction="none")
    nbytes = token_bytes[target_ids]
    mask = nbytes > 0
    return mx.sum(losses * mask.astype(losses.dtype)), mx.sum(nbytes.astype(mx.int64))
'''


ATTENTION_PRELUDE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Attention prelude
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's attention preparation up to, but not including, scaled dot-product
attention itself.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "attention_prelude"


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(x.dtype)


def kernel_fn(
    q_proj: mx.array,
    k_proj: mx.array,
    v_proj: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    q = _rms_norm(_apply_rotary(q_proj, cos, sin))
    k = _rms_norm(_apply_rotary(k_proj, cos, sin))
    v = v_proj
    return (
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
    )
'''


BLOCK_PRELUDE_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Block prelude
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
bounded block setup work before attention proper: residual blend, RMSNorm, and
the attention prelude up to, but not including, scaled dot-product attention.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "block_prelude"


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(x.dtype)


def kernel_fn(
    x: mx.array,
    x0: mx.array,
    q_proj: mx.array,
    k_proj: mx.array,
    v_proj: mx.array,
    cos: mx.array,
    sin: mx.array,
    resid_lambda: float,
    x0_lambda: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    x_blended = resid_lambda * x + x0_lambda * x0
    x_norm = _rms_norm(x_blended)
    q = _rms_norm(_apply_rotary(q_proj, cos, sin)).transpose(0, 2, 1, 3)
    k = _rms_norm(_apply_rotary(k_proj, cos, sin)).transpose(0, 2, 1, 3)
    v = v_proj.transpose(0, 2, 1, 3)
    return x_norm, q, k, v
'''


RMSNORM_BACKWARD_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: RMSNorm backward
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is an
analytical backward reference implementation.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "rmsnorm_backward"


def kernel_fn(
    x: mx.array,
    weight: mx.array,
    grad_out: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    g32 = grad_out.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    weighted_grad = g32 * w32
    dot = mx.mean(weighted_grad * x32, axis=-1, keepdims=True)
    dx = rms * weighted_grad - x32 * mx.power(rms, 3) * dot
    dweight = mx.sum(g32 * (x32 * rms), axis=0)
    return dx.astype(x.dtype), dweight.astype(weight.dtype)
'''


LAYERNORM_BACKWARD_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: LayerNorm backward
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path is an
analytical backward reference implementation.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "layernorm_backward"


def kernel_fn(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    grad_out: mx.array,
    eps: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array]:
    del bias
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    g32 = grad_out.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    centered = x32 - mean
    variance = mx.mean(mx.square(centered), axis=-1, keepdims=True)
    inv = mx.rsqrt(variance + eps)
    normalized = centered * inv
    weighted_grad = g32 * w32
    dim = x32.shape[-1]
    sum_u = mx.sum(weighted_grad, axis=-1, keepdims=True)
    sum_un = mx.sum(weighted_grad * normalized, axis=-1, keepdims=True)
    dx = (inv / dim) * (dim * weighted_grad - sum_u - normalized * sum_un)
    dweight = mx.sum(g32 * normalized, axis=0)
    dbias = mx.sum(g32, axis=0)
    return dx.astype(x.dtype), dweight.astype(weight.dtype), dbias.astype(weight.dtype)
'''


FUSED_MLP_TEMPLATE = '''"""
Autoresearch MLX kernel lab workspace.

Target: Fused MLP
Mutable file: yes

Replace `kernel_fn` with a faster implementation. The starter path matches the
repo's squared-ReLU MLP block without bias terms.
"""

from __future__ import annotations

import mlx.core as mx


KERNEL_TARGET = "fused_mlp"


def kernel_fn(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
    hidden = x @ w1
    hidden = mx.square(mx.maximum(hidden, 0))
    return hidden @ w2
'''


def _dtype(dtype_name: str):
    if dtype_name == "float16":
        return mx.float16
    if dtype_name == "float32":
        return mx.float32
    raise ValueError(f"Unsupported dtype for MLX lab: {dtype_name}")


def _numpy_dtype(dtype_name: str):
    return np.dtype(np.float16 if dtype_name == "float16" else np.float32)


def _throughput_gb_s(bytes_moved: int, latency_ms: float) -> float:
    return bytes_moved / (latency_ms / 1e3) / 1e9


def _tflops(flops: float, latency_ms: float) -> float:
    return flops / (latency_ms / 1e3) / 1e12


def _rng():
    return np.random.default_rng(42)


def _iter_leaves(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_leaves(item)
        return
    yield value


def _mx_array(shape: tuple[int, ...], dtype_name: str, scale: float = 1.0) -> mx.array:
    values = _rng().standard_normal(shape, dtype=np.float32) * scale
    return mx.array(values, dtype=_dtype(dtype_name))


def _mx_int_array(shape: tuple[int, ...], *, low: int, high: int, dtype=mx.int32) -> mx.array:
    values = _rng().integers(low, high, size=shape, dtype=np.int32)
    return mx.array(values, dtype=dtype)


def _rmsnorm_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((dim,), case.dtype)


def _rmsnorm_ref(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * rms * weight.astype(mx.float32)).astype(x.dtype)


def _rmsnorm_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((2 * rows * dim + dim) * itemsize, latency_ms)


def _layernorm_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((dim,), case.dtype), _mx_array((dim,), case.dtype)


def _layernorm_ref(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    inv = mx.rsqrt(variance + eps)
    y = (x32 - mean) * inv
    return (y * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(x.dtype)


def _layernorm_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((3 * rows * dim + 2 * dim) * itemsize, latency_ms)


def _rotary_inputs(case: LabCase):
    batch, seq, heads, dim = case.shape
    x = _mx_array((batch, seq, heads, dim), case.dtype)
    half = dim // 2
    cos = _mx_array((1, seq, 1, half), case.dtype)
    sin = _mx_array((1, seq, 1, half), case.dtype)
    return x, cos, sin


def _rotary_ref(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def _rotary_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, heads, dim = case.shape
    half = dim // 2
    itemsize = _numpy_dtype(case.dtype).itemsize
    bytes_moved = (2 * batch * seq * heads * dim + 2 * seq * half) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _reduce_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype),)


def _reduce_ref(x: mx.array) -> mx.array:
    return mx.sum(x, axis=-1)


def _reduce_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((rows * dim + rows) * itemsize, latency_ms)


def _softmax_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype),)


def _softmax_ref(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    shifted = x32 - mx.max(x32, axis=-1, keepdims=True)
    exp = mx.exp(shifted)
    return (exp / mx.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def _softmax_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((2 * rows * dim) * itemsize, latency_ms)


def _residual_blend_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((rows, dim), case.dtype), 1.0, 0.1


def _residual_blend_ref(x: mx.array, x0: mx.array, resid_lambda: float, x0_lambda: float) -> mx.array:
    return resid_lambda * x + x0_lambda * x0


def _residual_blend_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((3 * rows * dim) * itemsize, latency_ms)


def _residual_rmsnorm_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((rows, dim), case.dtype), 1.0, 0.1


def _residual_rmsnorm_ref(
    x: mx.array,
    x0: mx.array,
    resid_lambda: float,
    x0_lambda: float,
    eps: float = 1e-6,
) -> mx.array:
    blended = resid_lambda * x + x0_lambda * x0
    x32 = blended.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(blended.dtype)


def _residual_rmsnorm_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((4 * rows * dim) * itemsize, latency_ms)


def _qk_rmsnorm_inputs(case: LabCase):
    return _mx_array(case.shape, case.dtype), _mx_array(case.shape, case.dtype)


def _qk_rmsnorm_ref(q: mx.array, k: mx.array, eps: float = 1e-6) -> tuple[mx.array, mx.array]:
    return _residual_rmsnorm_ref(q, mx.zeros_like(q), 1.0, 0.0, eps=eps), _residual_rmsnorm_ref(
        k, mx.zeros_like(k), 1.0, 0.0, eps=eps
    )


def _qk_rmsnorm_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, heads, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((4 * batch * seq * heads * dim) * itemsize, latency_ms)


def _rope_qk_inputs(case: LabCase):
    batch, seq, heads, dim = case.shape
    q = _mx_array((batch, seq, heads, dim), case.dtype)
    k = _mx_array((batch, seq, heads, dim), case.dtype)
    half = dim // 2
    cos = _mx_array((1, seq, 1, half), case.dtype)
    sin = _mx_array((1, seq, 1, half), case.dtype)
    return q, k, cos, sin


def _rope_qk_ref(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    q = _rotary_ref(q, cos, sin)
    k = _rotary_ref(k, cos, sin)
    return _qk_rmsnorm_ref(q, k, eps=eps)


def _rope_qk_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, heads, dim = case.shape
    half = dim // 2
    itemsize = _numpy_dtype(case.dtype).itemsize
    bytes_moved = (6 * batch * seq * heads * dim + 4 * seq * half) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _logits_softcap_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype, scale=2.0),)


def _logits_softcap_ref(logits: mx.array, softcap: float = 15.0) -> mx.array:
    x32 = logits.astype(mx.float32)
    return (softcap * mx.tanh(x32 / softcap)).astype(logits.dtype)


def _logits_softcap_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((2 * rows * dim) * itemsize, latency_ms)


def _activation_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype, scale=0.2),)


def _activation_ref(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    return mx.square(mx.maximum(x32, 0)).astype(x.dtype)


def _activation_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((2 * rows * dim) * itemsize, latency_ms)


def _value_embed_gate_inputs(case: LabCase):
    batch, seq, kv_heads, head_dim = case.shape
    gate_channels = case.aux["gate_channels"] if case.aux else 32
    x_gate = _mx_array((batch, seq, gate_channels), case.dtype)
    gate_weight = _mx_array((gate_channels, kv_heads), case.dtype)
    v = _mx_array((batch, seq, kv_heads, head_dim), case.dtype)
    ve = _mx_array((batch, seq, kv_heads, head_dim), case.dtype)
    return x_gate, gate_weight, v, ve


def _value_embed_gate_ref(x_gate: mx.array, gate_weight: mx.array, v: mx.array, ve: mx.array) -> mx.array:
    gate = 2.0 * mx.sigmoid(x_gate @ gate_weight)
    return v + gate[..., None] * ve


def _value_embed_gate_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, kv_heads, head_dim = case.shape
    gate_channels = case.aux["gate_channels"] if case.aux else 32
    itemsize = _numpy_dtype(case.dtype).itemsize
    bytes_moved = (
        batch * seq * gate_channels
        + gate_channels * kv_heads
        + 3 * batch * seq * kv_heads * head_dim
        + batch * seq * kv_heads
    ) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _ve_lookup_reshape_inputs(case: LabCase):
    batch, seq, kv_heads, head_dim = case.shape
    vocab_size = case.aux["vocab_size"] if case.aux else 32768
    table = _mx_array((vocab_size, kv_heads * head_dim), case.dtype)
    idx = _mx_int_array((batch, seq), low=0, high=vocab_size)
    return table, idx, kv_heads, head_dim


def _ve_lookup_reshape_ref(value_embed_table: mx.array, idx: mx.array, n_kv_head: int, head_dim: int) -> mx.array:
    batch_size, seq_len = idx.shape
    ve = value_embed_table[idx]
    return ve.reshape(batch_size, seq_len, n_kv_head, head_dim)


def _ve_lookup_reshape_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, kv_heads, head_dim = case.shape
    vocab_size = case.aux["vocab_size"] if case.aux else 32768
    itemsize = _numpy_dtype(case.dtype).itemsize
    index_itemsize = np.dtype(np.int32).itemsize
    bytes_moved = (
        vocab_size * kv_heads * head_dim * itemsize
        + batch * seq * index_itemsize
        + batch * seq * kv_heads * head_dim * itemsize
    )
    return _throughput_gb_s(bytes_moved, latency_ms)


def _attention_mask_local_inputs(case: LabCase):
    seq_len, _ = case.shape
    window_size = case.aux["window_size"] if case.aux else seq_len // 2
    rows = mx.arange(seq_len, dtype=mx.int32)[:, None]
    cols = mx.arange(seq_len, dtype=mx.int32)[None, :]
    return rows, cols, window_size


def _attention_mask_local_ref(rows: mx.array, cols: mx.array, window_size: int) -> mx.array:
    causal = cols <= rows
    local = cols >= (rows - window_size + 1)
    allowed = causal & local
    mask = (~allowed).astype(mx.float32) * mx.finfo(mx.float32).min
    return mask[None, None, :, :]


def _attention_mask_local_metric(case: LabCase, latency_ms: float) -> float:
    seq_len, _ = case.shape
    itemsize = np.dtype(np.float32).itemsize
    bytes_moved = (3 * seq_len * seq_len + 2 * seq_len) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _proj_head_reshape_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype),)


def _proj_head_reshape_ref(y: mx.array) -> mx.array:
    batch_size, heads, seq_len, head_dim = y.shape
    return y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, heads * head_dim)


def _proj_head_reshape_metric(case: LabCase, latency_ms: float) -> float:
    batch, heads, seq, head_dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    bytes_moved = 2 * batch * heads * seq * head_dim * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _loss_logits_cast_softcap_inputs(case: LabCase):
    return (_mx_array(case.shape, case.dtype, scale=2.0),)


def _loss_logits_cast_softcap_ref(logits_bf16: mx.array, softcap: float = 15.0) -> mx.array:
    logits = logits_bf16.astype(mx.float32)
    return softcap * mx.tanh(logits / softcap)


def _loss_logits_cast_softcap_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    in_itemsize = _numpy_dtype(case.dtype).itemsize
    out_itemsize = np.dtype(np.float32).itemsize
    return _throughput_gb_s((rows * dim * (in_itemsize + out_itemsize)) * 2, latency_ms)


def _cross_entropy_prelude_inputs(case: LabCase):
    n = case.shape[0]
    vocab_size = case.aux["vocab_size"] if case.aux else 32768
    loss_flat = _mx_array((n,), case.dtype, scale=1.0)
    target_ids = _mx_int_array((n,), low=0, high=vocab_size)
    token_bytes_np = _rng().integers(0, 5, size=(vocab_size,), dtype=np.int32)
    token_bytes_np[0] = 0
    token_bytes = mx.array(token_bytes_np, dtype=mx.int32)
    return loss_flat, target_ids, token_bytes


def _cross_entropy_prelude_ref(
    loss_flat: mx.array,
    target_ids: mx.array,
    token_bytes: mx.array,
) -> tuple[mx.array, mx.array]:
    nbytes = token_bytes[target_ids]
    mask = nbytes > 0
    return mx.sum(loss_flat * mask.astype(loss_flat.dtype)), mx.sum(nbytes.astype(mx.int64))


def _cross_entropy_prelude_metric(case: LabCase, latency_ms: float) -> float:
    n = case.shape[0]
    vocab_size = case.aux["vocab_size"] if case.aux else 32768
    float_itemsize = _numpy_dtype(case.dtype).itemsize
    int_itemsize = np.dtype(np.int32).itemsize
    bytes_moved = n * float_itemsize + n * int_itemsize + vocab_size * int_itemsize + n * int_itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _cross_entropy_full_inputs(case: LabCase):
    rows, vocab_size = case.shape
    logits = _mx_array((rows, vocab_size), case.dtype, scale=2.0)
    target_ids = _mx_int_array((rows,), low=0, high=vocab_size)
    token_bytes_np = _rng().integers(0, 5, size=(vocab_size,), dtype=np.int32)
    token_bytes_np[0] = 0
    token_bytes = mx.array(token_bytes_np, dtype=mx.int32)
    return logits, target_ids, token_bytes


def _cross_entropy_full_ref(
    logits_bf16: mx.array,
    target_ids: mx.array,
    token_bytes: mx.array,
    softcap: float = 15.0,
) -> tuple[mx.array, mx.array]:
    logits = _loss_logits_cast_softcap_ref(logits_bf16, softcap=softcap)
    losses = nn.losses.cross_entropy(logits, target_ids, reduction="none")
    nbytes = token_bytes[target_ids]
    mask = nbytes > 0
    return mx.sum(losses * mask.astype(losses.dtype)), mx.sum(nbytes.astype(mx.int64))


def _cross_entropy_full_metric(case: LabCase, latency_ms: float) -> float:
    rows, vocab_size = case.shape
    in_itemsize = _numpy_dtype(case.dtype).itemsize
    out_itemsize = np.dtype(np.float32).itemsize
    int_itemsize = np.dtype(np.int32).itemsize
    bytes_moved = (
        rows * vocab_size * (in_itemsize + out_itemsize)
        + rows * int_itemsize
        + vocab_size * int_itemsize
        + rows * out_itemsize
    )
    return _throughput_gb_s(bytes_moved, latency_ms)


def _attention_prelude_inputs(case: LabCase):
    batch, seq, heads, dim = case.shape
    q = _mx_array((batch, seq, heads, dim), case.dtype)
    k = _mx_array((batch, seq, heads, dim), case.dtype)
    v = _mx_array((batch, seq, heads, dim), case.dtype)
    half = dim // 2
    cos = _mx_array((1, seq, 1, half), case.dtype)
    sin = _mx_array((1, seq, 1, half), case.dtype)
    return q, k, v, cos, sin


def _attention_prelude_ref(
    q_proj: mx.array,
    k_proj: mx.array,
    v_proj: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    q, k = _rope_qk_ref(q_proj, k_proj, cos, sin)
    return (
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v_proj.transpose(0, 2, 1, 3),
    )


def _attention_prelude_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, heads, dim = case.shape
    half = dim // 2
    itemsize = _numpy_dtype(case.dtype).itemsize
    bytes_moved = (8 * batch * seq * heads * dim + 4 * seq * half) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _block_prelude_inputs(case: LabCase):
    batch, seq, heads, dim = case.shape
    model_dim = heads * dim
    x = _mx_array((batch, seq, model_dim), case.dtype)
    x0 = _mx_array((batch, seq, model_dim), case.dtype)
    q = _mx_array((batch, seq, heads, dim), case.dtype)
    k = _mx_array((batch, seq, heads, dim), case.dtype)
    v = _mx_array((batch, seq, heads, dim), case.dtype)
    half = dim // 2
    cos = _mx_array((1, seq, 1, half), case.dtype)
    sin = _mx_array((1, seq, 1, half), case.dtype)
    return x, x0, q, k, v, cos, sin, 1.0, 0.1


def _block_prelude_ref(
    x: mx.array,
    x0: mx.array,
    q_proj: mx.array,
    k_proj: mx.array,
    v_proj: mx.array,
    cos: mx.array,
    sin: mx.array,
    resid_lambda: float,
    x0_lambda: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    x_norm = _residual_rmsnorm_ref(x, x0, resid_lambda, x0_lambda)
    q, k, v = _attention_prelude_ref(q_proj, k_proj, v_proj, cos, sin)
    return x_norm, q, k, v


def _block_prelude_metric(case: LabCase, latency_ms: float) -> float:
    batch, seq, heads, dim = case.shape
    half = dim // 2
    model_dim = heads * dim
    itemsize = _numpy_dtype(case.dtype).itemsize
    bytes_moved = (
        4 * batch * seq * model_dim
        + 8 * batch * seq * heads * dim
        + 4 * seq * half
    ) * itemsize
    return _throughput_gb_s(bytes_moved, latency_ms)


def _rmsnorm_backward_inputs(case: LabCase):
    rows, dim = case.shape
    return _mx_array((rows, dim), case.dtype), _mx_array((dim,), case.dtype), _mx_array((rows, dim), case.dtype)


def _rmsnorm_backward_ref(
    x: mx.array,
    weight: mx.array,
    grad_out: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    g32 = grad_out.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    weighted_grad = g32 * w32
    dot = mx.mean(weighted_grad * x32, axis=-1, keepdims=True)
    dx = rms * weighted_grad - x32 * mx.power(rms, 3) * dot
    dweight = mx.sum(g32 * (x32 * rms), axis=0)
    return dx.astype(x.dtype), dweight.astype(weight.dtype)


def _rmsnorm_backward_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((4 * rows * dim + 2 * dim) * itemsize, latency_ms)


def _layernorm_backward_inputs(case: LabCase):
    rows, dim = case.shape
    return (
        _mx_array((rows, dim), case.dtype),
        _mx_array((dim,), case.dtype),
        _mx_array((dim,), case.dtype),
        _mx_array((rows, dim), case.dtype),
    )


def _layernorm_backward_ref(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    grad_out: mx.array,
    eps: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array]:
    del bias
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    g32 = grad_out.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    centered = x32 - mean
    variance = mx.mean(mx.square(centered), axis=-1, keepdims=True)
    inv = mx.rsqrt(variance + eps)
    normalized = centered * inv
    weighted_grad = g32 * w32
    dim = x32.shape[-1]
    sum_u = mx.sum(weighted_grad, axis=-1, keepdims=True)
    sum_un = mx.sum(weighted_grad * normalized, axis=-1, keepdims=True)
    dx = (inv / dim) * (dim * weighted_grad - sum_u - normalized * sum_un)
    dweight = mx.sum(g32 * normalized, axis=0)
    dbias = mx.sum(g32, axis=0)
    return dx.astype(x.dtype), dweight.astype(weight.dtype), dbias.astype(weight.dtype)


def _layernorm_backward_metric(case: LabCase, latency_ms: float) -> float:
    rows, dim = case.shape
    itemsize = _numpy_dtype(case.dtype).itemsize
    return _throughput_gb_s((5 * rows * dim + 3 * dim) * itemsize, latency_ms)


def _fused_mlp_inputs(case: LabCase):
    rows, n_embd = case.shape
    hidden = 4 * n_embd
    return (
        _mx_array((rows, n_embd), case.dtype, scale=0.1),
        _mx_array((n_embd, hidden), case.dtype, scale=0.02),
        _mx_array((hidden, n_embd), case.dtype, scale=0.02),
    )


def _fused_mlp_ref(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
    hidden = x @ w1
    hidden = mx.square(mx.maximum(hidden, 0))
    return hidden @ w2


def _fused_mlp_metric(case: LabCase, latency_ms: float) -> float:
    rows, n_embd = case.shape
    hidden = 4 * n_embd
    fc1 = 2.0 * rows * n_embd * hidden
    act = 2.0 * rows * hidden
    fc2 = 2.0 * rows * hidden * n_embd
    return _tflops(fc1 + act + fc2, latency_ms)


class MLXKernelLab:
    name = "mlx"
    backend_family = "mlx"
    capabilities = LabCapabilities(
        supports_workspace_init=True,
        supports_fixed_bench=True,
        supports_profile=True,
        supports_extract=True,
        supports_orchestrate=True,
        supports_verify=True,
        supports_capture=True,
    )

    _SPECS: dict[str, TargetSpec] = {
        "rmsnorm": TargetSpec(
            info=LabTarget(
                key="rmsnorm",
                description="RMSNorm forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Good first MLX target: small, common, and replaceable with a custom Metal kernel.",
            ),
            template=RMSNORM_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((128, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((2048, 4096), "float16"),
                LabCase((512, 1024), "float32"),
            ),
            make_inputs=_rmsnorm_inputs,
            reference=_rmsnorm_ref,
            metric_value=_rmsnorm_metric,
        ),
        "layernorm": TargetSpec(
            info=LabTarget(
                key="layernorm",
                description="LayerNorm forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Natural second target after RMSNorm.",
            ),
            template=LAYERNORM_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((128, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((512, 1024), "float32"),
            ),
            make_inputs=_layernorm_inputs,
            reference=_layernorm_ref,
            metric_value=_layernorm_metric,
        ),
        "rmsnorm_backward": TargetSpec(
            info=LabTarget(
                key="rmsnorm_backward",
                description="RMSNorm backward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="First backward starter target for training-path kernel work.",
            ),
            template=RMSNORM_BACKWARD_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((128, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((512, 1024), "float32"),
            ),
            make_inputs=_rmsnorm_backward_inputs,
            reference=_rmsnorm_backward_ref,
            metric_value=_rmsnorm_backward_metric,
        ),
        "layernorm_backward": TargetSpec(
            info=LabTarget(
                key="layernorm_backward",
                description="LayerNorm backward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Companion backward starter target after RMSNorm backward.",
            ),
            template=LAYERNORM_BACKWARD_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((128, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((512, 1024), "float32"),
            ),
            make_inputs=_layernorm_backward_inputs,
            reference=_layernorm_backward_ref,
            metric_value=_layernorm_backward_metric,
        ),
        "rotary_embedding": TargetSpec(
            info=LabTarget(
                key="rotary_embedding",
                description="RoPE forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Good fit for a standalone MLX kernel harness.",
            ),
            template=ROTARY_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 256, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float32"),
            ),
            make_inputs=_rotary_inputs,
            reference=_rotary_ref,
            metric_value=_rotary_metric,
        ),
        "reduce": TargetSpec(
            info=LabTarget(
                key="reduce",
                description="Reduction kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Useful low-level primitive once the harness is stable.",
            ),
            template=REDUCE_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 1024), "float16"),
                LabCase((2048, 2048), "float16"),
            ),
            full_cases=(
                LabCase((256, 512), "float16"),
                LabCase((1024, 1024), "float16"),
                LabCase((2048, 2048), "float16"),
                LabCase((1024, 1024), "float32"),
            ),
            make_inputs=_reduce_inputs,
            reference=_reduce_ref,
            metric_value=_reduce_metric,
        ),
        "softmax": TargetSpec(
            info=LabTarget(
                key="softmax",
                description="Softmax forward kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Numerically sensitive but common enough to justify a dedicated lab.",
            ),
            template=SOFTMAX_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
            ),
            full_cases=(
                LabCase((256, 512), "float16"),
                LabCase((1024, 1024), "float16"),
                LabCase((1024, 2048), "float16"),
                LabCase((1024, 1024), "float32"),
            ),
            make_inputs=_softmax_inputs,
            reference=_softmax_ref,
            metric_value=_softmax_metric,
        ),
        "residual_blend": TargetSpec(
            info=LabTarget(
                key="residual_blend",
                description="Residual blend kernel lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the repo's per-layer resid/x0 scaling blend.",
            ),
            template=RESIDUAL_BLEND_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 768), "float16"),
                LabCase((2048, 768), "float16"),
            ),
            full_cases=(
                LabCase((512, 384), "float16"),
                LabCase((1024, 768), "float16"),
                LabCase((2048, 768), "float16"),
                LabCase((1024, 768), "float32"),
            ),
            make_inputs=_residual_blend_inputs,
            reference=_residual_blend_ref,
            metric_value=_residual_blend_metric,
        ),
        "residual_rmsnorm": TargetSpec(
            info=LabTarget(
                key="residual_rmsnorm",
                description="Residual blend + RMSNorm lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="A better training-path target than another standalone norm.",
            ),
            template=RESIDUAL_RMSNORM_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((512, 768), "float16"),
                LabCase((1024, 768), "float16"),
            ),
            full_cases=(
                LabCase((256, 384), "float16"),
                LabCase((512, 768), "float16"),
                LabCase((1024, 768), "float16"),
                LabCase((512, 768), "float32"),
            ),
            make_inputs=_residual_rmsnorm_inputs,
            reference=_residual_rmsnorm_ref,
            metric_value=_residual_rmsnorm_metric,
        ),
        "qk_rmsnorm": TargetSpec(
            info=LabTarget(
                key="qk_rmsnorm",
                description="Q/K RMSNorm lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the post-RoPE normalization path in attention.",
            ),
            template=QK_RMSNORM_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 256, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float32"),
            ),
            make_inputs=_qk_rmsnorm_inputs,
            reference=_qk_rmsnorm_ref,
            metric_value=_qk_rmsnorm_metric,
        ),
        "rope_qk_fused": TargetSpec(
            info=LabTarget(
                key="rope_qk_fused",
                description="RoPE + Q/K RMSNorm lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Useful medium-step before any attention-core kernel work.",
            ),
            template=ROPE_QK_FUSED_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 256, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float32"),
            ),
            make_inputs=_rope_qk_inputs,
            reference=_rope_qk_ref,
            metric_value=_rope_qk_metric,
        ),
        "logits_softcap": TargetSpec(
            info=LabTarget(
                key="logits_softcap",
                description="Logits softcap lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the final tanh-based logits clamp in the model head.",
            ),
            template=LOGITS_SOFTCAP_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 32768), "float16"),
                LabCase((2048, 32768), "float16"),
            ),
            full_cases=(
                LabCase((256, 8192), "float16"),
                LabCase((1024, 32768), "float16"),
                LabCase((2048, 32768), "float16"),
                LabCase((1024, 32768), "float32"),
            ),
            make_inputs=_logits_softcap_inputs,
            reference=_logits_softcap_ref,
            metric_value=_logits_softcap_metric,
        ),
        "activation_pointwise": TargetSpec(
            info=LabTarget(
                key="activation_pointwise",
                description="Squared-ReLU activation lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Useful family target if the MLP changes, including SwiGLU experiments.",
            ),
            template=ACTIVATION_POINTWISE_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((1024, 3072), "float16"),
                LabCase((2048, 4096), "float16"),
            ),
            full_cases=(
                LabCase((256, 1536), "float16"),
                LabCase((1024, 3072), "float16"),
                LabCase((2048, 4096), "float16"),
                LabCase((1024, 3072), "float32"),
            ),
            make_inputs=_activation_inputs,
            reference=_activation_ref,
            metric_value=_activation_metric,
        ),
        "value_embed_gate": TargetSpec(
            info=LabTarget(
                key="value_embed_gate",
                description="Value embed gate lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the value-embed gate path inside attention.",
            ),
            template=VALUE_EMBED_GATE_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 4, 64), "float16", {"gate_channels": 32}),
                LabCase((4, 1024, 4, 64), "float16", {"gate_channels": 64}),
            ),
            full_cases=(
                LabCase((1, 256, 4, 64), "float16", {"gate_channels": 32}),
                LabCase((2, 512, 4, 64), "float16", {"gate_channels": 32}),
                LabCase((4, 1024, 4, 64), "float16", {"gate_channels": 64}),
                LabCase((2, 512, 4, 64), "float32", {"gate_channels": 32}),
            ),
            make_inputs=_value_embed_gate_inputs,
            reference=_value_embed_gate_ref,
            metric_value=_value_embed_gate_metric,
        ),
        "ve_lookup_reshape": TargetSpec(
            info=LabTarget(
                key="ve_lookup_reshape",
                description="Value-embed lookup + reshape lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the embedding lookup and reshape path before value-embed gating.",
            ),
            template=VE_LOOKUP_RESHAPE_TEMPLATE,
            tolerance=0.0,
            quick_cases=(
                LabCase((2, 512, 4, 128), "float16", {"vocab_size": 8192}),
                LabCase((4, 1024, 4, 128), "float16", {"vocab_size": 8192}),
            ),
            full_cases=(
                LabCase((1, 256, 4, 128), "float16", {"vocab_size": 8192}),
                LabCase((2, 512, 4, 128), "float16", {"vocab_size": 8192}),
                LabCase((4, 1024, 4, 128), "float16", {"vocab_size": 8192}),
                LabCase((2, 512, 4, 128), "float32", {"vocab_size": 8192}),
            ),
            make_inputs=_ve_lookup_reshape_inputs,
            reference=_ve_lookup_reshape_ref,
            metric_value=_ve_lookup_reshape_metric,
        ),
        "attention_mask_local": TargetSpec(
            info=LabTarget(
                key="attention_mask_local",
                description="Local attention mask lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the local-causal mask construction used for windowed attention.",
            ),
            template=ATTENTION_MASK_LOCAL_TEMPLATE,
            tolerance=0.0,
            quick_cases=(
                LabCase((512, 512), "float32", {"window_size": 256}),
                LabCase((1024, 1024), "float32", {"window_size": 512}),
            ),
            full_cases=(
                LabCase((256, 256), "float32", {"window_size": 128}),
                LabCase((512, 512), "float32", {"window_size": 256}),
                LabCase((1024, 1024), "float32", {"window_size": 512}),
            ),
            make_inputs=_attention_mask_local_inputs,
            reference=_attention_mask_local_ref,
            metric_value=_attention_mask_local_metric,
        ),
        "proj_head_reshape": TargetSpec(
            info=LabTarget(
                key="proj_head_reshape",
                description="Attention output transpose + reshape lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the attention output staging just before the output projection.",
            ),
            template=PROJ_HEAD_RESHAPE_TEMPLATE,
            tolerance=0.0,
            quick_cases=(
                LabCase((2, 8, 512, 64), "float16"),
                LabCase((4, 8, 1024, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 8, 256, 64), "float16"),
                LabCase((2, 8, 512, 64), "float16"),
                LabCase((4, 8, 1024, 64), "float16"),
                LabCase((2, 8, 512, 64), "float32"),
            ),
            make_inputs=_proj_head_reshape_inputs,
            reference=_proj_head_reshape_ref,
            metric_value=_proj_head_reshape_metric,
        ),
        "loss_logits_cast_softcap": TargetSpec(
            info=LabTarget(
                key="loss_logits_cast_softcap",
                description="Final logits cast + softcap lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the float32 cast plus tanh softcap before cross-entropy.",
            ),
            template=CROSS_ENTROPY_SOFTCAP_TEMPLATE,
            tolerance=0.0,
            quick_cases=(
                LabCase((1024, 32768), "float16"),
                LabCase((2048, 32768), "float16"),
            ),
            full_cases=(
                LabCase((256, 8192), "float16"),
                LabCase((1024, 32768), "float16"),
                LabCase((2048, 32768), "float16"),
                LabCase((1024, 32768), "float32"),
            ),
            make_inputs=_loss_logits_cast_softcap_inputs,
            reference=_loss_logits_cast_softcap_ref,
            metric_value=_loss_logits_cast_softcap_metric,
        ),
        "cross_entropy_prelude": TargetSpec(
            info=LabTarget(
                key="cross_entropy_prelude",
                description="Cross-entropy prelude lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the byte-aware masked reduction around flattened loss output.",
            ),
            template=CROSS_ENTROPY_PRELUDE_TEMPLATE,
            tolerance=0.0,
            quick_cases=(
                LabCase((32768,), "float32", {"vocab_size": 8192}),
                LabCase((65536,), "float32", {"vocab_size": 8192}),
            ),
            full_cases=(
                LabCase((16384,), "float32", {"vocab_size": 8192}),
                LabCase((32768,), "float32", {"vocab_size": 8192}),
                LabCase((65536,), "float32", {"vocab_size": 8192}),
            ),
            make_inputs=_cross_entropy_prelude_inputs,
            reference=_cross_entropy_prelude_ref,
            metric_value=_cross_entropy_prelude_metric,
        ),
        "cross_entropy_full": TargetSpec(
            info=LabTarget(
                key="cross_entropy_full",
                description="Full cross-entropy loss path lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Matches the final logits cast + softcap + cross-entropy + byte-aware masked reduction.",
            ),
            template=CROSS_ENTROPY_FULL_TEMPLATE,
            tolerance=5e-4,
            quick_cases=(
                LabCase((1024, 8192), "float16"),
                LabCase((2048, 8192), "float16"),
            ),
            full_cases=(
                LabCase((512, 8192), "float16"),
                LabCase((1024, 8192), "float16"),
                LabCase((2048, 8192), "float16"),
                LabCase((1024, 8192), "float32"),
            ),
            make_inputs=_cross_entropy_full_inputs,
            reference=_cross_entropy_full_ref,
            metric_value=_cross_entropy_full_metric,
        ),
        "attention_prelude": TargetSpec(
            info=LabTarget(
                key="attention_prelude",
                description="Attention prelude lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Covers rotary, Q/K norm, and transpose staging up to SDPA.",
            ),
            template=ATTENTION_PRELUDE_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 256, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float32"),
            ),
            make_inputs=_attention_prelude_inputs,
            reference=_attention_prelude_ref,
            metric_value=_attention_prelude_metric,
        ),
        "block_prelude": TargetSpec(
            info=LabTarget(
                key="block_prelude",
                description="Residual + attention setup prelude lab",
                metric="throughput_gb_s",
                status="starter-ready",
                notes="Covers residual blend, RMSNorm, and attention staging up to SDPA, but stops before attention proper.",
            ),
            template=BLOCK_PRELUDE_TEMPLATE,
            tolerance=5e-3,
            quick_cases=(
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
            ),
            full_cases=(
                LabCase((1, 256, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float16"),
                LabCase((4, 1024, 8, 64), "float16"),
                LabCase((2, 512, 8, 64), "float32"),
            ),
            make_inputs=_block_prelude_inputs,
            reference=_block_prelude_ref,
            metric_value=_block_prelude_metric,
        ),
        "fused_mlp": TargetSpec(
            info=LabTarget(
                key="fused_mlp",
                description="Fused MLP lab",
                metric="throughput_tflops",
                status="starter-ready",
                notes="Higher-value target once the norm and elementwise labs are in place.",
            ),
            template=FUSED_MLP_TEMPLATE,
            tolerance=5e-2,
            quick_cases=(
                LabCase((256, 512), "float16"),
                LabCase((512, 1024), "float16"),
            ),
            full_cases=(
                LabCase((128, 256), "float16"),
                LabCase((256, 512), "float16"),
                LabCase((512, 1024), "float16"),
                LabCase((256, 512), "float32"),
            ),
            make_inputs=_fused_mlp_inputs,
            reference=_fused_mlp_ref,
            metric_value=_fused_mlp_metric,
        ),
        "flash_attention": TargetSpec(
            info=LabTarget(
                key="flash_attention",
                description="Attention kernel lab",
                metric="throughput_tflops",
                status="deferred",
                notes="Do not start here; MLX already has optimized attention primitives and custom backward is a worse first target.",
            ),
            template="",
            tolerance=0.0,
            quick_cases=(),
            full_cases=(),
            make_inputs=lambda case: (),
            reference=lambda: mx.array(0),
            metric_value=lambda case, latency_ms: 0.0,
        ),
    }

    def target_catalog(self) -> dict[str, LabTarget]:
        return {key: spec.info for key, spec in self._SPECS.items()}

    def init_workspace(self, *, target: str, workspace: Path) -> Path:
        spec = self._get_spec(target)
        if spec.info.status != "starter-ready":
            raise ValueError(
                f"Target {target} is {spec.info.status}, not starter-ready. "
                "Use `kernel-lab.py --engine mlx list-targets` to inspect the current catalog."
            )

        workspace.mkdir(parents=True, exist_ok=True)
        metadata = {
            "engine": self.name,
            "target": target,
            "metric": spec.info.metric,
            "status": spec.info.status,
            "template_version": 1,
        }
        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (workspace / "kernel.py").write_text(spec.template, encoding="utf-8")
        (workspace / "README.md").write_text(
            "\n".join(
                [
                    "# MLX Kernel Lab Workspace",
                    "",
                    f"Target: `{target}`",
                    "",
                    "- Edit `kernel.py` only.",
                    "- Benchmark with `uv run kernel-lab.py --engine mlx bench --workspace <this-dir>`.",
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
        spec = self._get_spec(target)
        kernel_mod = self._load_kernel_module(workspace / "kernel.py")
        if not hasattr(kernel_mod, "kernel_fn"):
            raise ValueError("Workspace kernel.py must define kernel_fn")

        cases = spec.quick_cases if quick else spec.full_cases
        start = time.perf_counter()
        max_abs_error = 0.0
        worst_case = None
        metric_values: list[float] = []
        latencies_ms: list[float] = []

        for case in cases:
            inputs = spec.make_inputs(case)
            ref = spec.reference(*inputs)
            out = kernel_mod.kernel_fn(*inputs)
            ref_leaves = tuple(_iter_leaves(ref))
            out_leaves = tuple(_iter_leaves(out))
            mx.eval(*ref_leaves, *out_leaves)
            if len(ref_leaves) != len(out_leaves):
                abs_error = float("inf")
            else:
                abs_errors: list[float] = []
                for ref_leaf, out_leaf in zip(ref_leaves, out_leaves):
                    ref_np = np.array(ref_leaf)
                    out_np = np.array(out_leaf)
                    if ref_np.shape != out_np.shape or not np.isfinite(ref_np).all() or not np.isfinite(out_np).all():
                        abs_errors = [float("inf")]
                        break
                    abs_errors.append(float(np.max(np.abs(out_np - ref_np))))
                abs_error = max(abs_errors) if abs_errors else 0.0
            max_abs_error = max(max_abs_error, abs_error)
            if worst_case is None or abs_error >= worst_case["max_abs_error"]:
                worst_case = {
                    "shape": list(case.shape),
                    "dtype": case.dtype,
                    "max_abs_error": abs_error,
                }

            latency_ms = self._bench_case(kernel_mod.kernel_fn, inputs)
            latencies_ms.append(latency_ms)
            metric_values.append(spec.metric_value(case, latency_ms))

        wall_seconds = time.perf_counter() - start
        status = "ok" if max_abs_error <= spec.tolerance else "fail"
        details = {
            "cases": len(cases),
            "max_abs_error": max_abs_error,
            "worst_case": worst_case,
            "median_latency_ms": statistics.median(latencies_ms),
            f"median_{spec.info.metric}": statistics.median(metric_values),
        }
        return LabBenchResult(
            target=target,
            status=status,
            metric_name=spec.info.metric,
            metric_value=statistics.median(metric_values),
            wall_seconds=wall_seconds,
            details=details,
        )

    def verify_workspace(self, *, workspace: Path, quick: bool = False) -> LabBenchResult:
        result = self.bench_workspace(workspace=workspace, quick=quick)
        append_lab_event(
            engine="mlx",
            backend_family="mlx",
            target=result.target,
            workspace=workspace,
            event_type="verify",
            status=result.status,
            metric_name=result.metric_name,
            metric_value=result.metric_value,
            details={
                "quick": quick,
                "wall_seconds": result.wall_seconds,
                "max_abs_error": result.details.get("max_abs_error"),
                "median_latency_ms": result.details.get("median_latency_ms"),
                "cases": result.details.get("cases"),
            },
        )
        return result

    def profile_targets(self, *, preset: str, top_k: int = 10) -> LabProfileResult:
        return profile_mlx_targets(target_catalog=self.target_catalog(), preset=preset, top_k=top_k)

    def extract_from_profile(self, *, profile_path: Path, workspace: Path, rank: int = 1) -> LabExtractResult:
        return extract_from_profile(
            init_workspace=self.init_workspace,
            profile_path=profile_path,
            workspace=workspace,
            rank=rank,
        )

    def orchestrate_from_profile(
        self,
        *,
        profile_path: Path,
        workspace_root: Path,
        rank: int = 1,
        trace_metadata_path: Path | None = None,
    ) -> LabOrchestrationPlan:
        return orchestrate_from_profile(
            profile_path=profile_path,
            workspace_root=workspace_root,
            rank=rank,
            trace_metadata_path=trace_metadata_path,
        )

    def capture_workspace(self, *, workspace: Path, output: Path, quick: bool = False) -> LabTraceResult:
        return capture_workspace_trace(workspace=workspace, output=output, quick=quick)

    @staticmethod
    def _resolve_runtime_preset(preset: str | None) -> tuple[str, str, dict | None]:
        if preset is not None:
            return preset, "manual", None
        hardware_key = detect_current_hardware_key()
        cached = load_platform_default_cache(engine_name="mlx", hardware_key=hardware_key)
        if cached is None:
            raise ValueError(
                "No calibrated MLX platform default was found for this device. "
                "Run `uv run calibrate.py --engine mlx --mode fast` first, or pass --preset explicitly."
            )
        candidate = cached.candidate_default
        if candidate.get("eval_semantics_signature") != current_eval_semantics_signature() or candidate.get(
            "runtime_shape_signature"
        ) != current_runtime_shape_signature():
            raise ValueError(
                "The cached MLX platform default for this device no longer matches the current code signatures. "
                "Rerun `uv run calibrate.py --engine mlx --mode fast`, or pass --preset explicitly."
            )
        resolved = candidate.get("preset")
        if not isinstance(resolved, str) or not resolved:
            raise ValueError(
                "The cached MLX platform default for this device is malformed. "
                "Rerun `uv run calibrate.py --engine mlx --mode fast`."
            )
        return resolved, "calibrated-platform-default", {
            "hardware_key": hardware_key,
            "generated_at": cached.generated_at,
            "source_output_dir": cached.source_output_dir,
            "source_report": cached.source_report,
            "candidate_default": candidate,
        }

    def summarize_evidence(self, *, target: str, preset: str | None = None) -> LabEvidenceResult:
        summary = summarize_lab_evidence(
            engine="mlx",
            backend_family="mlx",
            target=target,
            preset=preset,
        )
        return LabEvidenceResult(
            engine="mlx",
            backend_family="mlx",
            target=target,
            preset=preset,
            status=summary.promotion_status,
            details=asdict(summary),
        )

    def promotion_check(self, *, target: str, preset: str | None = None, workspace: Path | None = None) -> LabPromotionCheck:
        resolved_preset, preset_source, preset_context = self._resolve_runtime_preset(preset)
        summary = summarize_lab_evidence(
            engine="mlx",
            backend_family="mlx",
            target=target,
            preset=resolved_preset,
        )
        chosen_workspace = workspace.expanduser() if workspace is not None else None
        if chosen_workspace is None and summary.last_workspace is not None:
            candidate_workspace = Path(summary.last_workspace).expanduser()
            if candidate_workspace.exists():
                chosen_workspace = candidate_workspace

        if not supports_direct_integration(target):
            status = "needs-integration-adapter"
            commands = (
                f"# {target} is trace-backed at the lab level, but it still needs a direct training-path adapter before end-to-end A/B is meaningful",
            )
        elif summary.promotion_status == "integration-validated":
            status = "integration-validated"
            commands = (
                "# this target already has a successful end-to-end integration A/B result",
                "# review the integration logs and, if the win is meaningful, move to a real trainer integration patch",
            )
        elif summary.promotion_status == "integration-tested":
            status = "integration-tested"
            commands = (
                "# this target has completed an end-to-end integration A/B run",
                "# inspect the measured delta and decide whether to promote, refine, or rerun on a stronger preset",
            )
        elif summary.promotion_status == "integration-mixed":
            status = "integration-mixed"
            commands = (
                "# this target has mixed end-to-end integration A/B results",
                "# rerun on a stronger preset or a longer budget before promoting it into the trainer",
            )
        elif summary.promotion_status == "integration-regressed":
            status = "integration-regressed"
            commands = (
                "# this target regressed in end-to-end integration A/B",
                "# inspect the integration logs before spending more lab time on it",
            )
        elif summary.promotion_status == "ready-for-integration-test" and chosen_workspace is not None:
            status = "ready-for-integration-ab"
            cmd = f"uv run kernel-lab.py --engine mlx integration-ab --workspace {chosen_workspace} --time-budget 20 --benchmark-skip-eval --no-checkpoint"
            if preset_source == "manual":
                cmd += f" --preset {resolved_preset}"
            commands = (cmd,)
        elif summary.promotion_status == "trace-deprioritized":
            status = "deprioritized-after-trace"
            commands = (
                "# keep the workspace for reference, but prioritize a different target before integration work",
            )
        else:
            status = "not-ready"
            commands = (
                "# gather both verify and capture evidence before attempting an end-to-end integration A/B",
            )

        return LabPromotionCheck(
            engine="mlx",
            backend_family="mlx",
            target=target,
            preset=resolved_preset,
            workspace=str(chosen_workspace) if chosen_workspace is not None else None,
            status=status,
            commands=commands,
            details={
                "resolved_preset": resolved_preset,
                "preset_source": preset_source,
                "preset_context": preset_context,
                "evidence_summary": asdict(summary),
                "required_for_promotion": ["verify", "capture"],
            },
        )

    def run_integration_ab(
        self,
        *,
        workspace: Path,
        time_budget: float,
        preset: str | None = None,
        benchmark_skip_eval: bool = True,
        no_checkpoint: bool = True,
    ) -> LabIntegrationABResult:
        resolved_preset, preset_source, preset_context = self._resolve_runtime_preset(preset)
        workspace = workspace.expanduser().resolve()
        metadata = json.loads((workspace / "metadata.json").read_text(encoding="utf-8"))
        target = str(metadata["target"])
        if not supports_direct_integration(target):
            raise ValueError(
                f"Target {target!r} does not yet support direct training-path integration."
            )

        run_root = workspace / "integration-ab" / time.strftime("%Y%m%d-%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        warmup_budget = min(2.0, time_budget)

        def build_cmd(run_time_budget: float) -> list[str]:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "train.py"),
                "--engine",
                "mlx",
                "--preset",
                resolved_preset,
                "--time-budget",
                str(run_time_budget),
            ]
            if benchmark_skip_eval:
                cmd.append("--benchmark-skip-eval")
            if no_checkpoint:
                cmd.append("--no-checkpoint")
            return cmd

        warmup_cmd = build_cmd(warmup_budget)
        measured_cmd = build_cmd(time_budget)

        baseline_warmup_stdout = run_root / "baseline-warmup.stdout.log"
        baseline_warmup_stderr = run_root / "baseline-warmup.stderr.log"
        candidate_warmup_stdout = run_root / "candidate-warmup.stdout.log"
        candidate_warmup_stderr = run_root / "candidate-warmup.stderr.log"
        baseline_stdout = run_root / "baseline.stdout.log"
        baseline_stderr = run_root / "baseline.stderr.log"
        candidate_stdout = run_root / "candidate.stdout.log"
        candidate_stderr = run_root / "candidate.stderr.log"

        baseline_env = os.environ.copy()
        candidate_env = os.environ.copy()
        candidate_env.update(integration_environment(workspace=workspace))

        start = time.perf_counter()
        baseline_warmup = self._run_train_command(
            warmup_cmd,
            baseline_warmup_stdout,
            baseline_warmup_stderr,
            env=baseline_env,
        )
        candidate_warmup = self._run_train_command(
            warmup_cmd,
            candidate_warmup_stdout,
            candidate_warmup_stderr,
            env=candidate_env,
        )
        baseline = self._run_train_command(
            measured_cmd,
            baseline_stdout,
            baseline_stderr,
            env=baseline_env,
        )
        candidate = self._run_train_command(
            measured_cmd,
            candidate_stdout,
            candidate_stderr,
            env=candidate_env,
        )
        wall_seconds = time.perf_counter() - start

        baseline_warmup_summary = parse_summary(baseline_warmup.stdout) if baseline_warmup.returncode == 0 else {}
        candidate_warmup_summary = parse_summary(candidate_warmup.stdout) if candidate_warmup.returncode == 0 else {}
        baseline_summary = parse_summary(baseline.stdout) if baseline.returncode == 0 else {}
        candidate_summary = parse_summary(candidate.stdout) if candidate.returncode == 0 else {}

        status = (
            "ok"
            if all(
                result.returncode == 0
                for result in (baseline_warmup, candidate_warmup, baseline, candidate)
            )
            else "error"
        )
        details = {
            "warmup_budget": warmup_budget,
            "warmup": {
                "baseline_stdout": str(baseline_warmup_stdout),
                "baseline_stderr": str(baseline_warmup_stderr),
                "candidate_stdout": str(candidate_warmup_stdout),
                "candidate_stderr": str(candidate_warmup_stderr),
                "baseline_returncode": baseline_warmup.returncode,
                "candidate_returncode": candidate_warmup.returncode,
                "baseline": baseline_warmup_summary,
                "candidate": candidate_warmup_summary,
            },
            "baseline_stdout": str(baseline_stdout),
            "baseline_stderr": str(baseline_stderr),
            "candidate_stdout": str(candidate_stdout),
            "candidate_stderr": str(candidate_stderr),
            "baseline_returncode": baseline.returncode,
            "candidate_returncode": candidate.returncode,
            "benchmark_skip_eval": benchmark_skip_eval,
            "no_checkpoint": no_checkpoint,
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "delta": self._summary_delta(baseline_summary, candidate_summary),
        }

        append_lab_event(
            engine="mlx",
            backend_family="mlx",
            target=target,
            workspace=workspace,
            event_type="integration-ab",
            status=status,
            preset=resolved_preset,
            metric_name="steady_state_tok_per_sec_delta",
            metric_value=details["delta"].get("steady_state_tok_per_sec"),
            details={
                "preset": resolved_preset,
                "preset_source": preset_source,
                "preset_context": preset_context,
                "time_budget": time_budget,
                "warmup_budget": warmup_budget,
                "benchmark_skip_eval": benchmark_skip_eval,
                "no_checkpoint": no_checkpoint,
                "baseline_stdout": str(baseline_stdout),
                "candidate_stdout": str(candidate_stdout),
                "baseline_returncode": baseline.returncode,
                "candidate_returncode": candidate.returncode,
                "warmup_baseline_returncode": baseline_warmup.returncode,
                "warmup_candidate_returncode": candidate_warmup.returncode,
                "delta": details["delta"],
            },
        )

        return LabIntegrationABResult(
            engine="mlx",
            backend_family="mlx",
            target=target,
            preset=resolved_preset,
            workspace=str(workspace),
            status=status,
            wall_seconds=wall_seconds,
            details={
                "preset_source": preset_source,
                "preset_context": preset_context,
                **details,
            },
        )

    def _get_spec(self, target: str) -> TargetSpec:
        try:
            return self._SPECS[target]
        except KeyError as exc:
            raise ValueError(f"Unknown MLX lab target: {target}") from exc

    def _load_kernel_module(self, kernel_path: Path):
        spec = importlib.util.spec_from_file_location("autoresearch_mlx_lab_kernel", kernel_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load kernel module from {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def _bench_case(self, kernel_fn, inputs: tuple) -> float:
        for _ in range(3):
            out = kernel_fn(*inputs)
            mx.eval(*tuple(_iter_leaves(out)))
        times = []
        for _ in range(10):
            start = time.perf_counter()
            out = kernel_fn(*inputs)
            mx.eval(*tuple(_iter_leaves(out)))
            times.append((time.perf_counter() - start) * 1e3)
        return statistics.median(times)

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
        for key in ("steady_state_tok_per_sec", "peak_vram_mb", "val_bpb", "proxy_val_bpb"):
            baseline = baseline_summary.get(key)
            candidate = candidate_summary.get(key)
            if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
                deltas[key] = float(candidate) - float(baseline)
        return deltas
