import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    scale = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * scale).astype(x.dtype)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def count_params(tree) -> int:
    return sum(value.size for _, value in tree_flatten(tree))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5

        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_kv_head > self.n_head or self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def __call__(self, x: mx.array, ve: mx.array | None, cos_sin, attention_mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2.0 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate[..., None] * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = mx.square(mx.maximum(x, 0))
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, ve: mx.array | None, cos_sin, attention_mask):
        x = x + self.attn(rms_norm(x), ve, cos_sin, attention_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "window_sizes", self._compute_window_sizes(config))
        head_dim = config.n_embd // config.n_head
        object.__setattr__(self, "_rotary_head_dim", head_dim)
        object.__setattr__(self, "rotary_seq_len", min(config.sequence_len, 1024))
        object.__setattr__(self, "_mask_cache", {})

        self.transformer = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": [Block(config, layer_idx) for layer_idx in range(config.n_layer)],
        }
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = [
            nn.Embedding(config.vocab_size, kv_dim) if has_ve(layer_idx, config.n_layer) else None
            for layer_idx in range(config.n_layer)
        ]
        self._set_rotary_cache(self.rotary_seq_len)
        self.freeze(keys=["cos", "sin"], recurse=False, strict=True)

    def init_weights(self) -> None:
        self.transformer["wte"].weight = mx.random.normal(shape=self.transformer["wte"].weight.shape, scale=1.0)
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape, scale=0.001)

        n_embd = self.config.n_embd
        scale = math.sqrt(3.0) * (n_embd ** -0.5)
        for block in self.transformer["h"]:
            block.attn.c_q.weight = mx.random.uniform(low=-scale, high=scale, shape=block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(low=-scale, high=scale, shape=block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(low=-scale, high=scale, shape=block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(low=-scale, high=scale, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

        self.resid_lambdas = mx.ones_like(self.resid_lambdas)
        self.x0_lambdas = mx.full(self.x0_lambdas.shape, 0.1)

        for value_embed in self.value_embeds:
            if value_embed is not None:
                value_embed.weight = mx.random.uniform(low=-scale, high=scale, shape=value_embed.weight.shape)

        self._set_rotary_cache(self.rotary_seq_len)
        self.transformer["wte"].weight = self.transformer["wte"].weight.astype(mx.bfloat16)
        for value_embed in self.value_embeds:
            if value_embed is not None:
                value_embed.weight = value_embed.weight.astype(mx.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: float = 10000.0):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs).astype(mx.bfloat16)[None, :, None, :]
        sin = mx.sin(freqs).astype(mx.bfloat16)[None, :, None, :]
        return cos, sin

    def _set_rotary_cache(self, seq_len: int) -> None:
        cos, sin = self._precompute_rotary_embeddings(seq_len, self._rotary_head_dim)
        object.__setattr__(self, "rotary_seq_len", seq_len)
        self.cos = cos
        self.sin = sin

    def _ensure_rotary_cache(self, seq_len: int) -> None:
        if seq_len > self.config.sequence_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model config.sequence_len {self.config.sequence_len}."
            )
        current_seq_len = object.__getattribute__(self, "rotary_seq_len")
        if seq_len <= current_seq_len:
            return
        next_seq_len = min(self.config.sequence_len, max(seq_len, current_seq_len * 2))
        self._set_rotary_cache(next_seq_len)

    def _compute_window_sizes(self, config: GPTConfig):
        pattern = config.window_pattern.upper()
        if not all(char in "SL" for char in pattern):
            raise ValueError("window_pattern can only contain 'S' and 'L'")
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _get_attention_mask(self, seq_len: int, window_size: int):
        if window_size <= 0 or window_size >= seq_len:
            return "causal"

        cache = object.__getattribute__(self, "_mask_cache")
        mask = cache.get(window_size)
        cached_seq_len = 0 if mask is None else mask.shape[2]
        if cached_seq_len < seq_len:
            next_seq_len = max(seq_len, cached_seq_len * 2) if cached_seq_len else seq_len
            rows = mx.arange(next_seq_len)[:, None]
            cols = mx.arange(next_seq_len)[None, :]
            causal = cols <= rows
            local = cols >= (rows - window_size + 1)
            allowed = causal & local
            mask = (~allowed).astype(mx.float32) * mx.finfo(mx.float32).min
            cache[window_size] = mask[None, None, :, :]
            mask = cache[window_size]
        if mask.shape[2] == seq_len:
            return mask
        return mask[:, :, :seq_len, :seq_len]

    def ensure_runtime_caches(self, seq_len: int) -> None:
        self._ensure_rotary_cache(seq_len)
        for window_size, _ in set(self.window_sizes):
            if 0 < window_size < seq_len:
                self._get_attention_mask(seq_len, window_size)

    def estimate_flops(self) -> int:
        nparams = count_params(self.trainable_parameters())
        value_embeds_numel = count_params([embed.parameters() for embed in self.value_embeds if embed is not None])
        nparams_exclude = (
            count_params(self.transformer["wte"].parameters())
            + value_embeds_numel
            + self.resid_lambdas.size
            + self.x0_lambdas.size
        )
        heads = self.config.n_head
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        attn_flops = 0
        for window_size, _ in self.window_sizes:
            effective_seq = min(window_size, seq_len)
            attn_flops += 12 * heads * head_dim * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self) -> dict[str, int]:
        wte = count_params(self.transformer["wte"].parameters())
        value_embeds = count_params([embed.parameters() for embed in self.value_embeds if embed is not None])
        lm_head = count_params(self.lm_head.parameters())
        transformer_matrices = count_params([block.parameters() for block in self.transformer["h"]])
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def __call__(self, idx: mx.array, targets: mx.array | None = None, reduction: str = "mean"):
        seq_len = idx.shape[1]
        self._ensure_rotary_cache(seq_len)

        cos_sin = (self.cos[:, :seq_len], self.sin[:, :seq_len])
        x = self.transformer["wte"](idx)
        x = rms_norm(x)
        x0 = x

        for layer_idx, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            value_embed = self.value_embeds[layer_idx]
            ve = value_embed(idx) if value_embed is not None else None
            attention_mask = self._get_attention_mask(seq_len, self.window_sizes[layer_idx][0])
            x = block(x, ve, cos_sin, attention_mask)

        x = rms_norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        softcap = 15.0
        logits = softcap * mx.tanh(logits / softcap)

        if targets is None:
            return logits

        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        if reduction == "mean":
            return mx.mean(losses)
        if reduction == "sum":
            return mx.sum(losses)
        if reduction == "none":
            return losses
        raise ValueError(f"Invalid reduction: {reduction}")
