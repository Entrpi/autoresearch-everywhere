from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from autoresearch_platform.lr_profile import ResolvedLrProfile


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@dataclass(frozen=True)
class AdamWGroup:
    name: str
    paths: tuple[str, ...]
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class MuonGroup:
    paths: tuple[str, ...]
    shape: tuple[int, ...]
    red_dim: int
    beta2: float
    ns_steps: int


@dataclass(frozen=True)
class ParamSlot:
    path: str
    tokens: tuple[object, ...]
    safe_key: str
    parent: object
    leaf: object


def _safe_key(path: str) -> str:
    return path.replace(".", "__")


_MISSING = object()


def _parse_path(path: str) -> tuple[object, ...]:
    tokens = []
    for token in path.split("."):
        tokens.append(int(token) if token.isdigit() else token)
    return tuple(tokens)


def _descend(node, token):
    if isinstance(node, dict):
        return node[token]
    if isinstance(node, list):
        return node[token]
    return getattr(node, token)


def _resolve_slot(root, path: str) -> ParamSlot:
    tokens = _parse_path(path)
    parent = root
    for token in tokens[:-1]:
        parent = _descend(parent, token)
    return ParamSlot(path, tokens, _safe_key(path), parent, tokens[-1])


def _get_slot_value(slot: ParamSlot):
    if isinstance(slot.parent, dict):
        return slot.parent[slot.leaf]
    if isinstance(slot.parent, list):
        return slot.parent[slot.leaf]
    return getattr(slot.parent, slot.leaf)


def _set_slot_value(slot: ParamSlot, value) -> None:
    if isinstance(slot.parent, dict):
        slot.parent[slot.leaf] = value
    elif isinstance(slot.parent, list):
        slot.parent[slot.leaf] = value
    else:
        setattr(slot.parent, slot.leaf, value)


def _get_tree_value(root, tokens):
    node = root
    for token in tokens:
        if isinstance(node, dict):
            node = node.get(token, _MISSING)
        elif isinstance(node, list):
            if not isinstance(token, int) or token >= len(node):
                return _MISSING
            node = node[token]
        else:
            try:
                node = getattr(node, token)
            except AttributeError:
                return _MISSING
        if node is _MISSING:
            return _MISSING
    return node


def _adamw_update(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, weight_decay):
    lr = lr.astype(param.dtype)
    beta1 = beta1.astype(param.dtype)
    beta2 = beta2.astype(param.dtype)
    eps = eps.astype(param.dtype)
    weight_decay = weight_decay.astype(param.dtype)
    step = step.astype(param.dtype)

    param = param * (1 - lr * weight_decay)
    exp_avg = exp_avg + (1 - beta1) * (grad - exp_avg)
    exp_avg_sq = exp_avg_sq + (1 - beta2) * (mx.square(grad) - exp_avg_sq)
    bias1 = 1 - mx.power(beta1, step)
    bias2 = 1 - mx.power(beta2, step)
    denom = mx.sqrt(exp_avg_sq / bias2) + eps
    step_size = lr / bias1
    param = param - step_size * (exp_avg / denom)
    return param, exp_avg, exp_avg_sq


def _matrix_norm(x):
    return mx.sqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=(-2, -1), keepdims=True))


def _muon_update(stacked_params, stacked_grads, momentum_buffer, second_momentum_buffer, lr, momentum, weight_decay, beta2, ns_steps, red_dim):
    momentum = momentum.astype(stacked_grads.dtype)
    momentum_buffer = momentum_buffer + (1 - momentum) * (stacked_grads - momentum_buffer)
    g = stacked_grads + momentum * (momentum_buffer - stacked_grads)

    x = g.astype(mx.bfloat16)
    x = x / (_matrix_norm(x) * 1.02 + 1e-6)
    if g.shape[-2] > g.shape[-1]:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_matrix = x.swapaxes(-1, -2) @ x
            b_matrix = b * a_matrix + c * (a_matrix @ a_matrix)
            x = a * x + x @ b_matrix
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_matrix = x @ x.swapaxes(-1, -2)
            b_matrix = b * a_matrix + c * (a_matrix @ a_matrix)
            x = a * x + b_matrix @ x

    g = x
    beta2 = beta2.astype(g.dtype)
    v_mean = mx.mean(mx.square(g.astype(mx.float32)), axis=red_dim, keepdims=True)
    red_dim_size = g.shape[red_dim]
    v_norm_sq = mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
    v_norm = mx.sqrt(v_norm_sq)

    beta2_state = beta2.astype(second_momentum_buffer.dtype)
    second_momentum_buffer = second_momentum_buffer + (1 - beta2_state) * (
        v_mean.astype(second_momentum_buffer.dtype) - second_momentum_buffer
    )
    step_size = mx.rsqrt(mx.maximum(second_momentum_buffer, 1e-10))
    scaled_sq_sum = (v_mean * red_dim_size) * mx.square(step_size.astype(mx.float32))
    v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
    g = g * final_scale.astype(g.dtype)

    lr = lr.astype(g.dtype)
    weight_decay = weight_decay.astype(g.dtype)
    mask = (g * stacked_params) >= 0
    updated = stacked_params - (lr * g + lr * weight_decay * stacked_params * mask.astype(g.dtype))
    return updated, momentum_buffer, second_momentum_buffer


class MuonAdamW:
    def __init__(
        self,
        model,
        *,
        lr_profile: ResolvedLrProfile,
        weight_decay: float = 0.0,
        adam_betas=(0.8, 0.95),
    ):
        self.initial_lrs = {
            "lm_head": lr_profile.lm_head_lr,
            "embedding": lr_profile.embedding_lr,
            "value_embedding": lr_profile.value_embedding_lr,
            "resid": lr_profile.resid_lr,
            "x0": lr_profile.x0_lr,
            "muon": lr_profile.matrix_lr,
        }

        flat_params = dict(tree_flatten(model.trainable_parameters()))
        self.param_slots = {
            path: _resolve_slot(model, path)
            for path in flat_params
        }
        self.adamw_groups, self.muon_groups = self._build_groups(flat_params, adam_betas, weight_decay)
        self.adamw_group_slots = tuple(
            tuple(self.param_slots[path] for path in group.paths)
            for group in self.adamw_groups
        )
        self.muon_group_slots = tuple(
            tuple(self.param_slots[path] for path in group.paths)
            for group in self.muon_groups
        )
        self.state = self._init_state(flat_params)
        self.set_schedule(schedule_factor=1.0, muon_momentum=0.95, muon_weight_decay=weight_decay)

    def _build_groups(self, flat_params, adam_betas, weight_decay):
        all_paths = set(flat_params.keys())
        lm_head_paths = tuple(path for path in flat_params if path.startswith("lm_head."))
        embedding_paths = tuple(path for path in flat_params if path == "transformer.wte.weight")
        value_embedding_paths = tuple(path for path in flat_params if path.startswith("value_embeds."))
        resid_paths = tuple(path for path in flat_params if path == "resid_lambdas")
        x0_paths = tuple(path for path in flat_params if path == "x0_lambdas")
        matrix_paths = tuple(
            path
            for path, value in flat_params.items()
            if path.startswith("transformer.h.") and value.ndim >= 2
        )

        accounted_for = set(lm_head_paths) | set(embedding_paths) | set(value_embedding_paths) | set(resid_paths) | set(x0_paths) | set(matrix_paths)
        unhandled = sorted(all_paths - accounted_for)
        if unhandled:
            raise ValueError(f"Unhandled parameters for optimizer grouping: {unhandled}")

        adamw_groups = (
            AdamWGroup("lm_head", lm_head_paths, adam_betas[0], adam_betas[1], 1e-10, 0.0),
            AdamWGroup("embedding", embedding_paths, adam_betas[0], adam_betas[1], 1e-10, 0.0),
            AdamWGroup("value_embedding", value_embedding_paths, adam_betas[0], adam_betas[1], 1e-10, 0.0),
            AdamWGroup("resid", resid_paths, adam_betas[0], adam_betas[1], 1e-10, 0.0),
            AdamWGroup("x0", x0_paths, 0.96, 0.95, 1e-10, 0.0),
        )

        by_shape = {}
        for path in matrix_paths:
            by_shape.setdefault(tuple(flat_params[path].shape), []).append(path)
        muon_groups = []
        for shape, paths in sorted(by_shape.items()):
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            muon_groups.append(
                MuonGroup(tuple(paths), shape, red_dim, beta2=0.95, ns_steps=5)
            )
        return adamw_groups, tuple(muon_groups)

    def _init_state(self, flat_params):
        adamw_state = {}
        for group in self.adamw_groups:
            for path in group.paths:
                param = flat_params[path]
                adamw_state[_safe_key(path)] = {
                    "exp_avg": mx.zeros_like(param),
                    "exp_avg_sq": mx.zeros_like(param),
                }

        muon_state = []
        for group in self.muon_groups:
            param = flat_params[group.paths[0]]
            num_params = len(group.paths)
            state_shape = (
                (num_params, group.shape[-2], 1)
                if group.shape[-2] >= group.shape[-1]
                else (num_params, 1, group.shape[-1])
            )
            muon_state.append(
                {
                    "momentum_buffer": mx.zeros((num_params, *group.shape), dtype=param.dtype),
                    "second_momentum_buffer": mx.zeros(state_shape, dtype=param.dtype),
                }
            )

        return {
            "step": mx.array(0, dtype=mx.uint64),
            "hyperparams": {
                "lm_head_lr": mx.array(self.initial_lrs["lm_head"], dtype=mx.float32),
                "embedding_lr": mx.array(self.initial_lrs["embedding"], dtype=mx.float32),
                "value_embedding_lr": mx.array(self.initial_lrs["value_embedding"], dtype=mx.float32),
                "resid_lr": mx.array(self.initial_lrs["resid"], dtype=mx.float32),
                "x0_lr": mx.array(self.initial_lrs["x0"], dtype=mx.float32),
                "muon_lr": mx.array(self.initial_lrs["muon"], dtype=mx.float32),
                "muon_momentum": mx.array(0.95, dtype=mx.float32),
                "muon_weight_decay": mx.array(0.0, dtype=mx.float32),
            },
            "adamw": adamw_state,
            "muon": muon_state,
        }

    def set_schedule(self, *, schedule_factor: float, muon_momentum: float, muon_weight_decay: float) -> None:
        hyperparams = self.state["hyperparams"]
        hyperparams["lm_head_lr"] = mx.array(self.initial_lrs["lm_head"] * schedule_factor, dtype=mx.float32)
        hyperparams["embedding_lr"] = mx.array(self.initial_lrs["embedding"] * schedule_factor, dtype=mx.float32)
        hyperparams["value_embedding_lr"] = mx.array(self.initial_lrs["value_embedding"] * schedule_factor, dtype=mx.float32)
        hyperparams["resid_lr"] = mx.array(self.initial_lrs["resid"] * schedule_factor, dtype=mx.float32)
        hyperparams["x0_lr"] = mx.array(self.initial_lrs["x0"] * schedule_factor, dtype=mx.float32)
        hyperparams["muon_lr"] = mx.array(self.initial_lrs["muon"] * schedule_factor, dtype=mx.float32)
        hyperparams["muon_momentum"] = mx.array(muon_momentum, dtype=mx.float32)
        hyperparams["muon_weight_decay"] = mx.array(muon_weight_decay, dtype=mx.float32)

    def update(self, model, gradients) -> None:
        self.apply_gradients(gradients)

    def apply_gradients(self, gradients):
        step = self.state["step"] + 1
        self.state["step"] = step
        hyperparams = self.state["hyperparams"]

        group_lrs = {
            "lm_head": hyperparams["lm_head_lr"],
            "embedding": hyperparams["embedding_lr"],
            "value_embedding": hyperparams["value_embedding_lr"],
            "resid": hyperparams["resid_lr"],
            "x0": hyperparams["x0_lr"],
        }

        for group, slots in zip(self.adamw_groups, self.adamw_group_slots):
            lr = group_lrs[group.name]
            beta1 = mx.array(group.beta1, dtype=mx.float32)
            beta2 = mx.array(group.beta2, dtype=mx.float32)
            eps = mx.array(group.eps, dtype=mx.float32)
            weight_decay = mx.array(group.weight_decay, dtype=mx.float32)
            for slot in slots:
                grad = _get_tree_value(gradients, slot.tokens)
                if grad is _MISSING:
                    continue
                param = _get_slot_value(slot)
                state_entry = self.state["adamw"][slot.safe_key]
                new_param, exp_avg, exp_avg_sq = _adamw_update(
                    param,
                    grad,
                    state_entry["exp_avg"],
                    state_entry["exp_avg_sq"],
                    step,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                )
                state_entry["exp_avg"] = exp_avg
                state_entry["exp_avg_sq"] = exp_avg_sq
                _set_slot_value(slot, new_param)

        muon_lr = hyperparams["muon_lr"]
        muon_momentum = hyperparams["muon_momentum"]
        muon_weight_decay = hyperparams["muon_weight_decay"]
        for idx, (group, slots) in enumerate(zip(self.muon_groups, self.muon_group_slots)):
            grads = [_get_tree_value(gradients, slot.tokens) for slot in slots]
            if any(grad is _MISSING for grad in grads):
                continue
            stacked_params = mx.stack([_get_slot_value(slot) for slot in slots], axis=0)
            stacked_grads = mx.stack(grads, axis=0)
            state_entry = self.state["muon"][idx]
            updated_stack, momentum_buffer, second_momentum_buffer = _muon_update(
                stacked_params,
                stacked_grads,
                state_entry["momentum_buffer"],
                state_entry["second_momentum_buffer"],
                muon_lr * max(1.0, group.shape[-2] / group.shape[-1]) ** 0.5,
                muon_momentum,
                muon_weight_decay,
                mx.array(group.beta2, dtype=mx.float32),
                group.ns_steps,
                group.red_dim,
            )
            state_entry["momentum_buffer"] = momentum_buffer
            state_entry["second_momentum_buffer"] = second_momentum_buffer
            for param_idx, slot in enumerate(slots):
                _set_slot_value(slot, updated_stack[param_idx])
