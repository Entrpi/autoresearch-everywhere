from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


LR_MULTIPLIER_ARG_FIELDS = (
    "lr_multiplier",
    "embedding_lr_multiplier",
    "unembedding_lr_multiplier",
    "matrix_lr_multiplier",
    "scalar_lr_multiplier",
)


@dataclass(frozen=True)
class LrProfile:
    embedding_lr: float
    unembedding_lr: float
    matrix_lr: float
    scalar_lr: float
    value_embedding_lr_scale: float = 1.0
    resid_lr_scale: float = 0.01
    dmodel_reference: int = 768


@dataclass(frozen=True)
class LrMultipliers:
    lr_multiplier: float = 1.0
    embedding_lr_multiplier: float = 1.0
    unembedding_lr_multiplier: float = 1.0
    matrix_lr_multiplier: float = 1.0
    scalar_lr_multiplier: float = 1.0


@dataclass(frozen=True)
class ResolvedLrProfile:
    lm_head_lr: float
    embedding_lr: float
    value_embedding_lr: float
    resid_lr: float
    x0_lr: float
    matrix_lr: float
    dmodel_lr_scale: float


DEFAULT_LR_PROFILE = LrProfile(
    embedding_lr=0.6,
    unembedding_lr=0.004,
    matrix_lr=0.04,
    scalar_lr=0.5,
)
DEFAULT_LR_MULTIPLIERS = LrMultipliers()


def _require_positive(name: str, value: float) -> float:
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return numeric


def lr_profile_to_dict(profile: LrProfile) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in asdict(profile).items()
    }


def lr_multipliers_to_dict(multipliers: LrMultipliers) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in asdict(multipliers).items()
    }


def lr_profile_from_mapping(
    payload: Mapping[str, Any] | None,
    *,
    default_profile: LrProfile = DEFAULT_LR_PROFILE,
) -> LrProfile:
    if payload is None:
        return default_profile
    if "lr_profile" in payload and isinstance(payload["lr_profile"], Mapping):
        payload = payload["lr_profile"]
    return LrProfile(
        embedding_lr=float(payload.get("embedding_lr", default_profile.embedding_lr)),
        unembedding_lr=float(payload.get("unembedding_lr", default_profile.unembedding_lr)),
        matrix_lr=float(payload.get("matrix_lr", default_profile.matrix_lr)),
        scalar_lr=float(payload.get("scalar_lr", default_profile.scalar_lr)),
        value_embedding_lr_scale=float(
            payload.get("value_embedding_lr_scale", default_profile.value_embedding_lr_scale)
        ),
        resid_lr_scale=float(payload.get("resid_lr_scale", default_profile.resid_lr_scale)),
        dmodel_reference=int(payload.get("dmodel_reference", default_profile.dmodel_reference)),
    )


def lr_multipliers_from_mapping(payload: Mapping[str, Any] | None) -> LrMultipliers:
    if payload is None:
        return DEFAULT_LR_MULTIPLIERS
    if "lr_multipliers" in payload and isinstance(payload["lr_multipliers"], Mapping):
        payload = payload["lr_multipliers"]
    return LrMultipliers(
        lr_multiplier=float(payload.get("lr_multiplier", 1.0)),
        embedding_lr_multiplier=float(payload.get("embedding_lr_multiplier", 1.0)),
        unembedding_lr_multiplier=float(payload.get("unembedding_lr_multiplier", 1.0)),
        matrix_lr_multiplier=float(payload.get("matrix_lr_multiplier", 1.0)),
        scalar_lr_multiplier=float(payload.get("scalar_lr_multiplier", 1.0)),
    )


def apply_lr_multipliers(
    profile: LrProfile,
    multipliers: LrMultipliers = DEFAULT_LR_MULTIPLIERS,
) -> LrProfile:
    return LrProfile(
        embedding_lr=_require_positive(
            "embedding_lr",
            profile.embedding_lr * multipliers.lr_multiplier * multipliers.embedding_lr_multiplier,
        ),
        unembedding_lr=_require_positive(
            "unembedding_lr",
            profile.unembedding_lr * multipliers.lr_multiplier * multipliers.unembedding_lr_multiplier,
        ),
        matrix_lr=_require_positive(
            "matrix_lr",
            profile.matrix_lr * multipliers.lr_multiplier * multipliers.matrix_lr_multiplier,
        ),
        scalar_lr=_require_positive(
            "scalar_lr",
            profile.scalar_lr * multipliers.lr_multiplier * multipliers.scalar_lr_multiplier,
        ),
        value_embedding_lr_scale=_require_positive(
            "value_embedding_lr_scale",
            profile.value_embedding_lr_scale,
        ),
        resid_lr_scale=_require_positive(
            "resid_lr_scale",
            profile.resid_lr_scale,
        ),
        dmodel_reference=int(profile.dmodel_reference),
    )


def resolve_effective_lr_profile(
    profile: LrProfile,
    *,
    model_dim: int,
) -> ResolvedLrProfile:
    reference_dim = max(1, int(profile.dmodel_reference))
    dmodel_lr_scale = (float(model_dim) / float(reference_dim)) ** -0.5
    return ResolvedLrProfile(
        lm_head_lr=_require_positive("lm_head_lr", profile.unembedding_lr * dmodel_lr_scale),
        embedding_lr=_require_positive("embedding_lr", profile.embedding_lr * dmodel_lr_scale),
        value_embedding_lr=_require_positive(
            "value_embedding_lr",
            profile.embedding_lr * profile.value_embedding_lr_scale * dmodel_lr_scale,
        ),
        resid_lr=_require_positive("resid_lr", profile.scalar_lr * profile.resid_lr_scale),
        x0_lr=_require_positive("x0_lr", profile.scalar_lr),
        matrix_lr=_require_positive("matrix_lr", profile.matrix_lr),
        dmodel_lr_scale=dmodel_lr_scale,
    )
