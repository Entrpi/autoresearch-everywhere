from __future__ import annotations

import math
from dataclasses import asdict, dataclass


DEFAULT_STREAMING_EVAL_TOKENS = 3 * 524288
DEFAULT_STREAMING_EVAL_INTERVAL_STEPS = 25
STREAMING_EVAL_MODE_CHEAP = "cheap"
STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH = "subref-one-sixth"
STREAMING_EVAL_MODE_BOTH = "both"
STREAMING_EVAL_MODES = (
    STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH,
    STREAMING_EVAL_MODE_CHEAP,
    STREAMING_EVAL_MODE_BOTH,
)


def streaming_mode_uses_cheap(mode: str) -> bool:
    return mode in {STREAMING_EVAL_MODE_CHEAP, STREAMING_EVAL_MODE_BOTH}


def streaming_mode_uses_subref(mode: str) -> bool:
    return mode in {STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH, STREAMING_EVAL_MODE_BOTH}


@dataclass(frozen=True)
class StreamingEvalConfig:
    interval_steps: int
    eval_tokens: int
    batch_size: int
    seq_len: int
    mode: str = STREAMING_EVAL_MODE_SUBREF_ONE_SIXTH
    complete_cycle_on_budget: bool = False

    @property
    def step_tokens(self) -> int:
        return max(1, self.batch_size * self.seq_len)

    @property
    def cycle_batches(self) -> int:
        return max(1, math.ceil(self.eval_tokens / self.step_tokens))

    @property
    def cycle_steps(self) -> int:
        return self.cycle_batches * self.interval_steps


@dataclass(frozen=True)
class StreamingEvalPoint:
    step: int
    total_training_time: float
    total_tokens: int
    batch_bpb: float
    cycle_batch_index: int
    cycle_index: int
    cycle_completed: bool
    batch_nats: float | None = None
    batch_bytes: int | None = None
    honest_bpb: float | None = None
    cycle_nats: float | None = None
    cycle_bytes: int | None = None


@dataclass(frozen=True)
class StreamingEvalSummary:
    config: StreamingEvalConfig
    total_points: int
    cycle_batches: int
    cycles_completed: int
    honest_bpb: float | None
    auc: float | None
    min_batch_bpb: float | None
    last20_batch_bpb: float | None
    incomplete_cycle_fraction: float


@dataclass
class StreamingSupercycleAccumulator:
    subcycles_per_supercycle: int
    subcycles_completed: int = 0
    honest_subcycle_bpb: float | None = None
    supercycles_completed: int = 0
    honest_supercycle_bpb: float | None = None
    _current_supercycle_nats: float = 0.0
    _current_supercycle_bytes: int = 0

    def record_completed_subcycle(
        self,
        *,
        cycle_nats: float | None,
        cycle_bytes: int | None,
        honest_bpb: float | None,
    ) -> bool:
        self.subcycles_completed += 1
        self.honest_subcycle_bpb = honest_bpb
        if cycle_nats is None or cycle_bytes is None or cycle_bytes <= 0:
            return False
        self._current_supercycle_nats += cycle_nats
        self._current_supercycle_bytes += cycle_bytes
        if self.subcycles_completed % self.subcycles_per_supercycle != 0:
            return False
        self.supercycles_completed += 1
        self.honest_supercycle_bpb = self._current_supercycle_nats / (math.log(2) * self._current_supercycle_bytes)
        self._current_supercycle_nats = 0.0
        self._current_supercycle_bytes = 0
        return True

    def to_dict(self) -> dict:
        return {
            "subcycles_per_supercycle": self.subcycles_per_supercycle,
            "subcycles_completed": self.subcycles_completed,
            "honest_subcycle_bpb": self.honest_subcycle_bpb,
            "supercycles_completed": self.supercycles_completed,
            "honest_supercycle_bpb": self.honest_supercycle_bpb,
        }


def compute_auc(points: list[StreamingEvalPoint]) -> float | None:
    if len(points) < 2:
        return None
    area = 0.0
    for left, right in zip(points, points[1:]):
        width = max(0, right.step - left.step)
        area += width * (left.batch_bpb + right.batch_bpb) * 0.5
    total_width = points[-1].step - points[0].step
    if total_width <= 0:
        return None
    return area / total_width


def tail_mean(samples: list[float]) -> float | None:
    if not samples:
        return None
    tail_count = max(1, len(samples) // 5)
    return sum(samples[-tail_count:]) / tail_count


class StreamingEvalTracker:
    def __init__(self, config: StreamingEvalConfig):
        self.config = config
        self.points: list[StreamingEvalPoint] = []
        self._current_cycle_values: list[float] = []
        self._current_cycle_nats = 0.0
        self._current_cycle_bytes = 0
        self.cycles_completed = 0
        self.honest_bpb: float | None = None

    def should_eval_after_step(self, step: int) -> bool:
        return step > 0 and step % self.config.interval_steps == 0

    def record(
        self,
        *,
        step: int,
        total_training_time: float,
        total_tokens: int,
        batch_bpb: float,
        batch_nats: float | None = None,
        batch_bytes: int | None = None,
    ) -> StreamingEvalPoint:
        cycle_batch_index = len(self._current_cycle_values) + 1
        cycle_index = self.cycles_completed + 1
        self._current_cycle_values.append(batch_bpb)
        if batch_nats is not None and batch_bytes is not None:
            self._current_cycle_nats += batch_nats
            self._current_cycle_bytes += int(batch_bytes)
        cycle_completed = len(self._current_cycle_values) >= self.config.cycle_batches
        honest_bpb = None
        cycle_nats = None
        cycle_bytes = None
        if cycle_completed:
            cycle_nats = self._current_cycle_nats if self._current_cycle_bytes > 0 else None
            cycle_bytes = self._current_cycle_bytes if self._current_cycle_bytes > 0 else None
            if cycle_nats is not None and cycle_bytes is not None and cycle_bytes > 0:
                honest_bpb = cycle_nats / (math.log(2) * cycle_bytes)
            else:
                honest_bpb = sum(self._current_cycle_values) / len(self._current_cycle_values)
            self.honest_bpb = honest_bpb
            self.cycles_completed += 1
            self._current_cycle_values = []
            self._current_cycle_nats = 0.0
            self._current_cycle_bytes = 0
        point = StreamingEvalPoint(
            step=step,
            total_training_time=total_training_time,
            total_tokens=total_tokens,
            batch_bpb=batch_bpb,
            batch_nats=batch_nats,
            batch_bytes=batch_bytes,
            cycle_batch_index=cycle_batch_index,
            cycle_index=cycle_index,
            cycle_completed=cycle_completed,
            honest_bpb=honest_bpb,
            cycle_nats=cycle_nats,
            cycle_bytes=cycle_bytes,
        )
        self.points.append(point)
        return point

    def requires_cycle_completion(self) -> bool:
        if not self.config.complete_cycle_on_budget:
            return False
        point_count = len(self.points)
        return point_count == 0 or point_count % self.config.cycle_batches != 0

    def completed_cycle_boundary(self) -> bool:
        point_count = len(self.points)
        return point_count > 0 and point_count % self.config.cycle_batches == 0

    def summary(self) -> StreamingEvalSummary:
        batch_values = [point.batch_bpb for point in self.points]
        incomplete = len(self._current_cycle_values) / self.config.cycle_batches
        return StreamingEvalSummary(
            config=self.config,
            total_points=len(self.points),
            cycle_batches=self.config.cycle_batches,
            cycles_completed=self.cycles_completed,
            honest_bpb=self.honest_bpb,
            auc=compute_auc(self.points),
            min_batch_bpb=min(batch_values) if batch_values else None,
            last20_batch_bpb=tail_mean(batch_values),
            incomplete_cycle_fraction=incomplete,
        )

    def to_dict(self) -> dict:
        summary = self.summary()
        return {
            "config": asdict(self.config),
            "summary": asdict(summary),
            "points": [asdict(point) for point in self.points],
        }
