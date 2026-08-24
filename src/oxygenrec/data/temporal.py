"""Leak-resistant temporal splitting and next-item sample construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import random
from typing import Iterable, Mapping, Sequence

from .events import Behavior, InteractionEvent


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TemporalBoundaries:
    """Global timestamp boundaries shared by every user and item."""

    train_end_ms: int
    validation_end_ms: int

    def __post_init__(self) -> None:
        if self.train_end_ms < 0:
            raise ValueError("train_end_ms must be non-negative")
        if self.validation_end_ms <= self.train_end_ms:
            raise ValueError("validation_end_ms must be greater than train_end_ms")

    def split_for(self, timestamp_ms: int) -> Split:
        if timestamp_ms < self.train_end_ms:
            return Split.TRAIN
        if timestamp_ms < self.validation_end_ms:
            return Split.VALIDATION
        return Split.TEST


@dataclass(frozen=True)
class NextItemSample:
    """One autoregressive next-item target with strictly earlier history."""

    split: Split
    user_id: str
    history: tuple[InteractionEvent, ...]
    target: InteractionEvent

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("history must not be empty")
        if any(event.user_id != self.user_id for event in self.history):
            raise ValueError("all history events must belong to the sample user")
        if self.target.user_id != self.user_id:
            raise ValueError("target must belong to the sample user")
        if any(event.timestamp_ms >= self.target.timestamp_ms for event in self.history):
            raise ValueError("history must be strictly earlier than the target")


def training_item_ids(
    events: Iterable[InteractionEvent], boundaries: TemporalBoundaries
) -> frozenset[str]:
    """Return the item vocabulary visible before the global training cutoff."""

    return frozenset(
        event.item_id
        for event in events
        if boundaries.split_for(event.timestamp_ms) is Split.TRAIN
    )


def build_next_item_samples(
    events: Iterable[InteractionEvent],
    boundaries: TemporalBoundaries,
    *,
    target_behaviors: Sequence[Behavior] = (
        Behavior.VIEW,
        Behavior.ADD_TO_CART,
        Behavior.TRANSACTION,
    ),
    min_history: int = 1,
    max_history: int | None = None,
    require_target_in_training_items: bool = True,
    max_samples_per_split: Mapping[Split, int] | None = None,
    sample_seed: int = 0,
) -> list[NextItemSample]:
    """Build chronologically ordered samples without future-event leakage.

    Histories may cross split boundaries: a validation/test target can use all
    interactions strictly before its timestamp. The target label itself is never
    inserted into history. When timestamps tie, none of the tied events can see
    another tied event, because their causal order is unknown.
    """

    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    if max_history is not None and max_history < min_history:
        raise ValueError("max_history must be at least min_history")
    limits = dict(max_samples_per_split or {})
    if any(limit < 1 for limit in limits.values()):
        raise ValueError("max_samples_per_split limits must be positive")

    ordered_events = sorted(events)
    train_items = training_item_ids(ordered_events, boundaries)
    targets = frozenset(target_behaviors)
    by_user: dict[str, list[InteractionEvent]] = defaultdict(list)
    for event in ordered_events:
        by_user[event.user_id].append(event)

    samples: list[NextItemSample] = []
    reservoirs: dict[Split, list[NextItemSample]] = defaultdict(list)
    seen_by_split: Counter[Split] = Counter()
    generators = {
        split: random.Random(sample_seed + index)
        for index, split in enumerate(Split)
    }
    for user_id, user_events in sorted(by_user.items()):
        history: list[InteractionEvent] = []
        cursor = 0
        while cursor < len(user_events):
            timestamp = user_events[cursor].timestamp_ms
            group_end = cursor + 1
            while (
                group_end < len(user_events)
                and user_events[group_end].timestamp_ms == timestamp
            ):
                group_end += 1

            same_time_events = user_events[cursor:group_end]
            if len(history) >= min_history:
                selected_history = history[-max_history:] if max_history else history
                for target in same_time_events:
                    if target.behavior not in targets:
                        continue
                    if require_target_in_training_items and target.item_id not in train_items:
                        continue
                    sample = NextItemSample(
                        split=boundaries.split_for(target.timestamp_ms),
                        user_id=user_id,
                        history=tuple(selected_history),
                        target=target,
                    )
                    if sample.split not in limits:
                        samples.append(sample)
                    else:
                        seen_by_split[sample.split] += 1
                        bucket = reservoirs[sample.split]
                        limit = limits[sample.split]
                        if len(bucket) < limit:
                            bucket.append(sample)
                        else:
                            replacement = generators[sample.split].randrange(
                                seen_by_split[sample.split]
                            )
                            if replacement < limit:
                                bucket[replacement] = sample

            history.extend(same_time_events)
            cursor = group_end

    for split in Split:
        samples.extend(reservoirs[split])
    return samples
