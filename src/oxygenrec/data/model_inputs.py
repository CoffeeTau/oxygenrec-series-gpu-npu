"""Convert temporal samples to padded Semantic-ID model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..sid import SIDRegistry
from .temporal import NextItemSample


@dataclass(frozen=True)
class SIDModelBatch:
    history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    history_padding_mask: tuple[tuple[bool, ...], ...]
    target_sids: tuple[tuple[int, ...], ...]


def build_sid_model_batch(
    samples: Sequence[NextItemSample],
    registry: SIDRegistry,
    *,
    max_history_items: int,
) -> SIDModelBatch:
    """Map item IDs through one registry and left-pad recent known history.

    Unknown history items are excluded explicitly. Targets must be present in
    the train-fitted registry; cold-start targets belong to a separate protocol.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    if max_history_items < 1:
        raise ValueError("max_history_items must be positive")

    histories: list[tuple[tuple[int, ...], ...]] = []
    targets: list[tuple[int, ...]] = []
    for sample in samples:
        if sample.target.item_id not in registry.item_to_sid:
            raise ValueError(
                f"target item {sample.target.item_id!r} is absent from SID registry"
            )
        known_history = [
            registry.sid_for(event.item_id).codes
            for event in sample.history
            if event.item_id in registry.item_to_sid
        ][-max_history_items:]
        if not known_history:
            raise ValueError(
                f"sample for user {sample.user_id!r} has no known history items"
            )
        histories.append(tuple(known_history))
        targets.append(registry.sid_for(sample.target.item_id).codes)

    padded_history: list[tuple[tuple[int, ...], ...]] = []
    padding_masks: list[tuple[bool, ...]] = []
    pad_sid = (0,) * registry.levels
    for history in histories:
        padding = max_history_items - len(history)
        padded_history.append((pad_sid,) * padding + history)
        padding_masks.append((True,) * padding + (False,) * len(history))
    return SIDModelBatch(
        history_sids=tuple(padded_history),
        history_padding_mask=tuple(padding_masks),
        target_sids=tuple(targets),
    )
