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


@dataclass(frozen=True)
class LongShortSIDModelBatch:
    """Recent encoder history plus an older pool for instruction retrieval."""

    short_history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    short_history_padding_mask: tuple[tuple[bool, ...], ...]
    long_history_sids: tuple[tuple[tuple[int, ...], ...], ...]
    long_history_padding_mask: tuple[tuple[bool, ...], ...]
    target_sids: tuple[tuple[int, ...], ...]
    scenario_ids: tuple[int, ...]


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


def build_long_short_sid_model_batch(
    samples: Sequence[NextItemSample],
    registry: SIDRegistry,
    *,
    short_history_items: int,
    long_history_items: int,
    minimum_long_history_items: int = 1,
) -> LongShortSIDModelBatch:
    """Split known history into disjoint recent and older chronological windows.

    The recent tail is always assigned to the backbone. The immediately
    preceding window becomes the IGR candidate pool, so an interaction can
    never occur in both branches. Samples without enough known older items are
    rejected explicitly instead of silently retrieving padding.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    if short_history_items < 1 or long_history_items < 1:
        raise ValueError("history window sizes must be positive")
    if not 1 <= minimum_long_history_items <= long_history_items:
        raise ValueError("minimum_long_history_items must fit the long window")

    pad_sid = (0,) * registry.levels
    short_rows = []
    short_masks = []
    long_rows = []
    long_masks = []
    targets = []
    scenarios = []
    scenario_by_behavior = {"view": 0, "addtocart": 1, "transaction": 2}
    for sample in samples:
        if sample.target.item_id not in registry.item_to_sid:
            raise ValueError(
                f"target item {sample.target.item_id!r} is absent from SID registry"
            )
        known = [
            registry.sid_for(event.item_id).codes
            for event in sample.history
            if event.item_id in registry.item_to_sid
        ]
        short = known[-short_history_items:]
        long = known[: -len(short)][-long_history_items:] if short else []
        if not short:
            raise ValueError(f"sample for user {sample.user_id!r} has no known short history")
        if len(long) < minimum_long_history_items:
            raise ValueError(
                f"sample for user {sample.user_id!r} has only {len(long)} known long-history items"
            )
        short_pad = short_history_items - len(short)
        long_pad = long_history_items - len(long)
        short_rows.append((pad_sid,) * short_pad + tuple(short))
        short_masks.append((True,) * short_pad + (False,) * len(short))
        long_rows.append((pad_sid,) * long_pad + tuple(long))
        long_masks.append((True,) * long_pad + (False,) * len(long))
        targets.append(registry.sid_for(sample.target.item_id).codes)
        scenarios.append(scenario_by_behavior[sample.target.behavior.value])
    return LongShortSIDModelBatch(
        short_history_sids=tuple(short_rows),
        short_history_padding_mask=tuple(short_masks),
        long_history_sids=tuple(long_rows),
        long_history_padding_mask=tuple(long_masks),
        target_sids=tuple(targets),
        scenario_ids=tuple(scenarios),
    )
