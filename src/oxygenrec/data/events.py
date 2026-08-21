"""Canonical interaction events and source-specific readers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


class Behavior(str, Enum):
    """Behavior labels supported by the first OxygenREC benchmark."""

    VIEW = "view"
    ADD_TO_CART = "addtocart"
    TRANSACTION = "transaction"


@dataclass(frozen=True, order=True)
class InteractionEvent:
    """A dataset-neutral, time-ordered user-item interaction.

    ``source_row`` is a deterministic tie breaker for storage and diagnostics.
    Sample construction deliberately does not use same-timestamp rows as history,
    because their causal order is not established by RetailRocket.
    """

    timestamp_ms: int
    source_row: int
    user_id: str
    item_id: str
    behavior: Behavior
    transaction_id: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if self.source_row < 0:
            raise ValueError("source_row must be non-negative")
        if not self.user_id:
            raise ValueError("user_id must not be empty")
        if not self.item_id:
            raise ValueError("item_id must not be empty")


_RETAILROCKET_COLUMNS = {
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid",
}


def retailrocket_event_from_row(
    row: Mapping[str, str], *, source_row: int
) -> InteractionEvent:
    """Convert one ``events.csv`` row to the canonical schema."""

    try:
        behavior = Behavior(row["event"].strip().lower())
    except ValueError as error:
        raise ValueError(f"unsupported RetailRocket event {row.get('event')!r}") from error

    transaction_id = row.get("transactionid", "").strip() or None
    return InteractionEvent(
        timestamp_ms=int(row["timestamp"]),
        source_row=source_row,
        user_id=row["visitorid"].strip(),
        item_id=row["itemid"].strip(),
        behavior=behavior,
        transaction_id=transaction_id,
    )


def load_retailrocket_events(path: str | Path) -> Iterable[InteractionEvent]:
    """Stream canonical events from RetailRocket ``events.csv``.

    The generator keeps memory use bounded for the roughly 2.76M-row source file.
    """

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = _RETAILROCKET_COLUMNS - columns
        if missing:
            raise ValueError(f"RetailRocket events CSV is missing columns: {sorted(missing)}")
        for source_row, row in enumerate(reader, start=2):
            yield retailrocket_event_from_row(row, source_row=source_row)

