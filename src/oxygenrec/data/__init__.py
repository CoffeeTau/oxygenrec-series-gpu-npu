"""Dataset-neutral event schema and temporal sample construction."""

from .events import Behavior, InteractionEvent, load_retailrocket_events
from .temporal import (
    NextItemSample,
    Split,
    TemporalBoundaries,
    build_next_item_samples,
    training_item_ids,
)

__all__ = [
    "Behavior",
    "InteractionEvent",
    "NextItemSample",
    "Split",
    "TemporalBoundaries",
    "build_next_item_samples",
    "load_retailrocket_events",
    "training_item_ids",
]
from .model_inputs import SIDModelBatch, build_sid_model_batch

__all__ = ["SIDModelBatch", "build_sid_model_batch"]
