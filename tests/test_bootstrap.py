import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.data.bootstrap import build_frequency_bootstrap_registry
from oxygenrec.data.events import Behavior, InteractionEvent
from oxygenrec.data.temporal import TemporalBoundaries


def event(timestamp, row, item):
    return InteractionEvent(timestamp, row, "user", item, Behavior.VIEW)


class FrequencyBootstrapRegistryTest(unittest.TestCase):
    def test_selects_train_frequency_and_assigns_stable_unique_codes(self):
        events = [
            event(1, 1, "b"),
            event(2, 2, "b"),
            event(3, 3, "a"),
            event(4, 4, "c"),
            event(20, 5, "future-only"),
        ]
        registry = build_frequency_bootstrap_registry(
            events,
            TemporalBoundaries(10, 15),
            max_items=3,
            width=2,
        )
        self.assertEqual(set(registry.item_to_sid), {"a", "b", "c"})
        self.assertEqual(registry.sid_for("a").codes, (0, 0, 0))
        self.assertEqual(registry.sid_for("b").codes, (0, 0, 1))
        self.assertEqual(registry.sid_for("c").codes, (0, 1, 0))
        self.assertEqual(registry.collision_rate(), 0.0)

    def test_rejects_requested_vocabulary_above_capacity(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            build_frequency_bootstrap_registry(
                [event(1, 1, "a")],
                TemporalBoundaries(10, 20),
                max_items=9,
                levels=3,
                width=2,
            )
