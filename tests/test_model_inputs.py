import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.data.events import Behavior, InteractionEvent
from oxygenrec.data.model_inputs import build_sid_model_batch
from oxygenrec.data.temporal import NextItemSample, Split
from oxygenrec.sid import SIDRegistry


def event(timestamp, row, item):
    return InteractionEvent(timestamp, row, "user", item, Behavior.VIEW)


class SIDModelBatchTest(unittest.TestCase):
    def setUp(self):
        self.registry = SIDRegistry(
            {"a": (1, 2, 3), "b": (4, 5, 6), "target": (7, 8, 9)},
            version="toy-v1",
        )

    def test_maps_registry_truncates_recent_history_and_left_pads(self):
        sample = NextItemSample(
            Split.TRAIN,
            "user",
            (event(1, 1, "unknown"), event(2, 2, "a"), event(3, 3, "b")),
            event(4, 4, "target"),
        )
        batch = build_sid_model_batch([sample], self.registry, max_history_items=3)
        self.assertEqual(batch.history_sids, (((0, 0, 0), (1, 2, 3), (4, 5, 6)),))
        self.assertEqual(batch.history_padding_mask, ((True, False, False),))
        self.assertEqual(batch.target_sids, ((7, 8, 9),))

    def test_rejects_target_outside_training_registry(self):
        sample = NextItemSample(
            Split.VALIDATION,
            "user",
            (event(1, 1, "a"),),
            event(2, 2, "cold-start"),
        )
        with self.assertRaisesRegex(ValueError, "absent from SID registry"):
            build_sid_model_batch([sample], self.registry, max_history_items=2)

