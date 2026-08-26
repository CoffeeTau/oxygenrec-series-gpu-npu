import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.data.events import Behavior, InteractionEvent
from oxygenrec.data.model_inputs import build_long_short_sid_model_batch, build_sid_model_batch
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
        self.assertEqual(batch.history_behavior_ids, ((0, 0, 0),))
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

    def test_long_short_windows_are_disjoint_and_chronological(self):
        registry = SIDRegistry(
            {"a": (1, 1, 1), "b": (2, 2, 2), "c": (3, 3, 3),
             "d": (4, 4, 4), "target": (7, 8, 9)}, version="windows-v1"
        )
        sample = NextItemSample(
            Split.TRAIN, "user",
            (event(1, 1, "a"), event(2, 2, "b"), event(3, 3, "c"), event(4, 4, "d")),
            InteractionEvent(5, 5, "user", "target", Behavior.TRANSACTION),
        )
        batch = build_long_short_sid_model_batch(
            [sample], registry, short_history_items=2, long_history_items=3
        )
        self.assertEqual(batch.short_history_sids[0], ((3, 3, 3), (4, 4, 4)))
        self.assertEqual(batch.long_history_sids[0], ((0, 0, 0), (1, 1, 1), (2, 2, 2)))
        self.assertEqual(batch.long_history_padding_mask[0], (True, False, False))
        self.assertEqual(batch.short_history_behavior_ids[0], (0, 0))
        self.assertEqual(batch.long_history_behavior_ids[0], (0, 0, 0))
        self.assertEqual(batch.scenario_ids, (2,))

    def test_long_short_rejects_insufficient_retrieval_candidates(self):
        sample = NextItemSample(
            Split.TRAIN, "user", (event(1, 1, "a"), event(2, 2, "b")),
            event(3, 3, "target"),
        )
        with self.assertRaisesRegex(ValueError, "only 0 known long-history"):
            build_long_short_sid_model_batch(
                [sample], self.registry, short_history_items=2,
                long_history_items=3, minimum_long_history_items=1,
            )
