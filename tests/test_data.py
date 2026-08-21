import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.data import (
    Behavior,
    InteractionEvent,
    Split,
    TemporalBoundaries,
    build_next_item_samples,
    load_retailrocket_events,
    training_item_ids,
)


def event(timestamp, row, user, item, behavior=Behavior.VIEW):
    return InteractionEvent(timestamp, row, user, item, behavior)


class RetailRocketReaderTest(unittest.TestCase):
    def test_streams_and_normalizes_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["timestamp", "visitorid", "event", "itemid", "transactionid"]
                )
                writer.writerow(["1000", "u1", "view", "i1", ""])
                writer.writerow(["2000", "u1", "transaction", "i2", "tx9"])
            events = list(load_retailrocket_events(path))

        self.assertEqual(events[0].source_row, 2)
        self.assertEqual(events[0].behavior, Behavior.VIEW)
        self.assertIsNone(events[0].transaction_id)
        self.assertEqual(events[1].transaction_id, "tx9")


class TemporalSamplesTest(unittest.TestCase):
    def setUp(self):
        self.boundaries = TemporalBoundaries(train_end_ms=100, validation_end_ms=200)

    def test_uses_global_boundaries(self):
        self.assertEqual(self.boundaries.split_for(99), Split.TRAIN)
        self.assertEqual(self.boundaries.split_for(100), Split.VALIDATION)
        self.assertEqual(self.boundaries.split_for(199), Split.VALIDATION)
        self.assertEqual(self.boundaries.split_for(200), Split.TEST)

    def test_training_vocabulary_excludes_future_items(self):
        events = [event(10, 1, "u", "seen"), event(110, 2, "u", "future")]
        self.assertEqual(training_item_ids(events, self.boundaries), frozenset({"seen"}))

    def test_history_is_strictly_earlier_and_crosses_splits(self):
        events = [
            event(10, 1, "u1", "a"),
            event(90, 2, "u1", "b"),
            event(150, 3, "u1", "a", Behavior.ADD_TO_CART),
            event(210, 4, "u1", "b", Behavior.TRANSACTION),
        ]
        samples = build_next_item_samples(events, self.boundaries)

        validation = next(sample for sample in samples if sample.split is Split.VALIDATION)
        test = next(sample for sample in samples if sample.split is Split.TEST)
        self.assertEqual([item.item_id for item in validation.history], ["a", "b"])
        self.assertEqual([item.item_id for item in test.history], ["a", "b", "a"])
        self.assertTrue(
            all(item.timestamp_ms < test.target.timestamp_ms for item in test.history)
        )

    def test_same_timestamp_events_never_see_each_other(self):
        events = [
            event(10, 1, "u1", "a"),
            event(20, 2, "u1", "b"),
            event(20, 3, "u1", "c"),
        ]
        samples = build_next_item_samples(
            events, self.boundaries, require_target_in_training_items=False
        )
        tied = [sample for sample in samples if sample.target.timestamp_ms == 20]
        self.assertEqual(len(tied), 2)
        self.assertEqual([item.item_id for item in tied[0].history], ["a"])
        self.assertEqual([item.item_id for item in tied[1].history], ["a"])

    def test_unseen_targets_are_skipped_by_default(self):
        events = [event(10, 1, "u1", "a"), event(210, 2, "u1", "cold")]
        self.assertEqual(build_next_item_samples(events, self.boundaries), [])
        samples = build_next_item_samples(
            events, self.boundaries, require_target_in_training_items=False
        )
        self.assertEqual(samples[0].target.item_id, "cold")

    def test_history_truncation_keeps_most_recent_events(self):
        events = [
            event(10, 1, "u1", "a"),
            event(20, 2, "u1", "b"),
            event(30, 3, "u1", "c"),
        ]
        samples = build_next_item_samples(events, self.boundaries, max_history=1)
        self.assertEqual([item.item_id for item in samples[-1].history], ["b"])


if __name__ == "__main__":
    unittest.main()

