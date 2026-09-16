import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.migration_validation import (
    canonical_sha256,
    summarize_listwise_validation,
)


class MigrationValidationMetricsTest(unittest.TestCase):
    def test_canonical_hash_ignores_dictionary_key_order(self):
        self.assertEqual(
            canonical_sha256({"b": 2, "a": [1, 3]}),
            canonical_sha256({"a": [1, 3], "b": 2}),
        )

    def test_summarizes_listwise_and_exact_beam_ranking(self):
        targets = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 1, 2, 3],
        ]
        greedy = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 4, 5, 6],
        ]
        beams = [
            [targets[0], [4, 5, 6, 1, 2, 3]],
            [[7, 8, 9, 4, 5, 6], targets[1]],
        ]
        legal = {
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
        }
        metrics = summarize_listwise_validation(
            targets,
            greedy,
            beams,
            sid_levels=3,
            is_legal=lambda sid: tuple(sid) in legal,
        )
        self.assertEqual(metrics["lists"], 2)
        self.assertEqual(metrics["list_size"], 2)
        self.assertEqual(metrics["sid_recall"], 0.75)
        self.assertEqual(metrics["position_accuracy"], 0.75)
        self.assertEqual(metrics["exact_list_rate"], 0.5)
        self.assertEqual(metrics["beam_exact_list_hit_rate"], {"1": 0.5, "2": 1.0})
        self.assertAlmostEqual(metrics["beam_exact_list_mrr"], 0.75)
        self.assertAlmostEqual(
            metrics["beam_exact_list_ndcg"],
            (1.0 + 1.0 / math.log2(3)) / 2,
        )
        self.assertEqual(metrics["greedy_legal_item_rate"], 1.0)
        self.assertEqual(metrics["beam_legal_item_rate"], 1.0)
        self.assertEqual(metrics["sid_token_accuracy"], 0.75)

    def test_rejects_mismatched_input_sizes(self):
        with self.assertRaisesRegex(ValueError, "same positive size"):
            summarize_listwise_validation(
                [[1, 2, 3]],
                [],
                [[[1, 2, 3]]],
                sid_levels=3,
                is_legal=lambda sid: True,
            )


if __name__ == "__main__":
    unittest.main()
