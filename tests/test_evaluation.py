import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.evaluation import evaluate_sid_ranking
from oxygenrec.sid import SIDRegistry


class RankingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.registry = SIDRegistry(
            {
                "a": (1, 2, 3),
                "a-collision": (1, 2, 3),
                "b": (4, 5, 6),
                "c": (7, 8, 9),
            }
        )

    def test_computes_rank_metrics_and_legality(self):
        metrics = evaluate_sid_ranking(
            [
                [(1, 2, 3), (4, 5, 6)],
                [(1, 2, 3), (7, 8, 9)],
            ],
            ["a-collision", "c"],
            self.registry,
            ks=(1, 2),
        )
        self.assertEqual(metrics.hit_rate, {1: 0.5, 2: 1.0})
        self.assertEqual(metrics.recall, metrics.hit_rate)
        self.assertAlmostEqual(metrics.mrr, 0.75)
        self.assertAlmostEqual(metrics.ndcg, (1.0 + 1.0 / 1.5849625007) / 2)
        self.assertEqual(metrics.legal_sid_rate, 1.0)

    def test_reports_illegal_candidates_without_guessing_items(self):
        metrics = evaluate_sid_ranking(
            [[(9, 9, 9), (4, 5, 6)]], ["b"], self.registry, ks=(1, 2)
        )
        self.assertEqual(metrics.legal_sid_rate, 0.5)
        self.assertEqual(metrics.hit_rate, {1: 0.0, 2: 1.0})

    def test_requires_rankings_to_cover_requested_k(self):
        with self.assertRaisesRegex(ValueError, "largest K"):
            evaluate_sid_ranking(
                [[(1, 2, 3)]], ["a"], self.registry, ks=(1, 2)
            )
