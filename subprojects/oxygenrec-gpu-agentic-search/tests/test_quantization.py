import unittest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.quantization import ReferenceKMeans, ReferenceResidualKMeans
from oxygenrec.sid_metrics import compute_sid_diagnostics


class ReferenceKMeansTest(unittest.TestCase):
    def test_is_deterministic_and_separates_obvious_groups(self):
        vectors = [(0.0,), (0.2,), (9.8,), (10.0,)]
        first = ReferenceKMeans(2, seed=7).fit(vectors)
        second = ReferenceKMeans(2, seed=7).fit(vectors)
        self.assertEqual(first, second)
        self.assertEqual(first.assignments[0], first.assignments[1])
        self.assertEqual(first.assignments[2], first.assignments[3])
        self.assertNotEqual(first.assignments[0], first.assignments[2])

    def test_rejects_more_clusters_than_samples(self):
        with self.assertRaises(ValueError):
            ReferenceKMeans(3).fit([(0.0,), (1.0,)])


class ResidualKMeansTest(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            "a": (0.0, 0.0), "b": (0.0, 1.0),
            "c": (9.0, 9.0), "d": (10.0, 9.0),
        }

    def test_fits_three_levels_and_builds_registry(self):
        model = ReferenceResidualKMeans(levels=3, width=2, seed=11).fit(
            list(self.embeddings.values())
        )
        registry = model.registry_for(self.embeddings, version="toy-rq-v1")
        self.assertEqual(model.levels, 3)
        self.assertEqual(model.width, 2)
        self.assertEqual(len(registry.item_to_sid), 4)
        self.assertTrue(all(len(sid) == 3 for sid in registry.item_to_sid.values()))

    def test_residual_levels_do_not_increase_reconstruction_error(self):
        vectors = list(self.embeddings.values())
        one_level = ReferenceResidualKMeans(levels=1, width=2, seed=11).fit(vectors)
        three_levels = ReferenceResidualKMeans(levels=3, width=2, seed=11).fit(vectors)

        def error(model):
            reconstructions = model.reconstruct(model.encode(vectors))
            return sum(
                sum((value - estimate) ** 2 for value, estimate in zip(x, y))
                for x, y in zip(vectors, reconstructions)
            )

        self.assertLessEqual(error(three_levels), error(one_level) + 1e-12)

    def test_diagnostics_match_registry(self):
        model = ReferenceResidualKMeans(levels=3, width=2, seed=11).fit(
            list(self.embeddings.values())
        )
        registry = model.registry_for(self.embeddings, version="toy-rq-v1")
        metrics = compute_sid_diagnostics(registry)
        self.assertEqual(metrics.item_count, 4)
        self.assertEqual(len(metrics.prefix_coverage), 3)
        self.assertEqual(metrics.prefix_coverage[0].capacity, 2)
        self.assertEqual(metrics.prefix_coverage[2].capacity, 8)
        self.assertEqual(len(metrics.load_balance), 3)

    def test_codebook_json_round_trip_checks_version(self):
        model = ReferenceResidualKMeans(levels=3, width=2, seed=11).fit(
            list(self.embeddings.values())
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codebook.json"
            model.to_json(path, version="toy-codebook-v1")
            version, restored = model.from_json(
                path, expected_version="toy-codebook-v1"
            )
            with self.assertRaises(ValueError):
                model.from_json(path, expected_version="wrong-version")
        self.assertEqual(version, "toy-codebook-v1")
        self.assertEqual(restored, model)


if __name__ == "__main__":
    unittest.main()
