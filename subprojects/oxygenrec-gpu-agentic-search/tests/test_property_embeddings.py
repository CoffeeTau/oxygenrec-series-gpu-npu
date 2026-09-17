import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not installed in this environment")
class PropertyEmbeddingTest(unittest.TestCase):
    def test_uses_latest_pre_cutoff_values_and_normalizes(self):
        from oxygenrec.data.property_embeddings import build_property_hash_embeddings

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"part{part}.csv" for part in (1, 2)]
            for path in paths:
                with path.open("w", encoding="utf-8", newline="") as stream:
                    csv.writer(stream).writerow(("timestamp", "itemid", "property", "value"))
            with paths[0].open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(
                    [(10, "a", "color", "old"), (20, "a", "color", "new"), (40, "a", "color", "future")]
                )
            with paths[1].open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows([(15, "a", "category", "x"), (15, "b", "category", "y")])
            result = build_property_hash_embeddings(
                paths, ("a", "b", "missing"), train_end_ms=30, dimension=16
            )
        self.assertEqual(result.item_ids, ("a", "b"))
        self.assertEqual(result.retained_snapshot_count, 3)
        np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
        self.assertEqual(result.vectors.shape, (2, 16))

    def test_is_stable_across_repeated_runs(self):
        from oxygenrec.data.property_embeddings import build_property_hash_embeddings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "properties.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "itemid", "property", "value"))
                writer.writerow((10, "a", "category", "x"))
            first = build_property_hash_embeddings([path], ["a"], train_end_ms=20, dimension=8)
            second = build_property_hash_embeddings([path], ["a"], train_end_ms=20, dimension=8)
        np.testing.assert_array_equal(first.vectors, second.vectors)
