from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_v2_device_probes import compare_numbers, compare_summary_maps


class DeviceProbeComparisonTests(unittest.TestCase):
    def test_nested_values_use_tolerance(self) -> None:
        result = compare_numbers(
            [[1.0, 2.0], [3.0]],
            [[1.001, 2.001], [3.001]],
            atol=0.002,
            rtol=0.0,
        )
        self.assertTrue(result["all_close"])
        self.assertEqual(result["count"], 3)

    def test_nested_values_reject_shape_count_change(self) -> None:
        result = compare_numbers([1.0, 2.0], [1.0], atol=0.0, rtol=0.0)
        self.assertFalse(result["all_close"])

    def test_summary_map_checks_fixed_sample_indexes(self) -> None:
        summary = {
            "weight": {
                "shape": [2],
                "numel": 2,
                "finite": True,
                "min": 1.0,
                "max": 2.0,
                "mean": 1.5,
                "std": 0.5,
                "l1": 3.0,
                "l2": 2.236,
                "samples": [
                    {"index": 0, "value": 1.0},
                    {"index": 1, "value": 2.0},
                ],
            }
        }
        changed = {"weight": {**summary["weight"], "samples": [
            {"index": 0, "value": 1.0},
            {"index": 2, "value": 2.0},
        ]}}
        result = compare_summary_maps(summary, changed, atol=1e-6, rtol=1e-6)
        self.assertFalse(result["all_close"])
        self.assertFalse(result["tensors"]["weight"]["structure_match"])


if __name__ == "__main__":
    unittest.main()
