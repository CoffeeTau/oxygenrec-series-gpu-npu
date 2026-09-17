import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_v2_validation_cases import compare_payloads


def payload(platform, beam, *, precision="bf16"):
    return {
        "protocol": "oxygenrec_v2_full_fixed_validation_v1_cases",
        "platform": platform,
        "precision": precision,
        "source_commit": "commit-a",
        "inputs": {"events_sha256": "events-a"},
        "cases": [
            {
                "case_index": 0,
                "target": [1, 2, 3, 4, 5, 6],
                "greedy": [1, 2, 3, 7, 8, 9],
                "beam": beam,
                "beam_scores": [-1.0, -1.1],
            }
        ],
    }


class CompareV2ValidationCasesTest(unittest.TestCase):
    def test_emits_only_changed_beam_case(self):
        gpu = payload("gpu", [[1, 2, 3], [4, 5, 6]])
        npu = payload("npu", [[1, 2, 3], [4, 5, 7]])
        result = compare_payloads(gpu, npu)
        self.assertEqual(result["greedy_changed_cases"], 0)
        self.assertEqual(result["beam_changed_cases"], 1)
        self.assertEqual(result["changed_cases"][0]["case_index"], 0)

    def test_rejects_mismatched_inputs(self):
        gpu = payload("gpu", [[1, 2, 3], [4, 5, 6]])
        npu = payload("npu", [[1, 2, 3], [4, 5, 6]])
        npu["inputs"] = {"events_sha256": "events-b"}
        with self.assertRaisesRegex(ValueError, "inputs"):
            compare_payloads(gpu, npu)


if __name__ == "__main__":
    unittest.main()
