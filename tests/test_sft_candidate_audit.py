import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_reasoning_sft_candidates.py"
SPEC = importlib.util.spec_from_file_location("audit_reasoning_sft_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(case_id, behavior="view", reasoning_suffix=""):
    return {
        "case_id": case_id,
        "review_status": "pending",
        "input_evidence": {
            "history_length": 10,
            "behavior_counts": {behavior: 2},
            "repeated_item_kinds": 1,
        },
        "reasoning": {
            "intent": "历史兴趣" + reasoning_suffix,
            "evidence": [f"{behavior}=2"],
            "retrieval_strategy": "检索历史",
            "retrieval_plan": {
                "priority_behaviors": [behavior], "recency": "balanced",
                "prefer_repeated_items": True, "diversity": "medium",
            },
            "constraints": ["不猜测目标"],
        },
    }


class CandidateAuditTest(unittest.TestCase):
    def test_reports_validity_distribution_and_duplicates(self):
        summary = MODULE.audit([record("a"), record("b"), record("c", "addtocart", "x")], 2)
        self.assertEqual(summary["schema_valid"], 3)
        self.assertEqual(summary["executable_plans"], 3)
        self.assertEqual(summary["duplicate_reasoning_rows"], 1)
        self.assertEqual(len(summary["review_sample_case_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
