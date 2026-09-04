import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.sft_data import build_reasoning_sft_example


def valid_record(status="approved"):
    return {
        "case_id": "case-1",
        "review_status": status,
        "prompt": "历史长度: 3\n行为计数: view=2, addtocart=1, transaction=0",
        "input_evidence": {"behavior_counts": {"view": 2, "addtocart": 1}},
        "reasoning": {
            "intent": "浏览并存在加购线索",
            "evidence": ["view=2", "addtocart=1"],
            "retrieval_strategy": "检索浏览和加购历史",
            "retrieval_plan": {
                "priority_behaviors": ["view", "addtocart"],
                "recency": "balanced",
                "prefer_repeated_items": False,
                "diversity": "medium",
            },
            "constraints": ["不猜测目标商品"],
        },
    }


class SFTDataTest(unittest.TestCase):
    def test_approved_record_becomes_three_turn_messages(self):
        example = build_reasoning_sft_example(valid_record())
        self.assertEqual([item["role"] for item in example["messages"]], [
            "system", "user", "assistant",
        ])
        self.assertEqual(example["metadata"]["review_status"], "approved")

    def test_pending_record_is_not_train_ready_by_default(self):
        with self.assertRaisesRegex(ValueError, "not approved"):
            build_reasoning_sft_example(valid_record("pending"))
        candidate = build_reasoning_sft_example(
            valid_record("pending"), require_approved=False,
        )
        self.assertEqual(candidate["metadata"]["review_status"], "pending")

    def test_unexecutable_plan_is_rejected(self):
        record = valid_record()
        record["reasoning"]["retrieval_plan"]["priority_behaviors"] = ["transaction"]
        with self.assertRaisesRegex(ValueError, "no priority behavior observed"):
            build_reasoning_sft_example(record)


if __name__ == "__main__":
    unittest.main()
