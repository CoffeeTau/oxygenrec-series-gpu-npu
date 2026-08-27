import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.llm_features import build_behavior_prompt


class LLMFeaturePromptTest(unittest.TestCase):
    def test_prompt_contains_only_supplied_prior_evidence(self):
        prompt = build_behavior_prompt(
            history_length=12,
            behavior_counts={"view": 10, "addtocart": 2},
            recent_behaviors=("view", "addtocart"),
            repeated_item_kinds=3,
        )
        self.assertIn("历史长度: 12", prompt)
        self.assertIn("view=10", prompt)
        self.assertIn("不得猜测下一次", prompt)
        self.assertNotIn("目标SID", prompt)

    def test_prompt_rejects_empty_history(self):
        with self.assertRaises(ValueError):
            build_behavior_prompt(
                history_length=0, behavior_counts={}, recent_behaviors=(),
                repeated_item_kinds=0,
            )


if __name__ == "__main__":
    unittest.main()
