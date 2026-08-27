import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.retrieval_planning import compile_retrieval_plan

try:
    import torch
except ImportError:
    torch = None


class RetrievalPlanCompileTest(unittest.TestCase):
    def test_filters_unobserved_priority_behavior(self):
        compiled = compile_retrieval_plan(
            {"priority_behaviors": ["transaction", "view"], "recency": "long_term",
             "prefer_repeated_items": True, "diversity": "medium"},
            {"view": 41},
        )
        self.assertEqual(compiled.priority_behavior_ids, (0,))

    def test_rejects_plan_without_observed_priority(self):
        with self.assertRaisesRegex(ValueError, "no priority behavior observed"):
            compile_retrieval_plan(
                {"priority_behaviors": ["transaction"], "recency": "recent",
                 "prefer_repeated_items": False, "diversity": "low"},
                {"view": 41},
            )


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class RetrievalPlanExecuteTest(unittest.TestCase):
    def test_plan_changes_selection_and_high_diversity_suppresses_duplicate(self):
        from oxygenrec.retrieval_planning import execute_retrieval_plan

        scores = torch.tensor([[0.90, 0.89, 0.88, 0.87]])
        sids = torch.tensor([[[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3]]])
        behaviors = torch.tensor([[0, 0, 2, 1]])
        mask = torch.zeros(1, 4, dtype=torch.bool)
        plan = compile_retrieval_plan(
            {"priority_behaviors": ["transaction"], "recency": "balanced",
             "prefer_repeated_items": False, "diversity": "high"},
            {"view": 2, "transaction": 1, "addtocart": 1},
        )
        indices, _, _ = execute_retrieval_plan(
            scores, sids, behaviors, mask, [plan], top_k=2,
        )
        self.assertEqual(indices[0, 0].item(), 2)
        self.assertNotEqual(sids[0, indices[0, 0]].tolist(), sids[0, indices[0, 1]].tolist())


if __name__ == "__main__":
    unittest.main()
