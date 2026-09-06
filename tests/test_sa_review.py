import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.sa_review import select_sa_gcpo_trajectories


class SAGCPOReviewSelectionTest(unittest.TestCase):
    def test_selects_fixed_roles_and_merges_duplicates(self):
        rows = [
            {
                "cohort_index": 0,
                "target_covered": False,
                "reward_spread": 0.2,
                "policy_shift": 0.1,
                "suppressed_count": 1,
            },
            {
                "cohort_index": 1,
                "target_covered": True,
                "reward_spread": 0.8,
                "policy_shift": 0.3,
                "suppressed_count": 2,
            },
            {
                "cohort_index": 2,
                "target_covered": False,
                "reward_spread": 0.5,
                "policy_shift": 0.9,
                "suppressed_count": 3,
            },
        ]

        selected, coverage = select_sa_gcpo_trajectories(rows)

        self.assertEqual(coverage["target_covered"], 1)
        self.assertEqual(coverage["largest_reward_spread"], 1)
        self.assertEqual(coverage["largest_policy_shift"], 2)
        self.assertEqual(coverage["most_threshold_suppression"], 2)
        self.assertEqual([row["cohort_index"] for row in selected], [1, 2])
        self.assertEqual(
            selected[0]["selection_roles"],
            ["target_covered", "largest_reward_spread"],
        )

    def test_reports_missing_target_coverage(self):
        selected, coverage = select_sa_gcpo_trajectories([
            {
                "cohort_index": 0,
                "target_covered": False,
                "reward_spread": 0.0,
                "policy_shift": 0.0,
                "suppressed_count": 0,
            }
        ])

        self.assertIsNone(coverage["target_covered"])
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()

