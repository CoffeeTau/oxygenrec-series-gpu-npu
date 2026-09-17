import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class RewardMappingTest(unittest.TestCase):
    def test_maps_components_and_rewards_diverse_candidates(self):
        from oxygenrec.rewards import map_public_rewards

        candidates = torch.tensor([[[1,2,3],[1,2,3],[7,8,9]]])
        mapped = map_public_rewards(
            candidates, legal_mask=torch.tensor([[True, False, True]]),
            relative_scores=torch.tensor([[0.2, 0.1, 0.8]]),
            ranking_scores=torch.tensor([[0.0, 0.0, 1.0]]),
        )
        self.assertEqual(tuple(mapped.total.shape), (1, 3))
        self.assertEqual(float(mapped.format[0, 1]), 0.0)
        self.assertGreater(float(mapped.diversity[0, 2]), float(mapped.diversity[0, 0]))


if __name__ == "__main__":
    unittest.main()
