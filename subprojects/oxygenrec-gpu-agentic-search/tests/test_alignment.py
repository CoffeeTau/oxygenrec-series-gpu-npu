import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class SAGCPOTest(unittest.TestCase):
    def test_target_reward_threshold_suppresses_false_positive_advantage(self):
        from oxygenrec.alignment import sa_gcpo_loss

        current = torch.zeros(1, 3, 2, requires_grad=True)
        old = torch.zeros_like(current)
        output = sa_gcpo_loss(
            current, old, torch.tensor([[1.0, 2.0, 3.0]]),
            target_rewards=torch.tensor([3.5]),
        )
        self.assertEqual(float(output.thresholded_advantage[0, 2]), 0.0)
        self.assertLess(float(output.thresholded_advantage[0, 0]), 0.0)

    def test_soft_gate_has_unit_gradient_weight_at_ratio_one(self):
        from oxygenrec.alignment import sa_gcpo_loss

        log_probs = torch.zeros(1, 3, 2, requires_grad=True)
        output = sa_gcpo_loss(
            log_probs, torch.zeros_like(log_probs),
            torch.tensor([[1.0, 2.0, 4.0]]), torch.tensor([2.5]),
        )
        torch.testing.assert_close(output.gradient_weight, torch.ones_like(log_probs))
        output.loss.backward()
        self.assertTrue(torch.isfinite(log_probs.grad).all())

    def test_rejects_fully_masked_sequence(self):
        from oxygenrec.alignment import sa_gcpo_loss

        values = torch.zeros(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            sa_gcpo_loss(
                values, values, torch.tensor([[1.0, 2.0]]), torch.tensor([1.0]),
                token_mask=torch.zeros_like(values, dtype=torch.bool),
            )
