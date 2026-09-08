import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class EATOSDTest(unittest.TestCase):
    def test_geometric_reward_and_best_of_group(self):
        from oxygenrec.ea_tosd import select_best_verifiable_trajectory

        candidates = torch.tensor(
            [[[1, 2, 9], [1, 8, 3], [7, 2, 3]]], dtype=torch.long
        )
        gold = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = select_best_verifiable_trajectory(
            candidates, gold, geometric_decay=0.5
        )
        torch.testing.assert_close(
            result.token_weights, torch.tensor([4 / 7, 2 / 7, 1 / 7])
        )
        torch.testing.assert_close(
            result.rewards, torch.tensor([[(4 + 2) / 7, (4 + 1) / 7, (2 + 1) / 7]])
        )
        self.assertEqual(result.best_indices.tolist(), [0])
        self.assertEqual(result.selected_tokens.tolist(), [[1, 2, 9]])

    def test_low_and_high_entropy_branches_and_backward(self):
        from oxygenrec.ea_tosd import EATOSDConfig, ea_tosd_loss

        student = torch.zeros(1, 2, 3, requires_grad=True)
        teacher = torch.tensor(
            [[[12.0, -12.0, -12.0], [0.0, 0.0, 0.0]]], requires_grad=True
        )
        selected = torch.tensor([[0, 1]], dtype=torch.long)
        reward = torch.tensor([0.5])
        supervised = student.square().mean()
        config = EATOSDConfig(
            geometric_decay=0.5,
            verifiable_weight=0.1,
            self_distillation_weight=0.2,
            forward_kl_weight=0.3,
            supervised_weight=0.4,
            low_entropy_threshold=0.5,
            high_entropy_threshold=1.0,
        )
        output = ea_tosd_loss(
            student, teacher, selected, reward, supervised, config=config
        )
        self.assertAlmostEqual(float(output.low_entropy_weights[0, 0]), 1.0, places=5)
        self.assertEqual(float(output.low_entropy_weights[0, 1]), 0.0)
        self.assertEqual(float(output.high_entropy_weights[0, 0]), 0.0)
        self.assertAlmostEqual(float(output.high_entropy_weights[0, 1]), 1.0, places=5)
        expected = (
            0.1 * output.verifiable_loss
            + 0.2 * output.self_distillation_loss
            + 0.3 * output.forward_kl_loss
            + 0.4 * supervised
        )
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_middle_entropy_positions_receive_no_distillation_weight(self):
        from oxygenrec.ea_tosd import EATOSDConfig, ea_tosd_loss

        logits = torch.zeros(1, 2, 3, requires_grad=True)
        output = ea_tosd_loss(
            logits,
            logits.detach(),
            torch.tensor([[0, 1]]),
            torch.zeros(1),
            logits.square().mean(),
            config=EATOSDConfig(
                low_entropy_threshold=0.5, high_entropy_threshold=1.5
            ),
        )
        self.assertEqual(float(output.low_entropy_weights.sum()), 0.0)
        self.assertEqual(float(output.high_entropy_weights.sum()), 0.0)
        self.assertEqual(float(output.self_distillation_loss), 0.0)
        self.assertEqual(float(output.forward_kl_loss), 0.0)

    def test_rejects_overlapping_entropy_thresholds(self):
        from oxygenrec.ea_tosd import EATOSDConfig

        with self.assertRaisesRegex(ValueError, "smaller"):
            EATOSDConfig(low_entropy_threshold=2.0, high_entropy_threshold=1.0)


if __name__ == "__main__":
    unittest.main()
