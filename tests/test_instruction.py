import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class InstructionQ2IIGRTest(unittest.TestCase):
    def setUp(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(17)
        self.model = OxygenRECModel(OxygenRECConfig(
            sid_width=16, hidden_size=16, attention_heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_size=32,
            dropout=0.0, max_history_items=3, scenario_vocab_size=2,
            instruction_feature_size=6, q2i_dimension=8,
            q2i_weight=0.2, igr_top_k=2,
        ))
        self.short = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
        self.short_mask = torch.zeros(2, 2, dtype=torch.bool)
        self.long = torch.tensor([
            [[7, 8, 9], [3, 4, 5], [6, 7, 8], [0, 0, 0]],
            [[8, 9, 10], [1, 3, 5], [4, 6, 8], [0, 0, 0]],
        ])
        self.long_mask = torch.tensor([[False, False, False, True]] * 2)
        self.targets = torch.tensor([[7, 8, 9], [8, 9, 10]])
        self.features = torch.randn(2, 6)

    def test_joint_loss_backward_and_masked_igr(self):
        output = self.model(
            self.short, self.short_mask, target_sids=self.targets,
            scenario_ids=torch.tensor([0, 1]), instruction_features=self.features,
            long_history_sids=self.long,
            long_history_padding_mask=self.long_mask,
        )
        self.assertIsNotNone(output.q2i_loss)
        self.assertEqual(tuple(output.igr_indices.shape), (2, 2))
        self.assertTrue((output.igr_indices < 3).all())
        self.assertTrue((output.igr_scores[:, :-1] >= output.igr_scores[:, 1:]).all())
        torch.testing.assert_close(
            output.loss, output.ntp_loss + 0.2 * output.q2i_loss
        )
        output.loss.backward()
        self.assertIsNotNone(self.model.query_adapter[0].weight.grad)

    def test_dense_instruction_changes_predictions(self):
        self.model.eval()
        zeros = torch.zeros_like(self.features)
        ones = torch.ones_like(self.features)
        first = self.model(self.short, self.short_mask, instruction_features=zeros).logits
        second = self.model(self.short, self.short_mask, instruction_features=ones).logits
        self.assertFalse(torch.allclose(first[0], second[0]))


if __name__ == "__main__":
    unittest.main()
