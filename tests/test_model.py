import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class OxygenRECModelTest(unittest.TestCase):
    def setUp(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(7)
        self.model = OxygenRECModel(
            OxygenRECConfig(
                sid_width=11,
                hidden_size=16,
                attention_heads=4,
                encoder_layers=1,
                decoder_layers=1,
                feedforward_size=32,
                dropout=0.0,
                max_history_items=4,
            )
        )
        self.history = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]]
        )
        self.padding = torch.tensor([[False, False, True], [False, False, False]])
        self.targets = torch.tensor([[1, 2, 3], [7, 8, 9]])

    def test_logits_loss_and_backward(self):
        output = self.model(
            self.history,
            self.padding,
            target_sids=self.targets,
            level_weights=(1.0, 0.7, 0.4),
        )
        self.assertEqual([tuple(item.shape) for item in output.logits], [(2, 11)] * 3)
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_padding_codes_do_not_change_logits(self):
        self.model.eval()
        changed = self.history.clone()
        changed[0, 2] = torch.tensor([8, 8, 8])
        first = self.model(self.history, self.padding, target_sids=self.targets).logits
        second = self.model(changed, self.padding, target_sids=self.targets).logits
        for left, right in zip(first, second):
            torch.testing.assert_close(left[0], right[0])

    def test_future_target_codes_do_not_leak_into_earlier_levels(self):
        self.model.eval()
        changed = self.targets.clone()
        changed[:, 1:] = torch.tensor([[9, 10], [1, 2]])
        first = self.model(self.history, self.padding, target_sids=self.targets).logits
        second = self.model(self.history, self.padding, target_sids=changed).logits
        torch.testing.assert_close(first[0], second[0])

    def test_generation_follows_prefix_trie(self):
        from oxygenrec.sid import PrefixTrie

        self.model.eval()
        trie = PrefixTrie([(1, 2, 3), (1, 4, 5), (7, 8, 9)])
        generated = self.model.generate(self.history, self.padding, trie)
        self.assertEqual(tuple(generated.shape), (2, 3))
        for row in generated.tolist():
            self.assertTrue(trie.contains(row))

    def test_beam_search_returns_ranked_legal_paths(self):
        from oxygenrec.sid import PrefixTrie

        self.model.eval()
        trie = PrefixTrie([(1, 2, 3), (1, 4, 5), (7, 8, 9)])
        output = self.model.beam_search(
            self.history, self.padding, trie, beam_width=2
        )
        self.assertEqual(tuple(output.semantic_ids.shape), (2, 2, 3))
        self.assertEqual(tuple(output.scores.shape), (2, 2))
        for ranking, scores in zip(
            output.semantic_ids.tolist(), output.scores.tolist()
        ):
            self.assertTrue(all(trie.contains(row) for row in ranking))
            self.assertGreaterEqual(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
