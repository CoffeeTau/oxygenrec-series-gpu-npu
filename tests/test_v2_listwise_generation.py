import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class V2ListwiseGenerationTest(unittest.TestCase):
    def setUp(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel
        from oxygenrec.sid import PrefixTrie

        torch.manual_seed(23)
        self.model = OxygenRECModel(
            OxygenRECConfig(
                sid_width=11,
                behavior_vocab_size=3,
                behavior_instruction_vocab_size=3,
                hidden_size=16,
                attention_heads=4,
                encoder_layers=1,
                decoder_layers=1,
                feedforward_size=32,
                dropout=0.0,
                max_history_items=4,
                max_target_items=2,
            )
        )
        self.history = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                [[7, 8, 9], [1, 4, 5], [4, 5, 6]],
            ],
            dtype=torch.long,
        )
        self.padding = torch.tensor(
            [[False, False, True], [False, False, False]]
        )
        self.history_behaviors = torch.tensor([[0, 1, 0], [2, 0, 1]])
        self.behavior_instructions = torch.tensor([1, 2])
        self.targets = torch.tensor(
            [
                [[1, 2, 3], [1, 4, 5]],
                [[7, 8, 9], [4, 5, 6]],
            ],
            dtype=torch.long,
        )
        self.trie = PrefixTrie(
            [(1, 2, 3), (1, 4, 5), (4, 5, 6), (7, 8, 9)]
        )

    def _forward(self, targets=None):
        return self.model(
            self.history,
            self.padding,
            target_sids=self.targets if targets is None else targets,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.behavior_instructions,
            token_weights=torch.tensor(
                [[1.5] * 6, [2.0] * 6], dtype=torch.float32
            ),
        )

    def test_listwise_teacher_forcing_has_six_steps_and_backward(self):
        output = self._forward()
        self.assertEqual(len(output.logits), 6)
        self.assertEqual([tuple(item.shape) for item in output.logits], [(2, 11)] * 6)
        self.assertEqual(len(output.level_losses), 6)
        output.loss.backward()
        self.assertIsNotNone(self.model.behavior_instruction_adapter[0].weight.grad)
        reserved = self.model.sid_embeddings[0].weight.grad[11:14]
        self.assertGreater(float(reserved.abs().sum()), 0.0)

    def test_future_item_does_not_leak_into_first_item_logits(self):
        self.model.eval()
        changed = self.targets.clone()
        changed[:, 1] = torch.tensor([[7, 8, 9], [1, 2, 3]])
        first = self._forward().logits
        second = self._forward(changed).logits
        for step in range(3):
            torch.testing.assert_close(first[step], second[step])

    def test_greedy_generation_resets_trie_for_each_item(self):
        self.model.eval()
        generated = self.model.generate(
            self.history,
            self.padding,
            self.trie,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.behavior_instructions,
            output_items=2,
        )
        self.assertEqual(tuple(generated.shape), (2, 6))
        for row in generated.tolist():
            self.assertTrue(self.trie.contains(row[:3]))
            self.assertTrue(self.trie.contains(row[3:]))

    def test_beam_generation_returns_legal_two_item_paths(self):
        self.model.eval()
        output = self.model.beam_search(
            self.history,
            self.padding,
            self.trie,
            beam_width=2,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.behavior_instructions,
            output_items=2,
        )
        self.assertEqual(tuple(output.semantic_ids.shape), (2, 2, 6))
        for ranking in output.semantic_ids.tolist():
            for path in ranking:
                self.assertTrue(self.trie.contains(path[:3]))
                self.assertTrue(self.trie.contains(path[3:]))

    def test_candidate_log_probs_accepts_flattened_lists(self):
        candidates = torch.stack(
            (self.targets.reshape(2, 6), self.targets.flip(1).reshape(2, 6)), dim=1
        )
        scores = self.model.candidate_log_probs(
            self.history,
            self.padding,
            candidates,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.behavior_instructions,
        )
        self.assertEqual(tuple(scores.shape), (2, 2, 6))
        self.assertTrue(torch.isfinite(scores).all())

    def test_rejects_more_items_than_configured(self):
        too_many = torch.zeros(2, 3, 3, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "max_target_items"):
            self.model(
                self.history,
                self.padding,
                target_sids=too_many,
                history_behavior_ids=self.history_behaviors,
                behavior_instruction_ids=self.behavior_instructions,
            )


if __name__ == "__main__":
    unittest.main()
