import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class BehaviorInstructionTest(unittest.TestCase):
    def setUp(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(23)
        self.model = OxygenRECModel(OxygenRECConfig(
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
        ))
        self.history = torch.tensor([
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
        ])
        self.padding = torch.tensor([
            [False, False, True],
            [False, False, True],
        ])
        self.history_behaviors = torch.tensor([[0, 1, 0], [0, 1, 0]])
        self.targets = torch.tensor([[1, 2, 3], [7, 8, 9]])
        self.target_behaviors = torch.tensor([0, 2])

    def test_target_behavior_changes_decoder_but_not_encoder(self):
        """只切换 I_b 时，Encoder memory 不变而 Decoder logits 改变。"""
        self.model.eval()
        memory_before = self.model._encode(
            self.history, self.padding, self.history_behaviors
        )
        first = self.model(
            self.history,
            self.padding,
            target_sids=self.targets,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=torch.tensor([0, 0]),
        ).logits
        memory_after = self.model._encode(
            self.history, self.padding, self.history_behaviors
        )
        second = self.model(
            self.history,
            self.padding,
            target_sids=self.targets,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=torch.tensor([2, 2]),
        ).logits

        torch.testing.assert_close(memory_before, memory_after)
        self.assertTrue(
            any(not torch.equal(left, right) for left, right in zip(first, second))
        )

    def test_backward_reaches_behavior_projection_and_reserved_tokens(self):
        """NTP 梯度应同时到达两层 psi 和 SID 词表上方的行为 token 行。"""
        output = self.model(
            self.history,
            self.padding,
            target_sids=self.targets,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.target_behaviors,
        )
        output.loss.backward()

        adapter_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.model.behavior_instruction_adapter.parameters()
            if parameter.grad is not None
        )
        embedding_grad = self.model.sid_embeddings[0].weight.grad
        self.assertGreater(adapter_grad, 0.0)
        self.assertGreater(
            embedding_grad[self.model.config.sid_width :].abs().sum().item(), 0.0
        )
        self.assertEqual(self.model.sid_embeddings[0].num_embeddings, 14)
        self.assertEqual(self.model.sid_embeddings[1].num_embeddings, 11)

    def test_all_decoding_interfaces_accept_behavior_instruction(self):
        from oxygenrec.sid import PrefixTrie

        self.model.eval()
        trie = PrefixTrie([(1, 2, 3), (7, 8, 9)])
        generated = self.model.generate(
            self.history,
            self.padding,
            trie,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.target_behaviors,
        )
        beams = self.model.beam_search(
            self.history,
            self.padding,
            trie,
            beam_width=2,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.target_behaviors,
        )
        candidates = torch.tensor([
            [[1, 2, 3], [7, 8, 9]],
            [[1, 2, 3], [7, 8, 9]],
        ])
        log_probs = self.model.candidate_log_probs(
            self.history,
            self.padding,
            candidates,
            history_behavior_ids=self.history_behaviors,
            behavior_instruction_ids=self.target_behaviors,
        )

        self.assertEqual(tuple(generated.shape), (2, 3))
        self.assertEqual(tuple(beams.semantic_ids.shape), (2, 2, 3))
        self.assertEqual(tuple(log_probs.shape), (2, 2, 3))
        self.assertTrue(all(trie.contains(row) for row in generated.tolist()))

    def test_v2_configuration_requires_valid_behavior_ids(self):
        with self.assertRaisesRegex(ValueError, "are required"):
            self.model(self.history, self.padding, target_sids=self.targets)
        with self.assertRaisesRegex(ValueError, "unknown behavior"):
            self.model(
                self.history,
                self.padding,
                target_sids=self.targets,
                behavior_instruction_ids=torch.tensor([0, 3]),
            )


if __name__ == "__main__":
    unittest.main()
