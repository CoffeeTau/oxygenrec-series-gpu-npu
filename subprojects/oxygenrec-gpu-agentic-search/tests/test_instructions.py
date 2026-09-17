import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.instructions import (
    ContextualInstruction, InstructionStore, build_history_instruction, hash_instruction,
)


class ContextualInstructionTest(unittest.TestCase):
    def test_hash_is_deterministic_normalized_and_text_sensitive(self):
        first = hash_instruction("最近有购买行为，优先检索相关商品。", 32)
        second = hash_instruction("最近有购买行为，优先检索相关商品。", 32)
        different = hash_instruction("探索新的商品类别。", 32)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)

    def test_store_round_trip_preserves_provenance(self):
        record = ContextualInstruction(
            sample_id="review-001", text="优先检索高意图历史商品",
            evidence=("transaction=1", "addtocart=2"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instructions.jsonl"
            InstructionStore.save(path, [record])
            self.assertEqual(InstructionStore.load(path), [record])

    def test_history_instruction_uses_only_prior_behaviors(self):
        text, evidence = build_history_instruction(
            ["view", "view", "view", "addtocart"]
        )
        self.assertIn("高意图", text)
        self.assertIn("recent_high_intent=1", evidence)
        with self.assertRaises(ValueError):
            build_history_instruction([])


if __name__ == "__main__":
    unittest.main()
