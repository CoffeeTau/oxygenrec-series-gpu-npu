import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.sid import PrefixTrie, SIDRegistry, SemanticID


class SemanticIDTest(unittest.TestCase):
    def test_validates_level_count_and_range(self):
        self.assertEqual(SemanticID((1, 2, 3)).codes, (1, 2, 3))
        with self.assertRaises(ValueError):
            SemanticID((1, 2))
        with self.assertRaises(ValueError):
            SemanticID((1, 2, 8192))


class SIDRegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = SIDRegistry(
            {
                "item-a": (1, 2, 3),
                "item-b": (1, 2, 4),
                "item-c": (1, 2, 3),
                "item-d": (7, 8, 9),
            },
            version="toy-v1",
        )

    def test_preserves_collisions(self):
        self.assertEqual(
            self.registry.items_for((1, 2, 3)), ("item-a", "item-c")
        )
        self.assertEqual(len(self.registry.collisions()), 1)
        self.assertEqual(self.registry.collision_rate(), 0.5)

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sid_registry.json"
            self.registry.to_json(path)
            restored = SIDRegistry.from_json(path)
        self.assertEqual(restored.version, "toy-v1")
        self.assertEqual(restored.item_to_sid, self.registry.item_to_sid)


class PrefixTrieTest(unittest.TestCase):
    def setUp(self):
        registry = SIDRegistry(
            {"a": (1, 2, 3), "b": (1, 2, 4), "c": (1, 5, 0), "d": (7, 8, 9)}
        )
        self.trie = PrefixTrie.from_registry(registry)

    def test_allowed_tokens_follow_only_real_item_paths(self):
        self.assertEqual(self.trie.allowed_next(()), (1, 7))
        self.assertEqual(self.trie.allowed_next((1,)), (2, 5))
        self.assertEqual(self.trie.allowed_next((1, 2)), (3, 4))
        self.assertEqual(self.trie.allowed_next((9,)), ())

    def test_distinguishes_prefix_from_complete_sid(self):
        self.assertTrue(self.trie.is_valid_prefix((1, 2)))
        self.assertFalse(self.trie.contains((1, 2)))
        self.assertTrue(self.trie.contains((1, 2, 3)))
        self.assertFalse(self.trie.contains((1, 2, 8)))


if __name__ == "__main__":
    unittest.main()
