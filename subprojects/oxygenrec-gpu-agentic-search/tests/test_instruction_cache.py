import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.data import Behavior, InteractionEvent, NextItemSample, Split
from oxygenrec.instruction_cache import (
    instruction_sample_key,
    load_instruction_feature_cache,
    save_instruction_feature_cache,
)

try:
    import torch
except ImportError:
    torch = None


def sample(source_row: int, *, split: Split = Split.TRAIN) -> NextItemSample:
    history = InteractionEvent(10, source_row - 1, "private-user", "a", Behavior.VIEW)
    target = InteractionEvent(20, source_row, "private-user", "b", Behavior.VIEW)
    return NextItemSample(split, "private-user", (history,), target)


class InstructionSampleKeyTest(unittest.TestCase):
    def test_key_is_stable_and_does_not_expose_user_id(self):
        key = instruction_sample_key(sample(42, split=Split.VALIDATION))
        self.assertEqual(key, "validation:42")
        self.assertNotIn("private-user", key)


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class InstructionFeatureCacheTest(unittest.TestCase):
    def test_round_trip_preserves_rows_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
            save_instruction_feature_cache(
                path,
                sample_keys=("train:2", "validation:9"),
                features=features,
                metadata={"sid_registry_version": "rq-v1"},
            )
            loaded, index, metadata = load_instruction_feature_cache(path)

        torch.testing.assert_close(loaded, features)
        self.assertEqual(index, {"train:2": 0, "validation:9": 1})
        self.assertEqual(metadata["sid_registry_version"], "rq-v1")

    def test_rejects_duplicate_keys_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            features = torch.ones(2, 3)
            with self.assertRaisesRegex(ValueError, "unique"):
                save_instruction_feature_cache(
                    path,
                    sample_keys=("train:2", "train:2"),
                    features=features,
                    metadata={},
                )
            save_instruction_feature_cache(
                path,
                sample_keys=("train:2", "train:3"),
                features=features,
                metadata={},
            )
            with self.assertRaises(FileExistsError):
                save_instruction_feature_cache(
                    path,
                    sample_keys=("train:2", "train:3"),
                    features=features,
                    metadata={},
                )


if __name__ == "__main__":
    unittest.main()
