import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class MultimodalItemEncoderTest(unittest.TestCase):
    def test_shapes_gradients_and_both_modalities_affect_output(self):
        from oxygenrec.multimodal import MultimodalItemEncoder

        torch.manual_seed(17)
        model = MultimodalItemEncoder(
            text_size=12, image_size=10, hidden_size=16, query_tokens=4,
            qformer_layers=2, attention_heads=4, output_size=8,
        )
        text = torch.randn(3, 5, 12, requires_grad=True)
        image = torch.randn(3, 2, 10, requires_grad=True)
        output = model(text, image)
        self.assertEqual(tuple(output.item_embedding.shape), (3, 8))
        self.assertEqual(tuple(output.query_tokens.shape), (3, 4, 16))
        output.item_embedding.square().mean().backward()
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(image.grad)
        with torch.no_grad():
            baseline = model(text, image).item_embedding
            text_changed = model(text + 1.0, image).item_embedding
            image_changed = model(text, image + 1.0).item_embedding
        self.assertFalse(torch.equal(baseline, text_changed))
        self.assertFalse(torch.equal(baseline, image_changed))

    def test_rejects_mismatched_batches(self):
        from oxygenrec.multimodal import MultimodalItemEncoder

        model = MultimodalItemEncoder(
            text_size=4, image_size=4, hidden_size=8, attention_heads=2,
            query_tokens=2, qformer_layers=1, output_size=4,
        )
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            model(torch.randn(2, 1, 4), torch.randn(3, 1, 4))
