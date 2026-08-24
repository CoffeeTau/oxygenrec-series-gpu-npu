import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TorchResidualKMeansTest(unittest.TestCase):
    def test_fit_encode_reconstruct_registry_and_round_trip(self):
        from oxygenrec.quantization_torch import TorchResidualKMeans, TorchResidualKMeansModel

        generator = torch.Generator().manual_seed(3)
        vectors = torch.randn(64, 8, generator=generator)
        fitter = TorchResidualKMeans(
            levels=3, width=4, max_iterations=10, seed=5, assignment_chunk_size=13
        )
        model = fitter.fit(vectors)
        codes = model.encode(vectors, chunk_size=11)
        reconstruction = model.reconstruct(codes)
        self.assertEqual(tuple(codes.shape), (64, 3))
        self.assertLess(
            torch.mean((vectors - reconstruction) ** 2), torch.mean(vectors**2)
        )
        registry = model.registry_for(
            [f"item-{index}" for index in range(64)], vectors, version="toy"
        )
        self.assertEqual(len(registry.item_to_sid), 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codebooks.pt"
            model.save(path, version="toy")
            version, restored = TorchResidualKMeansModel.load(
                path, expected_version="toy"
            )
        self.assertEqual(version, "toy")
        torch.testing.assert_close(restored.codebooks, model.codebooks)

    def test_cpu_fit_is_seed_deterministic(self):
        from oxygenrec.quantization_torch import TorchResidualKMeans

        vectors = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        settings = dict(levels=2, width=3, max_iterations=5, seed=7)
        first = TorchResidualKMeans(**settings).fit(vectors)
        second = TorchResidualKMeans(**settings).fit(vectors)
        torch.testing.assert_close(first.codebooks, second.codebooks)

    def test_kmeans_plus_plus_is_seed_deterministic(self):
        from oxygenrec.quantization_torch import TorchResidualKMeans

        vectors = torch.randn(32, 5, generator=torch.Generator().manual_seed(4))
        settings = dict(
            levels=2,
            width=4,
            max_iterations=4,
            seed=12,
            initialization="kmeans++",
        )
        first = TorchResidualKMeans(**settings).fit(vectors)
        second = TorchResidualKMeans(**settings).fit(vectors)
        torch.testing.assert_close(first.codebooks, second.codebooks)
