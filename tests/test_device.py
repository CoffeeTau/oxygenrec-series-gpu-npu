import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class DeviceRuntimeTest(unittest.TestCase):
    def test_cpu_resolution_and_seed(self):
        from oxygenrec.device import resolve_device, seed_torch

        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        seed_torch(19, device)
        first = torch.rand(3)
        seed_torch(19, device)
        torch.testing.assert_close(first, torch.rand(3))

    def test_unsupported_device_is_rejected_before_torch_parsing(self):
        from oxygenrec.device import resolve_device

        with self.assertRaisesRegex(ValueError, "unsupported device type"):
            resolve_device("tpu:0")

    def test_npu_import_failure_is_actionable(self):
        from oxygenrec.device import resolve_device

        with mock.patch("importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "torch_npu cannot be imported"):
                resolve_device("npu:0")


if __name__ == "__main__":
    unittest.main()
