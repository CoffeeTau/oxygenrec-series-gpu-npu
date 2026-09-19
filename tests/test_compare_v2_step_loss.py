"""Contract tests for paired, per-step loss comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_v2_step_loss.py"
spec = importlib.util.spec_from_file_location("compare_v2_step_loss", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(platform: str, losses: list[float]) -> dict:
    return {
        "platform": platform,
        "protocol": "oxygenrec_v2_matched_step_diagnostic_v1",
        "precision": "bf16",
        "dropout_mode": "disabled",
        "configured_dropout": 0.1,
        "inputs": {"checkpoint_sha256": "same"},
        "seed": 17,
        "epoch": 1,
        "train_samples": 50_000,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "optimizer_state_loaded": True,
        "source": {"commit": "same"},
        "source_files_sha256": {"model.py": "same"},
        "steps": [
            {
                "step": index,
                "batch_sha256": f"batch-{index}",
                "loss_before_update": loss,
                "gradient_l2": 1.0,
            }
            for index, loss in enumerate(losses, start=1)
        ],
    }


class CompareV2StepLossTests(unittest.TestCase):
    def test_reports_first_step_above_each_tolerance(self) -> None:
        result = module.compare(
            payload("gpu", [5.0, 4.0]),
            payload("npu", [5.0005, 4.02]),
            absolute_tolerance=1e-3,
            relative_tolerance=1e-3,
        )
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["first_above_absolute_tolerance"]["step"], 2)
        self.assertEqual(result["first_above_relative_tolerance"]["step"], 2)

    def test_rejects_different_actual_batch(self) -> None:
        npu = payload("npu", [5.0])
        npu["steps"][0]["batch_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "actual batch differs"):
            module.compare(payload("gpu", [5.0]), npu, 1e-3, 1e-3)


if __name__ == "__main__":
    unittest.main()
