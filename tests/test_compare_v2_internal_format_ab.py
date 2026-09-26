"""Contract tests for the NPU internal-format performance A/B comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_v2_internal_format_ab.py"
spec = importlib.util.spec_from_file_location("compare_v2_internal_format_ab", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(internal_format: str, throughput: float) -> dict:
    return {
        "status": "passed",
        "platform": "npu",
        "device": "npu:0",
        "device_name": "Ascend950DT_9572",
        "protocol": "oxygenrec_v2_single_device_training_performance_v1",
        "precision": "bf16",
        "inputs": {"checkpoint_sha256": "same"},
        "workload": {
            "batch_sizes": [4096],
            "optimizer": "AdamW",
            "warmup_steps": 100,
            "zero_grad_mode": "set_to_none",
            "npu_internal_format": internal_format,
        },
        "source_files_sha256": {"model.py": "same"},
        "source": {"commit": "same", "tracked_dirty": False},
        "runs": [{
            "batch_size": 4096,
            "all_losses_finite": True,
            "first_loss": 5.4,
            "last_loss": 5.2,
            "mean_loss": 5.3,
        }],
        "aggregates": [{
            "batch_size": 4096,
            "samples_per_second_median": throughput,
            "mean_step_seconds_median": 4096 / throughput,
            "peak_memory_allocated_bytes_max": 100,
            "samples_per_second_cv": 0.01,
        }],
    }


class CompareV2InternalFormatABTests(unittest.TestCase):
    def test_reports_treatment_ratio(self) -> None:
        result = module.compare(
            payload("disable", 20_000.0),
            payload("enable", 25_000.0),
        )
        self.assertEqual(result["rows"][0]["treatment_to_control_throughput_ratio"], 1.25)
        self.assertEqual(result["factor"], "npu_internal_format")
        self.assertEqual(result["optimizer"], "AdamW")

    def test_rejects_workload_difference_beyond_factor(self) -> None:
        treatment = payload("enable", 25_000.0)
        treatment["workload"]["warmup_steps"] = 20
        with self.assertRaisesRegex(ValueError, "beyond npu_internal_format"):
            module.compare(payload("disable", 20_000.0), treatment)

    def test_requires_explicit_control_and_treatment_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "control npu_internal_format"):
            module.compare(payload("default", 20_000.0), payload("enable", 25_000.0))

    def test_rejects_nonfinite_loss_evidence(self) -> None:
        treatment = payload("enable", 25_000.0)
        treatment["runs"][0]["all_losses_finite"] = False
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.compare(payload("disable", 20_000.0), treatment)


if __name__ == "__main__":
    unittest.main()
