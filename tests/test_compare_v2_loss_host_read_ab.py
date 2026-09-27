"""Contract tests for deferred loss host-read performance comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_v2_loss_host_read_ab.py"
spec = importlib.util.spec_from_file_location("compare_v2_loss_host_read_ab", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(mode: str, materializations: int, throughput: float) -> dict:
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
            "loss_host_read": {
                "mode": mode,
                "timed_materializations_per_step": materializations,
                "post_timing_full_loss_check": True,
            },
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


class CompareV2LossHostReadABTests(unittest.TestCase):
    def test_reports_treatment_ratio_and_matching_losses(self) -> None:
        result = module.compare(
            payload("per_step", 1, 20_000.0),
            payload("deferred", 0, 25_000.0),
        )
        self.assertEqual(result["rows"][0]["treatment_to_control_throughput_ratio"], 1.25)
        self.assertTrue(result["rows"][0]["loss_medians_exactly_match"])

    def test_rejects_non_factor_workload_difference(self) -> None:
        treatment = payload("deferred", 0, 25_000.0)
        treatment["workload"]["warmup_steps"] = 20
        with self.assertRaisesRegex(ValueError, "beyond loss_host_read"):
            module.compare(payload("per_step", 1, 20_000.0), treatment)

    def test_rejects_wrong_treatment_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not materialize"):
            module.compare(
                payload("per_step", 1, 20_000.0),
                payload("deferred", 1, 25_000.0),
            )

    def test_rejects_nonfinite_loss_evidence(self) -> None:
        treatment = payload("deferred", 0, 25_000.0)
        treatment["runs"][0]["all_losses_finite"] = False
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.compare(payload("per_step", 1, 20_000.0), treatment)


if __name__ == "__main__":
    unittest.main()
