"""Contract tests for eager versus TorchAir NPU performance comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_v2_execution_mode_ab.py"
spec = importlib.util.spec_from_file_location("compare_v2_execution_mode_ab", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(mode: str, throughput: float) -> dict:
    execution = {
        "mode": mode,
        "backend": (
            None if mode == "eager"
            else "torch_npu.dynamo.torchair.get_npu_backend"
        ),
        "dynamic": None if mode == "eager" else False,
        "loss_only_wrapper": True,
    }
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
            "execution": execution,
        },
        "source_files_sha256": {"model.py": "same"},
        "source": {"commit": "same", "tracked_dirty": False},
        "runs": [{
            "batch_size": 4096,
            "warmup_elapsed_seconds": 20.0 if mode == "eager" else 50.0,
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


class CompareV2ExecutionModeABTests(unittest.TestCase):
    def test_reports_ratio_and_compile_warmup(self) -> None:
        result = module.compare(payload("eager", 20_000.0), payload("torchair", 25_000.0))
        row = result["rows"][0]
        self.assertEqual(row["treatment_to_control_throughput_ratio"], 1.25)
        self.assertEqual(row["control_warmup_elapsed_seconds_median"], 20.0)
        self.assertEqual(row["treatment_warmup_elapsed_seconds_median"], 50.0)
        self.assertEqual(row["treatment_warmup_elapsed_seconds_by_repeat"], [50.0])

    def test_rejects_non_factor_workload_difference(self) -> None:
        treatment = payload("torchair", 25_000.0)
        treatment["workload"]["warmup_steps"] = 20
        with self.assertRaisesRegex(ValueError, "beyond execution"):
            module.compare(payload("eager", 20_000.0), treatment)

    def test_rejects_wrong_backend(self) -> None:
        treatment = payload("torchair", 25_000.0)
        treatment["workload"]["execution"]["backend"] = "wrong"
        with self.assertRaisesRegex(ValueError, "expected TorchAir backend"):
            module.compare(payload("eager", 20_000.0), treatment)

    def test_rejects_nonfinite_loss_evidence(self) -> None:
        treatment = payload("torchair", 25_000.0)
        treatment["runs"][0]["all_losses_finite"] = False
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.compare(payload("eager", 20_000.0), treatment)


if __name__ == "__main__":
    unittest.main()
