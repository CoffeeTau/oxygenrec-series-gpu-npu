"""Contract tests for GPU/NPU performance-baseline comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_v2_performance_baselines.py"
spec = importlib.util.spec_from_file_location("compare_v2_performance_baselines", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(platform: str, throughput: float) -> dict:
    return {
        "platform": platform,
        "protocol": "oxygenrec_v2_single_device_training_performance_v1",
        "precision": "bf16",
        "device_name": platform,
        "inputs": {"checkpoint_sha256": "same"},
        "workload": {"batch_sizes": [64]},
        "source_files_sha256": {"model.py": "same"},
        "source": {"commit": "same"},
        "aggregates": [{
            "batch_size": 64,
            "samples_per_second_median": throughput,
            "mean_step_seconds_median": 64 / throughput,
            "peak_memory_allocated_bytes_max": 100,
            "samples_per_second_cv": 0.01,
        }],
    }


class CompareV2PerformanceBaselinesTests(unittest.TestCase):
    def test_reports_npu_to_gpu_ratio(self) -> None:
        result = module.compare(payload("gpu", 2000.0), payload("npu", 1500.0))
        self.assertEqual(result["rows"][0]["npu_to_gpu_throughput_ratio"], 0.75)

    def test_rejects_different_workload(self) -> None:
        npu = payload("npu", 1500.0)
        npu["workload"] = {"batch_sizes": [128]}
        with self.assertRaisesRegex(ValueError, "workload"):
            module.compare(payload("gpu", 2000.0), npu)


if __name__ == "__main__":
    unittest.main()
