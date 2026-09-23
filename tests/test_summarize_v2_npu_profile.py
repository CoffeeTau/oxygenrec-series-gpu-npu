"""Tests for the version-tolerant Ascend profiler CSV summarizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_v2_npu_profile.py"
spec = importlib.util.spec_from_file_location("summarize_v2_npu_profile", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummarizeV2NpuProfileTests(unittest.TestCase):
    def test_summarizes_operators_and_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator_details.csv"
            path.write_text(
                "Name,Call Count,Device Self Duration(us),Host Self Duration(us)\n"
                "aten::matmul,2,30,4\n"
                "aten::matmul,1,10,2\n"
                "aten::_local_scalar_dense,3,5,20\n",
                encoding="utf-8",
            )

            result = module.summarize_operator(path, top_k=10)

            self.assertEqual(result["top"][0]["name"], "aten::matmul")
            self.assertEqual(result["top"][0]["calls"], 3)
            keyword = next(
                row for row in result["keyword_summary"]
                if row["keyword"] == "_local_scalar_dense"
            )
            self.assertEqual(keyword["calls"], 3)
            self.assertEqual(keyword["primary_duration"], 5.0)

    def test_summarizes_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_details.csv"
            path.write_text(
                "Name,Type,Duration(us)\n"
                "CubeKernel,AI_CORE,10\n"
                "CubeKernel,AI_CORE,15\n"
                "TransData,AI_CORE,8\n",
                encoding="utf-8",
            )

            result = module.summarize_kernel(path, top_k=10)

            self.assertEqual(result["top"][0]["name"], "CubeKernel")
            self.assertEqual(result["top"][0]["calls"], 2)
            self.assertEqual(result["top"][0]["duration"], 25.0)
            keyword = next(
                row for row in result["keyword_summary"]
                if row["keyword"] == "transdata"
            )
            self.assertEqual(keyword["duration"], 8.0)


if __name__ == "__main__":
    unittest.main()
