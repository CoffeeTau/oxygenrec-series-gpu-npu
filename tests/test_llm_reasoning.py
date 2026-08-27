import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.llm_reasoning import parse_reasoning_json


class ReasoningJSONTest(unittest.TestCase):
    def test_parses_required_schema_with_surrounding_text(self):
        parsed = parse_reasoning_json(
            'prefix {"intent":"复购", "evidence":["购买=1"], '
            '"retrieval_strategy":"检索重复商品", '
            '"retrieval_plan":{"priority_behaviors":["transaction"],'
            '"recency":"balanced","prefer_repeated_items":true,"diversity":"low"},'
            '"constraints":["不猜目标"]} suffix'
        )
        self.assertEqual(parsed["intent"], "复购")

    def test_rejects_missing_or_wrong_fields(self):
        with self.assertRaises(ValueError):
            parse_reasoning_json('{"intent":"x"}')
        with self.assertRaises(ValueError):
            parse_reasoning_json(
                '{"intent":"x","evidence":"bad",'
                '"retrieval_strategy":"y","retrieval_plan":{},"constraints":[]}'
            )


if __name__ == "__main__":
    unittest.main()
