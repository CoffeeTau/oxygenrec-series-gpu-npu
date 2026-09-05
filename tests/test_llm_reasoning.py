import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.llm_reasoning import contextual_instruction_text, parse_reasoning_json


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

    def test_reports_truncated_nested_json_as_invalid(self):
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            parse_reasoning_json(
                '{"intent":"x","evidence":["view=41"],'
                '"retrieval_strategy":"y","retrieval_plan":'
                '{"priority_behaviors":["view"]}'
            )

    def test_ignores_text_after_first_complete_object(self):
        parsed = parse_reasoning_json(
            '{"intent":"x","evidence":["view=41"],'
            '"retrieval_strategy":"y",'
            '"retrieval_plan":{"priority_behaviors":["view"],'
            '"recency":"recent","prefer_repeated_items":false,'
            '"diversity":"high"},"constraints":["不猜目标"]}'
            ' trailing {not json}'
        )
        self.assertEqual(parsed["retrieval_plan"]["recency"], "recent")

    def test_paper_instruction_text_does_not_consume_agentic_plan(self):
        parsed = parse_reasoning_json(
            '{"intent":"识别长期兴趣","evidence":["view=41","重复商品=2"],'
            '"retrieval_strategy":"检索历史兴趣商品",'
            '"retrieval_plan":{"priority_behaviors":["view"],'
            '"recency":"recent","prefer_repeated_items":false,'
            '"diversity":"high"},"constraints":["不猜目标"]}'
        )
        first = contextual_instruction_text(parsed)
        parsed["retrieval_plan"]["recency"] = "long_term"
        parsed["retrieval_plan"]["diversity"] = "low"
        second = contextual_instruction_text(parsed)
        self.assertEqual(first, second)
        self.assertIn("当前意图：识别长期兴趣", first)
        self.assertIn("推理依据：view=41；重复商品=2", first)
        self.assertNotIn("recent", first)


if __name__ == "__main__":
    unittest.main()
