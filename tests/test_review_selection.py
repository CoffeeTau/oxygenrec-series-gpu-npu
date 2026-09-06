import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.review_selection import select_representative_rows


def row(key, *, behavior="view", cosine=0.0, eligible=False, hit=False,
        browse_only=False, beam_rank=None, history=30):
    return {
        "sample_key": key,
        "target_behavior": behavior,
        "q2i_cosine": cosine,
        "repeat_eligible": eligible,
        "igr_hit": hit,
        "browse_only": browse_only,
        "beam_hit_rank": beam_rank,
        "history_length": history,
    }


class RepresentativeSelectionTest(unittest.TestCase):
    def test_covers_positive_negative_and_behavior_roles(self):
        selected, coverage = select_representative_rows([
            row("a", eligible=True, hit=True, cosine=0.4),
            row("b", eligible=True, hit=False, cosine=-0.3),
            row("c", behavior="transaction", cosine=0.1, history=120),
            row("d", behavior="addtocart", cosine=0.2),
            row("e", browse_only=True, cosine=0.0),
            row("f", cosine=0.9, beam_rank=2),
        ])
        self.assertEqual(coverage["igr_hit"], "a")
        self.assertEqual(coverage["igr_miss"], "b")
        self.assertEqual(coverage["transaction_target"], "c")
        self.assertEqual(coverage["addtocart_target"], "d")
        self.assertEqual(coverage["q2i_best"], "f")
        self.assertEqual(coverage["q2i_worst"], "b")
        self.assertEqual(coverage["beam_hit"], "f")
        self.assertLess(len(selected), 8)

    def test_marks_unavailable_role_instead_of_inventing_case(self):
        _, coverage = select_representative_rows([
            row("only", browse_only=True, cosine=0.1),
        ])
        self.assertIsNone(coverage["igr_hit"])
        self.assertIsNone(coverage["transaction_target"])
        self.assertIsNone(coverage["beam_hit"])


if __name__ == "__main__":
    unittest.main()
