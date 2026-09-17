import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.data.events import Behavior, InteractionEvent
from oxygenrec.data.temporal import NextItemSample, Split
from oxygenrec.sft_sampling import evidence_cohort, select_stratified_sft_samples


def sample(index, behaviors, repeated=False):
    history = []
    for offset, behavior in enumerate(behaviors):
        history.append(InteractionEvent(
            timestamp_ms=index * 100 + offset, source_row=index * 10 + offset,
            user_id=f"u{index}", item_id="repeat" if repeated else f"i{index}-{offset}",
            behavior=Behavior(behavior),
        ))
    target = InteractionEvent(
        timestamp_ms=index * 100 + 99, source_row=index * 10 + 9,
        user_id=f"u{index}", item_id=f"target-{index}", behavior=Behavior.VIEW,
    )
    return NextItemSample(Split.TRAIN, f"u{index}", tuple(history), target)


class SFTSamplingTest(unittest.TestCase):
    def test_cohort_uses_history_only(self):
        self.assertEqual(evidence_cohort(sample(1, ["view", "transaction"])), "transaction_history")
        self.assertEqual(evidence_cohort(sample(2, ["view", "addtocart"])), "cart_history")
        self.assertEqual(evidence_cohort(sample(3, ["view", "view"], True)), "repeat_view")
        self.assertEqual(evidence_cohort(sample(4, ["view"])), "browse_only")

    def test_balances_available_cohorts(self):
        samples = []
        for index in range(4):
            samples.extend([
                sample(index * 10 + 1, ["view", "transaction"]),
                sample(index * 10 + 2, ["view", "addtocart"]),
                sample(index * 10 + 3, ["view", "view"], True),
                sample(index * 10 + 4, ["view"]),
            ])
        selected, counts = select_stratified_sft_samples(samples, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(set(counts.values()), {2})


if __name__ == "__main__":
    unittest.main()
