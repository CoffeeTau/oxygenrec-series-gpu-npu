#!/usr/bin/env python3
"""End-to-end CUDA check: constrained beam -> rewards -> SA-GCPO update."""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from oxygenrec.alignment import sa_gcpo_loss
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.rewards import map_public_rewards
from oxygenrec.sid import PrefixTrie


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(17)
    config = OxygenRECConfig(
        sid_width=12, hidden_size=24, attention_heads=4,
        encoder_layers=1, decoder_layers=1, feedforward_size=48,
        dropout=0.0, max_history_items=3,
    )
    policy = OxygenRECModel(config).to(device)
    old_policy = copy.deepcopy(policy).eval()
    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)
    history = torch.tensor([[[1,2,3],[4,5,6]], [[2,3,4],[5,6,7]]], device=device)
    padding = torch.zeros(2, 2, dtype=torch.bool, device=device)
    targets = torch.tensor([[7,8,9],[1,2,3]], device=device)
    trie = PrefixTrie([(1,2,3),(1,4,5),(7,8,9),(7,10,11)])
    candidates = old_policy.beam_search(history, padding, trie, beam_width=4).semantic_ids
    legal = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    relative = (candidates == targets[:, None]).to(torch.float32).mean(dim=-1)
    ranking = (candidates == targets[:, None]).all(dim=-1).to(torch.float32)
    mapped = map_public_rewards(
        candidates, legal_mask=legal, relative_scores=relative,
        ranking_scores=ranking,
    )
    target_rewards = mapped.total.max(dim=1).values
    with torch.no_grad():
        old_log_probs = old_policy.candidate_log_probs(history, padding, candidates)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    objectives = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        current_log_probs = policy.candidate_log_probs(history, padding, candidates)
        output = sa_gcpo_loss(
            current_log_probs, old_log_probs, mapped.total, target_rewards,
            tau_positive=2.0, tau_negative=5.0,
        )
        output.loss.backward()
        optimizer.step()
        objectives.append(float(output.objective.detach()))
    if not objectives[-1] > objectives[0]:
        raise RuntimeError("rollout objective did not improve")
    print(
        f"OK device={device} beams={tuple(candidates.shape)} "
        f"reward_min={float(mapped.total.min()):.6f} "
        f"reward_max={float(mapped.total.max()):.6f} "
        f"objective={objectives[0]:.6f}->{objectives[-1]:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
