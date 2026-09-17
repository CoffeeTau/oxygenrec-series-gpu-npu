#!/usr/bin/env python3
"""Exercise Phase-1 forward/backward, overfit, checkpoint, and decoding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(17)
    device = torch.device(args.device)
    config = OxygenRECConfig(
        sid_width=11,
        hidden_size=32,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_size=64,
        dropout=0.0,
        max_history_items=3,
    )
    model = OxygenRECModel(config).to(device)
    history = torch.tensor(
        [[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]],
        device=device,
    )
    padding = torch.tensor(
        [[False, False, True], [False, False, False]], device=device
    )
    targets = torch.tensor([[1, 2, 3], [7, 8, 9]], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    initial_loss = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(history, padding, target_sids=targets)
        if initial_loss is None:
            initial_loss = float(output.loss.detach())
        output.loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = model(history, padding, target_sids=targets)
    final_loss = float(final.loss)
    if final_loss >= initial_loss * 0.2:
        raise RuntimeError(
            f"toy batch did not overfit enough: {initial_loss:.6f} -> {final_loss:.6f}"
        )

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "toy.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = OxygenRECModel(config).to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        restored.eval()
        with torch.no_grad():
            restored_logits = restored(history, padding, target_sids=targets).logits
        for expected, actual in zip(final.logits, restored_logits):
            torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

    trie = PrefixTrie([(1, 2, 3), (1, 4, 5), (7, 8, 9)])
    generated = restored.generate(history, padding, trie)
    if not all(trie.contains(row) for row in generated.tolist()):
        raise RuntimeError("constrained generation emitted an invalid SID")
    beams = restored.beam_search(history, padding, trie, beam_width=2)
    if not all(
        trie.contains(path)
        for ranking in beams.semantic_ids.tolist()
        for path in ranking
    ):
        raise RuntimeError("constrained beam search emitted an invalid SID")
    print(
        f"OK device={device} loss={initial_loss:.6f}->{final_loss:.6f} "
        f"generated={generated.tolist()} beams={beams.semantic_ids.tolist()}"
    )


if __name__ == "__main__":
    main()
