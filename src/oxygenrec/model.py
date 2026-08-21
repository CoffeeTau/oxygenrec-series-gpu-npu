"""Minimal dense encoder-decoder for the Phase-1 OxygenREC loop.

The model consumes already-tokenized item Semantic IDs. Dataset code remains
responsible for mapping item IDs through a versioned ``SIDRegistry`` so model
checkpoints never silently select a different codebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .sid import PrefixTrie


@dataclass(frozen=True)
class OxygenRECConfig:
    """Explicit engineering choices for the first small dense model."""

    sid_width: int
    sid_levels: int = 3
    instruction_vocab_size: int = 1
    hidden_size: int = 128
    attention_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_size: int = 512
    dropout: float = 0.1
    max_history_items: int = 256

    def __post_init__(self) -> None:
        positive = {
            "sid_width": self.sid_width,
            "sid_levels": self.sid_levels,
            "instruction_vocab_size": self.instruction_vocab_size,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "feedforward_size": self.feedforward_size,
            "max_history_items": self.max_history_items,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class OxygenRECOutput:
    """Per-level logits and optional weighted next-token loss."""

    logits: tuple[Tensor, ...]
    loss: Tensor | None = None
    level_losses: tuple[Tensor, ...] | None = None


@dataclass(frozen=True)
class BeamSearchOutput:
    """Ranked legal SID paths and cumulative log-probability scores."""

    semantic_ids: Tensor  # [batch, beam, levels]
    scores: Tensor  # [batch, beam]


class OxygenRECModel(nn.Module):
    """Small Transformer encoder-decoder with one prediction head per SID level."""

    def __init__(self, config: OxygenRECConfig) -> None:
        super().__init__()
        self.config = config
        self.sid_embeddings = nn.ModuleList(
            nn.Embedding(config.sid_width, config.hidden_size)
            for _ in range(config.sid_levels)
        )
        self.history_positions = nn.Embedding(
            config.max_history_items, config.hidden_size
        )
        self.instruction_embeddings = nn.Embedding(
            config.instruction_vocab_size, config.hidden_size
        )
        self.bos_embedding = nn.Parameter(torch.empty(config.hidden_size))
        self.decoder_positions = nn.Embedding(
            config.sid_levels + 1, config.hidden_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.encoder_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, config.decoder_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        self.prediction_heads = nn.ModuleList(
            nn.Linear(config.hidden_size, config.sid_width, bias=False)
            for _ in range(config.sid_levels)
        )
        self.dropout = nn.Dropout(config.dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.history_positions.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_positions.weight, mean=0.0, std=0.02)

    def forward(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        *,
        target_sids: Tensor | None = None,
        instruction_ids: Tensor | None = None,
        level_weights: Sequence[float] | Tensor | None = None,
    ) -> OxygenRECOutput:
        """Predict all SID levels with teacher forcing.

        ``history_sids`` has shape ``[batch, history, levels]`` and
        ``history_padding_mask`` has shape ``[batch, history]`` where ``True``
        denotes padding. Without targets, prefixes are selected autoregressively
        from the preceding level head.
        """

        self._validate_inputs(history_sids, history_padding_mask, target_sids)
        batch_size = history_sids.shape[0]
        if instruction_ids is None:
            instruction_ids = torch.zeros(
                batch_size, dtype=torch.long, device=history_sids.device
            )
        if instruction_ids.shape != (batch_size,):
            raise ValueError("instruction_ids must have shape [batch]")

        memory = self._encode(history_sids, history_padding_mask)
        if target_sids is None:
            logits = self._autoregressive_logits(
                memory, history_padding_mask, instruction_ids
            )
            return OxygenRECOutput(logits=logits)
        prefix = target_sids[:, :-1]
        hidden = self._decode(memory, history_padding_mask, instruction_ids, prefix)
        logits = tuple(
            head(hidden[:, level + 1, :])
            for level, head in enumerate(self.prediction_heads)
        )
        loss, level_losses = self.weighted_ntp_loss(logits, target_sids, level_weights)
        return OxygenRECOutput(logits=logits, loss=loss, level_losses=level_losses)

    def _autoregressive_logits(
        self,
        memory: Tensor,
        memory_padding_mask: Tensor,
        instruction_ids: Tensor,
    ) -> tuple[Tensor, ...]:
        prefix = torch.empty(
            (memory.shape[0], 0), dtype=torch.long, device=memory.device
        )
        outputs = []
        for level, head in enumerate(self.prediction_heads):
            hidden = self._decode(
                memory, memory_padding_mask, instruction_ids, prefix
            )
            logits = head(hidden[:, level + 1, :])
            outputs.append(logits)
            prefix = torch.cat((prefix, logits.argmax(dim=-1, keepdim=True)), dim=1)
        return tuple(outputs)

    def _encode(self, history_sids: Tensor, padding_mask: Tensor) -> Tensor:
        _, history_length, _ = history_sids.shape
        positions = torch.arange(history_length, device=history_sids.device)
        hidden = self.history_positions(positions).unsqueeze(0)
        hidden = hidden + sum(
            embedding(history_sids[:, :, level])
            for level, embedding in enumerate(self.sid_embeddings)
        )
        return self.encoder(self.dropout(hidden), src_key_padding_mask=padding_mask)

    def _decode(
        self,
        memory: Tensor,
        memory_padding_mask: Tensor,
        instruction_ids: Tensor,
        prefix_codes: Tensor | None,
    ) -> Tensor:
        batch_size = memory.shape[0]
        tokens = [self.instruction_embeddings(instruction_ids)]
        tokens.append(self.bos_embedding.unsqueeze(0).expand(batch_size, -1))
        if prefix_codes is not None:
            tokens.extend(
                self.sid_embeddings[level](prefix_codes[:, level])
                for level in range(prefix_codes.shape[1])
            )
        hidden = torch.stack(tokens, dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.decoder_positions(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=hidden.device
            ),
            diagonal=1,
        )
        return self.decoder(
            self.dropout(hidden),
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

    @staticmethod
    def weighted_ntp_loss(
        logits: Sequence[Tensor],
        target_sids: Tensor,
        level_weights: Sequence[float] | Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return a weight-normalized cross-entropy over SID levels."""

        level_losses = tuple(
            F.cross_entropy(level_logits, target_sids[:, level])
            for level, level_logits in enumerate(logits)
        )
        if level_weights is None:
            weights = target_sids.new_ones(len(level_losses), dtype=torch.float32)
        else:
            weights = torch.as_tensor(
                level_weights, dtype=level_losses[0].dtype, device=target_sids.device
            )
        if weights.shape != (len(level_losses),):
            raise ValueError("level_weights must contain one value per SID level")
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("level_weights must be finite, non-negative, and sum positive")
        loss = sum(weight * item for weight, item in zip(weights, level_losses))
        return loss / weights.sum(), level_losses

    @torch.no_grad()
    def generate(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        trie: PrefixTrie,
        *,
        instruction_ids: Tensor | None = None,
    ) -> Tensor:
        """Greedily generate legal three-level SIDs using ``PrefixTrie`` masks."""

        self._validate_inputs(history_sids, history_padding_mask, None)
        batch_size = history_sids.shape[0]
        if instruction_ids is None:
            instruction_ids = torch.zeros(
                batch_size, dtype=torch.long, device=history_sids.device
            )
        memory = self._encode(history_sids, history_padding_mask)
        generated = torch.empty(
            (batch_size, 0), dtype=torch.long, device=history_sids.device
        )
        for level in range(self.config.sid_levels):
            hidden = self._decode(
                memory, history_padding_mask, instruction_ids, generated
            )
            logits = self.prediction_heads[level](hidden[:, level + 1, :])
            selected = []
            for row in range(batch_size):
                prefix = tuple(int(code) for code in generated[row].tolist())
                allowed = trie.allowed_next(prefix)
                if not allowed:
                    raise ValueError(f"trie has no legal continuation for prefix {prefix}")
                allowed_tensor = torch.tensor(
                    allowed, dtype=torch.long, device=logits.device
                )
                best = logits[row, allowed_tensor].argmax()
                selected.append(allowed_tensor[best])
            generated = torch.cat((generated, torch.stack(selected).unsqueeze(1)), dim=1)
        return generated

    @torch.no_grad()
    def beam_search(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        trie: PrefixTrie,
        *,
        beam_width: int,
        instruction_ids: Tensor | None = None,
    ) -> BeamSearchOutput:
        """Reference constrained beam search with deterministic tie breaking."""

        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self._validate_inputs(history_sids, history_padding_mask, None)
        batch_size = history_sids.shape[0]
        if instruction_ids is None:
            instruction_ids = torch.zeros(
                batch_size, dtype=torch.long, device=history_sids.device
            )
        if instruction_ids.shape != (batch_size,):
            raise ValueError("instruction_ids must have shape [batch]")
        memory = self._encode(history_sids, history_padding_mask)
        all_paths: list[list[tuple[int, ...]]] = []
        all_scores: list[list[float]] = []
        for row in range(batch_size):
            beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
            for level in range(self.config.sid_levels):
                candidates: list[tuple[tuple[int, ...], float]] = []
                for prefix, score in beams:
                    prefix_tensor = torch.tensor(
                        [prefix], dtype=torch.long, device=history_sids.device
                    )
                    hidden = self._decode(
                        memory[row : row + 1],
                        history_padding_mask[row : row + 1],
                        instruction_ids[row : row + 1],
                        prefix_tensor,
                    )
                    logits = self.prediction_heads[level](hidden[:, level + 1, :])
                    log_probabilities = F.log_softmax(logits[0], dim=-1)
                    allowed = trie.allowed_next(prefix)
                    if not allowed:
                        continue
                    candidates.extend(
                        (
                            prefix + (code,),
                            score + float(log_probabilities[code]),
                        )
                        for code in allowed
                    )
                if not candidates:
                    raise ValueError("trie has no complete path for beam search")
                candidates.sort(key=lambda item: (-item[1], item[0]))
                beams = candidates[:beam_width]
            all_paths.append([path for path, _ in beams])
            all_scores.append([score for _, score in beams])

        returned_beams = min(len(paths) for paths in all_paths)
        paths_tensor = torch.tensor(
            [paths[:returned_beams] for paths in all_paths],
            dtype=torch.long,
            device=history_sids.device,
        )
        scores_tensor = torch.tensor(
            [scores[:returned_beams] for scores in all_scores],
            dtype=torch.float32,
            device=history_sids.device,
        )
        return BeamSearchOutput(paths_tensor, scores_tensor)

    def _validate_inputs(
        self,
        history_sids: Tensor,
        history_padding_mask: Tensor,
        target_sids: Tensor | None,
    ) -> None:
        if history_sids.ndim != 3:
            raise ValueError("history_sids must have shape [batch, history, levels]")
        batch_size, history_length, levels = history_sids.shape
        if levels != self.config.sid_levels:
            raise ValueError(
                f"expected {self.config.sid_levels} SID levels, got {levels}"
            )
        if history_length > self.config.max_history_items:
            raise ValueError("history exceeds max_history_items")
        if history_padding_mask.shape != (batch_size, history_length):
            raise ValueError("history_padding_mask must have shape [batch, history]")
        if history_padding_mask.dtype != torch.bool:
            raise ValueError("history_padding_mask must be boolean")
        if history_padding_mask.all(dim=1).any():
            raise ValueError("every sample must contain at least one history item")
        if target_sids is not None and target_sids.shape != (
            batch_size,
            self.config.sid_levels,
        ):
            raise ValueError("target_sids must have shape [batch, levels]")
        for name, tensor in (("history_sids", history_sids), ("target_sids", target_sids)):
            if tensor is None:
                continue
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
            if (tensor < 0).any() or (tensor >= self.config.sid_width).any():
                raise ValueError(f"{name} contains a code outside the SID vocabulary")
