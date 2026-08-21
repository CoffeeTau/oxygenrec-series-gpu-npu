"""Auditable ranking metrics for generated Semantic-ID candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .sid import SIDRegistry


@dataclass(frozen=True)
class RankingMetrics:
    sample_count: int
    legal_sid_rate: float
    hit_rate: Mapping[int, float]
    recall: Mapping[int, float]
    mrr: float
    ndcg: float


def evaluate_sid_ranking(
    predictions: Sequence[Sequence[Sequence[int]]],
    target_item_ids: Sequence[str],
    registry: SIDRegistry,
    *,
    ks: Sequence[int] = (1, 10),
) -> RankingMetrics:
    """Evaluate ranked SID paths against one next-item target per sample.

    A colliding SID is a hit when its explicit registry item set contains the
    target. For a single relevant next-item target, Recall@K equals HR@K; both
    names are returned so experiment tables remain protocol-explicit.
    """

    if not predictions:
        raise ValueError("predictions must not be empty")
    if len(predictions) != len(target_item_ids):
        raise ValueError("predictions and target_item_ids must have equal length")
    normalized_ks = tuple(sorted(set(int(k) for k in ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("ks must contain positive cutoffs")
    if any(len(ranking) < normalized_ks[-1] for ranking in predictions):
        raise ValueError("every prediction ranking must cover the largest K")

    hits = {k: 0 for k in normalized_ks}
    reciprocal_rank = 0.0
    discounted_gain = 0.0
    legal = 0
    candidate_count = 0
    for ranking, raw_target in zip(predictions, target_item_ids, strict=True):
        target = str(raw_target)
        first_hit: int | None = None
        for rank, codes in enumerate(ranking, start=1):
            items = registry.items_for(codes)
            candidate_count += 1
            if items:
                legal += 1
            if first_hit is None and target in items:
                first_hit = rank
        if first_hit is not None:
            reciprocal_rank += 1.0 / first_hit
            discounted_gain += 1.0 / math.log2(first_hit + 1)
            for k in normalized_ks:
                if first_hit <= k:
                    hits[k] += 1

    count = len(predictions)
    rates = MappingProxyType({k: hits[k] / count for k in normalized_ks})
    return RankingMetrics(
        sample_count=count,
        legal_sid_rate=legal / candidate_count,
        hit_rate=rates,
        recall=MappingProxyType(dict(rates)),
        mrr=reciprocal_rank / count,
        ndcg=discounted_gain / count,
    )
