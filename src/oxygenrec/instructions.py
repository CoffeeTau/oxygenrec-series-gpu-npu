"""Auditable public proxies for contextual reasoning instructions.

The paper's Slow LLM and private instruction store are unavailable.  This
module deliberately provides a deterministic text-to-feature boundary so that
human-readable instructions can *actually* enter the reproduced fast model.
It is an engineering proxy, not a language-model reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence


_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u3400-\u9fff]")


@dataclass(frozen=True)
class ContextualInstruction:
    """One reviewable instruction and the evidence used to construct it."""

    sample_id: str
    text: str
    evidence: tuple[str, ...] = ()
    source: str = "deterministic_public_proxy"
    version: str = "v1"


def instruction_tokens(text: str) -> tuple[str, ...]:
    """Return simple language-independent tokens without external models."""

    normalized = " ".join(text.strip().lower().split())
    words = _ASCII_WORD.findall(normalized)
    cjk = _CJK.findall(normalized)
    cjk_bigrams = ["".join(cjk[index:index + 2]) for index in range(len(cjk) - 1)]
    return tuple(words + cjk + cjk_bigrams)


def hash_instruction(text: str, dimension: int = 64) -> tuple[float, ...]:
    """Encode text with signed feature hashing and L2 normalization.

    This function is deterministic across Python processes; unlike ``hash()``,
    BLAKE2 is unaffected by hash randomization.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    vector = [0.0] * dimension
    for token in instruction_tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=9).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        vector[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(vector)


def encode_instructions(
    instructions: Sequence[str], dimension: int = 64,
) -> list[tuple[float, ...]]:
    return [hash_instruction(text, dimension) for text in instructions]


class InstructionStore:
    """Small JSONL store that keeps proxy text and provenance reviewable."""

    @staticmethod
    def save(path: str | Path, records: Iterable[ContextualInstruction]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = asdict(record)
                payload["evidence"] = list(record.evidence)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[ContextualInstruction]:
        records = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload["evidence"] = tuple(payload.get("evidence", ()))
                records.append(ContextualInstruction(**payload))
        return records
