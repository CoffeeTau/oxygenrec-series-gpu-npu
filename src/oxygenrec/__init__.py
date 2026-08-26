"""Core components for the OxygenREC paper-method reimplementation."""

from .sid import PrefixTrie, SIDRegistry, SemanticID
from .quantization import ReferenceResidualKMeans, ResidualKMeansModel
from .sid_metrics import SIDDiagnostics, compute_sid_diagnostics
from .evaluation import RankingMetrics, evaluate_sid_ranking
from .instructions import ContextualInstruction, InstructionStore, hash_instruction

__all__ = [
    "PrefixTrie",
    "RankingMetrics",
    "ReferenceResidualKMeans",
    "ResidualKMeansModel",
    "SIDDiagnostics",
    "SIDRegistry",
    "SemanticID",
    "ContextualInstruction",
    "InstructionStore",
    "compute_sid_diagnostics",
    "evaluate_sid_ranking",
    "hash_instruction",
]
