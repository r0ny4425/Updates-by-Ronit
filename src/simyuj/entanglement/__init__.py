"""Public entangled-pair records and registry.

The package exports the core pair record and registry API. Read-only query
helpers remain available from ``simyuj.entanglement.queries`` so the top-level
surface stays small.
"""

from __future__ import annotations

from .build import pair_from_absorbs, swapped_pair_from_bsa
from .pair import EntangledPairRecord, PairState
from .registry import EntangledPairRegistry

__all__ = [
    "EntangledPairRecord",
    "EntangledPairRegistry",
    "PairState",
    "pair_from_absorbs",
    "swapped_pair_from_bsa",
]
