"""Public resource-bookkeeping records and manager.

The resources package exports the stable memory and reservation surface used by
callers that need protocol-neutral memory-slot ownership. Route-specific helper
functions remain available from ``simyuj.resources.route_requirements``.
"""

from __future__ import annotations

from .manager import ResourceManager, UnauthorizedError
from .memory import MemoryRef, MemorySlotState, MemorySlotView, memory_refs
from .reservation import Reservation, ReservationState

__all__ = [
    "MemoryRef",
    "MemorySlotState",
    "MemorySlotView",
    "Reservation",
    "ReservationState",
    "ResourceManager",
    "UnauthorizedError",
    "memory_refs",
]
