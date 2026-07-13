"""Public signal API for transport envelope records.

The signal namespace exposes the reusable primitives passed between sources,
channels, memories, detectors, and transport messages. Quantum-state storage
and operations remain in qstate and component layers.
"""

from __future__ import annotations

from .signal import EncodingScheme, Signal, SignalKind

__all__ = [
    "EncodingScheme",
    "Signal",
    "SignalKind",
]
