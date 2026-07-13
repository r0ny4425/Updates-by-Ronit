"""Shared detector primitives.

This package contains detector records and pure helpers for measurement
selection, readout mapping, gate windows, click resolution, and report
construction. Event-facing detector components import these primitives rather
than embedding that logic directly.

Only the small convenience surface below is re-exported here; most primitives
remain available from their focused modules.
"""

from .readout import MeasurementContext, run_qubit_readout

__all__ = [
    "MeasurementContext",
    "run_qubit_readout",
]
