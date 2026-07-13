from __future__ import annotations

"""Public subsystem, dimension, layout, and target-resolution helpers.

The ``space`` package defines how logical subsystems map to tensor axes.  A
``StateLayout`` stores subsystem order and local Hilbert-space dimensions; axis
``0`` is the first subsystem in that layout and is the most significant
computational-basis axis used by the dense state-vector helpers.
"""

from .dim import check_dim, check_dims, concat_dims, qubit_dims, remove_dims, total_dim
from .layout import StateLayout
from .subsystem import SubsystemId
from .target import Target, resolve_one, resolve_targets, resolve_two

# Public layout and target-resolution surface for ``simyuj.qstate.space``.
__all__ = [
    "StateLayout",
    "SubsystemId",
    "Target",
    "check_dim",
    "check_dims",
    "concat_dims",
    "qubit_dims",
    "remove_dims",
    "resolve_one",
    "resolve_targets",
    "resolve_two",
    "total_dim",
]
