from __future__ import annotations

"""Public import surface for the new qstate backend.

The top-level namespace exposes the manager, ownership records, layout objects,
core state payload types, measurement result records, operation records, and
domain exceptions. Lower-level constructors remain available from their
subpackage namespaces.
"""

from .errors import (
    DimensionError,
    InvalidLayoutError,
    InvalidOperationError,
    InvalidReprError,
    InvalidStateError,
    MeasurementError,
    NoiseError,
    QStateError,
    StateNotFoundError,
    StateOwnershipError,
    SubsystemNotFoundError,
)
from .ids import StateRef, SubsystemId
from .manager import QuantumStateManager
from .measure import (
    POVM,
    BellResult,
    MeasurementBasis,
    MeasurementResult,
    POVMElement,
    POVMResult,
)
from .ops import Unitary
from .record import QuantumStateRecord, SubsystemLocation
from .sampler import StateSample, StateSampler
from .space import StateLayout
from .state import BellDiagState, DensityState, KetState
from .store import QuantumStateStore

# Public top-level import surface for ``simyuj.qstate``.
__all__ = [
    "BellDiagState",
    "BellResult",
    "DensityState",
    "DimensionError",
    "InvalidLayoutError",
    "InvalidOperationError",
    "InvalidReprError",
    "InvalidStateError",
    "KetState",
    "MeasurementBasis",
    "MeasurementResult",
    "MeasurementError",
    "NoiseError",
    "POVM",
    "POVMElement",
    "POVMResult",
    "QStateError",
    "QuantumStateManager",
    "QuantumStateRecord",
    "QuantumStateStore",
    "StateLayout",
    "StateNotFoundError",
    "StateOwnershipError",
    "StateSample",
    "StateSampler",
    "StateRef",
    "SubsystemId",
    "SubsystemLocation",
    "SubsystemNotFoundError",
    "Unitary",
]
