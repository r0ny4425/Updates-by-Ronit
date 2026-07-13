from __future__ import annotations

"""Exception hierarchy for qstate ownership, layout, and operation errors."""


class QStateError(Exception):
    """Base class for qstate domain errors."""


class StateNotFoundError(QStateError):
    """Raised when a state reference is not live in the store."""


class SubsystemNotFoundError(QStateError):
    """Raised when a subsystem is not present in a layout or store."""


class StateOwnershipError(QStateError):
    """Raised when subsystem ownership would become ambiguous or duplicated."""


class InvalidStateError(QStateError):
    """Raised when a state record is structurally invalid."""


class InvalidLayoutError(QStateError):
    """Raised when a tensor layout is invalid."""


class InvalidReprError(QStateError):
    """Raised when an unsupported state representation is requested."""


class DimensionError(QStateError):
    """Raised when Hilbert-space dimensions are invalid or inconsistent."""


class InvalidOperationError(QStateError):
    """Raised when an operation request cannot be applied."""


class MeasurementError(QStateError):
    """Raised when a measurement request is invalid."""


class NoiseError(QStateError):
    """Raised when a noise-channel request is invalid."""


__all__ = [
    "QStateError",
    "StateNotFoundError",
    "SubsystemNotFoundError",
    "StateOwnershipError",
    "InvalidStateError",
    "InvalidLayoutError",
    "InvalidReprError",
    "DimensionError",
    "InvalidOperationError",
    "MeasurementError",
    "NoiseError",
]
