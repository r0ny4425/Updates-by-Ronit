"""
Shared simulator primitives used across subsystems.

This package exposes low-level records and validation helpers that are reused
by components, control-plane code, network models, and qstate-facing payloads.
Submodules are loaded on first attribute access to keep importing
``simyuj.primitives`` lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import coherent_state, ids, meta, subsystems, units, validation
    from .messages import ClassicalMessage, DeliveryReport, QuantumTransitPayload

_SUBMODULE_EXPORTS = {
    "coherent_state",
    "ids",
    "meta",
    "subsystems",
    "units",
    "validation",
}

_MESSAGE_EXPORTS = {
    "ClassicalMessage",
    "DeliveryReport",
    "QuantumTransitPayload",
}


def __getattr__(name: str) -> object:
    """Load exported submodules and message records on demand.

    Parameters
    ----------
    name : str
        Attribute requested from ``simyuj.primitives``.

    Returns
    -------
    object
        Exported submodule or message record class.

    Raises
    ------
    AttributeError
        If `name` is not exported by this package.
    """
    if name in _SUBMODULE_EXPORTS:
        value = import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    if name in _MESSAGE_EXPORTS:
        value = getattr(import_module(f"{__name__}.messages"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClassicalMessage",
    "DeliveryReport",
    "QuantumTransitPayload",
    "coherent_state",
    "ids",
    "meta",
    "subsystems",
    "units",
    "validation",
]
