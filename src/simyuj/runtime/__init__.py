"""Runtime lifecycle helpers for deterministic pre-execution setup.

The runtime package currently exposes the binding lifecycle used by simulation
objects to declare timeline-owned resources before simulation execution begins.
"""

from .binding import (
    BindableMixin,
    BindingContext,
    SupportsBind,
    bind_if_supported,
    bind_many,
)

__all__ = [
    "BindableMixin",
    "BindingContext",
    "SupportsBind",
    "bind_if_supported",
    "bind_many",
]
