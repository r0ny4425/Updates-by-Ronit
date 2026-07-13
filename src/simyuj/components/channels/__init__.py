"""Public channel component API.

The channel package exposes event-driven transport components for the
component port graph. Classical channels forward immutable
``ClassicalMessage`` records; quantum channels forward qstate-backed
``Signal`` records and are responsible for channel-level loss, timing metadata,
and optional noise-model application.

This module is the public channel import surface. Shared validation and port
construction helpers remain internal to ``_common``.
"""

from __future__ import annotations

from .classical import (
    ACTION_RECEIVE_CLASSICAL,
    ACTION_TRANSMIT_CLASSICAL,
    DEFAULT_FIBER_LIGHT_SPEED_M_PER_S,
    ClassicalChannel,
)
from .quantum import ACTION_TRANSMIT_QUANTUM, QuantumChannel

__all__ = [
    "ACTION_RECEIVE_CLASSICAL",
    "ACTION_TRANSMIT_CLASSICAL",
    "ACTION_TRANSMIT_QUANTUM",
    "ClassicalChannel",
    "DEFAULT_FIBER_LIGHT_SPEED_M_PER_S",
    "QuantumChannel",
]
