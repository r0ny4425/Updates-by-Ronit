from __future__ import annotations

"""Shared identifier aliases for qstate ownership records.

``StateRef`` is the integer handle assigned by
:class:`simyuj.qstate.store.QuantumStateStore`. ``SubsystemId`` is re-exported
from the space layer so ownership APIs can import both identifier types from one
small module.
"""

from typing import TypeAlias

from .space.subsystem import SubsystemId

StateRef: TypeAlias = int

__all__ = ["StateRef", "SubsystemId"]
