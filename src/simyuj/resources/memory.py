"""Memory-slot records used by protocol-neutral resource bookkeeping.

This module provides immutable resource-layer addresses and read-only slot
views.  The records describe where memory positions live in the network
namespace; they do not own quantum state, photons, or component runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import (
    validate_non_negative_int,
    validate_optional_non_negative_int,
    validate_positive_int,
)


class MemorySlotState(Enum):
    """Resource-layer lifecycle state for one addressable memory slot.

    This is bookkeeping state, not the physical MemoryPositionStatus used by
    QuantumMemory.
    """

    FREE = "free"
    RESERVED = "reserved"
    OCCUPIED = "occupied"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryRef:
    """Stable resource-layer address for one memory position.

    The resource layer addresses memory through the network namespace:

        node_id + device_id + position

    The underlying QuantumMemory still has its own memory_id. Later layers can
    resolve this reference to the actual component when they need to submit
    memory requests.

    Parameters
    ----------
    node_id : str
        Network node that owns the memory device.
    device_id : str
        Node-local device name for the memory component.
    position : int
        Zero-based position within the memory device.

    Raises
    ------
    TypeError
        If an identifier is not a string or ``position`` is not an integer.
    ValueError
        If an identifier is empty or ``position`` is negative.

    Examples
    --------
    >>> ref = MemoryRef("alice", "qmem", 0)
    >>> ref.key
    ('alice', 'qmem', 0)
    """

    node_id: str
    device_id: str
    position: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            ensure_nonempty_id(self.node_id, field_name="node_id"),
        )
        object.__setattr__(
            self,
            "device_id",
            ensure_nonempty_id(self.device_id, field_name="device_id"),
        )
        validate_non_negative_int(self.position, field_name="position")

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the deterministic tuple key for dictionaries and sorting."""

        return (self.node_id, self.device_id, self.position)

    def with_position(self, position: int) -> MemoryRef:
        """Return a copy of this reference with a different position.

        Parameters
        ----------
        position : int
            Zero-based memory-device position for the returned reference.

        Returns
        -------
        MemoryRef
            New reference with the same node and device identifiers.
        """

        return MemoryRef(
            node_id=self.node_id,
            device_id=self.device_id,
            position=position,
        )


@dataclass(frozen=True, slots=True)
class MemorySlotView:
    """Read-only resource-layer view of one memory slot.

    This is suitable for ResourceManager snapshots. It should not carry qstate
    objects, stored photons, signals, or component-owned mutable state.

    Parameters
    ----------
    ref : MemoryRef
        Resource-layer address of the slot.
    state : MemorySlotState
        Bookkeeping lifecycle state for the slot.
    ready_at : int or None, optional
        Optional non-negative tick at which the slot is expected to be ready.
    expires_at : int or None, optional
        Optional non-negative tick at which the slot's contents expire.
    metadata : tuple[tuple[str, object], ...], optional
        Immutable metadata entries copied from resource registration or
        component snapshots.

    Notes
    -----
    Metadata shape and keys are validated, but metadata values are stored as
    supplied. Mutable values remain mutable outside the frozen dataclass.

    Raises
    ------
    TypeError
        If ``ref``, ``state``, or ``metadata`` has the wrong shape.
    ValueError
        If optional times are negative or metadata keys are empty.
    """

    ref: MemoryRef
    state: MemorySlotState
    ready_at: int | None = None
    expires_at: int | None = None
    metadata: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ref, MemoryRef):
            raise TypeError("ref must be MemoryRef")

        if not isinstance(self.state, MemorySlotState):
            raise TypeError("state must be MemorySlotState")

        validate_optional_non_negative_int(
            self.ready_at,
            field_name="ready_at",
        )
        validate_optional_non_negative_int(
            self.expires_at,
            field_name="expires_at",
        )

        if not isinstance(self.metadata, tuple):
            raise TypeError("metadata must be tuple[tuple[str, object], ...]")

        for item in self.metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("metadata must contain only two-item tuple entries")
            key, _value = item
            ensure_nonempty_id(key, field_name="metadata key")

    @property
    def is_available(self) -> bool:
        """Return whether this slot's bookkeeping state is ``FREE``.

        This property does not evaluate readiness time, expiration time, or
        holders. Use ``ResourceManager.available_memories(now, ...)`` when
        callers need time-aware reservation availability.
        """

        return self.state is MemorySlotState.FREE


def memory_refs(
    node_id: str,
    device_id: str,
    *,
    num_positions: int,
) -> tuple[MemoryRef, ...]:
    """Build deterministic MemoryRef objects for one node-local memory device.

    Parameters
    ----------
    node_id : str
        Network node that owns the memory device.
    device_id : str
        Node-local device name for the memory component.
    num_positions : int
        Number of addressable positions to create, starting at position 0.

    Returns
    -------
    tuple[MemoryRef, ...]
        References ordered by increasing memory position.

    Raises
    ------
    TypeError
        If identifiers or ``num_positions`` have invalid types.
    ValueError
        If identifiers are empty or ``num_positions`` is not positive.

    Examples
    --------
    >>> memory_refs("alice", "qmem", num_positions=2)
    (MemoryRef(...), MemoryRef(...))
    """

    resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")
    resolved_device_id = ensure_nonempty_id(device_id, field_name="device_id")
    validate_positive_int(num_positions, field_name="num_positions")

    return tuple(
        MemoryRef(
            node_id=resolved_node_id,
            device_id=resolved_device_id,
            position=position,
        )
        for position in range(num_positions)
    )


__all__ = [
    "MemoryRef",
    "MemorySlotState",
    "MemorySlotView",
    "memory_refs",
]
