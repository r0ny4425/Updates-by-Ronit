"""Reservation records for protocol-neutral resource ownership.

The objects in this module capture caller intent to hold memory references or
links.  They are immutable bookkeeping records and do not perform physical
memory operations or create quantum state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import validate_optional_non_negative_int

from .memory import MemoryRef


class ReservationState(Enum):
    """Lifecycle state for a resource reservation.

    This is bookkeeping state. It does not imply that the underlying memory
    component performed any runtime operation, or that a memory slot is
    occupied. Slot state is tracked separately by MemorySlotState.
    """

    ACTIVE = "active"
    COMMITTED = "committed"
    RELEASED = "released"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Reservation:
    """Immutable record describing resources held for a caller.

    A reservation records intent and ownership. It does not allocate qstate,
    absorb photons, create entanglement, or modify QuantumMemory.

    Link-only reservations are valid records. ``ResourceManager`` currently
    creates reservations for memory slots; callers that need link ownership can
    construct or track link-bearing records at a higher layer.

    Parameters
    ----------
    reservation_id : str
        Stable identifier for the reservation.
    memory_refs : tuple[MemoryRef, ...]
        Exact memory positions held by the reservation. Duplicate positions are
        rejected.
    link_ids : tuple[str, ...], optional
        Link identifiers held by the reservation. Duplicate link IDs are
        rejected.
    owner : str
        Caller or subsystem that owns the reservation.
    created_at : int or None, optional
        Optional non-negative creation tick.
    expires_at : int or None, optional
        Optional non-negative expiration tick. When both times are present,
        ``expires_at`` must be greater than or equal to ``created_at``.
    state : ReservationState, optional
        Bookkeeping lifecycle state.
    metadata : tuple[tuple[str, object], ...], optional
        Immutable metadata entries for callers that need traceability.

    Notes
    -----
    State helper methods return replacement records only. Workflow transitions
    are enforced by ``ResourceManager``.

    Metadata shape and keys are validated, but metadata values are stored as
    supplied. Mutable values remain mutable outside the frozen dataclass.

    Raises
    ------
    TypeError
        If tuple fields, memory references, state, metadata, or identifiers use
        unsupported types.
    ValueError
        If identifiers are empty, entries are duplicated, times are invalid, or
        no memory refs or link IDs are supplied.

    Examples
    --------
    >>> ref = MemoryRef("alice", "qmem", 0)
    >>> reservation = Reservation("reservation:0", (ref,), owner="bb84")
    >>> reservation.state.value
    'active'
    """

    reservation_id: str
    memory_refs: tuple[MemoryRef, ...]
    owner: str
    link_ids: tuple[str, ...] = ()
    created_at: int | None = None
    expires_at: int | None = None
    state: ReservationState = ReservationState.ACTIVE
    metadata: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reservation_id",
            ensure_nonempty_id(
                self.reservation_id,
                field_name="reservation_id",
            ),
        )
        object.__setattr__(
            self,
            "link_ids",
            self._resolve_link_ids(self.link_ids),
        )
        object.__setattr__(
            self,
            "owner",
            self._resolve_owner(self.owner),
        )

        self._validate_memory_refs(self.memory_refs)
        self._validate_times()
        self._validate_state(self.state)
        self._validate_metadata(self.metadata)

        if not self.memory_refs and not self.link_ids:
            raise ValueError(
                "reservation must include at least one memory ref or link id"
            )

    @property
    def is_active(self) -> bool:
        """Return whether this reservation can still be committed or closed."""

        return self.state is ReservationState.ACTIVE

    @property
    def memory_ref_keys(self) -> tuple[tuple[str, str, int], ...]:
        """Return deterministic tuple keys for all reserved memory positions."""

        return tuple(memory_ref.key for memory_ref in self.memory_refs)

    def contains_memory(self, memory_ref: MemoryRef) -> bool:
        """Return whether this reservation includes ``memory_ref``.

        Parameters
        ----------
        memory_ref : MemoryRef
            Memory position to test.

        Returns
        -------
        bool
            ``True`` when the exact memory reference is held.
        """

        if not isinstance(memory_ref, MemoryRef):
            raise TypeError("memory_ref must be MemoryRef")

        return memory_ref in self.memory_refs

    def contains_link(self, link_id: str) -> bool:
        """Return whether this reservation includes ``link_id``."""

        resolved_link_id = ensure_nonempty_id(link_id, field_name="link_id")
        return resolved_link_id in self.link_ids

    def with_state(self, state: ReservationState) -> Reservation:
        """Return a copy of this reservation with ``state`` applied."""

        self._validate_state(state)
        return replace(self, state=state)

    def committed(self) -> Reservation:
        """Mark this reservation as handed off to runtime or protocol ownership.

        This does not imply that reserved memory is occupied.
        """

        return self.with_state(ReservationState.COMMITTED)

    def released(self) -> Reservation:
        """Return a copy marked released."""

        return self.with_state(ReservationState.RELEASED)

    def expired(self) -> Reservation:
        """Return a copy marked expired."""

        return self.with_state(ReservationState.EXPIRED)

    def cancelled(self) -> Reservation:
        """Return a copy marked cancelled."""

        return self.with_state(ReservationState.CANCELLED)

    @staticmethod
    def _validate_memory_refs(memory_refs: tuple[MemoryRef, ...]) -> None:
        if not isinstance(memory_refs, tuple):
            raise TypeError("memory_refs must be tuple[MemoryRef, ...]")

        seen: set[tuple[str, str, int]] = set()

        for memory_ref in memory_refs:
            if not isinstance(memory_ref, MemoryRef):
                raise TypeError("memory_refs must contain only MemoryRef instances")

            if memory_ref.key in seen:
                raise ValueError(f"duplicate memory ref {memory_ref.key!r}")

            seen.add(memory_ref.key)

    @staticmethod
    def _resolve_link_ids(link_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(link_ids, tuple):
            raise TypeError("link_ids must be tuple[str, ...]")

        resolved_link_ids: list[str] = []
        seen: set[str] = set()

        for link_id in link_ids:
            resolved_link_id = ensure_nonempty_id(link_id, field_name="link_id")

            if resolved_link_id in seen:
                raise ValueError(f"duplicate link id '{resolved_link_id}'")

            seen.add(resolved_link_id)
            resolved_link_ids.append(resolved_link_id)

        return tuple(resolved_link_ids)

    @staticmethod
    def _resolve_owner(owner: str) -> str:
        return ensure_nonempty_id(owner, field_name="owner")

    def _validate_times(self) -> None:
        validate_optional_non_negative_int(
            self.created_at,
            field_name="created_at",
        )
        validate_optional_non_negative_int(
            self.expires_at,
            field_name="expires_at",
        )

        if (
            self.created_at is not None
            and self.expires_at is not None
            and self.expires_at < self.created_at
        ):
            raise ValueError("expires_at cannot be earlier than created_at")

    @staticmethod
    def _validate_state(state: ReservationState) -> None:
        if not isinstance(state, ReservationState):
            raise TypeError("state must be ReservationState")

    @staticmethod
    def _validate_metadata(metadata: tuple[tuple[str, object], ...]) -> None:
        if not isinstance(metadata, tuple):
            raise TypeError("metadata must be tuple[tuple[str, object], ...]")

        for item in metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("metadata must contain only two-item tuple entries")

            key, _value = item
            ensure_nonempty_id(key, field_name="metadata key")


__all__ = [
    "Reservation",
    "ReservationState",
]
