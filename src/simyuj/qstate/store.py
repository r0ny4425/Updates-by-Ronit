from __future__ import annotations

"""Deterministic ownership store for qstate records and subsystem locations."""

from .check import check_state_ref
from .errors import StateNotFoundError, StateOwnershipError
from .ids import StateRef, SubsystemId
from .record import QuantumStateRecord, SubsystemLocation


class QuantumStateStore:
    """Store immutable state records and enforce unique subsystem ownership.

    The store assigns monotonically increasing integer references. Each live
    subsystem may be owned by at most one record, and the location index maps a
    subsystem to its owning state reference, axis, and dimension.
    """

    __slots__ = ("_records", "_locations", "_next_ref")

    def __init__(self) -> None:
        """Create an empty store with the next state reference set to zero."""
        self._records: dict[StateRef, QuantumStateRecord] = {}
        self._locations: dict[SubsystemId, SubsystemLocation] = {}
        self._next_ref: StateRef = 0

    def put(self, record: QuantumStateRecord) -> StateRef:
        """Insert a new record and return its state reference.

        Parameters
        ----------
        record : QuantumStateRecord
            Record whose subsystems must not already be owned by another live
            record.

        Returns
        -------
        StateRef
            Newly allocated state reference.

        Raises
        ------
        TypeError
            If ``record`` is not a ``QuantumStateRecord``.
        StateOwnershipError
            If any output subsystem is already owned.
        """
        self._check_record(record)
        self._check_output_ownership(record.layout.subsystems, allowed_refs=())
        state_ref = self._allocate_ref()
        self._records[state_ref] = record
        self._add_locations(state_ref, record)
        return state_ref

    def get(self, state_ref: StateRef) -> object:
        """Return the payload for a live state reference."""
        return self.record(state_ref).payload

    def record(self, state_ref: StateRef) -> QuantumStateRecord:
        """Return the record for a live state reference.

        Parameters
        ----------
        state_ref : StateRef
            Candidate state reference.

        Returns
        -------
        QuantumStateRecord
            Live record stored under ``state_ref``.

        Raises
        ------
        TypeError
            If ``state_ref`` is not exactly an ``int``.
        ValueError
            If ``state_ref`` is negative.
        StateNotFoundError
            If ``state_ref`` is not live.
        """
        state_ref = check_state_ref(state_ref)
        try:
            return self._records[state_ref]
        except KeyError:
            raise StateNotFoundError(f"state_ref is not live: {state_ref}") from None

    def get_record(self, state_ref: StateRef) -> QuantumStateRecord:
        """Return the record for ``state_ref``.

        This is an alias for :meth:`record`.
        """
        return self.record(state_ref)

    def replace(self, state_ref: StateRef, record: QuantumStateRecord) -> None:
        """Replace a live record while preserving its state reference.

        Existing subsystems owned by ``state_ref`` may appear in the replacement
        record. Subsystems owned by other records remain protected.

        Parameters
        ----------
        state_ref : StateRef
            Live state reference to replace.
        record : QuantumStateRecord
            Replacement record.

        Raises
        ------
        TypeError
            If ``state_ref`` or ``record`` has the wrong type.
        ValueError
            If ``state_ref`` is negative.
        StateNotFoundError
            If ``state_ref`` is not live.
        StateOwnershipError
            If the replacement would claim a subsystem owned by another state.
        """
        state_ref = check_state_ref(state_ref)
        old_record = self.record(state_ref)
        self._check_record(record)
        self._check_output_ownership(
            record.layout.subsystems, allowed_refs=(state_ref,)
        )
        self._drop_locations(old_record)
        self._records[state_ref] = record
        self._add_locations(state_ref, record)

    def _replace_same_layout(
        self,
        state_ref: StateRef,
        record: QuantumStateRecord,
    ) -> None:
        """Replace a live record whose layout is unchanged."""
        state_ref = check_state_ref(state_ref)
        old_record = self.record(state_ref)
        self._check_record(record)
        if record.layout != old_record.layout:
            raise ValueError("same-layout replacement requires unchanged layout")
        self._records[state_ref] = record

    def consume_and_put(
        self,
        consumed_refs: tuple[StateRef, ...],
        record: QuantumStateRecord,
    ) -> StateRef:
        """Remove live records and insert one replacement record.

        This operation is used when multiple owned states are combined into a
        single tensor record. Consumed references are not reused.

        Parameters
        ----------
        consumed_refs : tuple[StateRef, ...]
            Non-empty tuple of unique live references to remove.
        record : QuantumStateRecord
            Replacement record whose subsystems may come from the consumed
            references.

        Returns
        -------
        StateRef
            Newly allocated state reference for ``record``.

        Raises
        ------
        TypeError
            If ``consumed_refs`` or ``record`` has the wrong type.
        ValueError
            If references are empty, negative, or duplicated.
        StateNotFoundError
            If any consumed reference is not live.
        StateOwnershipError
            If the replacement would claim a subsystem outside the consumed set.
        """
        consumed_refs = self._check_consumed_refs(consumed_refs)
        self._check_record(record)
        self._check_output_ownership(
            record.layout.subsystems,
            allowed_refs=consumed_refs,
        )

        for state_ref in consumed_refs:
            old_record = self._records[state_ref]
            self._drop_locations(old_record)
            del self._records[state_ref]

        state_ref = self._allocate_ref()
        self._records[state_ref] = record
        self._add_locations(state_ref, record)
        return state_ref

    def delete(self, state_ref: StateRef) -> None:
        """Delete a live record and remove its subsystem locations."""
        state_ref = check_state_ref(state_ref)
        record = self.record(state_ref)
        self._drop_locations(record)
        del self._records[state_ref]

    def state_of(self, subsystem: SubsystemId) -> StateRef:
        """Return the state reference that owns ``subsystem``."""
        return self.location_of(subsystem).state_ref

    def location_of(self, subsystem: SubsystemId) -> SubsystemLocation:
        """Return ownership location metadata for ``subsystem``.

        Raises
        ------
        TypeError
            If ``subsystem`` is not a ``SubsystemId``.
        StateNotFoundError
            If ``subsystem`` is not owned by a live record.
        """
        self._check_subsystem(subsystem)
        try:
            return self._locations[subsystem]
        except KeyError:
            raise StateNotFoundError(f"subsystem is not owned: {subsystem}") from None

    def contains_state(self, state_ref: StateRef) -> bool:
        """Return whether ``state_ref`` is live after scalar validation."""
        state_ref = check_state_ref(state_ref)
        return state_ref in self._records

    def contains_subsystem(self, subsystem: SubsystemId) -> bool:
        """Return whether ``subsystem`` is owned by a live record."""
        self._check_subsystem(subsystem)
        return subsystem in self._locations

    def clear(self) -> None:
        """Remove all live records and locations.

        The allocation counter is not reset, so references remain monotonic
        across a clear.
        """
        self._records.clear()
        self._locations.clear()

    def size(self) -> int:
        """Return the number of live records in the store."""
        return len(self._records)

    def assert_consistent(self) -> None:
        """Check that records and the subsystem location index agree.

        Raises
        ------
        StateOwnershipError
            If a subsystem appears in multiple live records or the location
            index does not match the records.
        """
        expected: dict[SubsystemId, SubsystemLocation] = {}
        for state_ref, record in self._records.items():
            for axis, subsystem in enumerate(record.layout.subsystems):
                if subsystem in expected:
                    raise StateOwnershipError(
                        "subsystem appears in more than one live state"
                    )
                expected[subsystem] = SubsystemLocation(
                    state_ref=state_ref,
                    axis=axis,
                    dim=record.layout.dims[axis],
                )
        if self._locations != expected:
            raise StateOwnershipError("location index does not match records")

    def _allocate_ref(self) -> StateRef:
        state_ref = self._next_ref
        self._next_ref += 1
        return state_ref

    @staticmethod
    def _check_record(record: QuantumStateRecord) -> None:
        if not isinstance(record, QuantumStateRecord):
            raise TypeError("record must be QuantumStateRecord")

    @staticmethod
    def _check_subsystem(subsystem: SubsystemId) -> None:
        if not isinstance(subsystem, SubsystemId):
            raise TypeError("subsystem must be SubsystemId")

    def _check_consumed_refs(
        self,
        consumed_refs: tuple[StateRef, ...],
    ) -> tuple[StateRef, ...]:
        if not isinstance(consumed_refs, tuple):
            raise TypeError("consumed_refs must be tuple")
        if not consumed_refs:
            raise ValueError("consumed_refs must be non-empty")

        refs = tuple(check_state_ref(state_ref) for state_ref in consumed_refs)
        if len(set(refs)) != len(refs):
            raise ValueError("consumed_refs must be unique")
        for state_ref in refs:
            if state_ref not in self._records:
                raise StateNotFoundError(f"state_ref is not live: {state_ref}")
        return refs

    def _check_output_ownership(
        self,
        subsystems: tuple[SubsystemId, ...],
        *,
        allowed_refs: tuple[StateRef, ...],
    ) -> None:
        allowed = set(allowed_refs)
        for subsystem in subsystems:
            location = self._locations.get(subsystem)
            if location is None:
                continue
            if location.state_ref not in allowed:
                raise StateOwnershipError(
                    f"subsystem is already owned by state {location.state_ref}: "
                    f"{subsystem}"
                )

    def _drop_locations(self, record: QuantumStateRecord) -> None:
        for subsystem in record.layout.subsystems:
            self._locations.pop(subsystem, None)

    def _add_locations(
        self,
        state_ref: StateRef,
        record: QuantumStateRecord,
    ) -> None:
        for axis, subsystem in enumerate(record.layout.subsystems):
            self._locations[subsystem] = SubsystemLocation(
                state_ref=state_ref,
                axis=axis,
                dim=record.layout.dims[axis],
            )


__all__ = ["QuantumStateStore"]
