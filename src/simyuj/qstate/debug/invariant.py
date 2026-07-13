from __future__ import annotations

"""Invariant checks for qstate records and store-like objects.

These helpers are intended for tests and diagnostics. They perform explicit
validation and raise qstate domain errors when record payloads, layouts, or
ownership indexes do not agree.
"""

from collections.abc import Iterable, Mapping

from ..errors import InvalidLayoutError, InvalidStateError, StateOwnershipError
from ..ids import StateRef, SubsystemId
from ..record import QuantumStateRecord
from ..space.layout import StateLayout
from ..state.check import assert_payload_layout_compatible, check_payload


def assert_record_ok(record: QuantumStateRecord) -> None:
    """Assert that a record has a valid payload and matching layout.

    Parameters
    ----------
    record : QuantumStateRecord
        Record to validate.

    Raises
    ------
    TypeError
        If ``record`` is not a ``QuantumStateRecord``.
    InvalidStateError
        If the representation field is not a non-empty string.
    InvalidLayoutError
        If the layout field is not a ``StateLayout`` or does not match the
        payload.
    """
    if not isinstance(record, QuantumStateRecord):
        raise TypeError("record must be QuantumStateRecord")

    if not isinstance(record.rep, str) or not record.rep:
        raise InvalidStateError("record rep must be non-empty str")
    if not isinstance(record.layout, StateLayout):
        raise InvalidLayoutError("record layout must be StateLayout")

    assert_valid_state(record)
    assert_layout_matches_payload(record)


def assert_valid_state(record: QuantumStateRecord) -> None:
    """Assert that a record payload is valid for its representation.

    Parameters
    ----------
    record : QuantumStateRecord
        Record whose payload should be checked.

    Raises
    ------
    TypeError
        If ``record`` is not a ``QuantumStateRecord`` or the payload type does
        not match ``record.rep``.
    InvalidStateError
        If representation-specific payload validation fails.
    """
    if not isinstance(record, QuantumStateRecord):
        raise TypeError("record must be QuantumStateRecord")

    check_payload(record.payload, rep=record.rep)


def assert_layout_matches_payload(record: QuantumStateRecord) -> None:
    """Assert that a record layout is compatible with its payload.

    Parameters
    ----------
    record : QuantumStateRecord
        Record whose payload-layout relationship should be checked.

    Raises
    ------
    TypeError
        If ``record`` is not a ``QuantumStateRecord``.
    InvalidLayoutError
        If the payload and layout dimensions do not agree.
    """
    if not isinstance(record, QuantumStateRecord):
        raise TypeError("record must be QuantumStateRecord")

    assert_payload_layout_compatible(record.payload, record.layout)


def assert_unique_subsystems(store: object) -> None:
    """Assert that each subsystem is owned by at most one live state.

    Store-like objects are read through :func:`iter_store_records`. If the store
    also exposes ``state_of`` or ``location_of``, those indexes are checked
    against the records.

    Parameters
    ----------
    store : object
        Store-like object to inspect.

    Raises
    ------
    TypeError
        If ``store`` does not expose a supported record-iteration shape.
    StateOwnershipError
        If a subsystem appears in multiple records or a store index disagrees
        with the live records.
    """
    seen: dict[SubsystemId, tuple[StateRef, int]] = {}

    for state_ref, record in iter_store_records(store):
        for axis, subsystem in enumerate(record.layout.subsystems):
            if subsystem in seen:
                old_state_ref, old_axis = seen[subsystem]
                raise StateOwnershipError(
                    "subsystem appears in multiple live locations: "
                    f"{subsystem!r} at ({old_state_ref}, {old_axis}) and "
                    f"({state_ref}, {axis})"
                )

            seen[subsystem] = (state_ref, axis)

            if hasattr(store, "state_of"):
                owner = store.state_of(subsystem)
                if owner != state_ref:
                    raise StateOwnershipError(
                        f"store owner mismatch for {subsystem!r}: "
                        f"expected {state_ref}, got {owner}"
                    )

            if hasattr(store, "location_of"):
                location = store.location_of(subsystem)
                if location.state_ref != state_ref:
                    raise StateOwnershipError(
                        f"location state_ref mismatch for {subsystem!r}"
                    )
                if location.axis != axis:
                    raise StateOwnershipError(
                        f"location axis mismatch for {subsystem!r}: "
                        f"expected {axis}, got {location.axis}"
                    )
                if location.dim != record.layout.dim_at(axis):
                    raise StateOwnershipError(
                        f"location dim mismatch for {subsystem!r}: "
                        f"expected {record.layout.dim_at(axis)}, got {location.dim}"
                    )


def assert_store_ok(store: object) -> None:
    """Assert that a store-like object is internally consistent.

    The helper calls ``store.assert_consistent()`` when available, checks
    optional ``size()`` output, validates state references, validates each
    record, and verifies subsystem ownership uniqueness.

    Parameters
    ----------
    store : object
        Store-like object to inspect.

    Raises
    ------
    TypeError
        If ``store`` does not expose a supported record-iteration shape.
    StateOwnershipError
        If record indexes, state references, size, or subsystem ownership are
        inconsistent.
    InvalidLayoutError
        If a record layout is invalid or incompatible with its payload.
    InvalidStateError
        If a record payload is invalid for its representation.
    """
    if hasattr(store, "assert_consistent") and callable(store.assert_consistent):
        store.assert_consistent()

    records = tuple(iter_store_records(store))

    if hasattr(store, "size") and callable(store.size):
        size = store.size()
        if size != len(records):
            raise StateOwnershipError(
                f"store size mismatch: size()={size}, records={len(records)}"
            )

    for state_ref, record in records:
        if type(state_ref) is not int:
            raise StateOwnershipError("state refs must be ints")
        if state_ref < 0:
            raise StateOwnershipError("state refs must be non-negative")
        assert_record_ok(record)

    assert_unique_subsystems(store)


def iter_store_records(
    store: object,
) -> tuple[tuple[StateRef, QuantumStateRecord], ...]:
    """Return live records from a store-like object.

    The lookup order supports direct ``items()``, a ``records()`` method,
    ``_records``, and then ``_states``. Results are sorted by state reference so
    diagnostics are deterministic.

    Parameters
    ----------
    store : object
        Store-like object exposing one supported record shape.

    Returns
    -------
    tuple[tuple[StateRef, QuantumStateRecord], ...]
        Sorted ``(state_ref, record)`` pairs.

    Raises
    ------
    TypeError
        If no supported record shape exists or record items are malformed.
    """
    if hasattr(store, "items") and callable(store.items):
        return _coerce_record_items(store.items())

    if hasattr(store, "records") and callable(store.records):
        records = store.records()
        if isinstance(records, Mapping):
            return _coerce_record_items(records.items())
        return _coerce_record_items(records)

    if hasattr(store, "_records"):
        records = getattr(store, "_records")
        if isinstance(records, Mapping):
            return _coerce_record_items(records.items())

    if hasattr(store, "_states"):
        records = getattr(store, "_states")
        if isinstance(records, Mapping):
            return _coerce_record_items(records.items())

    raise TypeError("store does not expose records/items/_records/_states")


def _coerce_record_items(
    items: Iterable[object],
) -> tuple[tuple[StateRef, QuantumStateRecord], ...]:
    """Validate and sort raw ``(state_ref, record)`` pairs."""
    result: list[tuple[StateRef, QuantumStateRecord]] = []

    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("record items must be (state_ref, record) tuples")

        state_ref, record = item
        if type(state_ref) is not int:
            raise TypeError("state_ref must be int")
        if not isinstance(record, QuantumStateRecord):
            raise TypeError("record item value must be QuantumStateRecord")

        result.append((state_ref, record))

    return tuple(sorted(result, key=lambda pair: pair[0]))


__all__ = [
    "assert_layout_matches_payload",
    "assert_record_ok",
    "assert_store_ok",
    "assert_unique_subsystems",
    "assert_valid_state",
    "iter_store_records",
]
