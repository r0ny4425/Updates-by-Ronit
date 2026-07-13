from __future__ import annotations

import pytest

from simyuj.resources import MemoryRef, Reservation, ReservationState


def ref(position: int = 0) -> MemoryRef:
    return MemoryRef(node_id="alice", device_id="qmem", position=position)


def test_reservation_records_memory_and_link_resources() -> None:
    memory_ref = ref()
    reservation = Reservation(
        reservation_id="reservation-1",
        memory_refs=(memory_ref,),
        owner="session-1",
        link_ids=("q_link",),
        created_at=10,
        expires_at=20,
        metadata=(("purpose", "test"),),
    )

    assert reservation.reservation_id == "reservation-1"
    assert reservation.memory_refs == (memory_ref,)
    assert reservation.memory_ref_keys == (memory_ref.key,)
    assert reservation.link_ids == ("q_link",)
    assert reservation.owner == "session-1"
    assert reservation.created_at == 10
    assert reservation.expires_at == 20
    assert reservation.state is ReservationState.ACTIVE
    assert reservation.is_active
    assert reservation.metadata == (("purpose", "test"),)
    assert reservation.contains_memory(memory_ref)
    assert reservation.contains_link("q_link")


def test_reservation_allows_link_only_resources() -> None:
    reservation = Reservation(
        reservation_id="reservation-1",
        memory_refs=(),
        owner="test-owner",
        link_ids=("q_link",),
    )

    assert reservation.memory_refs == ()
    assert reservation.link_ids == ("q_link",)


def test_reservation_requires_at_least_one_resource() -> None:
    with pytest.raises(ValueError, match="at least one memory ref or link id"):
        Reservation(reservation_id="reservation-1", memory_refs=(), owner="test")


def test_reservation_rejects_duplicate_memory_refs() -> None:
    memory_ref = ref()

    with pytest.raises(ValueError, match="duplicate memory ref"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(memory_ref, memory_ref),
            owner="test-owner",
        )


def test_reservation_rejects_duplicate_link_ids() -> None:
    with pytest.raises(ValueError, match="duplicate link id"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(),
            owner="test-owner",
            link_ids=("q_link", "q_link"),
        )


def test_reservation_rejects_invalid_resource_collections() -> None:
    with pytest.raises(TypeError, match="memory_refs must be tuple"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=[ref()],  # type: ignore[arg-type]
            owner="test-owner",
        )

    with pytest.raises(TypeError, match="MemoryRef instances"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=("not-ref",),  # type: ignore[arg-type]
            owner="test-owner",
        )

    with pytest.raises(TypeError, match="link_ids must be tuple"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(),
            owner="test-owner",
            link_ids=["q_link"],  # type: ignore[arg-type]
        )


def test_reservation_rejects_invalid_owner_and_times() -> None:
    with pytest.raises(ValueError, match="owner must be non-empty"):
        Reservation(reservation_id="reservation-1", memory_refs=(ref(),), owner="")

    with pytest.raises(ValueError, match="created_at must be non-negative"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            created_at=-1,
        )

    with pytest.raises(ValueError, match="expires_at must be non-negative"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            expires_at=-1,
        )

    with pytest.raises(ValueError, match="expires_at cannot be earlier"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            created_at=10,
            expires_at=9,
        )


def test_reservation_rejects_invalid_state() -> None:
    with pytest.raises(TypeError, match="state must be ReservationState"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            state="active",  # type: ignore[arg-type]
        )


def test_reservation_rejects_invalid_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be tuple"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            metadata={},  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="two-item tuple"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            metadata=(("key",),),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="metadata key must be non-empty"):
        Reservation(
            reservation_id="reservation-1",
            memory_refs=(ref(),),
            owner="test-owner",
            metadata=(("", "value"),),
        )


def test_reservation_state_helpers_return_updated_records() -> None:
    reservation = Reservation(
        reservation_id="reservation-1", memory_refs=(ref(),), owner="test-owner"
    )

    committed = reservation.committed()
    released = reservation.released()
    expired = reservation.expired()
    cancelled = reservation.cancelled()

    assert committed.state is ReservationState.COMMITTED
    assert released.state is ReservationState.RELEASED
    assert expired.state is ReservationState.EXPIRED
    assert cancelled.state is ReservationState.CANCELLED
    assert not committed.is_active
    assert reservation.state is ReservationState.ACTIVE


def test_reservation_contains_methods_validate_inputs() -> None:
    reservation = Reservation(
        reservation_id="reservation-1",
        memory_refs=(ref(),),
        owner="test-owner",
        link_ids=("q_link",),
    )

    with pytest.raises(TypeError, match="memory_ref must be MemoryRef"):
        reservation.contains_memory("not-ref")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="link_id must be non-empty"):
        reservation.contains_link("")
