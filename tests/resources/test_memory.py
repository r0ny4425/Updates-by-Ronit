from __future__ import annotations

import pytest

from simyuj.resources import MemoryRef, MemorySlotState, MemorySlotView, memory_refs


def test_memory_ref_exposes_key_and_position_replacement() -> None:
    ref = MemoryRef(node_id="alice", device_id="qmem", position=1)

    assert ref.key == ("alice", "qmem", 1)
    assert ref.with_position(2) == MemoryRef(
        node_id="alice",
        device_id="qmem",
        position=2,
    )


def test_memory_ref_rejects_invalid_position() -> None:
    with pytest.raises(ValueError, match="position must be non-negative"):
        MemoryRef(node_id="alice", device_id="qmem", position=-1)


def test_memory_slot_view_accepts_resource_state_snapshot_fields() -> None:
    ref = MemoryRef(node_id="alice", device_id="qmem", position=0)
    view = MemorySlotView(
        ref=ref,
        state=MemorySlotState.RESERVED,
        ready_at=10,
        expires_at=20,
        metadata=(("reservation_id", "reservation-1"),),
    )

    assert view.ref is ref
    assert view.state is MemorySlotState.RESERVED
    assert view.ready_at == 10
    assert view.expires_at == 20
    assert view.metadata == (("reservation_id", "reservation-1"),)
    assert not view.is_available


def test_memory_slot_view_is_available_for_free_state() -> None:
    ref = MemoryRef(node_id="alice", device_id="qmem", position=0)

    assert MemorySlotView(ref=ref, state=MemorySlotState.FREE).is_available
    assert not MemorySlotView(ref=ref, state=MemorySlotState.OCCUPIED).is_available


def test_memory_slot_view_rejects_non_resource_state() -> None:
    with pytest.raises(TypeError, match="state must be MemorySlotState"):
        MemorySlotView(
            ref=MemoryRef(node_id="alice", device_id="qmem", position=0),
            state="free",  # type: ignore[arg-type]
        )


def test_memory_slot_view_rejects_negative_times() -> None:
    ref = MemoryRef(node_id="alice", device_id="qmem", position=0)

    with pytest.raises(ValueError, match="ready_at must be non-negative"):
        MemorySlotView(ref=ref, state=MemorySlotState.FREE, ready_at=-1)

    with pytest.raises(ValueError, match="expires_at must be non-negative"):
        MemorySlotView(ref=ref, state=MemorySlotState.FREE, expires_at=-1)


def test_memory_slot_view_rejects_invalid_metadata_shape() -> None:
    ref = MemoryRef(node_id="alice", device_id="qmem", position=0)

    with pytest.raises(TypeError, match="metadata must be tuple"):
        MemorySlotView(
            ref=ref,
            state=MemorySlotState.FREE,
            metadata={},  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="two-item tuple"):
        MemorySlotView(
            ref=ref,
            state=MemorySlotState.FREE,
            metadata=(("key",),),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="metadata key must be non-empty"):
        MemorySlotView(ref=ref, state=MemorySlotState.FREE, metadata=(("", 1),))


def test_memory_refs_builds_deterministic_refs() -> None:
    assert memory_refs("alice", "qmem", num_positions=3) == (
        MemoryRef(node_id="alice", device_id="qmem", position=0),
        MemoryRef(node_id="alice", device_id="qmem", position=1),
        MemoryRef(node_id="alice", device_id="qmem", position=2),
    )


def test_memory_refs_requires_positive_num_positions() -> None:
    with pytest.raises(ValueError, match="num_positions must be positive"):
        memory_refs("alice", "qmem", num_positions=0)

    with pytest.raises(ValueError, match="num_positions must be positive"):
        memory_refs("alice", "qmem", num_positions=-1)
