from __future__ import annotations

from dataclasses import replace

import pytest

from simyuj.components.memories import (
    MemoryPositionRecord,
    MemoryPositionStatus,
    QuantumMemory,
)
from simyuj.network import Network, Node
from simyuj.qstate import SubsystemId
from simyuj.resources import (
    MemoryRef,
    MemorySlotState,
    ReservationState,
    ResourceManager,
    UnauthorizedError,
)


def test_register_memory_creates_deterministic_free_slots() -> None:
    manager = ResourceManager()
    refs = manager.register_memory(
        "alice", "qmem", num_positions=2, metadata=(("memory_id", "alice.mem"),)
    )
    assert refs == (MemoryRef("alice", "qmem", 0), MemoryRef("alice", "qmem", 1))
    assert manager.registered_memories() == refs
    assert manager.available_memories(10, "alice") == refs
    assert manager.get_slot(refs[0]).state is MemorySlotState.FREE
    assert manager.get_slot(refs[0]).metadata == (("memory_id", "alice.mem"),)


def test_register_memory_rejects_duplicate_refs() -> None:
    manager = ResourceManager()
    manager.register_memory("alice", "qmem", num_positions=1)
    with pytest.raises(ValueError, match="already registered"):
        manager.register_memory("alice", "qmem", num_positions=1)


def test_available_memories_filters_by_node_and_device() -> None:
    manager = ResourceManager()
    alice_refs = manager.register_memory("alice", "qmem", num_positions=2)
    manager.register_memory("alice", "other", num_positions=1)
    manager.register_memory("bob", "qmem", num_positions=1)
    manager.reserve_memory_refs(10, (alice_refs[0],), owner="test-owner")
    assert manager.available_memories(10, "alice", device_id="qmem") == (alice_refs[1],)


def test_available_memories_requires_now() -> None:
    manager = ResourceManager()
    manager.register_memory("alice", "qmem", num_positions=1)

    with pytest.raises(TypeError, match="now must be int"):
        manager.available_memories("not-a-tick", "alice")  # type: ignore[arg-type]


def test_available_memories_filters_by_ready_time() -> None:
    network = Network("resources")
    alice = Node("alice")
    memory = QuantumMemory(memory_id="alice.mem", num_positions=2)
    memory.positions = (
        replace(memory.positions[0], ready_at=7),
        replace(memory.positions[1], ready_at=3),
    )
    alice.add_device("qmem", memory)
    network.add_node(alice)
    manager = ResourceManager.from_network(network)
    refs = manager.registered_memories("alice")

    assert manager.available_memories(4, "alice") == (refs[1],)
    assert manager.available_memories(7, "alice") == refs


def test_reserve_memories_selects_first_deterministic_refs() -> None:
    manager = ResourceManager()
    alice_refs = manager.register_memory("alice", "qmem", num_positions=3)
    bob_refs = manager.register_memory("bob", "qmem", num_positions=1)
    reservation = manager.reserve_memories(
        10, {"bob": 1, "alice": 2}, owner="test-owner"
    )
    assert reservation.memory_refs == (alice_refs[0], alice_refs[1], bob_refs[0])
    assert all(
        (
            manager.get_slot(ref).state is MemorySlotState.RESERVED
            for ref in reservation.memory_refs
        )
    )


def test_reserve_memories_rejects_unknown_or_insufficient_nodes() -> None:
    manager = ResourceManager()
    manager.register_memory("alice", "qmem", num_positions=1)
    with pytest.raises(KeyError, match="unknown node id"):
        manager.reserve_memories(10, {"bob": 1}, owner="test-owner")
    with pytest.raises(ValueError, match="1 available"):
        manager.reserve_memories(10, {"alice": 2}, owner="test-owner")


def test_reserve_memories_targets_specific_devices() -> None:
    manager = ResourceManager()
    qmem_a = manager.register_memory("relay", "qmem_a", num_positions=2)
    qmem_b = manager.register_memory("relay", "qmem_b", num_positions=2)

    reservation = manager.reserve_memories(
        10,
        {"relay": {"qmem_b": 1, "qmem_a": 2}},
        owner="test-owner",
    )

    assert reservation.memory_refs == (*qmem_a, qmem_b[0])


def test_reserve_memories_rejects_insufficient_targeted_device_slots() -> None:
    manager = ResourceManager()
    manager.register_memory("relay", "qmem_a", num_positions=1)
    manager.register_memory("relay", "qmem_b", num_positions=2)

    with pytest.raises(ValueError, match="device 'qmem_a'"):
        manager.reserve_memories(
            10,
            {"relay": {"qmem_a": 2}},
            owner="test-owner",
        )


def test_reserve_memory_refs_rejects_bad_refs() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=1)
    with pytest.raises(ValueError, match="non-empty"):
        manager.reserve_memory_refs(10, (), owner="test-owner")
    with pytest.raises(ValueError, match="duplicate memory ref"):
        manager.reserve_memory_refs(10, (refs[0], refs[0]), owner="test-owner")
    with pytest.raises(KeyError, match="unknown memory ref"):
        manager.reserve_memory_refs(
            10, (MemoryRef("alice", "other", 0),), owner="test-owner"
        )
    manager.mark_occupied(refs[0])
    with pytest.raises(ValueError, match="not available"):
        manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")


def test_failed_auto_reservation_does_not_skip_generated_id() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=1)
    manager.mark_occupied(refs[0])
    with pytest.raises(ValueError, match="not available"):
        manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")
    manager.mark_free(refs[0])
    reservation = manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")
    assert reservation.reservation_id == "reservation:0"


def test_commit_reservation_changes_state_but_keeps_slot_reserved() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=1)
    reservation = manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")
    committed = manager.commit_reservation(
        reservation.reservation_id, owner="test-owner"
    )
    assert committed.state is ReservationState.COMMITTED
    assert manager.get_slot(refs[0]).state is MemorySlotState.RESERVED
    assert manager.reservation_for_memory(refs[0]) == committed


def test_reservation_lifecycle_rejects_wrong_owner() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=3)
    commit_reservation = manager.reserve_memory_refs(10, (refs[0],), owner="owner-a")
    release_reservation = manager.reserve_memory_refs(10, (refs[1],), owner="owner-a")
    cancel_reservation = manager.reserve_memory_refs(10, (refs[2],), owner="owner-a")

    with pytest.raises(UnauthorizedError, match="does not own reservation"):
        manager.commit_reservation(commit_reservation.reservation_id, owner="owner-b")
    with pytest.raises(UnauthorizedError, match="does not own reservation"):
        manager.release_reservation(release_reservation.reservation_id, owner="owner-b")
    with pytest.raises(UnauthorizedError, match="does not own reservation"):
        manager.cancel_reservation(cancel_reservation.reservation_id, owner="owner-b")


def test_release_reservation_frees_only_reserved_slots() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=2)
    first = manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")
    second = manager.reserve_memory_refs(10, (refs[1],), owner="test-owner")
    manager.mark_occupied(refs[1])
    released_first = manager.release_reservation(
        first.reservation_id, owner="test-owner"
    )
    released_second = manager.release_reservation(
        second.reservation_id, owner="test-owner"
    )
    assert released_first.state is ReservationState.RELEASED
    assert released_second.state is ReservationState.RELEASED
    assert manager.get_slot(refs[0]).state is MemorySlotState.FREE
    assert manager.get_slot(refs[1]).state is MemorySlotState.OCCUPIED
    assert manager.reservation_for_memory(refs[0]) is None
    assert manager.reservation_for_memory(refs[1]) is None


def test_mark_free_respects_active_holder() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=1)
    reservation = manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")
    manager.mark_occupied(refs[0])
    held_free = manager.mark_free(refs[0])
    manager.release_reservation(reservation.reservation_id, owner="test-owner")
    released_free = manager.mark_free(refs[0])
    assert held_free.state is MemorySlotState.RESERVED
    assert released_free.state is MemorySlotState.FREE


def test_mark_consumed_and_expired_reject_invalid_states() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=4)
    reserved = manager.reserve_memory_refs(
        10, (refs[1],), owner="test-owner"
    ).memory_refs[0]
    occupied = refs[2]
    manager.mark_occupied(occupied)
    manager.mark_failed(refs[3])
    with pytest.raises(ValueError, match="cannot consume"):
        manager.mark_consumed(refs[0])
    with pytest.raises(ValueError, match="cannot consume"):
        manager.mark_consumed(reserved)
    with pytest.raises(ValueError, match="cannot consume"):
        manager.mark_consumed(refs[3])
    assert manager.mark_consumed(occupied).state is MemorySlotState.CONSUMED
    with pytest.raises(ValueError, match="cannot expire"):
        manager.mark_expired(refs[0])
    with pytest.raises(ValueError, match="cannot expire"):
        manager.mark_expired(occupied)
    with pytest.raises(ValueError, match="cannot expire"):
        manager.mark_expired(refs[3])
    assert manager.mark_expired(reserved).state is MemorySlotState.EXPIRED


def test_cancel_and_expire_close_reservations() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=2)
    cancelled = manager.reserve_memory_refs(10, (refs[0],), owner="test-owner")
    expired = manager.reserve_memory_refs(10, (refs[1],), owner="test-owner")
    assert (
        manager.cancel_reservation(cancelled.reservation_id, owner="test-owner").state
        is ReservationState.CANCELLED
    )
    assert (
        manager.expire_reservation(expired.reservation_id).state
        is ReservationState.EXPIRED
    )
    assert manager.get_slot(refs[0]).state is MemorySlotState.FREE
    assert manager.get_slot(refs[1]).state is MemorySlotState.FREE


def test_get_reservation_reports_unknown_ids() -> None:
    manager = ResourceManager()
    with pytest.raises(KeyError, match="unknown reservation id"):
        manager.get_reservation("missing")


def test_from_network_registers_only_quantum_memory_devices() -> None:
    network = Network("resources")
    alice = Node("alice")
    memory = QuantumMemory(memory_id="alice.mem", num_positions=2)
    alice.add_device("qmem", memory)
    alice.add_device("not_memory", object())
    network.add_node(alice)
    manager = ResourceManager.from_network(network)
    refs = manager.registered_memories()
    assert refs == (MemoryRef("alice", "qmem", 0), MemoryRef("alice", "qmem", 1))
    assert manager.get_slot(refs[0]).metadata == (("memory_id", "alice.mem"),)


def test_from_network_maps_position_status_and_times() -> None:
    network = Network("resources")
    alice = Node("alice")
    memory = QuantumMemory(memory_id="alice.mem", num_positions=2)
    memory.positions = (
        replace(memory.positions[0], ready_at=7),
        MemoryPositionRecord(
            position=1,
            status=MemoryPositionStatus.OCCUPIED,
            memory_subsystem=SubsystemId("memory:alice.mem:position:1"),
            stored_time=3,
            last_noise_update_time=3,
            expires_at=11,
        ),
    )
    alice.add_device("qmem", memory)
    network.add_node(alice)
    manager = ResourceManager.from_network(network)
    free_ref = MemoryRef("alice", "qmem", 0)
    occupied_ref = MemoryRef("alice", "qmem", 1)
    assert manager.get_slot(free_ref).state is MemorySlotState.FREE
    assert manager.get_slot(free_ref).ready_at == 7
    assert manager.get_slot(occupied_ref).state is MemorySlotState.OCCUPIED
    assert manager.get_slot(occupied_ref).expires_at == 11


def test_expire_before_sweeps_active_and_committed_reservations() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=4)
    res_10 = manager.reserve_memory_refs(
        10, (refs[0],), expires_at=10, owner="test-owner"
    )
    res_15 = manager.reserve_memory_refs(
        10, (refs[1],), expires_at=15, owner="test-owner"
    )
    manager.commit_reservation(res_15.reservation_id, owner="test-owner")
    res_20 = manager.reserve_memory_refs(
        10, (refs[2],), expires_at=20, owner="test-owner"
    )
    res_none = manager.reserve_memory_refs(10, (refs[3],), owner="test-owner")
    expired = manager.expire_before(15)
    assert len(expired) == 2
    assert expired[0].reservation_id == res_10.reservation_id
    assert expired[0].state is ReservationState.EXPIRED
    assert expired[1].reservation_id == res_15.reservation_id
    assert expired[1].state is ReservationState.EXPIRED
    assert (
        manager.get_reservation(res_20.reservation_id).state is ReservationState.ACTIVE
    )
    assert (
        manager.get_reservation(res_none.reservation_id).state
        is ReservationState.ACTIVE
    )
    assert manager.get_slot(refs[0]).state is MemorySlotState.FREE
    assert manager.get_slot(refs[1]).state is MemorySlotState.FREE
    assert manager.get_slot(refs[2]).state is MemorySlotState.RESERVED
    assert manager.get_slot(refs[3]).state is MemorySlotState.RESERVED


def test_expire_before_ignores_occupied_slots() -> None:
    manager = ResourceManager()
    refs = manager.register_memory("alice", "qmem", num_positions=1)
    res = manager.reserve_memory_refs(10, (refs[0],), expires_at=10, owner="test-owner")
    manager.mark_occupied(refs[0])
    manager.expire_before(10)
    assert manager.get_reservation(res.reservation_id).state is ReservationState.EXPIRED
    assert manager.get_slot(refs[0]).state is MemorySlotState.OCCUPIED


def test_expire_before_validates_now() -> None:
    manager = ResourceManager()
    with pytest.raises(ValueError, match="must be non-negative"):
        manager.expire_before(-1)


def test_available_memories_filters_by_link_id() -> None:
    manager = ResourceManager()
    # Register memory without link_id metadata
    manager.register_memory("alice", "qmem_gen", num_positions=1)

    # Register memory with link_id metadata
    manager.register_memory(
        "alice", "qmem_link_1", num_positions=1, metadata=(("link_id", "link-1"),)
    )

    # Register memory with different link_id metadata
    manager.register_memory(
        "alice", "qmem_link_2", num_positions=1, metadata=(("link_id", "link-2"),)
    )

    # Without filter, all are returned
    all_memories = manager.available_memories(now=0)
    assert len(all_memories) == 3

    # Filter by link-1
    link_1_memories = manager.available_memories(now=0, link_id="link-1")
    assert len(link_1_memories) == 1
    assert link_1_memories[0].device_id == "qmem_link_1"

    # Filter by link-2
    link_2_memories = manager.available_memories(now=0, link_id="link-2")
    assert len(link_2_memories) == 1
    assert link_2_memories[0].device_id == "qmem_link_2"


def test_available_memories_combines_public_link_device_and_state_filters() -> None:
    manager = ResourceManager()
    ready = manager.register_memory(
        "alice",
        "ready",
        num_positions=1,
        metadata=(("link_id", "link-1"),),
    )[0]
    delayed = manager.register_memory(
        "alice",
        "delayed",
        num_positions=1,
        metadata=(("link_id", "link-1"),),
    )[0]
    other_link = manager.register_memory(
        "alice",
        "other_link",
        num_positions=1,
        metadata=(("link_id", "link-2"),),
    )[0]
    manager.mark_occupied(delayed)
    manager.mark_free(delayed)

    assert manager.available_memories(
        5,
        "alice",
        device_id="ready",
        link_id="link-1",
    ) == (ready,)
    assert manager.available_memories(
        5,
        "alice",
        device_id="delayed",
        link_id="link-1",
    ) == (delayed,)
    assert manager.available_memories(5, "alice", link_id="link-1") == (
        delayed,
        ready,
    )
    assert manager.available_memories(
        5,
        "alice",
        link_id="link-2",
    ) == (other_link,)
