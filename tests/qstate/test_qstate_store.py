import pytest

from simyuj.engine.timeline import Timeline
from simyuj.qstate import (
    QuantumStateRecord,
    QuantumStateStore,
    StateLayout,
    StateNotFoundError,
    StateOwnershipError,
    SubsystemId,
)


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def rec(payload: object, subsystems: tuple[SubsystemId, ...]) -> QuantumStateRecord:
    return QuantumStateRecord(
        payload=payload,
        rep="ket",
        layout=StateLayout(subsystems, (2,) * len(subsystems)),
    )


def test_store_refs_are_deterministic_and_get_delete_work() -> None:
    store = QuantumStateStore()
    q0 = q("q0")
    q1 = q("q1")
    record0 = rec("payload0", (q0,))
    record1 = rec("payload1", (q1,))

    ref0 = store.put(record0)
    ref1 = store.put(record1)

    assert ref0 == 0
    assert ref1 == 1
    assert store.get(ref0) == "payload0"
    assert store.record(ref1) is record1
    assert store.size() == 2

    store.delete(ref0)

    assert store.size() == 1
    assert not store.contains_state(ref0)
    assert not store.contains_subsystem(q0)
    with pytest.raises(StateNotFoundError):
        store.record(ref0)


def test_store_enforces_unique_live_subsystem_ownership() -> None:
    store = QuantumStateStore()
    q0 = q("q0")
    store.put(rec("payload0", (q0,)))

    with pytest.raises(StateOwnershipError):
        store.put(rec("payload1", (q0,)))


def test_store_replace_keeps_owned_subsystem_and_reindexes_new_layout() -> None:
    store = QuantumStateStore()
    q0 = q("q0")
    q1 = q("q1")
    state_ref = store.put(rec("payload0", (q0,)))

    store.replace(state_ref, rec("replacement", (q0, q1)))

    assert store.get(state_ref) == "replacement"
    assert store.state_of(q0) == state_ref
    assert store.state_of(q1) == state_ref
    assert store.location_of(q0).axis == 0
    assert store.location_of(q1).axis == 1
    store.assert_consistent()


def test_consume_and_put_removes_old_refs_and_reindexes_locations() -> None:
    store = QuantumStateStore()
    q0 = q("q0")
    q1 = q("q1")
    ref0 = store.put(rec("payload0", (q0,)))
    ref1 = store.put(rec("payload1", (q1,)))

    merged_ref = store.consume_and_put((ref0, ref1), rec("merged", (q0, q1)))

    assert merged_ref == 2
    assert not store.contains_state(ref0)
    assert not store.contains_state(ref1)
    assert store.contains_state(merged_ref)
    assert store.state_of(q0) == merged_ref
    assert store.location_of(q0).axis == 0
    assert store.location_of(q1).axis == 1
    store.assert_consistent()


def test_consume_and_put_rejects_non_consumed_live_owner() -> None:
    store = QuantumStateStore()
    q0 = q("q0")
    q1 = q("q1")
    ref0 = store.put(rec("payload0", (q0,)))
    store.put(rec("payload1", (q1,)))

    with pytest.raises(StateOwnershipError):
        store.consume_and_put((ref0,), rec("merged", (q0, q1)))


def test_timeline_owned_qstate_refs_are_local_to_each_timeline() -> None:
    t1 = Timeline()
    t2 = Timeline()
    q1 = q("q0")
    q2 = q("q0")

    ref1 = t1.qstate.prepare("|0>", subsystems=(q1,))
    ref2 = t2.qstate.prepare("|1>", subsystems=(q2,))

    assert ref1 == 0
    assert ref2 == 0
    assert t1.qstate.state_of(q1) == ref1
    assert t2.qstate.state_of(q2) == ref2
