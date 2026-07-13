from __future__ import annotations

import pytest

from simyuj.qstate import QuantumStateManager, QuantumStateRecord, StateLayout
from simyuj.qstate.debug import (
    assert_layout_matches_payload,
    assert_record_ok,
    assert_store_ok,
    assert_unique_subsystems,
    assert_valid_state,
    iter_store_records,
)
from simyuj.qstate.errors import InvalidLayoutError, StateOwnershipError
from simyuj.qstate.record import SubsystemLocation
from simyuj.qstate.space import SubsystemId
from simyuj.qstate.state.make import basis


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def _layout(*names: str) -> StateLayout:
    return StateLayout(tuple(q(name) for name in names), (2,) * len(names))


def _record(state: object, *names: str) -> QuantumStateRecord:
    return QuantumStateRecord(state, "ket", _layout(*names))


def test_record_invariants_accept_valid_record() -> None:
    record = _record(basis("0"), "q0")

    assert_valid_state(record)
    assert_layout_matches_payload(record)
    assert_record_ok(record)


def test_record_invariants_reject_layout_payload_mismatch() -> None:
    record = _record(basis("0"), "q0", "q1")

    with pytest.raises(InvalidLayoutError, match="layout size"):
        assert_layout_matches_payload(record)
    with pytest.raises(InvalidLayoutError, match="layout size"):
        assert_record_ok(record)


def test_record_invariants_reject_invalid_payload_for_rep() -> None:
    record = QuantumStateRecord("not-a-ket", "ket", _layout("q0"))

    with pytest.raises(TypeError, match="KetState"):
        assert_valid_state(record)
    with pytest.raises(TypeError, match="KetState"):
        assert_record_ok(record)


def test_store_invariants_accept_real_store() -> None:
    manager = QuantumStateManager()
    manager.prepare("|0>", subsystems=(q("q0"),))
    manager.prepare("|1>", subsystems=(q("q1"),))

    assert_store_ok(manager.store)
    assert iter_store_records(manager.store) == tuple(
        sorted(manager.store._records.items(), key=lambda pair: pair[0])
    )


def test_store_invariants_pass_after_common_lifecycle_operations() -> None:
    import random

    from simyuj.qstate.ops import CNOT

    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")

    manager.prepare("|+>", subsystems=(q0,))
    manager.prepare("|0>", subsystems=(q1,))
    assert_store_ok(manager.store)

    manager.apply(CNOT, targets=(q0, q1))
    assert_store_ok(manager.store)

    manager.convert(manager.state_of(q0), "density")
    assert_store_ok(manager.store)

    result = manager.measure(
        targets=(q0,), basis="z", collapse=True, rng=random.Random(1)
    )
    assert result.state_ref is not None
    assert_store_ok(manager.store)


def test_store_invariants_pass_after_bell_discard_and_reset_operations() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")

    manager.prepare("phi+", subsystems=(q0, q1))
    assert_store_ok(manager.store)

    manager.measure_bell(targets=(q0, q1))
    assert_store_ok(manager.store)

    manager.reset(targets=(q1,), state="|0>")
    assert_store_ok(manager.store)

    manager.discard(targets=(q1,))
    assert_store_ok(manager.store)


def test_iter_store_records_accepts_items_and_records_shapes() -> None:
    record0 = _record(basis("0"), "q0")
    record1 = _record(basis("1"), "q1")

    class ItemsStore:
        def items(self):
            return ((4, record1), (2, record0))

    class RecordsMappingStore:
        def records(self):
            return {7: record1, 3: record0}

    class RecordsIterableStore:
        def records(self):
            return ((9, record1), (1, record0))

    assert iter_store_records(ItemsStore()) == ((2, record0), (4, record1))
    assert iter_store_records(RecordsMappingStore()) == ((3, record0), (7, record1))
    assert iter_store_records(RecordsIterableStore()) == ((1, record0), (9, record1))


def test_store_invariants_reject_duplicate_subsystems() -> None:
    q0 = q("q0")

    class FakeStore:
        _records = {
            0: QuantumStateRecord(basis("0"), "ket", StateLayout((q0,), (2,))),
            1: QuantumStateRecord(basis("1"), "ket", StateLayout((q0,), (2,))),
        }

    with pytest.raises(StateOwnershipError, match="multiple live locations"):
        assert_unique_subsystems(FakeStore())


def test_store_invariants_reject_owner_and_location_mismatch() -> None:
    q0 = q("q0")
    record = QuantumStateRecord(basis("0"), "ket", StateLayout((q0,), (2,)))

    class OwnerMismatchStore:
        _records = {0: record}

        def state_of(self, subsystem: SubsystemId) -> int:
            return 1

    class LocationMismatchStore:
        _records = {0: record}

        def location_of(self, subsystem: SubsystemId) -> SubsystemLocation:
            return SubsystemLocation(state_ref=0, axis=1, dim=2)

    with pytest.raises(StateOwnershipError, match="owner mismatch"):
        assert_unique_subsystems(OwnerMismatchStore())
    with pytest.raises(StateOwnershipError, match="axis mismatch"):
        assert_unique_subsystems(LocationMismatchStore())


def test_store_invariants_reject_size_and_state_ref_mismatch() -> None:
    record = _record(basis("0"), "q0")

    class SizeMismatchStore:
        _records = {0: record}

        def size(self) -> int:
            return 2

    class BadRefStore:
        _records = {-1: record}

    with pytest.raises(StateOwnershipError, match="size mismatch"):
        assert_store_ok(SizeMismatchStore())
    with pytest.raises(StateOwnershipError, match="non-negative"):
        assert_store_ok(BadRefStore())


def test_iter_store_records_rejects_unknown_or_malformed_store() -> None:
    record = _record(basis("0"), "q0")

    class BadItemsStore:
        def items(self):
            return ((0, record, "extra"),)

    with pytest.raises(TypeError, match="does not expose"):
        iter_store_records(object())
    with pytest.raises(TypeError, match="record items"):
        iter_store_records(BadItemsStore())
