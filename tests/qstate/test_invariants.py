from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate import (
    QuantumStateManager,
    QuantumStateRecord,
    StateLayout,
    SubsystemId,
)
from simyuj.qstate.debug import (
    assert_layout_matches_payload,
    assert_record_ok,
    assert_store_ok,
    assert_unique_subsystems,
)
from simyuj.qstate.errors import InvalidLayoutError, StateOwnershipError


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def test_store_invariants_pass_after_common_lifecycle_operations() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")

    manager.prepare("|+>", subsystems=(q0,))
    manager.prepare("|0>", subsystems=(q1,))
    assert_store_ok(manager.store)

    from simyuj.qstate.ops import CNOT

    manager.apply(CNOT, targets=(q0, q1))
    assert_store_ok(manager.store)

    manager.convert(manager.state_of(q0), "density")
    assert_store_ok(manager.store)

    result = manager.measure(
        targets=(q0,),
        basis="z",
        collapse=True,
        rng=np.random.default_rng(1),
    )
    assert result.state_ref is not None
    assert_store_ok(manager.store)


def test_store_invariants_pass_after_bell_and_reset_operations() -> None:
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


def test_record_invariant_rejects_payload_layout_mismatch() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")

    state_ref = manager.prepare("|0>", subsystems=(q0,))
    payload = manager.get(state_ref)

    bad_record = QuantumStateRecord(
        payload=payload,
        rep="ket",
        layout=StateLayout((q("a"), q("b")), (2, 2)),
    )

    with pytest.raises(InvalidLayoutError):
        assert_layout_matches_payload(bad_record)


def test_unique_subsystem_check_catches_duplicate_in_fake_store() -> None:
    q0 = q("q0")

    manager = QuantumStateManager()
    ref0 = manager.prepare("|0>", subsystems=(q("a"),))
    ref1 = manager.prepare("|1>", subsystems=(q("b"),))
    rec0 = manager.record(ref0)
    rec1 = manager.record(ref1)

    duplicate_rec0 = QuantumStateRecord(
        payload=rec0.payload,
        rep=rec0.rep,
        layout=StateLayout((q0,), (2,)),
    )
    duplicate_rec1 = QuantumStateRecord(
        payload=rec1.payload,
        rep=rec1.rep,
        layout=StateLayout((q0,), (2,)),
    )

    class FakeStore:
        _records = {
            0: duplicate_rec0,
            1: duplicate_rec1,
        }

    with pytest.raises(StateOwnershipError, match="multiple live locations"):
        assert_unique_subsystems(FakeStore())


def test_record_ok_accepts_real_records() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")

    state_ref = manager.prepare("|0>", subsystems=(q0,))
    record = manager.record(state_ref)

    assert_record_ok(record)
