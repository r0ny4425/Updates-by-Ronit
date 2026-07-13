from __future__ import annotations

import random
from collections.abc import Iterable
from typing import cast

import pytest

from simyuj.qstate import KetState, QuantumStateManager, StateRef, SubsystemId
from simyuj.qstate.errors import DimensionError, InvalidOperationError
from simyuj.qstate.ops import (
    CNOT,
    CRX,
    CRY,
    CRZ,
    RX,
    RY,
    RZ,
    SWAP,
    CPhase,
    H,
    Phase,
    X,
    Z,
)

PI = 3.141592653589793
INV_SQRT2 = 0.5**0.5
ATOL = 1e-12


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def _ket(manager: QuantumStateManager, state_ref: StateRef) -> KetState:
    payload = manager.get(state_ref)
    assert isinstance(payload, KetState)
    return payload


def _assert_vector(actual: object, expected: tuple[complex, ...]) -> None:
    observed = tuple(complex(value) for value in cast(Iterable[complex], actual))
    assert observed == pytest.approx(expected, abs=ATOL)


def _assert_same_up_to_phase(actual: object, expected: tuple[complex, ...]) -> None:
    observed = tuple(complex(value) for value in cast(Iterable[complex], actual))
    pivot = next(index for index, value in enumerate(expected) if abs(value) > ATOL)
    phase = observed[pivot] / expected[pivot]
    assert abs(phase) == pytest.approx(1.0, abs=ATOL)
    assert observed == pytest.approx(
        tuple(phase * value for value in expected),
        abs=ATOL,
    )


def test_prepare_and_measure_builtin_single_qubit_kets() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    q2 = q("q2")

    manager.prepare("|0>", subsystems=(q0,))
    manager.prepare("|+>", subsystems=(q1,))
    manager.prepare("|+i>", subsystems=(q2,))

    assert manager.measure(targets=(q0,), basis="z").label == "0"
    assert manager.measure(targets=(q1,), basis="x").label == "+"
    assert manager.measure(targets=(q2,), basis="y").label == "+i"


def test_single_qubit_gates_update_ket_payloads() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|0>", subsystems=(q0,))
    manager.apply(X, targets=(q0,))
    _assert_vector(_ket(manager, ref).vector, (0.0 + 0.0j, 1.0 + 0.0j))

    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|+>", subsystems=(q0,))
    manager.apply(Z, targets=(q0,))
    _assert_vector(
        _ket(manager, ref).vector,
        (INV_SQRT2 + 0.0j, -INV_SQRT2 + 0.0j),
    )

    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|0>", subsystems=(q0,))
    manager.apply(H, targets=(q0,))
    _assert_vector(
        _ket(manager, ref).vector,
        (INV_SQRT2 + 0.0j, INV_SQRT2 + 0.0j),
    )


def test_prepare_validates_required_layout_against_ket_payload() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")

    with pytest.raises(ValueError, match="subsystems or layout"):
        manager.prepare("|0>")

    with pytest.raises(DimensionError, match="layout does not match"):
        manager.prepare("|01>", subsystems=(q0,))


def test_apply_rejects_target_count_that_does_not_match_gate_arity() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    manager.prepare("|0>", subsystems=(q0,))

    with pytest.raises(InvalidOperationError, match="target count"):
        manager.apply(CNOT, targets=(q0,))


def test_rotations_are_supported_on_ket_states() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|0>", subsystems=(q0,))
    manager.apply(RX(PI), targets=(q0,))
    _assert_same_up_to_phase(
        _ket(manager, ref).vector,
        (0.0 + 0.0j, 1.0 + 0.0j),
    )

    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|0>", subsystems=(q0,))
    manager.apply(RY(PI), targets=(q0,))
    _assert_same_up_to_phase(
        _ket(manager, ref).vector,
        (0.0 + 0.0j, 1.0 + 0.0j),
    )

    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|+>", subsystems=(q0,))
    manager.apply(RZ(PI), targets=(q0,))
    _assert_same_up_to_phase(
        _ket(manager, ref).vector,
        (INV_SQRT2 + 0.0j, -INV_SQRT2 + 0.0j),
    )


def test_phase_gate_is_supported_on_ket_states() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    ref = manager.prepare("|+>", subsystems=(q0,))

    manager.apply(Phase(PI), targets=(q0,))

    _assert_vector(
        _ket(manager, ref).vector,
        (INV_SQRT2 + 0.0j, -INV_SQRT2 + 0.0j),
    )


def test_controlled_parameterized_gates_are_supported_on_ket_states() -> None:
    control = q("control")
    target = q("target")

    manager = QuantumStateManager()
    ref = manager.prepare("|11>", subsystems=(control, target))
    manager.apply(CPhase(PI), targets=(control, target))
    _assert_vector(_ket(manager, ref).vector, (0.0, 0.0, 0.0, -1.0))

    manager = QuantumStateManager()
    ref = manager.prepare("|10>", subsystems=(control, target))
    manager.apply(CRX(PI), targets=(control, target))
    _assert_same_up_to_phase(_ket(manager, ref).vector, (0.0, 0.0, 0.0, 1.0))

    manager = QuantumStateManager()
    ref = manager.prepare("|10>", subsystems=(control, target))
    manager.apply(CRY(PI), targets=(control, target))
    _assert_same_up_to_phase(_ket(manager, ref).vector, (0.0, 0.0, 0.0, 1.0))

    manager = QuantumStateManager()
    ref = manager.prepare(
        (0.0, 0.0, INV_SQRT2, INV_SQRT2),
        subsystems=(control, target),
    )
    manager.apply(CRZ(PI), targets=(control, target))
    _assert_same_up_to_phase(
        _ket(manager, ref).vector,
        (0.0, 0.0, INV_SQRT2, -INV_SQRT2),
    )


def test_two_qubit_gate_across_separate_states_tensors_and_consumes_inputs() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    ref0 = manager.prepare("|1>", subsystems=(q0,))
    ref1 = manager.prepare("|0>", subsystems=(q1,))

    merged_ref = manager.apply(CNOT, targets=(q0, q1))

    assert merged_ref == 2
    assert not manager.store.contains_state(ref0)
    assert not manager.store.contains_state(ref1)
    assert manager.state_of(q0) == merged_ref
    assert manager.state_of(q1) == merged_ref
    assert manager.location_of(q0).axis == 0
    assert manager.location_of(q1).axis == 1
    assert manager.measure(targets=(q0, q1), basis="z").outcome == (1, 1)


def test_swap_preserves_target_order_inside_combined_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    manager.prepare("|1>", subsystems=(q0,))
    manager.prepare("|0>", subsystems=(q1,))

    manager.apply(SWAP, targets=(q0, q1))

    assert manager.measure(targets=(q0, q1), basis="z").outcome == (0, 1)


def test_measurement_targets_must_already_share_one_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    manager.prepare("|0>", subsystems=(q0,))
    manager.prepare("|0>", subsystems=(q1,))

    with pytest.raises(InvalidOperationError, match="one live state"):
        manager.measure(targets=(q0, q1), basis="z")


def test_probabilistic_measurement_requires_rng_and_collapse_is_stored() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare("|+>", subsystems=(q0,))

    with pytest.raises(ValueError, match="explicit rng"):
        manager.measure(targets=(q0,), basis="z")

    result = manager.measure(targets=(q0,), basis="z", rng=random.Random(3))

    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    assert result.probability == pytest.approx(0.5, abs=ATOL)
    assert manager.measure(targets=(q0,), basis="z").outcome == result.outcome


def test_measure_without_collapse_does_not_replace_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare("|+>", subsystems=(q0,))
    stored_state = manager.get(state_ref)

    result = manager.measure(
        targets=(q0,),
        basis="z",
        rng=random.Random(8),
        collapse=False,
    )

    assert result.post_state_ref is None
    assert result.post_state is None
    assert manager.get(state_ref) is stored_state
