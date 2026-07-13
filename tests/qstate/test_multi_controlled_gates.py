from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate import KetState, QuantumStateManager, StateRef, SubsystemId
from simyuj.qstate.ops import CCX, CCZ, CNOT, CSWAP, CZ, FREDKIN, MCX, MCZ, TOFFOLI
from simyuj.qstate.state import basis


def _sid(name: str) -> SubsystemId:
    return SubsystemId(name)


def _ket(manager: QuantumStateManager, state_ref: StateRef) -> KetState:
    state = manager.get(state_ref)
    assert isinstance(state, KetState)
    return state


def test_ccx_and_toffoli_flip_target_when_both_controls_are_one() -> None:
    c0 = _sid("c0")
    c1 = _sid("c1")
    target = _sid("target")

    manager = QuantumStateManager()
    ref = manager.prepare("110", subsystems=(c0, c1, target))

    out_ref = manager.apply(CCX, targets=(c0, c1, target))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("111").vector)

    manager = QuantumStateManager()
    ref = manager.prepare("110", subsystems=(c0, c1, target))

    out_ref = manager.apply(TOFFOLI, targets=(c0, c1, target))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("111").vector)


def test_ccx_does_not_flip_when_any_control_is_zero() -> None:
    c0 = _sid("c0")
    c1 = _sid("c1")
    target = _sid("target")

    manager = QuantumStateManager()
    ref = manager.prepare("100", subsystems=(c0, c1, target))

    out_ref = manager.apply(CCX, targets=(c0, c1, target))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("100").vector)


def test_ccz_applies_phase_to_all_ones_component() -> None:
    c0 = _sid("c0")
    c1 = _sid("c1")
    target = _sid("target")
    vector = np.ones(8, dtype=np.complex128) / np.sqrt(8.0)
    expected = vector.copy()
    expected[-1] *= -1.0

    manager = QuantumStateManager()
    ref = manager.prepare(KetState(vector), subsystems=(c0, c1, target))

    out_ref = manager.apply(CCZ, targets=(c0, c1, target))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(expected)


def test_fredkin_and_cswap_swap_only_when_control_is_one() -> None:
    control = _sid("control")
    left = _sid("left")
    right = _sid("right")

    manager = QuantumStateManager()
    ref = manager.prepare("101", subsystems=(control, left, right))

    out_ref = manager.apply(CSWAP, targets=(control, left, right))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("110").vector)

    manager = QuantumStateManager()
    ref = manager.prepare("101", subsystems=(control, left, right))

    out_ref = manager.apply(FREDKIN, targets=(control, left, right))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("110").vector)


def test_fredkin_does_not_swap_when_control_is_zero() -> None:
    control = _sid("control")
    left = _sid("left")
    right = _sid("right")

    manager = QuantumStateManager()
    ref = manager.prepare("001", subsystems=(control, left, right))

    out_ref = manager.apply(CSWAP, targets=(control, left, right))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("001").vector)


def test_mcx_supports_more_than_two_controls() -> None:
    c0 = _sid("c0")
    c1 = _sid("c1")
    c2 = _sid("c2")
    target = _sid("target")

    manager = QuantumStateManager()
    ref = manager.prepare("1110", subsystems=(c0, c1, c2, target))

    out_ref = manager.apply(MCX(3), targets=(c0, c1, c2, target))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(basis("1111").vector)


def test_mcz_supports_more_than_two_controls() -> None:
    c0 = _sid("c0")
    c1 = _sid("c1")
    c2 = _sid("c2")
    target = _sid("target")
    vector = np.ones(16, dtype=np.complex128) / np.sqrt(16.0)
    expected = vector.copy()
    expected[-1] *= -1.0

    manager = QuantumStateManager()
    ref = manager.prepare(KetState(vector), subsystems=(c0, c1, c2, target))

    out_ref = manager.apply(MCZ(3), targets=(c0, c1, c2, target))

    assert out_ref == ref
    assert _ket(manager, out_ref).vector == pytest.approx(expected)


def test_single_control_factories_match_existing_two_qubit_gates() -> None:
    mcx = MCX(1)
    assert mcx.name == "CNOT"
    assert mcx.arity == CNOT.arity
    assert tuple(mcx.matrix.ravel()) == pytest.approx(tuple(CNOT.matrix.ravel()))

    mcz = MCZ(1)
    assert mcz.name == "CZ"
    assert mcz.arity == CZ.arity
    assert tuple(mcz.matrix.ravel()) == pytest.approx(tuple(CZ.matrix.ravel()))


def test_multi_controlled_gate_factories_validate_control_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        MCX(0)

    with pytest.raises(ValueError, match="positive"):
        MCZ(0)

    with pytest.raises(TypeError, match="int"):
        MCX(1.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="int"):
        MCZ(1.0)  # type: ignore[arg-type]
