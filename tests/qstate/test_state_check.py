from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate import QuantumStateManager
from simyuj.qstate.errors import DimensionError, InvalidLayoutError, InvalidStateError
from simyuj.qstate.space import StateLayout, SubsystemId
from simyuj.qstate.state.bell_diag import BellDiagState
from simyuj.qstate.state.check import (
    assert_payload_layout_compatible,
    check_bell_diag,
    check_density,
    check_ket,
    check_payload,
    density_is_pure,
    density_purity,
    is_bell_diag,
    is_density,
    is_ket,
    ket_global_phase_aligned_vector,
    payload_hilbert_dim,
    payload_num_qubits,
)
from simyuj.qstate.state.density import DensityState
from simyuj.qstate.state.ket import KetState
from simyuj.qstate.state.make import basis, plus


def _layout(*names: str, dims: tuple[int, ...] | None = None) -> StateLayout:
    return StateLayout(
        tuple(SubsystemId(name) for name in names),
        dims if dims is not None else (2,) * len(names),
    )


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def _unchecked_ket(vector: object) -> KetState:
    state = object.__new__(KetState)
    object.__setattr__(state, "vector", np.asarray(vector, dtype=np.complex128))
    return state


def _unchecked_density(rho: object) -> DensityState:
    state = object.__new__(DensityState)
    object.__setattr__(state, "rho", np.asarray(rho, dtype=np.complex128))
    return state


def test_state_type_predicates_and_payload_dispatch() -> None:
    ket = basis("0")
    density = DensityState([[1, 0], [0, 0]])
    bell_diag = BellDiagState.from_label("phi+")

    assert is_ket(ket)
    assert is_density(density)
    assert is_bell_diag(bell_diag)
    assert check_payload(ket) is ket
    assert check_payload(density) is density
    assert check_payload(bell_diag) is bell_diag
    assert check_payload(ket, rep="ket") is ket

    with pytest.raises(TypeError, match="KetState"):
        check_payload(density, rep="ket")
    with pytest.raises(InvalidStateError, match="unsupported"):
        check_payload(object())
    with pytest.raises(InvalidStateError, match="unsupported"):
        check_payload(ket, rep="graph")


def test_state_checks_accept_manager_prepared_payloads() -> None:
    manager = QuantumStateManager()

    ket_ref = manager.prepare("|+>", subsystems=(q("q0"),))
    density_ref = manager.prepare("|0>", rep="density", subsystems=(q("q1"),))
    bell_ref = manager.prepare(
        "phi+",
        rep="bell_diag",
        subsystems=(q("a"), q("b")),
    )

    ket_record = manager.record(ket_ref)
    density_record = manager.record(density_ref)
    bell_record = manager.record(bell_ref)

    assert is_ket(ket_record.payload)
    assert is_density(density_record.payload)
    assert is_bell_diag(bell_record.payload)

    assert check_ket(ket_record.payload) is ket_record.payload
    assert check_density(density_record.payload) is density_record.payload
    assert check_bell_diag(bell_record.payload) is bell_record.payload

    assert check_payload(ket_record.payload) is ket_record.payload
    assert check_payload(density_record.payload) is density_record.payload
    assert check_payload(bell_record.payload) is bell_record.payload

    assert payload_num_qubits(bell_record.payload) == 2
    assert payload_hilbert_dim(bell_record.payload) == 4
    assert_payload_layout_compatible(ket_record.payload, ket_record.layout)
    assert_payload_layout_compatible(density_record.payload, density_record.layout)
    assert_payload_layout_compatible(bell_record.payload, bell_record.layout)


def test_check_ket_validates_vector_invariants_and_phase_alignment() -> None:
    ket = KetState([0, 1j])

    assert check_ket(ket) is ket
    np.testing.assert_allclose(ket_global_phase_aligned_vector(ket), [0, 1])

    with pytest.raises(TypeError, match="KetState"):
        check_ket(object())
    with pytest.raises(InvalidStateError, match="one-dimensional"):
        check_ket(_unchecked_ket([[1, 0]]))
    with pytest.raises(DimensionError, match="power of two"):
        check_ket(_unchecked_ket([1, 0, 0]))
    with pytest.raises(InvalidStateError, match="finite"):
        check_ket(_unchecked_ket([np.inf, 0]))
    with pytest.raises(InvalidStateError, match="normalized"):
        check_ket(_unchecked_ket([1, 1]))


def test_check_density_validates_matrix_invariants_and_purity() -> None:
    pure = DensityState([[1, 0], [0, 0]])
    mixed = DensityState([[0.5, 0], [0, 0.5]])

    assert check_density(pure) is pure
    assert density_purity(pure) == pytest.approx(1.0)
    assert density_is_pure(pure)
    assert density_purity(mixed) == pytest.approx(0.5)
    assert not density_is_pure(mixed)

    with pytest.raises(TypeError, match="DensityState"):
        check_density(object())
    with pytest.raises(InvalidStateError, match="two-dimensional"):
        check_density(_unchecked_density([1, 0]))
    with pytest.raises(DimensionError, match="square"):
        check_density(_unchecked_density([[1, 0]]))
    with pytest.raises(InvalidStateError, match="finite"):
        check_density(_unchecked_density([[np.inf, 0], [0, 1]]))
    with pytest.raises(InvalidStateError, match="Hermitian"):
        check_density(_unchecked_density([[1, 1], [0, 0]]))
    with pytest.raises(InvalidStateError, match="trace"):
        check_density(_unchecked_density([[0.5, 0], [0, 0]]))
    with pytest.raises(InvalidStateError, match="positive"):
        check_density(_unchecked_density([[1.1, 0], [0, -0.1]]))


def test_check_bell_diag_validates_probabilities() -> None:
    state = BellDiagState((0.5, 0.5, 0.0, 0.0))

    assert check_bell_diag(state) is state

    with pytest.raises(TypeError, match="BellDiagState"):
        check_bell_diag(object())

    bad = object.__new__(BellDiagState)
    object.__setattr__(bad, "probs", (1.0, 0.0, 0.0))
    with pytest.raises(InvalidStateError, match="four"):
        check_bell_diag(bad)

    bad = object.__new__(BellDiagState)
    object.__setattr__(bad, "probs", (0.5, 0.5, 0.5, 0.0))
    with pytest.raises(InvalidStateError, match="sum"):
        check_bell_diag(bad)


def test_payload_dimension_and_layout_compatibility_helpers() -> None:
    ket = basis("01")
    density = DensityState([[1.0]])

    assert payload_num_qubits(ket) == 2
    assert payload_hilbert_dim(ket) == 4
    assert payload_num_qubits(density) == 0
    assert payload_hilbert_dim(density) == 1

    assert_payload_layout_compatible(ket, _layout("q0", "q1"))
    assert_payload_layout_compatible(density, StateLayout((), ()))

    with pytest.raises(TypeError, match="StateLayout"):
        assert_payload_layout_compatible(ket, object())  # type: ignore[arg-type]
    with pytest.raises(InvalidLayoutError, match="layout size"):
        assert_payload_layout_compatible(ket, _layout("q0"))
    with pytest.raises(InvalidLayoutError, match="qubit axes"):
        assert_payload_layout_compatible(ket, _layout("q0", "q1", dims=(1, 4)))


def test_phase_aligned_vector_rejects_empty_amplitude_only_defensively() -> None:
    bad = _unchecked_ket([0, 0])

    with pytest.raises(InvalidStateError, match="normalized"):
        ket_global_phase_aligned_vector(bad)

    aligned = ket_global_phase_aligned_vector(plus())
    np.testing.assert_allclose(aligned, [2**-0.5, 2**-0.5])
