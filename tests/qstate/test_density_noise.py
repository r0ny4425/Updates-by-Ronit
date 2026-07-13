from __future__ import annotations

import random
from typing import cast

import numpy as np
import pytest

from simyuj.qstate import DensityState, KetState, QuantumStateManager, SubsystemId
from simyuj.qstate.errors import InvalidStateError, NoiseError
from simyuj.qstate.noise import amplitude_damping, depolarizing, phase_flip
from simyuj.qstate.ops import CNOT, X
from simyuj.qstate.state import density_to_ket_if_pure, ket_to_density

ATOL = 1e-12


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def _density(manager: QuantumStateManager, state_ref: int) -> DensityState:
    payload = manager.get(state_ref)
    assert isinstance(payload, DensityState)
    return cast(DensityState, payload)


def test_ket_to_density_conversion_and_back_for_pure_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare("|+>", subsystems=(q0,))

    assert manager.convert(state_ref, "density") == state_ref
    record = manager.record(state_ref)
    assert record.rep == "density"
    assert isinstance(record.payload, DensityState)
    assert record.payload.rho == pytest.approx(
        np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),
        abs=ATOL,
    )

    assert manager.convert(state_ref, "ket") == state_ref
    assert isinstance(manager.get(state_ref), KetState)


@pytest.mark.parametrize(
    "rho",
    (
        [[1, 1], [0, 0]],
        [[2, 0], [0, 0]],
        [[1.1, 0], [0, -0.1]],
    ),
)
def test_density_state_validation_rejects_non_physical_matrices(
    rho: object,
) -> None:
    with pytest.raises(InvalidStateError):
        DensityState(rho)


def test_prepare_density_and_apply_unitary_on_density_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare("|0>", rep="density", subsystems=(q0,))

    manager.apply(X, targets=(q0,))

    assert _density(manager, state_ref).rho == pytest.approx(
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
        abs=ATOL,
    )
    assert manager.measure(targets=(q0,), basis="z").label == "1"


def test_density_two_qubit_gate_across_separate_states_tensors_inputs() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    ref0 = manager.prepare("|1>", rep="density", subsystems=(q0,))
    ref1 = manager.prepare("|0>", rep="density", subsystems=(q1,))

    merged_ref = manager.apply(CNOT, targets=(q0, q1))

    assert merged_ref == 2
    assert not manager.store.contains_state(ref0)
    assert not manager.store.contains_state(ref1)
    assert manager.measure(targets=(q0, q1), basis="z").outcome == (1, 1)


def test_density_measurement_collapse_updates_stored_density_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare("|+>", rep="density", subsystems=(q0,))

    with pytest.raises(ValueError, match="explicit rng"):
        manager.measure(targets=(q0,), basis="z")

    result = manager.measure(targets=(q0,), basis="z", rng=random.Random(3))

    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    assert result.probability == pytest.approx(0.5, abs=ATOL)
    assert manager.measure(targets=(q0,), basis="z").outcome == result.outcome


def test_phase_flip_noise_on_density_state_changes_x_eigenstate() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    manager.prepare("|+>", rep="density", subsystems=(q0,))

    manager.apply_noise(phase_flip(1.0), targets=(q0,))

    assert manager.measure(targets=(q0,), basis="x").label == "-"


def test_amplitude_damping_on_density_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare("|1>", rep="density", subsystems=(q0,))

    manager.apply_noise(amplitude_damping(0.25), targets=(q0,))

    rho = _density(manager, state_ref).rho
    assert rho[0, 0] == pytest.approx(0.25, abs=ATOL)
    assert rho[1, 1] == pytest.approx(0.75, abs=ATOL)
    assert rho[0, 1] == pytest.approx(0.0, abs=ATOL)


def test_depolarizing_noise_and_noise_requires_density_representation() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    state_ref = manager.prepare("|0>", rep="density", subsystems=(q0,))

    manager.apply_noise(depolarizing(1.0), targets=(q0,))

    assert _density(manager, state_ref).rho == pytest.approx(
        np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
        abs=ATOL,
    )

    manager.prepare("|0>", subsystems=(q1,))
    with pytest.raises(NoiseError, match="density"):
        manager.apply_noise(phase_flip(1.0), targets=(q1,))


def test_density_to_ket_rejects_mixed_state() -> None:
    mixed = DensityState(np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128))

    with pytest.raises(InvalidStateError):
        density_to_ket_if_pure(mixed)

    pure = ket_to_density(KetState([1, 0]))
    assert isinstance(density_to_ket_if_pure(pure), KetState)


def test_manager_convert_rejects_mixed_density_state_to_ket() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    state_ref = manager.prepare(
        np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
        rep="density",
        subsystems=(q0,),
    )

    with pytest.raises(InvalidStateError):
        manager.convert(state_ref, "ket")
