from __future__ import annotations

import random
from typing import cast

import numpy as np
import pytest

from simyuj.qstate import (
    BellDiagState,
    BellResult,
    DensityState,
    KetState,
    QuantumStateManager,
    SubsystemId,
)
from simyuj.qstate.errors import InvalidStateError, MeasurementError
from simyuj.qstate.measure.bell import bell_density_matrix, bell_projector, bell_vector
from simyuj.qstate.ops import (
    PauliFrame,
    X,
    Z,
    correction_for_bell,
    correction_for_entanglement_swap,
    identity_frame,
    update_after_bsm,
    update_after_swap,
    update_after_teleport,
)
from simyuj.qstate.state import (
    bell,
    bell_diag_to_density,
    bell_fidelity,
    density_to_bell_diag_if_exact,
    fidelity,
    make_bell_diag,
    make_ket,
    purity,
)

ATOL = 1e-12


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def _array(value: object) -> np.ndarray:
    return cast(np.ndarray, value)


def test_bell_ket_constructors_and_projectors_are_canonical() -> None:
    phi_plus = make_ket("phi+")
    psi_minus = bell("psi-")
    expected_psi_minus = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0)

    assert isinstance(phi_plus, KetState)
    assert isinstance(psi_minus, KetState)
    assert _array(phi_plus.vector) == pytest.approx(bell_vector("phi+"))
    assert _array(psi_minus.vector) == pytest.approx(expected_psi_minus)
    assert bell_projector("phi+") == pytest.approx(bell_density_matrix("phi+"))


def test_manager_measures_prepared_bell_ket_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    state_ref = manager.prepare("phi+", subsystems=(q0, q1))

    result = manager.measure_bell(targets=(q0, q1))

    assert isinstance(result, BellResult)
    assert result.label == "phi+"
    assert result.outcome == (0, 0)
    assert result.probability == pytest.approx(1.0, abs=ATOL)
    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    assert isinstance(manager.get(state_ref), KetState)


def test_manager_bell_measurement_combines_separate_input_states() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    ref0 = manager.prepare("|0>", subsystems=(q0,))
    ref1 = manager.prepare("|0>", subsystems=(q1,))

    result = manager.measure_bell(
        targets=(q0, q1),
        rng=random.Random(3),
        collapse=False,
    )

    assert result.label in {"phi+", "phi-"}
    assert result.probability == pytest.approx(0.5, abs=ATOL)
    assert result.state_ref == 2
    assert result.state_ref not in (ref0, ref1)
    assert result.post_state_ref is None
    assert manager.state_of(q0) == 2
    assert manager.state_of(q1) == 2
    assert manager.size() == 1


def test_density_bell_measurement_and_bell_fidelity() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    state_ref = manager.prepare("psi-", rep="density", subsystems=(q0, q1))
    payload = manager.get(state_ref)

    assert isinstance(payload, DensityState)
    assert bell_fidelity(payload, "psi-") == pytest.approx(1.0, abs=ATOL)

    result = manager.measure_bell(targets=(q0, q1))

    assert result.label == "psi-"
    assert result.outcome == (1, 1)
    assert isinstance(manager.get(state_ref), DensityState)


def test_bell_diagonal_prepare_measure_and_conversion() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    state_ref = manager.prepare(
        {"phi+": 0.7, "phi-": 0.3},
        rep="bell_diag",
        subsystems=(q0, q1),
    )
    payload = manager.get(state_ref)

    assert isinstance(payload, BellDiagState)
    assert payload.fidelity("phi+") == pytest.approx(0.7, abs=ATOL)
    assert bell_fidelity(payload) == pytest.approx(0.7, abs=ATOL)
    assert purity(payload) == pytest.approx(0.58, abs=ATOL)

    result = manager.measure_bell(
        targets=(q0, q1),
        rng=random.Random(3),
        collapse=False,
    )
    assert result.label == "phi+"

    expected_density = bell_diag_to_density(payload)
    manager.convert(state_ref, "density")
    density = manager.get(state_ref)
    assert isinstance(density, DensityState)
    assert _array(density.rho) == pytest.approx(expected_density.rho)

    round_trip = density_to_bell_diag_if_exact(density)
    assert round_trip.probs == pytest.approx((0.7, 0.3, 0.0, 0.0), abs=ATOL)


def test_entanglement_swap_correction_accounts_for_input_bell_frames() -> None:
    manager = QuantumStateManager()
    alice = q("alice")
    relay_left = q("relay_left")
    relay_right = q("relay_right")
    bob = q("bob")

    left_ref = manager.prepare(
        "phi+",
        rep="bell_diag",
        subsystems=(alice, relay_left),
    )
    right_ref = manager.prepare(
        "phi-",
        rep="bell_diag",
        subsystems=(bob, relay_right),
    )

    manager.convert(left_ref, "density")
    manager.convert(right_ref, "density")
    bsm = manager.measure_bell(
        targets=(relay_left, relay_right),
        rng=np.random.default_rng(42),
    )
    assert bsm.label == "psi-"

    manager.discard(targets=(relay_left, relay_right))
    ab_ref = manager.state_of(alice)
    assert manager.state_of(bob) == ab_ref

    ab_payload = manager.get(ab_ref)
    assert isinstance(ab_payload, DensityState)
    uncorrected = density_to_bell_diag_if_exact(ab_payload)
    assert uncorrected.probs == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=ATOL)

    x_bit, z_bit = correction_for_entanglement_swap("phi+", "phi-", bsm.label)
    assert (x_bit, z_bit) == (1, 0)
    if x_bit:
        manager.apply(X, targets=(bob,))
    if z_bit:
        manager.apply(Z, targets=(bob,))

    corrected_payload = manager.get(ab_ref)
    assert isinstance(corrected_payload, DensityState)
    corrected = density_to_bell_diag_if_exact(corrected_payload)
    assert corrected.probs == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=ATOL)


def test_pure_bell_diagonal_converts_to_density_and_ket() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    state_ref = manager.prepare("phi-", rep="bell_diag", subsystems=(q0, q1))

    manager.convert(state_ref, "density")
    assert isinstance(manager.get(state_ref), DensityState)
    manager.convert(state_ref, "bell_diag")
    assert isinstance(manager.get(state_ref), BellDiagState)
    manager.convert(state_ref, "ket")
    ket = manager.get(state_ref)

    assert isinstance(ket, KetState)
    assert bell_fidelity(ket, "phi-") == pytest.approx(1.0, abs=ATOL)


def test_mixed_bell_diagonal_to_ket_conversion_is_rejected() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")
    q1 = q("q1")
    mixed = make_bell_diag((0.5, 0.5, 0.0, 0.0))
    state_ref = manager.prepare(mixed, rep="bell_diag", subsystems=(q0, q1))

    with pytest.raises(InvalidStateError, match="pure"):
        manager.convert(state_ref, "ket")


def test_bell_result_and_invalid_target_contracts() -> None:
    result = BellResult(
        label="Phi+",
        outcome=(0, 0),
        probability=1,
        probabilities=(("phi+", 1.0),),
        meta={"kind": "bsm"},
    )

    assert result.label == "phi+"
    assert result.outcome_label == "phi+"
    assert result.probability == pytest.approx(1.0, abs=ATOL)
    assert result.meta == (("kind", "bsm"),)

    manager = QuantumStateManager()
    with pytest.raises(MeasurementError, match="exactly two"):
        manager.measure_bell(targets=(q("q0"),))


def test_pauli_frame_updates_from_bell_outcomes() -> None:
    frame = identity_frame(2)

    assert isinstance(frame, PauliFrame)
    assert correction_for_bell("phi+") == (0, 0)
    assert correction_for_bell("psi-") == (1, 1)
    assert update_after_bsm(frame, "psi-", qubit=1) == PauliFrame(
        (0, 1),
        (0, 1),
    )
    assert update_after_teleport(frame, "phi-", qubit=0) == PauliFrame(
        (0, 0),
        (1, 0),
    )

    swap_x, swap_z = update_after_swap("psi+", "phi-")
    assert frame.with_correction(0, x=swap_x, z=swap_z) == PauliFrame(
        (1, 0),
        (1, 0),
    )


def test_fidelity_helpers_cover_supported_bell_states() -> None:
    phi_plus = bell("phi+")
    phi_minus = bell("phi-")
    bell_diag = make_bell_diag({"phi+": 0.7, "phi-": 0.3})

    assert fidelity(phi_plus, phi_plus) == pytest.approx(1.0, abs=ATOL)
    assert fidelity(phi_plus, phi_minus) == pytest.approx(0.0, abs=ATOL)
    assert fidelity(bell_diag, bell_diag) == pytest.approx(1.0, abs=ATOL)
