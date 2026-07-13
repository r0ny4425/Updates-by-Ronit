import random

import numpy as np
import pytest

from simyuj.qstate import (
    POVM,
    BellDiagState,
    DensityState,
    POVMElement,
    QuantumStateManager,
    QuantumStateRecord,
    StateLayout,
    SubsystemId,
)
from simyuj.qstate.errors import (
    InvalidOperationError,
    MeasurementError,
    NoiseError,
    StateNotFoundError,
    StateOwnershipError,
)
from simyuj.qstate.noise import bit_flip, imperfect_cnot, imperfect_cz, phase_flip
from simyuj.qstate.ops import X
from simyuj.qstate.state import KetState, ghz
from simyuj.qstate.state.convert import ket_to_density
from simyuj.qstate.state.make import basis


def _z_povm() -> POVM:
    return POVM(
        (
            POVMElement("zero", [[1, 0], [0, 0]]),
            POVMElement("one", [[0, 0], [0, 1]]),
        ),
        name="z",
    )


def _computational_two_qubit_povm() -> POVM:
    elements = []
    for index, label in enumerate(("00", "01", "10", "11")):
        effect = np.zeros((4, 4), dtype=np.complex128)
        effect[index, index] = 1.0
        elements.append(POVMElement(label, effect))
    return POVM(tuple(elements), name="zz")


def _assert_density(actual: DensityState, expected: DensityState) -> None:
    np.testing.assert_allclose(actual.rho, expected.rho, atol=1e-12)


def test_manager_put_exposes_store_ownership_lookup() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    record = QuantumStateRecord("payload", "ket", StateLayout((q0,), (2,)))

    state_ref = manager.put(record)

    assert state_ref == 0
    assert manager.get(state_ref) == "payload"
    assert manager.state_of(q0) == state_ref
    assert manager.location_of(q0).axis == 0


def test_relabel_subsystem_updates_single_qubit_ownership() -> None:
    manager = QuantumStateManager()
    photon = SubsystemId("photon:A:10")
    memory = SubsystemId("memory:A:0")
    state_ref = manager.prepare("|0>", subsystems=(photon,))

    assert manager.relabel_subsystem(photon, memory) == state_ref

    record = manager.record(state_ref)
    assert record.layout.subsystems == (memory,)
    assert record.layout.dims == (2,)
    assert manager.state_of(memory) == state_ref
    assert manager.location_of(memory).axis == 0
    assert manager.location_of(memory).dim == 2
    with pytest.raises(StateNotFoundError, match="not owned"):
        manager.state_of(photon)


def test_relabel_subsystem_preserves_entangled_joint_payload() -> None:
    manager = QuantumStateManager()
    photon = SubsystemId("photon:A:10")
    partner = SubsystemId("photon:B:10")
    memory = SubsystemId("memory:A:0")
    emitted = SubsystemId("photon:A:emit:1")
    state_ref = manager.prepare("phi+", subsystems=(photon, partner))
    payload = manager.get(state_ref)

    assert manager.relabel_subsystem(photon, memory) == state_ref
    assert manager.get(state_ref) is payload
    assert manager.record(state_ref).layout.subsystems == (memory, partner)
    assert manager.location_of(memory).axis == 0
    assert manager.location_of(partner).axis == 1

    assert manager.relabel_subsystem(memory, emitted) == state_ref
    assert manager.get(state_ref) is payload
    assert manager.record(state_ref).layout.subsystems == (emitted, partner)
    assert manager.location_of(emitted).axis == 0
    assert manager.state_of(partner) == state_ref


def test_relabel_subsystem_rejects_owned_new_subsystem() -> None:
    manager = QuantumStateManager()
    photon = SubsystemId("photon:A:10")
    occupied = SubsystemId("memory:A:0")
    state_ref = manager.prepare("|0>", subsystems=(photon,))
    occupied_ref = manager.prepare("|1>", subsystems=(occupied,))

    with pytest.raises(StateOwnershipError, match="already owned"):
        manager.relabel_subsystem(photon, occupied)

    assert manager.state_of(photon) == state_ref
    assert manager.state_of(occupied) == occupied_ref


def test_relabel_subsystem_rejects_missing_old_subsystem() -> None:
    manager = QuantumStateManager()
    missing = SubsystemId("photon:A:missing")
    memory = SubsystemId("memory:A:0")

    with pytest.raises(StateNotFoundError, match="not owned"):
        manager.relabel_subsystem(missing, memory)


def test_manager_prepare_accepts_explicit_ghz_state() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    q2 = SubsystemId("q2")

    state_ref = manager.prepare(ghz(3), subsystems=(q0, q1, q2))
    payload = manager.get(state_ref)

    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0 / np.sqrt(2.0)
    expected[-1] = 1.0 / np.sqrt(2.0)

    assert isinstance(payload, KetState)
    np.testing.assert_allclose(payload.vector, expected, atol=1e-12)


def test_measure_bell_on_prepared_bell_ket_updates_store_refs() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("phi+", subsystems=(q0, q1))

    result = manager.measure_bell(targets=(q0, q1))

    assert result.label == "phi+"
    assert result.outcome == (0, 0)
    assert result.probability == pytest.approx(1.0)
    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    assert isinstance(manager.get(state_ref), KetState)


def test_measure_bell_combines_separate_ket_states_before_dispatch() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    manager.prepare("|0>", subsystems=(q0,))
    manager.prepare("|0>", subsystems=(q1,))

    result = manager.measure_bell(targets=(q0, q1), rng=random.Random(3))

    assert result.label == "phi+"
    assert result.probability == pytest.approx(0.5)
    assert result.state_ref is not None
    assert result.post_state_ref == result.state_ref
    assert manager.size() == 1
    assert manager.state_of(q0) == result.state_ref
    assert manager.state_of(q1) == result.state_ref
    assert isinstance(manager.get(result.state_ref), KetState)


def test_measure_bell_uses_bell_diag_handler_and_preserves_rep() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare(
        {"phi+": 0.7, "phi-": 0.3},
        rep="bell_diag",
        subsystems=(q0, q1),
    )

    result = manager.measure_bell(targets=(q0, q1), rng=random.Random(3))

    assert result.label == "phi+"
    assert result.probability == pytest.approx(0.7)
    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    record = manager.record(state_ref)
    assert record.rep == "bell_diag"
    assert isinstance(record.payload, BellDiagState)
    assert record.payload.probs == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_measure_bell_rejects_combining_different_representations() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    manager.prepare("|0>", subsystems=(q0,))
    manager.prepare("|0>", rep="density", subsystems=(q1,))

    with pytest.raises(InvalidOperationError, match="different representations"):
        manager.measure_bell(targets=(q0, q1), rng=random.Random(3))


def test_measure_bell_requires_exactly_two_targets() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    q2 = SubsystemId("q2")

    with pytest.raises(MeasurementError, match="exactly two"):
        manager.measure_bell(targets=(q0,))
    with pytest.raises(MeasurementError, match="exactly two"):
        manager.measure_bell(targets=(q0, q1, q2))


def test_measure_povm_converts_ket_to_density_and_updates_refs() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|1>", subsystems=(q0,))

    result = manager.measure_povm(_z_povm(), targets=(q0,))

    assert result.outcome == 1
    assert result.label == "one"
    assert result.probability == pytest.approx(1.0)
    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    record = manager.record(state_ref)
    assert record.rep == "density"
    assert isinstance(record.payload, DensityState)
    _assert_density(record.payload, ket_to_density(basis("1")))


def test_measure_povm_combines_separate_ket_states_before_density_measure() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    ref0 = manager.prepare("|0>", subsystems=(q0,))
    ref1 = manager.prepare("|1>", subsystems=(q1,))

    result = manager.measure_povm(
        _computational_two_qubit_povm(),
        targets=(q0, q1),
    )

    assert result.label == "01"
    assert result.outcome == 1
    assert result.state_ref is not None
    assert result.post_state_ref == result.state_ref
    assert manager.store.contains_state(result.state_ref)
    assert not manager.store.contains_state(ref0)
    assert not manager.store.contains_state(ref1)
    assert manager.size() == 1
    assert manager.state_of(q0) == result.state_ref
    assert manager.state_of(q1) == result.state_ref
    record = manager.record(result.state_ref)
    assert record.rep == "density"
    _assert_density(record.payload, ket_to_density(basis("01")))


def test_measure_povm_without_collapse_preserves_stored_representation() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|1>", subsystems=(q0,))

    result = manager.measure_povm(_z_povm(), targets=(q0,), collapse=False)

    assert result.label == "one"
    assert result.state_ref == state_ref
    assert result.post_state_ref is None
    assert result.post_state is None
    assert manager.record(state_ref).rep == "ket"
    assert isinstance(manager.get(state_ref), KetState)


def test_discard_reduces_density_payload_and_updates_ownership() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("|10>", subsystems=(q0, q1))

    assert manager.discard(targets=(q1,)) == state_ref

    record = manager.record(state_ref)
    assert record.rep == "density"
    assert record.layout.subsystems == (q0,)
    assert manager.state_of(q0) == state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        manager.state_of(q1)
    _assert_density(record.payload, ket_to_density(basis("1")))


def test_measure_and_discard_matches_measure_then_discard() -> None:
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")

    expected = QuantumStateManager()
    expected_ref = expected.prepare("phi+", subsystems=(q0, q1))
    expected_result = expected.measure(targets=(q0,), rng=random.Random(2))
    assert expected.discard(targets=(q0,)) == expected_ref

    actual = QuantumStateManager()
    actual_ref = actual.prepare("phi+", subsystems=(q0, q1))
    actual_result = actual.measure_and_discard(targets=(q0,), rng=random.Random(2))

    assert actual_result.outcome == expected_result.outcome
    assert actual_result.outcome_labels == expected_result.outcome_labels
    assert actual_result.probability == pytest.approx(expected_result.probability)
    assert actual_result.probabilities == expected_result.probabilities
    assert actual_result.state_ref == expected_result.state_ref
    assert actual_result.post_state_ref == expected_result.post_state_ref
    assert actual_result.post_state is not None
    assert actual.record(actual_ref).layout == expected.record(expected_ref).layout
    actual_payload = actual.get(actual_ref)
    expected_payload = expected.get(expected_ref)
    assert isinstance(actual_payload, DensityState)
    assert isinstance(expected_payload, DensityState)
    _assert_density(actual_payload, expected_payload)


def test_measure_and_discard_without_collapse_matches_measure_then_discard() -> None:
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")

    expected = QuantumStateManager()
    expected_ref = expected.prepare("phi+", subsystems=(q0, q1))
    expected_result = expected.measure(
        targets=(q0,),
        rng=random.Random(2),
        collapse=False,
    )
    assert expected.discard(targets=(q0,)) == expected_ref

    actual = QuantumStateManager()
    actual_ref = actual.prepare("phi+", subsystems=(q0, q1))
    actual_result = actual.measure_and_discard(
        targets=(q0,),
        rng=random.Random(2),
        collapse=False,
    )

    assert actual_result.outcome == expected_result.outcome
    assert actual_result.post_state is None
    assert actual_result.post_state_ref is None
    assert actual.record(actual_ref).layout == expected.record(expected_ref).layout
    actual_payload = actual.get(actual_ref)
    expected_payload = expected.get(expected_ref)
    assert isinstance(actual_payload, DensityState)
    assert isinstance(expected_payload, DensityState)
    _assert_density(actual_payload, expected_payload)


def test_measure_and_discard_all_targets_deletes_state_after_measurement() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|0>", subsystems=(q0,))

    result = manager.measure_and_discard(targets=(q0,))

    assert result.label == "0"
    assert result.state_ref == state_ref
    assert result.post_state_ref == state_ref
    assert manager.size() == 0
    with pytest.raises(StateNotFoundError):
        manager.record(state_ref)


def test_discarded_state_can_still_be_measured_and_updated() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("|10>", subsystems=(q0, q1))

    assert manager.discard(targets=(q1,)) == state_ref
    assert manager.apply(X, targets=(q0,)) == state_ref
    result = manager.measure(targets=(q0,), basis="z")

    assert result.label == "0"
    assert result.state_ref == state_ref
    assert manager.state_of(q0) == state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        manager.state_of(q1)


def test_discard_all_targets_deletes_state() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|0>", subsystems=(q0,))

    assert manager.discard(targets=(q0,)) is None

    assert manager.size() == 0
    with pytest.raises(StateNotFoundError):
        manager.record(state_ref)
    with pytest.raises(StateNotFoundError):
        manager.state_of(q0)


def test_discard_requires_targets_from_one_state() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    manager.prepare("|0>", subsystems=(q0,))
    manager.prepare("|1>", subsystems=(q1,))

    with pytest.raises(InvalidOperationError, match="one live state"):
        manager.discard(targets=(q0, q1))


def test_reset_converts_to_density_and_preserves_layout() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("|10>", subsystems=(q0, q1))

    assert manager.reset(targets=(q1,), state="1") == state_ref

    record = manager.record(state_ref)
    assert record.rep == "density"
    assert record.layout.subsystems == (q0, q1)
    _assert_density(record.payload, ket_to_density(basis("11")))


def test_reset_entangled_state_uses_density_math() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("phi+", subsystems=(q0, q1))

    assert manager.reset(targets=(q0,), state="0") == state_ref

    record = manager.record(state_ref)
    assert record.rep == "density"
    assert record.layout.subsystems == (q0, q1)
    _assert_density(record.payload, DensityState(np.diag([0.5, 0.5, 0.0, 0.0])))


def test_reset_requires_targets_from_one_state() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    manager.prepare("|0>", subsystems=(q0,))
    manager.prepare("|1>", subsystems=(q1,))

    with pytest.raises(InvalidOperationError, match="one live state"):
        manager.reset(targets=(q0, q1))


def test_manager_converts_between_ket_and_density_representations() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|0>", subsystems=(q0,), meta={"source": "ket"})

    assert manager.convert(state_ref, "density") == state_ref
    density_record = manager.record(state_ref)
    assert density_record.rep == "density"
    assert isinstance(density_record.payload, DensityState)
    assert density_record.meta == (("source", "ket"),)

    assert manager.convert(state_ref, "ket", meta={"source": "density"}) == state_ref
    ket_record = manager.record(state_ref)
    assert ket_record.rep == "ket"
    assert isinstance(ket_record.payload, KetState)
    assert ket_record.meta == (("source", "density"),)

    assert manager.convert(state_ref, "ket", meta={"ignored": True}) == state_ref
    assert manager.record(state_ref).meta == (("source", "density"),)


def test_apply_noise_rejects_ket_representation() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    manager.prepare("|0>", subsystems=(q0,))

    with pytest.raises(NoiseError, match="requires density"):
        manager.apply_noise(phase_flip(0.1), targets=(q0,))


def test_apply_noise_updates_density_payload() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|0>", rep="density", subsystems=(q0,))

    assert manager.apply_noise(bit_flip(0.25), targets=(q0,)) == state_ref

    payload = manager.get(state_ref)
    assert isinstance(payload, DensityState)
    assert payload.rho == pytest.approx(np.array([[0.75, 0.0], [0.0, 0.25]]))


def test_apply_noise_updates_bell_diag_with_pauli_channel() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("phi+", rep="bell_diag", subsystems=(q0, q1))

    assert manager.apply_noise(phase_flip(1.0), targets=(q1,)) == state_ref

    payload = manager.get(state_ref)
    assert isinstance(payload, BellDiagState)
    assert payload.probs == pytest.approx((0.0, 1.0, 0.0, 0.0))


def test_apply_noise_validates_target_count_against_channel_arity() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    manager.prepare("|00>", rep="density", subsystems=(q0, q1))

    with pytest.raises(NoiseError, match="target count"):
        manager.apply_noise(bit_flip(0.1), targets=(q0, q1))


def test_manager_applies_imperfect_cnot_channel() -> None:
    manager = QuantumStateManager()
    control = SubsystemId("q0")
    target = SubsystemId("q1")
    state_ref_0 = manager.prepare("|1>", rep="density", subsystems=(control,))
    state_ref_1 = manager.prepare("|0>", rep="density", subsystems=(target,))

    state_ref = manager.apply_noise(
        imperfect_cnot(0.0),
        targets=(control, target),
    )

    assert state_ref not in (state_ref_0, state_ref_1)

    payload = manager.get(state_ref)
    assert isinstance(payload, DensityState)
    assert payload.rho == pytest.approx(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
    )


def test_manager_applies_imperfect_cz_channel() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare(
        "phi+",
        rep="density",
        subsystems=(q0, q1),
    )

    result_ref = manager.apply_noise(
        imperfect_cz(0.0),
        targets=(q0, q1),
    )

    assert result_ref == state_ref

    payload = manager.get(result_ref)
    assert isinstance(payload, DensityState)

    expected_vector = np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2.0)
    expected = np.outer(expected_vector, expected_vector.conj())
    assert payload.rho == pytest.approx(expected)
