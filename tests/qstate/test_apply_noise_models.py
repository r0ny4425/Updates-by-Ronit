from math import expm1

import numpy as np
import pytest

from simyuj.engine import Timeline
from simyuj.qstate import (
    BellDiagState,
    DensityState,
    KetState,
    NoiseError,
    QuantumStateManager,
    SubsystemId,
)
from simyuj.qstate.noise import (
    DepolarizingNoise,
    T1T2Noise,
    amplitude_damping,
    bit_flip,
    dephasing,
    depolarizing,
    phase_flip,
    two_qubit_depolarizing,
)

ATOL = 1e-12


def _density_payload(manager: QuantumStateManager, state_ref: int) -> DensityState:
    payload = manager.get(state_ref)
    assert isinstance(payload, DensityState)
    return payload


def _ket_payload(manager: QuantumStateManager, state_ref: int) -> KetState:
    payload = manager.get(state_ref)
    assert isinstance(payload, KetState)
    return payload


class FixedRNG:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def test_apply_noise_models_empty_returns_owner_without_conversion():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    returned = manager.apply_noise_models([], targets=(q0,))

    assert returned == state_ref
    assert manager.record(state_ref).rep == "ket"


def test_apply_noise_models_auto_converts_ket_to_density():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    returned = manager.apply_noise_models(
        [
            depolarizing(0.01),
            dephasing(0.02),
        ],
        targets=(q0,),
    )

    assert returned == state_ref
    assert manager.record(state_ref).rep == "density"
    assert _density_payload(manager, state_ref).rho == pytest.approx(
        np.array([[0.995, 0.0], [0.0, 0.005]], dtype=np.complex128),
        abs=ATOL,
    )


def test_apply_noise_models_rejects_wrong_arity_before_conversion():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    with pytest.raises(NoiseError):
        manager.apply_noise_models(
            [two_qubit_depolarizing(0.01)],
            targets=(q0,),
        )

    assert manager.record(state_ref).rep == "ket"


def test_apply_noise_models_rejects_non_sequence():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    manager.prepare("|0>", rep="ket", subsystems=(q0,))

    with pytest.raises(TypeError):
        manager.apply_noise_models(
            depolarizing(0.01),
            targets=(q0,),
        )


def test_apply_noise_models_accepts_time_dependent_noise_with_duration():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    rate_hz = 1e6
    duration_s = 1e-9

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    returned = manager.apply_noise_models(
        [DepolarizingNoise(rate_hz=rate_hz)],
        targets=(q0,),
        duration_s=duration_s,
    )

    assert returned == state_ref
    assert manager.record(state_ref).rep == "density"
    p = -expm1(-rate_hz * duration_s)
    assert _density_payload(manager, state_ref).rho == pytest.approx(
        np.array([[1.0 - p / 2.0, 0.0], [0.0, p / 2.0]], dtype=np.complex128),
        abs=ATOL,
    )


def test_apply_noise_models_rejects_time_dependent_noise_without_duration():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    with pytest.raises(ValueError, match="requires duration_s"):
        manager.apply_noise_models(
            [DepolarizingNoise(rate_hz=1e6)],
            targets=(q0,),
        )

    assert manager.record(state_ref).rep == "ket"


def test_apply_noise_models_accepts_mixed_fixed_and_time_dependent_noise():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    returned = manager.apply_noise_models(
        [
            depolarizing(0.001),
            DepolarizingNoise(rate_hz=1e6),
            T1T2Noise(T1=1e-3, T2=5e-4),
        ],
        targets=(q0,),
        duration_s=1e-9,
    )

    assert returned == state_ref
    assert manager.record(state_ref).rep == "density"


def test_apply_noise_models_rejects_negative_duration():
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")

    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))

    with pytest.raises(ValueError, match="duration_s must be non-negative"):
        manager.apply_noise_models(
            [DepolarizingNoise(rate_hz=1e6)],
            targets=(q0,),
            duration_s=-1.0,
        )

    assert manager.record(state_ref).rep == "ket"


def test_manager_rejects_invalid_noise_mode() -> None:
    with pytest.raises(ValueError, match="noise_mode"):
        QuantumStateManager(noise_mode="sample")


def test_sampled_ket_phase_flip_remains_ket() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket", noise_rng=FixedRNG(0.0))
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|+>", rep="ket", subsystems=(q0,))

    returned = manager.apply_noise_models([phase_flip(1.0)], targets=(q0,))

    assert returned == state_ref
    assert manager.record(state_ref).rep == "ket"
    assert _ket_payload(manager, state_ref).vector == pytest.approx(
        np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0),
        abs=ATOL,
    )


def test_sampled_ket_amplitude_damping_full_decay_remains_ket() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket", noise_rng=FixedRNG(0.0))
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|1>", rep="ket", subsystems=(q0,))

    manager.apply_noise_models([amplitude_damping(1.0)], targets=(q0,))

    assert manager.record(state_ref).rep == "ket"
    assert _ket_payload(manager, state_ref).vector == pytest.approx(
        np.array([1.0, 0.0], dtype=np.complex128),
        abs=ATOL,
    )


def test_sampled_ket_amplitude_damping_superposition_branches() -> None:
    q0 = SubsystemId("q0")
    gamma = 0.25
    no_jump_probability = 0.875
    expected_no_jump = np.array(
        [1.0, np.sqrt(1.0 - gamma)], dtype=np.complex128
    ) / np.sqrt(2.0 * no_jump_probability)

    no_jump = QuantumStateManager(
        noise_mode="sampled_ket",
        noise_rng=FixedRNG(0.1),
    )
    no_jump_ref = no_jump.prepare("|+>", rep="ket", subsystems=(q0,))
    no_jump.apply_noise_models([amplitude_damping(gamma)], targets=(q0,))
    assert _ket_payload(no_jump, no_jump_ref).vector == pytest.approx(
        expected_no_jump,
        abs=ATOL,
    )

    jump = QuantumStateManager(noise_mode="sampled_ket", noise_rng=FixedRNG(0.9))
    jump_ref = jump.prepare("|+>", rep="ket", subsystems=(q0,))
    jump.apply_noise_models([amplitude_damping(gamma)], targets=(q0,))
    assert _ket_payload(jump, jump_ref).vector == pytest.approx(
        np.array([1.0, 0.0], dtype=np.complex128),
        abs=ATOL,
    )


def test_sampled_ket_noise_on_entangled_axis_updates_shared_ket() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket", noise_rng=FixedRNG(0.75))
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("phi+", rep="ket", subsystems=(q0, q1))

    manager.apply_noise_models([amplitude_damping(1.0)], targets=(q1,))

    assert _ket_payload(manager, state_ref).vector == pytest.approx(
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
        abs=ATOL,
    )


def test_sampled_ket_noise_requires_rng_only_for_non_empty_ket_noise() -> None:
    q0 = SubsystemId("q0")

    manager = QuantumStateManager(noise_mode="sampled_ket")
    state_ref = manager.prepare("|0>", rep="ket", subsystems=(q0,))
    assert manager.apply_noise_models([], targets=(q0,)) == state_ref

    with pytest.raises(ValueError, match="noise_rng"):
        manager.apply_noise_models([phase_flip(0.1)], targets=(q0,))

    density_manager = QuantumStateManager(noise_mode="sampled_ket")
    density_ref = density_manager.prepare("|0>", rep="density", subsystems=(q0,))
    assert (
        density_manager.apply_noise_models([bit_flip(1.0)], targets=(q0,))
        == density_ref
    )
    assert density_manager.record(density_ref).rep == "density"


def test_sampled_ket_mode_keeps_density_noise_exact() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket")
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|0>", rep="density", subsystems=(q0,))

    manager.apply_noise_models([bit_flip(0.25)], targets=(q0,))

    assert _density_payload(manager, state_ref).rho == pytest.approx(
        np.array([[0.75, 0.0], [0.0, 0.25]], dtype=np.complex128),
        abs=ATOL,
    )


def test_sampled_ket_mode_keeps_supported_bell_diag_noise_compact() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket")
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("phi+", rep="bell_diag", subsystems=(q0, q1))

    manager.apply_noise_models([phase_flip(1.0)], targets=(q1,))

    payload = manager.get(state_ref)
    assert isinstance(payload, BellDiagState)
    assert payload.probs == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=ATOL)


def test_sampled_ket_mode_converts_unsupported_bell_diag_noise_to_density() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket")
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = manager.prepare("phi+", rep="bell_diag", subsystems=(q0, q1))

    manager.apply_noise_models([amplitude_damping(0.25)], targets=(q0,))

    assert manager.record(state_ref).rep == "density"
    assert isinstance(manager.get(state_ref), DensityState)


def test_sampled_ket_mode_resolves_time_dependent_noise_before_sampling() -> None:
    manager = QuantumStateManager(noise_mode="sampled_ket", noise_rng=FixedRNG(0.99))
    q0 = SubsystemId("q0")
    state_ref = manager.prepare("|1>", rep="ket", subsystems=(q0,))

    manager.apply_noise_models(
        [T1T2Noise(T1=1.0, T2=0.0)],
        targets=(q0,),
        duration_s=1.0,
    )

    assert manager.record(state_ref).rep == "ket"
    assert _ket_payload(manager, state_ref).vector == pytest.approx(
        np.array([1.0, 0.0], dtype=np.complex128),
        abs=ATOL,
    )


def test_timeline_sampled_ket_noise_mode_predeclares_noise_rng() -> None:
    timeline = Timeline(master_seed=7, qstate_noise_mode="sampled_ket")
    q0 = SubsystemId("q0")
    state_ref = timeline.qstate.prepare("|1>", rep="ket", subsystems=(q0,))

    timeline.qstate.apply_noise_models([amplitude_damping(1.0)], targets=(q0,))

    assert timeline.qstate.record(state_ref).rep == "ket"
    assert _ket_payload(timeline.qstate, state_ref).vector == pytest.approx(
        np.array([1.0, 0.0], dtype=np.complex128),
        abs=ATOL,
    )
