from __future__ import annotations

import numpy as np
import pytest

import simyuj.qstate.noise.kraus as kraus_module
from simyuj.qstate.errors import DimensionError, NoiseError
from simyuj.qstate.noise.base import (
    KrausChannel,
    NoiseChannel,
    check_kraus_channel,
    check_noise_channel,
)
from simyuj.qstate.noise.damping import amplitude_damping, generalized_amplitude_damping
from simyuj.qstate.noise.dephase import common_mode_dephasing, dephasing, phase_damping
from simyuj.qstate.noise.depolarize import depolarizing, two_qubit_depolarizing
from simyuj.qstate.noise.kraus import (
    apply_kraus_density,
    apply_kraus_ket_sampled,
    check_kraus,
)
from simyuj.qstate.noise.noisy_gates import imperfect_cnot, imperfect_cz
from simyuj.qstate.noise.pauli import (
    bit_flip,
    bit_phase_flip,
    pauli_channel,
    phase_flip,
    two_qubit_pauli_channel,
)
from simyuj.qstate.noise.t1t2 import T1T2NoiseModel, t1t2_noise_model
from simyuj.qstate.noise.time import DepolarizingNoise, T1T2Noise
from simyuj.qstate.space import StateLayout, SubsystemId
from simyuj.qstate.state.density import DensityHandler, DensityState
from simyuj.qstate.state.ket import KetState

ATOL = 1e-12
PSD_ATOL = 1e-10


def _pure_density(vector: np.ndarray) -> DensityState:
    return DensityState(np.outer(vector, vector.conj()))


def _assert_valid_density_matrix(
    rho: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> None:
    assert rho.shape == expected_shape
    assert rho == pytest.approx(rho.conj().T, abs=ATOL)
    assert np.trace(rho) == pytest.approx(1.0, abs=ATOL)
    assert np.min(np.linalg.eigvalsh(rho)) >= -PSD_ATOL


ZERO = DensityState([[1.0, 0.0], [0.0, 0.0]])
ONE = DensityState([[0.0, 0.0], [0.0, 1.0]])
PLUS = _pure_density(np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0))
BELL_PHI_PLUS = _pure_density(
    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
)
TWO_QUBIT_ZERO = DensityState(
    np.diag(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128))
)


@pytest.mark.parametrize(
    ("channel", "state", "axes"),
    [
        pytest.param(bit_flip(0.25), ZERO, (0,), id="bit_flip"),
        pytest.param(phase_flip(0.25), PLUS, (0,), id="phase_flip"),
        pytest.param(bit_phase_flip(0.25), ZERO, (0,), id="bit_phase_flip"),
        pytest.param(pauli_channel(0.1, 0.2, 0.3), PLUS, (0,), id="pauli"),
        pytest.param(depolarizing(0.4), ZERO, (0,), id="depolarizing"),
        pytest.param(dephasing(0.4), PLUS, (0,), id="dephasing"),
        pytest.param(phase_damping(0.4), PLUS, (0,), id="phase_damping"),
        pytest.param(amplitude_damping(0.25), ONE, (0,), id="amplitude_damping"),
        pytest.param(
            generalized_amplitude_damping(gamma=0.25, prob=0.65),
            ONE,
            (0,),
            id="generalized_amplitude_damping",
        ),
        pytest.param(
            T1T2NoiseModel(T1=4.0, T2=4.0, duration=1.0),
            PLUS,
            (0,),
            id="T1T2NoiseModel",
        ),
        pytest.param(
            t1t2_noise_model(T1=4.0, T2=4.0, duration=1.0),
            PLUS,
            (0,),
            id="t1t2_noise_model",
        ),
        pytest.param(
            DepolarizingNoise(rate_hz=2.0).resolve(duration_s=0.5),
            ZERO,
            (0,),
            id="DepolarizingNoise.resolve",
        ),
        pytest.param(
            T1T2Noise(T1=4.0, T2=4.0).resolve(duration_s=1.0),
            PLUS,
            (0,),
            id="T1T2Noise.resolve",
        ),
        pytest.param(
            two_qubit_pauli_channel({"XX": 0.2, "ZZ": 0.1}),
            TWO_QUBIT_ZERO,
            (0, 1),
            id="two_qubit_pauli_channel",
        ),
        pytest.param(
            two_qubit_depolarizing(0.32),
            TWO_QUBIT_ZERO,
            (0, 1),
            id="two_qubit_depolarizing",
        ),
        pytest.param(
            common_mode_dephasing(0.25),
            BELL_PHI_PLUS,
            (0, 1),
            id="common_mode_dephasing",
        ),
        pytest.param(
            imperfect_cnot(0.2),
            TWO_QUBIT_ZERO,
            (0, 1),
            id="imperfect_cnot",
        ),
        pytest.param(
            imperfect_cz(0.2),
            BELL_PHI_PLUS,
            (0, 1),
            id="imperfect_cz",
        ),
    ],
)
def test_public_noise_channels_preserve_density_matrix_invariants(
    channel: KrausChannel,
    state: DensityState,
    axes: tuple[int, ...],
) -> None:
    result = apply_kraus_density(state, channel, axes=axes)

    _assert_valid_density_matrix(result.rho, expected_shape=state.rho.shape)


def test_noise_channel_validates_name_and_arity() -> None:
    channel = NoiseChannel(name=" identity ", arity=1)

    assert channel.name == "identity"
    assert channel.arity == 1
    assert check_noise_channel(channel) is channel

    with pytest.raises(TypeError, match="name"):
        NoiseChannel(name=1, arity=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        NoiseChannel(name=" ", arity=1)
    with pytest.raises(TypeError, match="arity"):
        NoiseChannel(name="bad", arity=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        NoiseChannel(name="bad", arity=0)
    with pytest.raises(TypeError, match="NoiseChannel"):
        check_noise_channel("bad")


def test_kraus_channel_validates_ops_and_completeness() -> None:
    identity = np.eye(2, dtype=np.complex128)
    channel = KrausChannel((identity,), name=" identity ", arity=1)

    assert channel.name == "identity"
    assert channel.arity == 1
    assert len(channel.ops) == 1
    assert channel.ops[0] == pytest.approx(identity)
    assert not channel.ops[0].flags.writeable
    assert check_noise_channel(channel) is channel
    assert check_kraus_channel(channel) is channel

    with pytest.raises(TypeError, match="tuple"):
        KrausChannel([identity])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        KrausChannel(())
    with pytest.raises(DimensionError, match="arity"):
        KrausChannel((np.eye(4),), arity=1)
    with pytest.raises(NoiseError, match="completeness"):
        KrausChannel((0.5 * identity,), arity=1)
    with pytest.raises(TypeError, match="KrausChannel"):
        check_kraus_channel(NoiseChannel(name="generic", arity=1))


def test_kraus_channel_supports_multi_qubit_arity() -> None:
    identity = np.eye(4, dtype=np.complex128)
    channel = KrausChannel((identity,), arity=2)

    assert channel.arity == 2
    assert channel.ops[0].shape == (4, 4)


def test_apply_kraus_density_applies_bit_flip_channel() -> None:
    probability = 0.25
    identity = np.eye(2, dtype=np.complex128)
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    channel = KrausChannel(
        (
            np.sqrt(1.0 - probability) * identity,
            np.sqrt(probability) * x_gate,
        ),
        name="bit_flip",
    )
    state = DensityState([[1, 0], [0, 0]])

    result = apply_kraus_density(state, channel, axes=(0,))

    assert check_kraus(channel) is channel
    assert result.rho == pytest.approx(np.array([[0.75, 0.0], [0.0, 0.25]]))


def test_apply_kraus_ket_sampled_uses_direct_axis_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = KetState([1, 0, 0, 0])
    channel = bit_flip(1.0)

    def fail_expand(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("sampled ket noise should not expand full operators")

    monkeypatch.setattr(kraus_module, "expand_operator", fail_expand)

    result = apply_kraus_ket_sampled(
        state,
        channel,
        axes=(1,),
        rng=None,
    )

    assert result.vector == pytest.approx(
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),
        abs=ATOL,
    )


def test_apply_kraus_density_rejects_trace_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = KrausChannel((np.eye(2, dtype=np.complex128),), arity=1)
    state = DensityState([[1, 0], [0, 0]])

    def scaled_expand(
        matrix: object,
        *,
        axes: tuple[int, ...],
        num_qubits: int,
    ) -> np.ndarray:
        del axes, num_qubits
        return 0.5 * np.asarray(matrix, dtype=np.complex128)

    monkeypatch.setattr(kraus_module, "expand_operator", scaled_expand)

    with pytest.raises(NoiseError, match="did not preserve trace"):
        apply_kraus_density(state, channel, axes=(0,))


def test_apply_kraus_density_validates_inputs() -> None:
    channel = KrausChannel((np.eye(2, dtype=np.complex128),), arity=1)
    state = DensityState([[1, 0], [0, 0]])

    with pytest.raises(TypeError, match="DensityState"):
        apply_kraus_density("bad", channel, axes=(0,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KrausChannel"):
        apply_kraus_density(
            state,
            NoiseChannel(name="generic", arity=1),
            axes=(0,),
        )
    with pytest.raises(TypeError, match="tuple"):
        apply_kraus_density(state, channel, axes=[0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arity"):
        apply_kraus_density(state, channel, axes=(0, 1))


def test_density_handler_channel_delegates_to_kraus_application() -> None:
    identity = np.eye(2, dtype=np.complex128)
    channel = KrausChannel((identity,), arity=1)
    state = DensityState([[0, 0], [0, 1]])
    handler = DensityHandler()
    layout = StateLayout((SubsystemId("q0"),), (2,))

    result = handler.channel(state, channel, layout=layout, axes=(0,))

    assert isinstance(result, DensityState)
    assert result.rho == pytest.approx(state.rho)


def test_pauli_noise_constructors_build_one_qubit_kraus_channels() -> None:
    for channel, name in (
        (bit_flip(0.25), "bit_flip"),
        (phase_flip(0.25), "phase_flip"),
        (bit_phase_flip(0.25), "bit_phase_flip"),
    ):
        assert channel.name == name
        assert channel.arity == 1
        assert len(channel.ops) == 2

    state = DensityState([[1, 0], [0, 0]])
    result = apply_kraus_density(state, bit_flip(0.25), axes=(0,))

    assert result.rho == pytest.approx(np.array([[0.75, 0.0], [0.0, 0.25]]))


def test_pauli_channel_validates_probability_total() -> None:
    channel = pauli_channel(0.1, 0.2, 0.3)

    assert channel.name == "pauli"
    assert channel.arity == 1
    assert len(channel.ops) == 4

    with pytest.raises(TypeError, match="px"):
        pauli_channel("bad", 0.0, 0.0)
    with pytest.raises(ValueError, match="py"):
        pauli_channel(0.0, -0.1, 0.0)
    with pytest.raises(ValueError, match="at most 1"):
        pauli_channel(0.4, 0.4, 0.3)


def test_two_qubit_pauli_channel_builds_correlated_kraus_channel() -> None:
    channel = two_qubit_pauli_channel({"XX": 0.25, "ZZ": 0.10})

    assert channel.name == "two_qubit_pauli"
    assert channel.arity == 2
    assert len(channel.ops) == 3
    for op in channel.ops:
        assert op.shape == (4, 4)


def test_two_qubit_pauli_channel_applies_correlated_xx_error() -> None:
    channel = two_qubit_pauli_channel({"XX": 0.25})
    state = DensityState(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(
        np.array(
            [
                [0.75, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0.25],
            ],
            dtype=np.complex128,
        )
    )


def test_two_qubit_pauli_channel_labels_follow_target_order() -> None:
    channel = two_qubit_pauli_channel({"IX": 1.0})
    state = DensityState(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )


def test_two_qubit_pauli_channel_validates_inputs() -> None:
    with pytest.raises(TypeError, match="mapping"):
        two_qubit_pauli_channel([("XX", 0.1)])

    with pytest.raises(TypeError, match="str"):
        two_qubit_pauli_channel({1: 0.1})

    with pytest.raises(ValueError, match="unsupported"):
        two_qubit_pauli_channel({"AA": 0.1})

    with pytest.raises(ValueError, match="implicit"):
        two_qubit_pauli_channel({"II": 0.1})

    with pytest.raises(ValueError, match="duplicate"):
        two_qubit_pauli_channel({"xx": 0.1, "XX": 0.2})

    with pytest.raises(TypeError, match=r"p\[XX\]"):
        two_qubit_pauli_channel({"XX": "bad"})

    with pytest.raises(ValueError, match="at most 1"):
        two_qubit_pauli_channel({"XX": 0.6, "ZZ": 0.5})


def test_depolarizing_builds_one_qubit_mixing_channel() -> None:
    channel = depolarizing(0.3)

    assert channel.name == "pauli"
    assert channel.arity == 1
    assert len(channel.ops) == 4
    assert channel.ops[1] == pytest.approx(bit_flip(0.075).ops[1])
    assert channel.ops[2] == pytest.approx(bit_phase_flip(0.075).ops[1])
    assert channel.ops[3] == pytest.approx(phase_flip(0.075).ops[1])

    with pytest.raises(TypeError, match="p"):
        depolarizing("bad")
    with pytest.raises(ValueError, match="p"):
        depolarizing(1.1)


def test_depolarizing_p_one_returns_single_qubit_maximally_mixed() -> None:
    channel = depolarizing(1.0)
    state = DensityState([[1, 0], [0, 0]])

    result = apply_kraus_density(state, channel, axes=(0,))

    assert result.rho == pytest.approx(
        np.array(
            [
                [0.5, 0.0],
                [0.0, 0.5],
            ],
            dtype=np.complex128,
        )
    )


def test_two_qubit_depolarizing_builds_two_qubit_mixing_channel() -> None:
    channel = two_qubit_depolarizing(0.16)

    assert channel.name == "two_qubit_pauli"
    assert channel.arity == 2
    assert len(channel.ops) == 16
    for op in channel.ops:
        assert op.shape == (4, 4)


def test_two_qubit_depolarizing_applies_mixing_channel() -> None:
    channel = two_qubit_depolarizing(0.16)
    state = DensityState(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(
        np.array(
            [
                [0.88, 0.0, 0.0, 0.0],
                [0.0, 0.04, 0.0, 0.0],
                [0.0, 0.0, 0.04, 0.0],
                [0.0, 0.0, 0.0, 0.04],
            ],
            dtype=np.complex128,
        )
    )


def test_two_qubit_depolarizing_validates_probability() -> None:
    with pytest.raises(TypeError, match="p"):
        two_qubit_depolarizing("bad")

    with pytest.raises(ValueError, match="p"):
        two_qubit_depolarizing(-0.1)

    with pytest.raises(ValueError, match="p"):
        two_qubit_depolarizing(1.1)


def test_two_qubit_depolarizing_p_one_returns_two_qubit_maximally_mixed() -> None:
    channel = two_qubit_depolarizing(1.0)
    state = DensityState(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(np.eye(4, dtype=np.complex128) / 4.0)


def test_dephasing_aliases_phase_flip() -> None:
    expected = phase_flip(0.25)

    for channel in (dephasing(0.25), phase_damping(0.25)):
        assert channel.name == expected.name
        assert channel.arity == expected.arity
        assert len(channel.ops) == len(expected.ops)
        for observed, expected_op in zip(channel.ops, expected.ops):
            assert observed == pytest.approx(expected_op)

    with pytest.raises(TypeError, match="p"):
        dephasing("bad")
    with pytest.raises(ValueError, match="p"):
        phase_damping(1.1)


def test_common_mode_dephasing_builds_two_qubit_correlated_phase_channel() -> None:
    channel = common_mode_dephasing(0.25)

    assert channel.name == "two_qubit_pauli"
    assert channel.arity == 2
    assert len(channel.ops) == 2
    for op in channel.ops:
        assert op.shape == (4, 4)


def test_common_mode_dephasing_reduces_cross_parity_coherence() -> None:
    p = 0.25
    channel = common_mode_dephasing(p)
    vector = np.array([1, 1, 0, 0], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0, 1))

    expected = np.array(
        [
            [0.5, 0.25, 0.0, 0.0],
            [0.25, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    assert result.rho == pytest.approx(expected)


def test_common_mode_dephasing_preserves_phi_plus_bell_state() -> None:
    channel = common_mode_dephasing(0.5)
    phi_plus = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(phi_plus, phi_plus.conj()))

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(state.rho)


def test_common_mode_dephasing_validates_probability() -> None:
    with pytest.raises(TypeError, match="p"):
        common_mode_dephasing("bad")

    with pytest.raises(ValueError, match="p"):
        common_mode_dephasing(-0.1)

    with pytest.raises(ValueError, match="p"):
        common_mode_dephasing(1.1)


def test_imperfect_cnot_builds_two_qubit_kraus_channel() -> None:
    channel = imperfect_cnot(0.16)

    assert channel.name == "imperfect_cnot"
    assert channel.arity == 2
    assert len(channel.ops) == 16
    for op in channel.ops:
        assert op.shape == (4, 4)


def test_imperfect_cnot_p_zero_matches_ideal_cnot() -> None:
    channel = imperfect_cnot(0.0)
    state = DensityState(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    assert result.rho == pytest.approx(expected)


def test_imperfect_cnot_p_one_returns_two_qubit_maximally_mixed() -> None:
    channel = imperfect_cnot(1.0)
    state = DensityState(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(np.eye(4, dtype=np.complex128) / 4.0)


def test_imperfect_cnot_applies_cnot_then_depolarizing_noise() -> None:
    channel = imperfect_cnot(0.16)
    state = DensityState(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0, 1))

    expected = np.array(
        [
            [0.04, 0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0, 0.0],
            [0.0, 0.0, 0.04, 0.0],
            [0.0, 0.0, 0.0, 0.88],
        ],
        dtype=np.complex128,
    )
    assert result.rho == pytest.approx(expected)


def test_imperfect_cnot_validates_probability() -> None:
    with pytest.raises(TypeError, match="p"):
        imperfect_cnot("bad")

    with pytest.raises(ValueError, match="p"):
        imperfect_cnot(-0.1)

    with pytest.raises(ValueError, match="p"):
        imperfect_cnot(1.1)


def test_imperfect_cz_builds_two_qubit_kraus_channel() -> None:
    channel = imperfect_cz(0.16)

    assert channel.name == "imperfect_cz"
    assert channel.arity == 2
    assert len(channel.ops) == 16
    for op in channel.ops:
        assert op.shape == (4, 4)


def test_imperfect_cz_p_zero_matches_ideal_cz() -> None:
    channel = imperfect_cz(0.0)
    vector = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0, 1))

    expected_vector = np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2.0)
    expected = np.outer(expected_vector, expected_vector.conj())
    assert result.rho == pytest.approx(expected)


def test_imperfect_cz_p_one_returns_two_qubit_maximally_mixed() -> None:
    channel = imperfect_cz(1.0)
    vector = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0, 1))

    assert result.rho == pytest.approx(np.eye(4, dtype=np.complex128) / 4.0)


def test_imperfect_cz_applies_cz_then_depolarizing_noise() -> None:
    channel = imperfect_cz(0.16)
    vector = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0, 1))

    expected = np.array(
        [
            [0.46, 0.0, 0.0, -0.42],
            [0.0, 0.04, 0.0, 0.0],
            [0.0, 0.0, 0.04, 0.0],
            [-0.42, 0.0, 0.0, 0.46],
        ],
        dtype=np.complex128,
    )
    assert result.rho == pytest.approx(expected)


def test_imperfect_cz_validates_probability() -> None:
    with pytest.raises(TypeError, match="p"):
        imperfect_cz("bad")

    with pytest.raises(ValueError, match="p"):
        imperfect_cz(-0.1)

    with pytest.raises(ValueError, match="p"):
        imperfect_cz(1.1)


def test_amplitude_damping_decays_excited_population() -> None:
    channel = amplitude_damping(0.25)
    state = DensityState([[0, 0], [0, 1]])

    result = apply_kraus_density(state, channel, axes=(0,))

    assert channel.name == "amplitude_damping"
    assert channel.arity == 1
    assert len(channel.ops) == 2
    assert result.rho == pytest.approx(np.array([[0.25, 0.0], [0.0, 0.75]]))

    with pytest.raises(TypeError, match="gamma"):
        amplitude_damping("bad")
    with pytest.raises(ValueError, match="gamma"):
        amplitude_damping(-0.1)


def test_t1t2_noise_model_both_zero_is_identity() -> None:
    channel = T1T2NoiseModel(T1=0.0, T2=0.0, duration=1.0)
    vector = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0,))

    assert channel.name == "t1t2_noise_model"
    assert channel.arity == 1
    assert len(channel.ops) == 1
    assert result.rho == pytest.approx(state.rho)


def test_t1t2_noise_model_t1_zero_applies_t2_only_dephasing() -> None:
    duration = 1.0
    T2 = 4.0
    channel = T1T2NoiseModel(T1=0.0, T2=T2, duration=duration)
    vector = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0,))

    coherence_factor = np.exp(-duration / T2)

    assert result.rho == pytest.approx(
        np.array(
            [
                [0.5, 0.5 * coherence_factor],
                [0.5 * coherence_factor, 0.5],
            ],
            dtype=np.complex128,
        )
    )


def test_t1t2_noise_model_t2_zero_applies_t1_only_damping() -> None:
    duration = 1.0
    T1 = 4.0
    channel = T1T2NoiseModel(T1=T1, T2=0.0, duration=duration)
    state = DensityState([[0.0, 0.0], [0.0, 1.0]])

    result = apply_kraus_density(state, channel, axes=(0,))

    gamma = -np.expm1(-duration / T1)

    assert result.rho == pytest.approx(
        np.array(
            [
                [gamma, 0.0],
                [0.0, 1.0 - gamma],
            ],
            dtype=np.complex128,
        )
    )


def test_t1t2_noise_model_applies_damping_then_pure_dephasing() -> None:
    duration = 1.0
    T1 = 4.0
    T2 = 4.0
    channel = t1t2_noise_model(T1=T1, T2=T2, duration=duration)
    vector = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    state = DensityState(np.outer(vector, vector.conj()))

    result = apply_kraus_density(state, channel, axes=(0,))

    gamma = -np.expm1(-duration / T1)
    coherence_factor = np.exp(-duration / T2)

    assert result.rho == pytest.approx(
        np.array(
            [
                [0.5 * (1.0 + gamma), 0.5 * coherence_factor],
                [0.5 * coherence_factor, 0.5 * (1.0 - gamma)],
            ],
            dtype=np.complex128,
        )
    )


def test_t1t2_noise_model_validates_inputs() -> None:
    with pytest.raises(TypeError, match="T1"):
        T1T2NoiseModel(T1="bad", T2=0.0)

    with pytest.raises(ValueError, match="T1"):
        T1T2NoiseModel(T1=-1.0, T2=0.0)

    with pytest.raises(TypeError, match="T2"):
        T1T2NoiseModel(T1=0.0, T2="bad")

    with pytest.raises(ValueError, match="T2"):
        T1T2NoiseModel(T1=0.0, T2=-1.0)

    with pytest.raises(TypeError, match="duration"):
        T1T2NoiseModel(T1=1.0, T2=1.0, duration="bad")

    with pytest.raises(ValueError, match="duration"):
        T1T2NoiseModel(T1=1.0, T2=1.0, duration=-1.0)

    with pytest.raises(ValueError, match="T2 must be <= 2 \\* T1"):
        T1T2NoiseModel(T1=1.0, T2=3.0)


def test_t1t2_noise_model_uses_stable_small_probability_behavior() -> None:
    duration = 1.0e-12
    T1 = 1.0
    T2 = 1.0

    channel = T1T2NoiseModel(T1=T1, T2=T2, duration=duration)

    gamma = -np.expm1(-duration / T1)

    assert gamma > 0.0
    assert channel.arity == 1
    assert channel.name == "t1t2_noise_model"


def test_generalized_amplitude_damping_builds_four_kraus_ops() -> None:
    channel = generalized_amplitude_damping(gamma=0.25, prob=0.75)

    assert channel.name == "generalized_amplitude_damping"
    assert channel.arity == 1
    assert len(channel.ops) == 4


def test_generalized_amplitude_damping_stationary_state() -> None:
    gamma = 0.25
    prob = 0.75
    channel = generalized_amplitude_damping(gamma=gamma, prob=prob)
    state = DensityState(
        np.array(
            [[prob, 0.0], [0.0, 1.0 - prob]],
            dtype=np.complex128,
        )
    )

    result = apply_kraus_density(state, channel, axes=(0,))

    assert result.rho == pytest.approx(state.rho)


def test_generalized_amplitude_damping_prob_one_matches_ordinary_channel() -> None:
    gamma = 0.25
    ordinary = amplitude_damping(gamma)
    generalized = generalized_amplitude_damping(gamma, prob=1.0)
    zero = np.zeros((2, 2), dtype=np.complex128)

    assert len(ordinary.ops) == 2
    assert len(generalized.ops) == 4
    assert generalized.ops[0] == pytest.approx(ordinary.ops[0])
    assert generalized.ops[1] == pytest.approx(ordinary.ops[1])
    assert generalized.ops[2] == pytest.approx(zero)
    assert generalized.ops[3] == pytest.approx(zero)


def test_generalized_amplitude_damping_validates_inputs() -> None:
    with pytest.raises(TypeError, match="gamma"):
        generalized_amplitude_damping("bad", prob=1.0)
    with pytest.raises(ValueError, match="gamma"):
        generalized_amplitude_damping(-0.1, prob=1.0)
    with pytest.raises(TypeError, match="prob"):
        generalized_amplitude_damping(0.1, prob="bad")
    with pytest.raises(ValueError, match="prob"):
        generalized_amplitude_damping(0.1, prob=1.1)
