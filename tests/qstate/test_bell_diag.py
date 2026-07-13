from __future__ import annotations

import random

import pytest

from simyuj.qstate.errors import InvalidOperationError, MeasurementError, NoiseError
from simyuj.qstate.measure.result import BellResult
from simyuj.qstate.noise import amplitude_damping, pauli_channel, phase_flip
from simyuj.qstate.ops import CNOT, H, X, Z
from simyuj.qstate.space import StateLayout, SubsystemId
from simyuj.qstate.state.bell_diag import (
    BELL_LABELS,
    BellDiagHandler,
    BellDiagState,
    bell_index,
    bits_to_label,
    is_pure_bell_diag,
    label_to_bits,
    make_bell_diag,
    normalize_bell_label,
    werner,
)


def _layout(*names: str, dims: tuple[int, ...] | None = None) -> StateLayout:
    return StateLayout(
        tuple(SubsystemId(name) for name in names),
        dims if dims is not None else (2,) * len(names),
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("phi+", "phi+"),
        ("phi_plus", "phi+"),
        ("phi-plus", "phi+"),
        ("phiplus", "phi+"),
        ("Φ+", "phi+"),
        ("φ+", "phi+"),
        ("phi-", "phi-"),
        ("phi_minus", "phi-"),
        ("phi-minus", "phi-"),
        ("phiminus", "phi-"),
        ("Φ-", "phi-"),
        ("φ-", "phi-"),
        ("psi+", "psi+"),
        ("psi_plus", "psi+"),
        ("psi-plus", "psi+"),
        ("psiplus", "psi+"),
        ("Ψ+", "psi+"),
        ("ψ+", "psi+"),
        ("psi-", "psi-"),
        ("psi_minus", "psi-"),
        ("psi-minus", "psi-"),
        ("psiminus", "psi-"),
        ("Ψ-", "psi-"),
        ("ψ-", "psi-"),
    ],
)
def test_bell_label_aliases_normalize_to_canonical_labels(
    alias: str,
    canonical: str,
) -> None:
    assert normalize_bell_label(alias) == canonical


def test_bell_label_validation_and_index_mapping() -> None:
    assert BELL_LABELS == ("phi+", "phi-", "psi+", "psi-")
    assert bell_index("phi+") == 0
    assert bell_index("psi-") == 3

    with pytest.raises(TypeError, match="str"):
        normalize_bell_label(1)
    with pytest.raises(ValueError, match="non-empty"):
        normalize_bell_label(" ")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_bell_label("omega+")


def test_bell_label_bit_mapping_validates_inputs() -> None:
    assert label_to_bits("phi+") == (0, 0)
    assert label_to_bits("phi-") == (0, 1)
    assert label_to_bits("psi+") == (1, 0)
    assert label_to_bits("psi-") == (1, 1)
    assert bits_to_label((0, 0)) == "phi+"
    assert bits_to_label((1, 1)) == "psi-"

    with pytest.raises(TypeError, match="tuple"):
        bits_to_label([0, 0])
    with pytest.raises(ValueError, match="exactly two"):
        bits_to_label((0,))
    with pytest.raises(ValueError, match="0 or 1"):
        bits_to_label((0, 2))
    with pytest.raises(ValueError, match="0 or 1"):
        bits_to_label((True, 0))


def test_bell_diag_state_validates_and_reports_probabilities() -> None:
    state = BellDiagState((0.7, 0.1, 0.1, 0.1))

    assert state.num_qubits == 2
    assert state.probs == pytest.approx((0.7, 0.1, 0.1, 0.1))
    assert state.probabilities == (
        ("phi+", 0.7),
        ("phi-", 0.1),
        ("psi+", 0.1),
        ("psi-", 0.1),
    )
    assert state.probability("phi_plus") == pytest.approx(0.7)
    assert state.fidelity() == pytest.approx(0.7)

    with pytest.raises(ValueError, match="length four"):
        BellDiagState((0.5, 0.5))
    with pytest.raises(ValueError, match="sum"):
        BellDiagState((0.5, 0.2, 0.2, 0.2))


def test_bell_diag_state_from_label_and_purity_check() -> None:
    state = BellDiagState.from_label("psi-")

    assert state.probs == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert is_pure_bell_diag(state)
    assert not is_pure_bell_diag(BellDiagState((0.25, 0.25, 0.25, 0.25)))


def test_make_bell_diag_accepts_state_label_mapping_and_sequence() -> None:
    state = BellDiagState.from_label("phi+")

    assert make_bell_diag(state) is state
    assert make_bell_diag("psi_plus").probs == pytest.approx((0, 0, 1, 0))
    assert make_bell_diag({"phi+": 0.7, "phi-": 0.3}).probs == pytest.approx(
        (0.7, 0.3, 0.0, 0.0)
    )
    assert make_bell_diag((0.25, 0.25, 0.25, 0.25)).probs == pytest.approx(
        (0.25, 0.25, 0.25, 0.25)
    )

    with pytest.raises(ValueError, match="duplicate"):
        make_bell_diag({"phi+": 0.5, "phi_plus": 0.5})
    with pytest.raises(TypeError, match="state"):
        make_bell_diag(1)


def test_werner_creates_bell_diagonal_state_from_target_fidelity() -> None:
    state = werner(0.85)

    assert isinstance(state, BellDiagState)
    assert state.num_qubits == 2
    assert state.probs == pytest.approx((0.85, 0.05, 0.05, 0.05))
    assert state.fidelity("phi+") == pytest.approx(0.85)


def test_werner_supports_non_default_target_bell_label() -> None:
    state = werner(0.7, label="psi-")

    assert state.probs == pytest.approx((0.1, 0.1, 0.1, 0.7))
    assert state.fidelity("psi-") == pytest.approx(0.7)


def test_werner_validates_fidelity_and_label() -> None:
    with pytest.raises(TypeError, match="fidelity"):
        werner("0.9")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        werner(1.2)
    with pytest.raises(ValueError, match="unsupported"):
        werner(0.9, label="omega+")


def test_bell_diag_handler_supports_pauli_unitaries_and_rejects_nonclosed_ops() -> None:
    handler = BellDiagHandler()
    state = BellDiagState.from_label("phi+")

    assert handler.make("phi+") == state
    x_result = handler.apply(state, X, layout=_layout("a", "b"), axes=(0,))
    assert isinstance(x_result, BellDiagState)
    assert x_result.probs == pytest.approx((0.0, 0.0, 1.0, 0.0))
    z_result = handler.apply(state, Z, layout=_layout("a", "b"), axes=(1,))
    assert isinstance(z_result, BellDiagState)
    assert z_result.probs == pytest.approx((0.0, 1.0, 0.0, 0.0))
    with pytest.raises(InvalidOperationError, match="tensoring"):
        handler.tensor(state, state)
    with pytest.raises(InvalidOperationError, match="Pauli"):
        handler.apply(state, H, layout=_layout("a", "b"), axes=(0,))
    with pytest.raises(InvalidOperationError, match="Pauli"):
        handler.apply(state, CNOT, layout=_layout("a", "b"), axes=(0, 1))
    with pytest.raises(MeasurementError, match="projective"):
        handler.measure(state, layout=_layout("a", "b"), axes=(0,))


def test_bell_diag_handler_supports_pauli_noise_channels() -> None:
    handler = BellDiagHandler()
    state = BellDiagState.from_label("phi+")

    bit_and_phase = pauli_channel(0.25, 0.0, 0.50)
    result = handler.channel(
        state,
        bit_and_phase,
        layout=_layout("a", "b"),
        axes=(0,),
    )

    assert isinstance(result, BellDiagState)
    assert result.probs == pytest.approx((0.25, 0.50, 0.25, 0.0))
    phase_result = handler.channel(
        state,
        phase_flip(1.0),
        layout=_layout("a", "b"),
        axes=(1,),
    )
    assert isinstance(phase_result, BellDiagState)
    assert phase_result.probs == pytest.approx((0.0, 1.0, 0.0, 0.0))

    with pytest.raises(NoiseError, match="Pauli"):
        handler.channel(
            state,
            amplitude_damping(0.25),
            layout=_layout("a", "b"),
            axes=(0,),
        )


def test_bell_diag_handler_measure_bell_samples_and_collapses() -> None:
    handler = BellDiagHandler()
    state = BellDiagState((0.0, 0.0, 0.0, 1.0))

    result = handler.measure_bell(
        state,
        layout=_layout("a", "b"),
        axes=(0, 1),
        collapse=True,
    )

    assert isinstance(result, BellResult)
    assert result.label == "psi-"
    assert result.outcome == (1, 1)
    assert result.probability == pytest.approx(1.0)
    assert isinstance(result.post_state, BellDiagState)
    assert result.post_state.probs == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert result.collapsed


def test_bell_diag_handler_measure_bell_can_sample_without_collapse() -> None:
    handler = BellDiagHandler()
    state = BellDiagState((0.25, 0.25, 0.25, 0.25))

    result = handler.measure_bell(
        state,
        layout=_layout("a", "b"),
        axes=(1, 0),
        rng=random.Random(3),
        collapse=False,
    )

    assert result.label == "phi+"
    assert result.outcome == (0, 0)
    assert result.probability == pytest.approx(0.25)
    assert result.probabilities == state.probabilities
    assert result.post_state is None
    assert not result.collapsed


def test_bell_diag_handler_measure_bell_validates_inputs() -> None:
    handler = BellDiagHandler()
    state = BellDiagState.from_label("phi+")

    with pytest.raises(TypeError, match="payload"):
        handler.measure_bell("bad", layout=_layout("a", "b"), axes=(0, 1))
    with pytest.raises(TypeError, match="StateLayout"):
        handler.measure_bell(state, layout="bad", axes=(0, 1))  # type: ignore[arg-type]
    with pytest.raises(MeasurementError, match="two-axis"):
        handler.measure_bell(state, layout=_layout("a"), axes=(0, 1))
    with pytest.raises(MeasurementError, match="qubit axes"):
        handler.measure_bell(
            state,
            layout=_layout("a", "b", dims=(2, 3)),
            axes=(0, 1),
        )
    with pytest.raises(TypeError, match="tuple"):
        handler.measure_bell(
            state,
            layout=_layout("a", "b"),
            axes=[0, 1],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="axes entries"):
        handler.measure_bell(
            state,
            layout=_layout("a", "b"),
            axes=(0, 1.0),  # type: ignore[arg-type]
        )
    with pytest.raises(MeasurementError, match="both stored"):
        handler.measure_bell(state, layout=_layout("a", "b"), axes=(0, 0))
    with pytest.raises(TypeError, match="collapse"):
        handler.measure_bell(
            state,
            layout=_layout("a", "b"),
            axes=(0, 1),
            collapse=1,  # type: ignore[arg-type]
        )
