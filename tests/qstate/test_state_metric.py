from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate.errors import InvalidOperationError
from simyuj.qstate.measure.bell import bell_density_matrix, bell_vector
from simyuj.qstate.state.bell_diag import BellDiagState
from simyuj.qstate.state.density import DensityState
from simyuj.qstate.state.ket import KetState
from simyuj.qstate.state.metric import (
    bell_fidelity,
    concurrence,
    entropy,
    fidelity,
    log_negativity,
    max_chsh_value,
    negativity,
    purity,
)


def test_purity_supports_ket_density_and_bell_diag_states() -> None:
    assert purity(KetState([1, 0])) == pytest.approx(1.0)
    assert purity(DensityState([[0.5, 0.0], [0.0, 0.5]])) == pytest.approx(0.5)
    assert purity(BellDiagState((0.5, 0.5, 0.0, 0.0))) == pytest.approx(0.5)


def test_purity_rejects_unsupported_payloads() -> None:
    with pytest.raises(InvalidOperationError, match="purity"):
        purity("bad")


def test_entropy_supports_ket_density_and_bell_diag_states() -> None:
    assert entropy(KetState([1, 0])) == pytest.approx(0.0)
    assert entropy(DensityState([[0.5, 0.0], [0.0, 0.5]])) == pytest.approx(1.0)
    assert entropy(BellDiagState.from_label("phi+")) == pytest.approx(0.0)
    assert entropy(BellDiagState((0.25, 0.25, 0.25, 0.25))) == pytest.approx(2.0)
    assert entropy(BellDiagState((0.5, 0.5, 0.0, 0.0)), base=np.e) == pytest.approx(
        np.log(2.0),
    )


def test_entropy_validates_base_and_payload() -> None:
    with pytest.raises(TypeError, match="base"):
        entropy(BellDiagState.from_label("phi+"), base="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        entropy(BellDiagState.from_label("phi+"), base=0.0)
    with pytest.raises(ValueError, match="not be one"):
        entropy(BellDiagState.from_label("phi+"), base=1.0)
    with pytest.raises(InvalidOperationError, match="entropy"):
        entropy("bad")


def test_bell_fidelity_supports_bell_diag_ket_and_density_states() -> None:
    bell_diag = BellDiagState((0.7, 0.3, 0.0, 0.0))
    ket = KetState(bell_vector("psi+"))
    density = DensityState(bell_density_matrix("phi-"))

    assert bell_fidelity(bell_diag, "phi+") == pytest.approx(0.7)
    assert bell_fidelity(ket, "psi+") == pytest.approx(1.0)
    assert bell_fidelity(ket, "psi-") == pytest.approx(0.0)
    assert bell_fidelity(density, "phi-") == pytest.approx(1.0)


def test_bell_fidelity_requires_two_qubit_ket_or_density_state() -> None:
    with pytest.raises(InvalidOperationError, match="exactly two"):
        bell_fidelity(KetState([1, 0]))
    with pytest.raises(InvalidOperationError, match="exactly two"):
        bell_fidelity(DensityState([[1, 0], [0, 0]]))
    with pytest.raises(InvalidOperationError, match="Bell fidelity"):
        bell_fidelity("bad")


def test_concurrence_supports_two_qubit_ket_density_and_bell_diag_states() -> None:
    assert concurrence(KetState([1, 0, 0, 0])) == pytest.approx(0.0)
    assert concurrence(KetState(bell_vector("phi+"))) == pytest.approx(1.0)
    assert concurrence(DensityState(bell_density_matrix("psi-"))) == pytest.approx(1.0)
    assert concurrence(BellDiagState.from_label("phi+")) == pytest.approx(1.0)
    assert concurrence(BellDiagState((0.6, 0.2, 0.1, 0.1))) == pytest.approx(0.2)
    assert concurrence(BellDiagState((0.5, 0.5, 0.0, 0.0))) == pytest.approx(0.0)


def test_concurrence_requires_supported_two_qubit_state() -> None:
    with pytest.raises(InvalidOperationError, match="exactly two"):
        concurrence(KetState([1, 0]))
    with pytest.raises(InvalidOperationError, match="exactly two"):
        concurrence(DensityState([[1, 0], [0, 0]]))
    with pytest.raises(InvalidOperationError, match="concurrence"):
        concurrence("bad")


def test_negativity_supports_two_qubit_ket_density_and_bell_diag_states() -> None:
    assert negativity(KetState([1, 0, 0, 0])) == pytest.approx(0.0)
    assert negativity(KetState(bell_vector("phi+"))) == pytest.approx(0.5)
    assert negativity(DensityState(bell_density_matrix("psi-"))) == pytest.approx(0.5)
    assert negativity(BellDiagState.from_label("phi+")) == pytest.approx(0.5)
    assert negativity(BellDiagState((0.6, 0.2, 0.1, 0.1))) == pytest.approx(0.1)
    assert negativity(BellDiagState((0.5, 0.5, 0.0, 0.0))) == pytest.approx(0.0)


def test_log_negativity_supports_two_qubit_states_and_log_base() -> None:
    bell_diag = BellDiagState.from_label("phi+")

    assert log_negativity(KetState([1, 0, 0, 0])) == pytest.approx(0.0)
    assert log_negativity(KetState(bell_vector("phi+"))) == pytest.approx(1.0)
    assert log_negativity(DensityState(bell_density_matrix("psi-"))) == pytest.approx(
        1.0,
    )
    assert log_negativity(bell_diag) == pytest.approx(1.0)
    assert log_negativity(bell_diag, base=np.e) == pytest.approx(np.log(2.0))


def test_negativity_metrics_validate_inputs() -> None:
    state = BellDiagState.from_label("phi+")

    with pytest.raises(TypeError, match="subsystem"):
        negativity(state, subsystem=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subsystem"):
        negativity(state, subsystem=2)
    with pytest.raises(TypeError, match="base"):
        log_negativity(state, base="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not be one"):
        log_negativity(state, base=1.0)
    with pytest.raises(InvalidOperationError, match="negativity"):
        negativity("bad")
    with pytest.raises(InvalidOperationError, match="log negativity"):
        log_negativity("bad")
    with pytest.raises(InvalidOperationError, match="exactly two"):
        negativity(KetState([1, 0]))


def test_max_chsh_value_supports_ket_density_and_bell_diag_states() -> None:
    product = KetState([1, 0, 0, 0])
    bell_ket = KetState(bell_vector("phi+"))
    bell_density = DensityState(bell_density_matrix("psi-"))
    bell_diag = BellDiagState.from_label("phi+")

    assert max_chsh_value(product) == pytest.approx(2.0, abs=1e-12)
    assert max_chsh_value(bell_ket) == pytest.approx(2.0 * np.sqrt(2.0), abs=1e-12)
    assert max_chsh_value(bell_density) == pytest.approx(
        2.0 * np.sqrt(2.0),
        abs=1e-12,
    )
    assert max_chsh_value(bell_diag) == pytest.approx(
        2.0 * np.sqrt(2.0),
        abs=1e-12,
    )


def test_max_chsh_value_uniform_bell_diagonal_has_no_correlation() -> None:
    state = BellDiagState((0.25, 0.25, 0.25, 0.25))

    assert max_chsh_value(state) == pytest.approx(0.0, abs=1e-12)


def test_max_chsh_value_matches_werner_visibility_scaling() -> None:
    visibility = 0.8
    state = BellDiagState((0.85, 0.05, 0.05, 0.05))

    assert max_chsh_value(state) == pytest.approx(
        2.0 * np.sqrt(2.0) * visibility,
        abs=1e-12,
    )


def test_max_chsh_value_requires_supported_two_qubit_state() -> None:
    with pytest.raises(InvalidOperationError, match="exactly two"):
        max_chsh_value(KetState([1, 0]))
    with pytest.raises(InvalidOperationError, match="exactly two"):
        max_chsh_value(DensityState([[1, 0], [0, 0]]))
    with pytest.raises(InvalidOperationError, match="max CHSH value"):
        max_chsh_value("bad")


def test_fidelity_supports_ket_ket_and_ket_density_pairs() -> None:
    zero = KetState([1, 0])
    one = KetState([0, 1])
    mixed = DensityState([[0.25, 0.0], [0.0, 0.75]])

    assert fidelity(zero, zero) == pytest.approx(1.0)
    assert fidelity(zero, one) == pytest.approx(0.0)
    assert fidelity(zero, mixed) == pytest.approx(0.25)
    assert fidelity(mixed, one) == pytest.approx(0.75)


def test_fidelity_supports_classical_bell_diag_fidelity() -> None:
    left = BellDiagState((0.5, 0.5, 0.0, 0.0))
    right = BellDiagState((0.25, 0.75, 0.0, 0.0))
    expected = (np.sqrt(0.5 * 0.25) + np.sqrt(0.5 * 0.75)) ** 2

    assert fidelity(left, right) == pytest.approx(expected)


def test_fidelity_rejects_unsupported_pairs() -> None:
    with pytest.raises(InvalidOperationError, match="fidelity currently supports"):
        fidelity(DensityState([[1, 0], [0, 0]]), DensityState([[1, 0], [0, 0]]))
    with pytest.raises(InvalidOperationError, match="fidelity currently supports"):
        fidelity(BellDiagState.from_label("phi+"), KetState(bell_vector("phi+")))
