from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate.errors import DimensionError, InvalidReprError, InvalidStateError
from simyuj.qstate.measure.bell import bell_density_matrix, bell_vector
from simyuj.qstate.state.bell_diag import BellDiagState
from simyuj.qstate.state.convert import (
    as_rep,
    bell_diag_to_density,
    bell_diag_to_ket_if_pure,
    density_to_bell_diag_if_exact,
    ket_to_bell_diag_if_exact,
)
from simyuj.qstate.state.density import DensityState
from simyuj.qstate.state.ket import KetState


def test_bell_diag_to_density_builds_probability_weighted_projector_sum() -> None:
    state = BellDiagState((0.7, 0.3, 0.0, 0.0))

    density = bell_diag_to_density(state)

    expected = 0.7 * bell_density_matrix("phi+")
    expected = expected + 0.3 * bell_density_matrix("phi-")
    assert density.rho == pytest.approx(expected)


def test_bell_diag_to_ket_accepts_only_pure_bell_diagonal_states() -> None:
    state = BellDiagState.from_label("psi-")

    ket = bell_diag_to_ket_if_pure(state)

    assert ket.vector == pytest.approx(bell_vector("psi-"))
    with pytest.raises(InvalidStateError, match="not pure"):
        bell_diag_to_ket_if_pure(BellDiagState((0.25, 0.25, 0.25, 0.25)))


def test_density_to_bell_diag_if_exact_projects_bell_diagonal_density() -> None:
    density = DensityState(
        0.1 * bell_density_matrix("phi+")
        + 0.2 * bell_density_matrix("phi-")
        + 0.3 * bell_density_matrix("psi+")
        + 0.4 * bell_density_matrix("psi-")
    )

    state = density_to_bell_diag_if_exact(density)

    assert state.probs == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_density_to_bell_diag_if_exact_rejects_invalid_inputs() -> None:
    with pytest.raises(DimensionError, match="exactly two"):
        density_to_bell_diag_if_exact(DensityState([[1, 0], [0, 0]]))

    not_bell_diagonal = DensityState(np.diag([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(InvalidStateError, match="not exactly Bell diagonal"):
        density_to_bell_diag_if_exact(not_bell_diagonal)


def test_ket_to_bell_diag_if_exact_round_trips_pure_bell_state() -> None:
    state = ket_to_bell_diag_if_exact(KetState(bell_vector("psi+")))

    assert state.probs == pytest.approx((0.0, 0.0, 1.0, 0.0))


def test_ket_to_bell_diag_if_exact_rejects_non_bell_state() -> None:
    with pytest.raises(InvalidStateError, match="not exactly Bell diagonal"):
        ket_to_bell_diag_if_exact(KetState([1, 0, 0, 0]))


def test_as_rep_supports_bell_diag_density_and_pure_ket_paths() -> None:
    bell_diag = BellDiagState.from_label("phi+")

    assert as_rep(bell_diag, "bell_diag") is bell_diag
    assert isinstance(as_rep(bell_diag, "density"), DensityState)
    assert isinstance(as_rep(bell_diag, "ket"), KetState)
    from_density = as_rep(as_rep(bell_diag, "density"), "bell_diag")
    from_ket = as_rep(as_rep(bell_diag, "ket"), "bell_diag")

    assert isinstance(from_density, BellDiagState)
    assert isinstance(from_ket, BellDiagState)
    assert from_density.probs == pytest.approx(bell_diag.probs)
    assert from_ket.probs == pytest.approx(bell_diag.probs)


def test_as_rep_rejects_mixed_bell_diag_to_ket_and_unknown_representation() -> None:
    with pytest.raises(InvalidStateError, match="not pure"):
        as_rep(BellDiagState((0.5, 0.5, 0.0, 0.0)), "ket")

    with pytest.raises(InvalidReprError, match="unsupported"):
        as_rep(BellDiagState.from_label("phi+"), "graph")


def test_bell_diag_conversion_helpers_validate_payload_types() -> None:
    with pytest.raises(TypeError, match="BellDiagState"):
        bell_diag_to_density("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BellDiagState"):
        bell_diag_to_ket_if_pure("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DensityState"):
        density_to_bell_diag_if_exact("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KetState"):
        ket_to_bell_diag_if_exact("bad")  # type: ignore[arg-type]
