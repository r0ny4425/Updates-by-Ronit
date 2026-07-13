from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate.errors import DimensionError
from simyuj.qstate.state.bell_diag import BellDiagState
from simyuj.qstate.state.convert import ket_to_density
from simyuj.qstate.state.density import DensityState
from simyuj.qstate.state.make import basis, bell
from simyuj.qstate.state.reduce import (
    discard_density,
    drop_axes,
    keep_axes,
    keep_axes_density,
    partial_trace,
    reorder_density_axes,
    reset_density,
)


def _assert_density(actual: DensityState, expected: DensityState) -> None:
    np.testing.assert_allclose(actual.rho, expected.rho, atol=1e-12)


def test_partial_trace_accepts_density_ket_and_bell_diag_states() -> None:
    density = ket_to_density(basis("01"))

    _assert_density(partial_trace(density, keep_axes=(0,)), ket_to_density(basis("0")))
    _assert_density(partial_trace(density, drop_axes=(0,)), ket_to_density(basis("1")))

    reduced_bell = partial_trace(bell("phi+"), keep_axes=(0,))
    np.testing.assert_allclose(reduced_bell.rho, np.eye(2) / 2, atol=1e-12)

    reduced_diag = partial_trace(BellDiagState.from_label("psi-"), drop_axes=(1,))
    np.testing.assert_allclose(reduced_diag.rho, np.eye(2) / 2, atol=1e-12)


def test_keep_drop_and_discard_density_delegates_to_partial_trace() -> None:
    density = ket_to_density(basis("10"))

    _assert_density(keep_axes(density, (0,)), ket_to_density(basis("1")))
    _assert_density(drop_axes(density, (0,)), ket_to_density(basis("0")))
    _assert_density(
        discard_density(density, drop_axes=(1,)), ket_to_density(basis("1"))
    )


def test_keep_axes_density_handles_noop_and_scalar_reductions() -> None:
    density = ket_to_density(basis("01"))

    assert keep_axes_density(density, (0, 1)) is density

    scalar = keep_axes_density(density, ())
    assert scalar.num_qubits == 0
    np.testing.assert_allclose(scalar.rho, [[1.0]], atol=1e-12)


def test_keep_axes_density_preserves_requested_remaining_axis_order() -> None:
    density = ket_to_density(basis("010"))

    _assert_density(keep_axes_density(density, (2, 0)), ket_to_density(basis("00")))
    _assert_density(keep_axes_density(density, (1, 0)), ket_to_density(basis("10")))


def test_reset_density_restores_original_axis_order() -> None:
    density = ket_to_density(basis("101"))

    result = reset_density(density, axes=(2, 0), state="10")

    _assert_density(result, ket_to_density(basis("001")))


def test_reset_density_expands_named_zero_and_one_for_multi_axis_reset() -> None:
    density = ket_to_density(basis("00"))

    _assert_density(
        reset_density(density, axes=(0, 1), state="1"),
        ket_to_density(basis("11")),
    )
    _assert_density(
        reset_density(density, axes=(1, 0), state="0"),
        ket_to_density(basis("00")),
    )


def test_reset_density_allows_single_axis_plus_reset() -> None:
    density = ket_to_density(basis("10"))
    expected_vector = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.complex128) / np.sqrt(2)

    _assert_density(
        reset_density(density, axes=(1,), state="plus"),
        DensityState(np.outer(expected_vector, np.conjugate(expected_vector))),
    )


def test_reset_density_on_entangled_state_uses_density_reduction() -> None:
    density = ket_to_density(bell("phi+"))

    result = reset_density(density, axes=(0,), state="0")

    _assert_density(result, DensityState(np.diag([0.5, 0.5, 0.0, 0.0])))


def test_reorder_density_axes_swaps_bra_and_ket_halves() -> None:
    density = ket_to_density(basis("01"))

    result = reorder_density_axes(
        density,
        current_order=(0, 1),
        target_order=(1, 0),
    )

    _assert_density(result, ket_to_density(basis("10")))

    labeled = reorder_density_axes(
        density,
        current_order=(2, 4),
        target_order=(4, 2),
    )
    _assert_density(labeled, ket_to_density(basis("10")))


def test_partial_trace_validates_axis_selection() -> None:
    density = ket_to_density(basis("00"))

    with pytest.raises(ValueError, match="required"):
        partial_trace(density)
    with pytest.raises(ValueError, match="either"):
        partial_trace(density, keep_axes=(0,), drop_axes=(1,))
    with pytest.raises(TypeError, match="keep_axes"):
        partial_trace(density, keep_axes=[0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entries"):
        partial_trace(density, keep_axes=(True,))  # type: ignore[arg-type]
    with pytest.raises(DimensionError, match="range"):
        partial_trace(density, keep_axes=(2,))
    with pytest.raises(ValueError, match="unique"):
        partial_trace(density, drop_axes=(1, 1))
    with pytest.raises(TypeError, match="state"):
        partial_trace(object(), keep_axes=(0,))


def test_density_reduce_helpers_validate_inputs() -> None:
    density = ket_to_density(basis("00"))

    with pytest.raises(TypeError, match="state"):
        discard_density(basis("00"), drop_axes=(0,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state"):
        keep_axes_density(basis("00"), (0,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="density"):
        reset_density(basis("00"), axes=(0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        reset_density(density, axes=())
    with pytest.raises(DimensionError, match="width"):
        reset_density(density, axes=(0, 1), state="+")


def test_reorder_density_axes_validates_orders() -> None:
    density = ket_to_density(basis("00"))

    with pytest.raises(TypeError, match="current_order"):
        reorder_density_axes(
            density,
            current_order=[0, 1],  # type: ignore[arg-type]
            target_order=(0, 1),
        )
    with pytest.raises(TypeError, match="target_order"):
        reorder_density_axes(
            density,
            current_order=(0, 1),
            target_order=[0, 1],  # type: ignore[arg-type]
        )
    with pytest.raises(DimensionError, match="length"):
        reorder_density_axes(density, current_order=(0,), target_order=(0,))
    with pytest.raises(ValueError, match="same axes"):
        reorder_density_axes(density, current_order=(0, 2), target_order=(0, 1))
    with pytest.raises(ValueError, match="unique"):
        reorder_density_axes(density, current_order=(0, 0), target_order=(0, 1))
