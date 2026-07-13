from __future__ import annotations

import numpy as np

from simyuj.qstate.ops.reset import (
    discard_and_prepare,
    reset_one,
    reset_plus,
    reset_zero,
)
from simyuj.qstate.state.convert import ket_to_density
from simyuj.qstate.state.density import DensityState
from simyuj.qstate.state.make import basis


def _assert_density(actual: DensityState, expected: DensityState) -> None:
    np.testing.assert_allclose(actual.rho, expected.rho, atol=1e-12)


def test_reset_wrappers_replace_selected_density_axes() -> None:
    density = ket_to_density(basis("10"))

    _assert_density(reset_zero(density, axes=(0,)), ket_to_density(basis("00")))
    _assert_density(reset_one(density, axes=(1,)), ket_to_density(basis("11")))

    plus_expected = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
    _assert_density(
        reset_plus(density, axes=(0,)),
        DensityState(np.outer(plus_expected, np.conjugate(plus_expected))),
    )

    _assert_density(
        discard_and_prepare(density, axes=(1,), prepared="1"),
        ket_to_density(basis("11")),
    )
