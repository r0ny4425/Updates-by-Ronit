from __future__ import annotations

import random
from typing import cast

import numpy as np
import pytest

from simyuj.qstate.errors import DimensionError, MeasurementError
from simyuj.qstate.measure.bell import (
    bell_density_matrix,
    bell_projector,
    bell_projectors,
    bell_vector,
    bell_vectors,
    measure_bell_density,
    measure_bell_ket,
)
from simyuj.qstate.measure.result import BellResult
from simyuj.qstate.space import StateLayout, SubsystemId
from simyuj.qstate.state.density import DensityHandler, DensityState
from simyuj.qstate.state.ket import KetHandler, KetState


def _array(value: object) -> np.ndarray:
    return cast(np.ndarray, value)


def _layout(*names: str, dims: tuple[int, ...] | None = None) -> StateLayout:
    return StateLayout(
        tuple(SubsystemId(name) for name in names),
        dims if dims is not None else (2,) * len(names),
    )


def test_bell_vectors_are_canonical_readonly_length_four_vectors() -> None:
    inv = 1.0 / np.sqrt(2.0)

    assert bell_vector("phi+") == pytest.approx((inv, 0, 0, inv))
    assert bell_vector("phi-") == pytest.approx((inv, 0, 0, -inv))
    assert bell_vector("psi+") == pytest.approx((0, inv, inv, 0))
    assert bell_vector("psi-") == pytest.approx((0, inv, -inv, 0))
    assert not bell_vector("phi+").flags.writeable
    assert tuple(vector.shape for vector in bell_vectors()) == ((4,),) * 4


def test_bell_projectors_and_density_matrix_are_outer_products() -> None:
    vector = bell_vector("psi-").reshape(-1, 1)
    projector = bell_projector("psi-")

    assert projector == pytest.approx(vector @ np.conjugate(vector.T))
    assert bell_density_matrix("psi-") == pytest.approx(projector)
    assert len(bell_projectors()) == 4

    resolution = sum(bell_projectors())
    assert resolution == pytest.approx(np.eye(4))


def test_measure_bell_ket_deterministic_outcome_needs_no_rng_and_collapses() -> None:
    state = KetState(bell_vector("psi-"))

    result = measure_bell_ket(
        state,
        layout=_layout("a", "b"),
        axes=(0, 1),
    )

    assert isinstance(result, BellResult)
    assert result.label == "psi-"
    assert result.outcome == (1, 1)
    assert result.probability == pytest.approx(1.0)
    assert dict(result.probabilities)["psi-"] == pytest.approx(1.0)
    assert isinstance(result.post_state, KetState)
    assert _array(result.post_state.vector) == pytest.approx(bell_vector("psi-"))


def test_ket_handler_measure_bell_delegates_to_ket_measurement() -> None:
    handler = KetHandler()
    state = KetState(bell_vector("phi+"))

    result = handler.measure_bell(
        state,
        layout=_layout("a", "b"),
        axes=(0, 1),
    )

    assert result.label == "phi+"
    assert result.outcome == (0, 0)
    assert isinstance(result.post_state, KetState)


def test_measure_bell_ket_samples_probabilistic_state_with_explicit_rng() -> None:
    state = KetState([1, 0, 0, 0])

    with pytest.raises(ValueError, match="explicit rng"):
        measure_bell_ket(state, layout=_layout("a", "b"), axes=(0, 1))

    result = measure_bell_ket(
        state,
        layout=_layout("a", "b"),
        axes=(0, 1),
        rng=random.Random(0),
        collapse=False,
    )

    assert result.label == "phi-"
    assert result.outcome == (0, 1)
    assert result.probability == pytest.approx(0.5)
    assert result.post_state is None
    assert not result.collapsed


def test_measure_bell_ket_expands_projectors_for_non_adjacent_axes() -> None:
    inv = 1.0 / np.sqrt(2.0)
    vector = np.zeros(8, dtype=np.complex128)
    vector[0] = inv
    vector[5] = inv
    state = KetState(vector)

    result = measure_bell_ket(
        state,
        layout=_layout("a", "b", "c"),
        axes=(2, 0),
    )

    assert result.label == "phi+"
    assert result.probability == pytest.approx(1.0)
    assert isinstance(result.post_state, KetState)
    assert _array(result.post_state.vector) == pytest.approx(vector)


def test_measure_bell_density_expands_projectors_for_non_adjacent_axes() -> None:
    inv = 1.0 / np.sqrt(2.0)
    vector = np.zeros(8, dtype=np.complex128)
    vector[0] = inv
    vector[5] = inv
    density = DensityState(np.outer(vector, np.conjugate(vector)))

    result = measure_bell_density(
        density,
        layout=_layout("a", "b", "c"),
        axes=(2, 0),
    )

    assert result.label == "phi+"
    assert result.probability == pytest.approx(1.0)
    assert isinstance(result.post_state, DensityState)
    assert _array(result.post_state.rho) == pytest.approx(density.rho)


def test_measure_bell_density_deterministic_outcome_collapses() -> None:
    density = DensityState(bell_density_matrix("phi-"))

    result = measure_bell_density(
        density,
        layout=_layout("a", "b"),
        axes=(0, 1),
    )

    assert result.label == "phi-"
    assert result.outcome == (0, 1)
    assert result.probability == pytest.approx(1.0)
    assert isinstance(result.post_state, DensityState)
    assert _array(result.post_state.rho) == pytest.approx(bell_density_matrix("phi-"))


def test_density_handler_measure_bell_delegates_to_density_measurement() -> None:
    handler = DensityHandler()
    state = DensityState(bell_density_matrix("psi+"))

    result = handler.measure_bell(
        state,
        layout=_layout("a", "b"),
        axes=(0, 1),
    )

    assert result.label == "psi+"
    assert result.outcome == (1, 0)
    assert isinstance(result.post_state, DensityState)


def test_measure_bell_density_samples_and_collapses_mixed_state() -> None:
    density = DensityState(np.diag([1.0, 0.0, 0.0, 0.0]))

    result = measure_bell_density(
        density,
        layout=_layout("a", "b"),
        axes=(0, 1),
        rng=random.Random(0),
        collapse=True,
    )

    assert result.label == "phi-"
    assert result.probability == pytest.approx(0.5)
    assert isinstance(result.post_state, DensityState)
    assert _array(result.post_state.rho) == pytest.approx(bell_density_matrix("phi-"))


def test_measure_bell_validates_state_layout_axes_and_collapse() -> None:
    state = KetState(bell_vector("phi+"))

    with pytest.raises(TypeError, match="KetState"):
        measure_bell_ket(
            "bad",  # type: ignore[arg-type]
            layout=_layout("a", "b"),
            axes=(0, 1),
        )
    with pytest.raises(TypeError, match="DensityState"):
        measure_bell_density(
            state,  # type: ignore[arg-type]
            layout=_layout("a", "b"),
            axes=(0, 1),
        )
    with pytest.raises(TypeError, match="StateLayout"):
        measure_bell_ket(state, layout="bad", axes=(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        measure_bell_ket(
            state,
            layout=_layout("a", "b"),
            axes=[0, 1],  # type: ignore[arg-type]
        )
    with pytest.raises(MeasurementError, match="exactly two"):
        measure_bell_ket(
            state,
            layout=_layout("a", "b"),
            axes=(0,),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="axes entries"):
        measure_bell_ket(
            state,
            layout=_layout("a", "b"),
            axes=(0, 1.0),  # type: ignore[arg-type]
        )
    with pytest.raises(MeasurementError, match="unique"):
        measure_bell_ket(state, layout=_layout("a", "b"), axes=(0, 0))
    with pytest.raises(MeasurementError, match="range"):
        measure_bell_ket(state, layout=_layout("a", "b"), axes=(0, 2))
    with pytest.raises(DimensionError, match="qubit axes"):
        measure_bell_ket(
            state,
            layout=_layout("a", "b", dims=(2, 3)),
            axes=(0, 1),
        )
    with pytest.raises(DimensionError, match="layout"):
        measure_bell_ket(state, layout=_layout("a", "b", "c"), axes=(0, 1))
    with pytest.raises(TypeError, match="collapse"):
        measure_bell_ket(
            state,
            layout=_layout("a", "b"),
            axes=(0, 1),
            collapse=1,  # type: ignore[arg-type]
        )
