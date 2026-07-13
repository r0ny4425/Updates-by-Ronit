from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate.errors import DimensionError, MeasurementError
from simyuj.qstate.measure.povm import (
    POVM,
    POVMElement,
    measure_povm_density,
    measure_povm_ket,
)
from simyuj.qstate.measure.result import POVMResult
from simyuj.qstate.state.convert import ket_to_density
from simyuj.qstate.state.density import DensityState
from simyuj.qstate.state.make import basis


class FixedRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _z_povm() -> POVM:
    return POVM(
        (
            POVMElement("zero", [[1, 0], [0, 0]]),
            POVMElement("one", [[0, 0], [0, 1]]),
        ),
        name="z",
    )


def _assert_density(actual: DensityState, expected: DensityState) -> None:
    np.testing.assert_allclose(actual.rho, expected.rho, atol=1e-12)


def test_povm_element_normalizes_label_default_op_and_readonly_storage() -> None:
    element = POVMElement(" zero ", [[1, 0], [0, 0]])

    assert element.label == "zero"
    assert element.arity == 1
    np.testing.assert_allclose(element.op, element.effect, atol=1e-12)
    assert not element.effect.flags.writeable
    assert not element.op.flags.writeable


def test_povm_element_accepts_custom_collapse_op() -> None:
    effect = np.array([[0.25, 0.0], [0.0, 0.0]], dtype=np.complex128)
    op = np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.complex128)

    element = POVMElement("dim", effect, op)

    np.testing.assert_allclose(element.effect, effect, atol=1e-12)
    np.testing.assert_allclose(element.op, op, atol=1e-12)


def test_povm_validates_elements_and_exposes_labels() -> None:
    povm = _z_povm()

    assert povm.name == "z"
    assert povm.arity == 1
    assert povm.labels == ("zero", "one")


def test_measure_povm_ket_delegates_through_density_measurement() -> None:
    result = measure_povm_ket(basis("1"), _z_povm(), axes=(0,))

    assert isinstance(result, POVMResult)
    assert result.outcome == 1
    assert result.label == "one"
    assert result.probability == 1.0
    assert result.probabilities == (("zero", 0.0), ("one", 1.0))
    _assert_density(
        result.post_state,  # type: ignore[arg-type]
        ket_to_density(basis("1")),
    )


def test_measure_povm_density_expands_effects_to_selected_axis() -> None:
    density = ket_to_density(basis("10"))

    result = measure_povm_density(density, _z_povm(), axes=(1,))

    assert result.outcome == 0
    assert result.label == "zero"
    _assert_density(result.post_state, density)  # type: ignore[arg-type]


def test_measure_povm_density_supports_non_projective_custom_collapse() -> None:
    density = ket_to_density(basis("0"))
    povm = POVM(
        (
            POVMElement("dim", [[0.25, 0], [0, 0.25]], [[0.5, 0], [0, 0.5]]),
            POVMElement("bright", [[0.75, 0], [0, 0.75]]),
        )
    )

    result = measure_povm_density(density, povm, axes=(0,), rng=FixedRng(0.1))

    assert result.outcome == 0
    assert result.label == "dim"
    assert result.probability == pytest.approx(0.25)
    assert result.probabilities[0][0] == "dim"
    assert result.probabilities[0][1] == pytest.approx(0.25)
    assert result.probabilities[1][0] == "bright"
    assert result.probabilities[1][1] == pytest.approx(0.75)
    _assert_density(result.post_state, density)  # type: ignore[arg-type]


def test_measure_povm_density_can_skip_collapse() -> None:
    result = measure_povm_density(
        ket_to_density(basis("0")),
        _z_povm(),
        axes=(0,),
        collapse=False,
    )

    assert result.post_state is None
    assert not result.collapsed


def test_povm_element_validates_label_effect_and_custom_op() -> None:
    with pytest.raises(TypeError, match="label"):
        POVMElement(1, [[1]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        POVMElement(" ", [[1]])
    with pytest.raises(DimensionError, match="square"):
        POVMElement("bad", [1, 0])
    with pytest.raises(DimensionError, match="power of two"):
        POVMElement("bad", np.eye(3))
    with pytest.raises(MeasurementError, match="Hermitian"):
        POVMElement("bad", [[1, 1], [0, 0]])
    with pytest.raises(MeasurementError, match="positive"):
        POVMElement("bad", [[1, 0], [0, -1]])
    with pytest.raises(DimensionError, match="shape"):
        POVMElement("bad", np.eye(2), np.eye(4))
    with pytest.raises(MeasurementError, match="M†M"):
        POVMElement("bad", np.eye(2), np.diag([1, 0]))


def test_povm_validates_container_invariants() -> None:
    zero = POVMElement("zero", [[1, 0], [0, 0]])
    one = POVMElement("one", [[0, 0], [0, 1]])

    with pytest.raises(TypeError, match="elements"):
        POVM([zero, one])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        POVM(())
    with pytest.raises(TypeError, match="POVMElement"):
        POVM((zero, object()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name"):
        POVM((zero, one), name=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        POVM((zero, one), name=" ")
    with pytest.raises(DimensionError, match="arities"):
        POVM((zero, POVMElement("two", np.eye(4))))
    with pytest.raises(ValueError, match="unique"):
        POVM((zero, POVMElement("zero", [[0, 0], [0, 1]])))
    with pytest.raises(MeasurementError, match="identity"):
        POVM((zero,))


def test_measure_povm_density_validates_inputs() -> None:
    density = ket_to_density(basis("0"))
    povm = _z_povm()

    with pytest.raises(TypeError, match="DensityState"):
        measure_povm_density(basis("0"), povm, axes=(0,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="POVM"):
        measure_povm_density(density, object(), axes=(0,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="axes"):
        measure_povm_density(density, povm, axes=[0])  # type: ignore[arg-type]
    with pytest.raises(MeasurementError, match="arity"):
        measure_povm_density(density, povm, axes=())
    with pytest.raises(TypeError, match="entries"):
        measure_povm_density(density, povm, axes=(True,))  # type: ignore[arg-type]
    with pytest.raises(MeasurementError, match="range"):
        measure_povm_density(density, povm, axes=(1,))
    two_axis_povm = POVM((POVMElement("id", np.eye(4)),))
    with pytest.raises(MeasurementError, match="unique"):
        measure_povm_density(
            ket_to_density(basis("00")),
            two_axis_povm,
            axes=(0, 0),
        )
    with pytest.raises(TypeError, match="collapse"):
        measure_povm_density(
            density,
            povm,
            axes=(0,),
            collapse=1,  # type: ignore[arg-type]
        )


def test_measure_povm_ket_validates_state() -> None:
    with pytest.raises(TypeError, match="KetState"):
        measure_povm_ket(
            ket_to_density(basis("0")),  # type: ignore[arg-type]
            _z_povm(),
            axes=(0,),
        )
