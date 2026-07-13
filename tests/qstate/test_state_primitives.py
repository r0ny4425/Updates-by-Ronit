from __future__ import annotations

import random
from typing import cast

import numpy as np
import pytest

from simyuj.qstate.errors import (
    DimensionError,
    InvalidOperationError,
    InvalidReprError,
    InvalidStateError,
    MeasurementError,
)
from simyuj.qstate.math.linalg import is_hermitian, is_psd, normalize_density, trace
from simyuj.qstate.math.tensor import (
    apply_operator_to_axes,
    apply_unitary_to_axes,
    expand_operator,
)
from simyuj.qstate.measure.basis import MeasurementBasis, basis_for
from simyuj.qstate.measure.projective import measure_density
from simyuj.qstate.measure.result import BellResult, MeasurementResult, POVMResult
from simyuj.qstate.measure.sample import normalize_probs, sample_probs
from simyuj.qstate.ops import CNOT, X, unitary
from simyuj.qstate.space import StateLayout, SubsystemId
from simyuj.qstate.state import KetState, basis, bell, ghz, make_ket, zero
from simyuj.qstate.state.convert import as_rep, density_to_ket_if_pure, ket_to_density
from simyuj.qstate.state.density import DensityHandler, DensityState


def _array(value: object) -> np.ndarray:
    return cast(np.ndarray, value)


def _layout(*names: str, dims: tuple[int, ...] | None = None) -> StateLayout:
    return StateLayout(
        tuple(SubsystemId(name) for name in names),
        dims if dims is not None else (2,) * len(names),
    )


def test_ket_state_validation_and_readonly_storage() -> None:
    state = zero()

    assert isinstance(state, KetState)
    assert state.num_qubits == 1
    assert not _array(state.vector).flags.writeable

    with pytest.raises(DimensionError, match="one-dimensional"):
        KetState([[1, 0]])
    with pytest.raises(DimensionError, match="power of two"):
        KetState([1, 0, 0])
    with pytest.raises(InvalidStateError, match="positive and finite"):
        KetState([0, 0])
    with pytest.raises(InvalidStateError, match="normalized"):
        KetState([1, 1])


def test_ket_constructors_distinguish_basis_bits_from_raw_vectors() -> None:
    assert tuple(_array(basis("01").vector)) == pytest.approx((0, 1, 0, 0))
    assert tuple(_array(basis([1, 0]).vector)) == pytest.approx((0, 0, 1, 0))
    assert tuple(_array(make_ket([1, 0]).vector)) == pytest.approx((1, 0))

    with pytest.raises(InvalidStateError, match="basis bits"):
        basis("02")
    with pytest.raises(InvalidStateError, match="unsupported ket state"):
        make_ket("cat")


def test_ket_constructors_support_bell_labels() -> None:
    inv = 1.0 / np.sqrt(2.0)

    assert tuple(_array(bell("phi+").vector)) == pytest.approx((inv, 0, 0, inv))
    assert tuple(_array(make_ket("psi-").vector)) == pytest.approx((0, inv, -inv, 0))
    assert tuple(_array(make_ket("psi_minus").vector)) == pytest.approx(
        (0, inv, -inv, 0)
    )


def test_ket_constructors_support_explicit_ghz_state() -> None:
    inv = 1.0 / np.sqrt(2.0)

    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = inv
    expected[-1] = inv

    assert tuple(_array(ghz(3).vector)) == pytest.approx(tuple(expected))

    expected_four = np.zeros(16, dtype=np.complex128)
    expected_four[0] = inv
    expected_four[-1] = inv

    assert tuple(_array(ghz(4).vector)) == pytest.approx(tuple(expected_four))

    with pytest.raises(TypeError, match="num_qubits"):
        ghz(3.0)  # type: ignore[arg-type]

    with pytest.raises(InvalidStateError, match="at least three"):
        ghz(2)


def test_unitary_validation_and_readonly_storage() -> None:
    gate = unitary(X.matrix, name="custom_x", arity=1)

    assert gate.name == "custom_x"
    assert gate.arity == 1
    assert not _array(gate.matrix).flags.writeable

    with pytest.raises(DimensionError, match="does not match arity"):
        unitary(X.matrix, arity=2)
    with pytest.raises(InvalidOperationError, match="unitary"):
        unitary([[1, 1], [0, 1]])


def test_density_linalg_helpers_validate_matrix_properties() -> None:
    density = np.array([[2, 1j], [-1j, 2]], dtype=complex)
    normalized = normalize_density(density)

    assert trace(density) == pytest.approx(4.0)
    assert trace(normalized) == pytest.approx(1.0)
    assert is_hermitian(density)
    assert is_psd(density)
    assert not is_hermitian([[1, 1], [0, 1]])
    assert not is_psd([[1, 0], [0, -1]])

    with pytest.raises(ValueError, match="square"):
        normalize_density([[1, 0]])
    with pytest.raises(ValueError, match="finite and non-zero"):
        normalize_density([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="finite and non-zero"):
        normalize_density([[np.inf, 0], [0, 1]])


def test_density_state_validation_and_readonly_storage() -> None:
    state = DensityState([[1, 0], [0, 0]])

    assert state.num_qubits == 1
    assert state.matrix is state.rho
    assert not _array(state.rho).flags.writeable

    with pytest.raises(DimensionError, match="square"):
        DensityState([[1, 0]])
    with pytest.raises(DimensionError, match="power of two"):
        DensityState(np.eye(3) / 3)
    with pytest.raises(InvalidStateError, match="Hermitian"):
        DensityState([[0.5, 1], [0, 0.5]])
    with pytest.raises(InvalidStateError, match="trace"):
        DensityState([[0.5, 0], [0, 0]])
    with pytest.raises(InvalidStateError, match="positive semidefinite"):
        DensityState([[1.1, 0], [0, -0.1]])


def test_density_handler_tensors_and_applies_unitaries() -> None:
    handler = DensityHandler()
    zero_density = DensityState([[1, 0], [0, 0]])
    one_density = DensityState([[0, 0], [0, 1]])

    assert handler.make(zero_density) is zero_density
    made = handler.make([[1, 0], [0, 0]])
    assert isinstance(made, DensityState)
    assert made.rho == pytest.approx(zero_density.rho)

    combined = handler.tensor(zero_density, one_density)
    assert combined.rho == pytest.approx(np.kron(zero_density.rho, one_density.rho))

    applied = handler.apply(
        zero_density,
        X,
        layout=_layout("q0"),
        axes=(0,),
    )
    assert applied.rho == pytest.approx(one_density.rho)

    with pytest.raises(TypeError, match="payload"):
        handler.apply("bad", X, layout=_layout("q0"), axes=(0,))
    with pytest.raises(DimensionError, match="layout"):
        handler.apply(zero_density, X, layout=_layout("q0", "q1"), axes=(0,))
    with pytest.raises(DimensionError, match="qubit axes"):
        handler.apply(
            combined,
            X,
            layout=_layout("a", "b", dims=(1, 4)),
            axes=(0,),
        )


def test_state_conversion_between_ket_and_density_representations() -> None:
    ket = KetState([1j / np.sqrt(2), -1j / np.sqrt(2)])
    density = ket_to_density(ket)

    expected_density = np.outer(ket.vector, np.conjugate(ket.vector))
    assert density.rho == pytest.approx(expected_density)
    assert as_rep(ket, "ket") is ket
    assert as_rep(density, "density") is density
    converted_density = as_rep(ket, "density")
    assert isinstance(converted_density, DensityState)
    assert converted_density.rho == pytest.approx(density.rho)

    restored = density_to_ket_if_pure(density)
    assert restored.vector[0].imag == pytest.approx(0.0)
    assert restored.vector[0].real > 0.0
    assert ket_to_density(restored).rho == pytest.approx(density.rho)
    converted_ket = as_rep(density, "ket")
    assert isinstance(converted_ket, KetState)
    assert converted_ket.vector == pytest.approx(restored.vector)

    mixed = DensityState([[0.5, 0], [0, 0.5]])
    with pytest.raises(InvalidStateError, match="not pure"):
        density_to_ket_if_pure(mixed)
    with pytest.raises(TypeError, match="KetState"):
        ket_to_density(density)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DensityState"):
        density_to_ket_if_pure(ket)  # type: ignore[arg-type]
    with pytest.raises(InvalidReprError, match="unsupported"):
        as_rep(ket, "bad")


def test_apply_unitary_to_axes_keeps_order_for_non_adjacent_axes() -> None:
    state = basis("001")
    result = apply_unitary_to_axes(
        state.vector,
        CNOT.matrix,
        axes=(2, 0),
        num_qubits=3,
    )

    assert tuple(result) == pytest.approx(tuple(_array(basis("101").vector)))


def test_apply_operator_to_axes_supports_non_unitary_local_ops() -> None:
    state = basis("10")
    damping_no_jump = np.array(
        [[1.0, 0.0], [0.0, 0.5]],
        dtype=np.complex128,
    )

    result = apply_operator_to_axes(
        state.vector,
        damping_no_jump,
        axes=(0,),
        num_qubits=2,
    )
    expanded = expand_operator(damping_no_jump, axes=(0,), num_qubits=2)

    assert result == pytest.approx(expanded @ state.vector)
    assert tuple(result) == pytest.approx((0.0, 0.0, 0.5, 0.0))


def test_expand_operator_builds_dense_operator_from_axis_action() -> None:
    expanded = expand_operator(X.matrix, axes=(1,), num_qubits=2)

    assert expanded == pytest.approx(np.kron(np.eye(2), X.matrix))

    with pytest.raises(TypeError, match="int"):
        expand_operator(
            X.matrix,
            axes=(0,),
            num_qubits=1.0,  # type: ignore[arg-type]
        )
    with pytest.raises(DimensionError, match="positive"):
        expand_operator(X.matrix, axes=(0,), num_qubits=0)
    with pytest.raises(DimensionError, match="matrix shape"):
        expand_operator(X.matrix, axes=(0, 1), num_qubits=2)


def test_measure_density_computes_probabilities_without_collapse() -> None:
    state = DensityState([[0.25, 0], [0, 0.75]])

    result = measure_density(
        state,
        layout=_layout("q0"),
        axes=(0,),
        rng=random.Random(0),
        collapse=False,
    )

    probabilities = dict(result.probabilities)
    assert result.outcome == (1,)
    assert result.probability == pytest.approx(0.75)
    assert probabilities[("0",)] == pytest.approx(0.25)
    assert probabilities[("1",)] == pytest.approx(0.75)
    assert result.post_state is None
    assert not result.collapsed


def test_measure_density_collapses_in_requested_basis() -> None:
    plus_density = DensityState([[0.5, 0.5], [0.5, 0.5]])

    result = measure_density(
        plus_density,
        layout=_layout("q0"),
        axes=(0,),
        basis="x",
        collapse=True,
    )

    assert result.outcome == (0,)
    assert result.label == "+"
    assert result.probability == pytest.approx(1.0)
    assert isinstance(result.post_state, DensityState)
    assert result.post_state.rho == pytest.approx(plus_density.rho)


def test_density_handler_measure_delegates_to_density_projective() -> None:
    state = DensityState([[0, 0], [0, 1]])
    handler = DensityHandler()

    result = handler.measure(state, layout=_layout("q0"), axes=(0,))

    assert result.outcome == (1,)
    assert result.probability == pytest.approx(1.0)
    assert isinstance(result.post_state, DensityState)
    assert result.post_state.rho == pytest.approx(state.rho)


def test_measurement_basis_validation_and_resolution() -> None:
    custom = MeasurementBasis(
        name=" Custom ",
        vectors=([1, 0], [0, 1]),
        labels=("left", "right"),
    )

    assert custom.name == "custom"
    assert basis_for(custom) is custom
    assert basis_for("computational").name == "z"
    assert not _array(custom.vectors[0]).flags.writeable

    with pytest.raises(MeasurementError, match="orthonormal"):
        MeasurementBasis(name="bad", vectors=([1, 0], [1, 0]), labels=("0", "1"))
    with pytest.raises(MeasurementError, match="unsupported measurement basis"):
        basis_for("bad")


def test_probability_sampling_allows_certain_outcomes_without_rng() -> None:
    assert normalize_probs([-1.0, 2.0]) == (0.0, 1.0)
    assert sample_probs([0.0, 3.0], rng=None) == 1

    with pytest.raises(ValueError, match="positive total"):
        normalize_probs([-1.0, 0.0])
    with pytest.raises(ValueError, match="explicit rng"):
        sample_probs([0.5, 0.5], rng=None)
    assert sample_probs([0.25, 0.75], rng=random.Random(3)) == 0


def test_measurement_result_normalizes_meta_and_labels() -> None:
    result = MeasurementResult(
        outcome=(0, 1),
        outcome_labels=("0", "1"),
        probability=1,
        probabilities=((("0", "1"), 1.0),),
        meta={"basis": "z"},
    )

    assert result.label == ("0", "1")
    assert result.probability == 1.0
    assert result.meta == (("basis", "z"),)

    with pytest.raises(ValueError, match="bits"):
        MeasurementResult(
            outcome=(2,),
            outcome_labels=("2",),
            probability=1.0,
            probabilities=(),
        )


def test_bell_result_normalizes_label_probabilities_refs_and_meta() -> None:
    result = BellResult(
        label=" Phi+ ",
        outcome=(0, 0),
        probability=1,
        probabilities=((" Phi+ ", 1), ("psi-", 0.0)),
        state_ref=0,
        post_state_ref=1,
        collapsed=False,
        meta={"basis": "bell"},
    )

    assert result.label == "phi+"
    assert result.outcome_label == "phi+"
    assert result.probability == 1.0
    assert result.probabilities == (("phi+", 1.0), ("psi-", 0.0))
    assert result.state_ref == 0
    assert result.post_state_ref == 1
    assert result.meta == (("basis", "bell"),)
    assert not result.collapsed


def test_povm_result_normalizes_label_probabilities_refs_and_meta() -> None:
    result = POVMResult(
        outcome=2,
        label=" click-2 ",
        probability=1,
        probabilities=((" click-2 ", 1), ("dark", 0.0)),
        state_ref=0,
        post_state_ref=1,
        collapsed=False,
        meta={"basis": "povm"},
    )

    assert result.outcome == 2
    assert result.label == "click-2"
    assert result.probability == 1.0
    assert result.probabilities == (("click-2", 1.0), ("dark", 0.0))
    assert result.state_ref == 0
    assert result.post_state_ref == 1
    assert result.meta == (("basis", "povm"),)
    assert not result.collapsed


def test_bell_result_validates_shape_and_probability_fields() -> None:
    with pytest.raises(TypeError, match="label"):
        BellResult(
            label=1,  # type: ignore[arg-type]
            outcome=(0, 0),
            probability=1.0,
            probabilities=(),
        )
    with pytest.raises(ValueError, match="non-empty"):
        BellResult(label=" ", outcome=(0, 0), probability=1.0, probabilities=())
    with pytest.raises(TypeError, match="outcome"):
        BellResult(
            label="phi+",
            outcome=[0, 0],  # type: ignore[arg-type]
            probability=1.0,
            probabilities=(),
        )
    with pytest.raises(ValueError, match="exactly two"):
        BellResult(
            label="phi+",
            outcome=(0,),  # type: ignore[arg-type]
            probability=1.0,
            probabilities=(),
        )
    with pytest.raises(ValueError, match="bits"):
        BellResult(label="phi+", outcome=(0, 2), probability=1.0, probabilities=())
    with pytest.raises(TypeError, match="probability"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability="1",  # type: ignore[arg-type]
            probabilities=(),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BellResult(label="phi+", outcome=(0, 0), probability=1.1, probabilities=())


def test_bell_result_validates_probability_table_refs_collapse_and_meta() -> None:
    with pytest.raises(TypeError, match="probabilities"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=[],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="2-tuples"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=(("phi+", 1.0, "extra"),),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="labels"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=((1, 1.0),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=((" ", 1.0),),
        )
    with pytest.raises(TypeError, match="values"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=(("phi+", "1"),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=(("phi+", -0.1),),
        )
    with pytest.raises(TypeError, match="state_ref"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=(),
            state_ref="0",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="collapsed"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=(),
            collapsed=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="meta"):
        BellResult(
            label="phi+",
            outcome=(0, 0),
            probability=1.0,
            probabilities=(),
            meta=object(),  # type: ignore[arg-type]
        )


def test_povm_result_validates_shape_and_probability_fields() -> None:
    with pytest.raises(TypeError, match="outcome"):
        POVMResult(
            outcome="0",  # type: ignore[arg-type]
            label="click",
            probability=1.0,
            probabilities=(),
        )
    with pytest.raises(ValueError, match="non-negative"):
        POVMResult(outcome=-1, label="click", probability=1.0, probabilities=())
    with pytest.raises(TypeError, match="label"):
        POVMResult(
            outcome=0,
            label=1,  # type: ignore[arg-type]
            probability=1.0,
            probabilities=(),
        )
    with pytest.raises(ValueError, match="non-empty"):
        POVMResult(outcome=0, label=" ", probability=1.0, probabilities=())
    with pytest.raises(TypeError, match="probability"):
        POVMResult(
            outcome=0,
            label="click",
            probability="1",  # type: ignore[arg-type]
            probabilities=(),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        POVMResult(outcome=0, label="click", probability=1.1, probabilities=())


def test_povm_result_validates_probability_table_refs_collapse_and_meta() -> None:
    with pytest.raises(TypeError, match="probabilities"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=[],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="2-tuples"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(("click", 1.0, "extra"),),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="labels"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=((1, 1.0),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=((" ", 1.0),),
        )
    with pytest.raises(TypeError, match="values"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(("click", "1"),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(("click", -0.1),),
        )
    with pytest.raises(TypeError, match="state_ref"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(),
            state_ref="0",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="post_state_ref"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(),
            post_state_ref="0",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="collapsed"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(),
            collapsed=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="meta"):
        POVMResult(
            outcome=0,
            label="click",
            probability=1.0,
            probabilities=(),
            meta=object(),  # type: ignore[arg-type]
        )
