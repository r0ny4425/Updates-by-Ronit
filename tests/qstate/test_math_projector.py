from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate.errors import DimensionError, InvalidStateError
from simyuj.qstate.math.projector import (
    basis_projectors,
    computational_projectors,
    is_projector,
    outer,
    tensor_projectors,
    vector_projector,
)


def test_outer_and_vector_projector_build_rank_one_matrices() -> None:
    left = np.array([1.0, 1.0j])
    right = np.array([1.0, -1.0j])

    np.testing.assert_allclose(
        outer(left, right),
        np.outer(left, np.conjugate(right)),
    )
    np.testing.assert_allclose(
        vector_projector([1.0, 1.0j]),
        outer([1.0, 1.0j]),
    )
    np.testing.assert_allclose(
        vector_projector([2.0, 0.0], normalize=True),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    )


def test_basis_and_computational_projectors_are_canonical() -> None:
    projectors = basis_projectors(3)

    assert len(projectors) == 3
    np.testing.assert_allclose(sum(projectors), np.eye(3))
    assert all(is_projector(candidate) for candidate in projectors)

    qubit_projectors = computational_projectors(2)
    assert len(qubit_projectors) == 4
    np.testing.assert_allclose(sum(qubit_projectors), np.eye(4))


def test_tensor_projectors_build_cartesian_product_order() -> None:
    zero, one = basis_projectors(2)

    product = tensor_projectors(((zero, one), (zero, one)))

    assert len(product) == 4
    np.testing.assert_allclose(product[0], np.diag([1, 0, 0, 0]))
    np.testing.assert_allclose(product[1], np.diag([0, 1, 0, 0]))
    np.testing.assert_allclose(product[2], np.diag([0, 0, 1, 0]))
    np.testing.assert_allclose(product[3], np.diag([0, 0, 0, 1]))


def test_is_projector_rejects_non_projectors() -> None:
    assert is_projector([[1, 0], [0, 0]])
    assert not is_projector([[0.5, 0], [0, 0.5]])
    assert not is_projector([[1, 1], [0, 0]])


def test_projector_construction_rejects_invalid_state_vectors() -> None:
    with pytest.raises(DimensionError, match="one-dimensional"):
        vector_projector([[1, 0]])
    with pytest.raises(InvalidStateError, match="finite"):
        vector_projector([np.inf, 0])
    with pytest.raises(InvalidStateError, match="cannot normalize"):
        vector_projector([0, 0], normalize=True)

    with pytest.raises(DimensionError, match="positive"):
        basis_projectors(0)

    with pytest.raises(DimensionError, match="positive"):
        computational_projectors(0)


def test_tensor_projectors_reject_invalid_local_projector_sets() -> None:
    with pytest.raises(TypeError, match="local_projectors"):
        tensor_projectors([basis_projectors(2)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        tensor_projectors(())
    with pytest.raises(TypeError, match="projector set"):
        tensor_projectors(((),))
    with pytest.raises(DimensionError, match="square matrix"):
        tensor_projectors(((np.array([1, 0]),),))
