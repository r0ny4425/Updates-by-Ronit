from __future__ import annotations

"""Tensor-product and axis-application helpers for qubit state vectors.

Qubit axes are ordered consistently with state-vector reshaping: axis ``0`` is
the most significant computational-basis bit.  When a local operator is applied
to multiple axes, the order of the ``axes`` tuple defines the operand order of
the local matrix.
"""

from collections.abc import Sequence

import numpy as np

from ..errors import DimensionError
from .const import COMPLEX_DTYPE
from .linalg import as_complex_array


def kron(left: object, right: object) -> np.ndarray:
    """Return the Kronecker product of two complex arrays.

    Parameters
    ----------
    left, right : object
        Array-like operands accepted by ``numpy.kron`` after ``complex128``
        coercion.

    Returns
    -------
    ndarray
        Kronecker product ``kron(left, right)`` with dtype ``complex128``.
    """
    return np.kron(as_complex_array(left), as_complex_array(right))


def kron_all(parts: Sequence[object]) -> np.ndarray:
    """Return the left-to-right Kronecker product of multiple operands.

    Parameters
    ----------
    parts : Sequence[object]
        Non-empty sequence of array-like operands.

    Returns
    -------
    ndarray
        Left-associated Kronecker product with dtype ``complex128``.

    Raises
    ------
    ValueError
        If ``parts`` is empty.
    """
    if not parts:
        raise ValueError("parts must be non-empty")
    result = as_complex_array(parts[0])
    for part in parts[1:]:
        result = np.kron(result, as_complex_array(part))
    return np.asarray(result, dtype=COMPLEX_DTYPE)


def apply_operator_to_axes(
    vector: object,
    matrix: object,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Apply a local qubit operator to selected state-vector axes.

    Parameters
    ----------
    vector : object
        State vector with shape ``(2**num_qubits,)``.
    matrix : object
        Local operator with shape ``(2**len(axes), 2**len(axes))``.  The
        implementation does not check that the matrix is unitary.
    axes : tuple of int
        Unique target axes.  The order of this tuple is the operand order seen
        by ``matrix``.
    num_qubits : int
        Positive number of qubits represented by ``vector``.

    Returns
    -------
    ndarray of complex, shape ``(2**num_qubits,)``
        Updated state vector after applying ``matrix``.

    Raises
    ------
    TypeError
        If ``num_qubits`` is not exactly an ``int``, ``axes`` is not a tuple, or
        an axis is not exactly an ``int``.
    ValueError
        If ``axes`` is empty, contains duplicates, or contains an out-of-range
        axis.
    DimensionError
        If ``vector`` or ``matrix`` has an incompatible shape.

    Notes
    -----
    Axis ``0`` is the most significant basis bit.  For ``axes=(2, 0)``, the
    first operand of ``matrix`` acts on axis ``2`` and the second on axis ``0``.

    Examples
    --------
    >>> from simyuj.qstate.math.matrix import X
    >>> apply_operator_to_axes([1, 0, 0, 0], X, axes=(1,), num_qubits=2)
    array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])
    """
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be int")
    if num_qubits <= 0:
        raise DimensionError("num_qubits must be positive")
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if not axes:
        raise ValueError("axes must be non-empty")
    for axis in axes:
        if type(axis) is not int:
            raise TypeError("axes must be ints")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must be unique")
    for axis in axes:
        if axis < 0 or axis >= num_qubits:
            raise ValueError("axes must be in range")

    state = as_complex_array(vector)
    state_dim = 2**num_qubits
    if state.ndim != 1 or state.shape[0] != state_dim:
        raise DimensionError("vector length must be 2 ** num_qubits")

    op = as_complex_array(matrix)
    target_dim = 2 ** len(axes)
    if op.shape != (target_dim, target_dim):
        raise DimensionError("matrix shape must be (2 ** len(axes), 2 ** len(axes))")

    rest_axes = tuple(axis for axis in range(num_qubits) if axis not in axes)
    permutation = axes + rest_axes
    inverse = np.argsort(permutation)

    tensor = state.reshape((2,) * num_qubits)
    moved = np.transpose(tensor, permutation).reshape(target_dim, -1)
    applied = op @ moved
    restored = applied.reshape((2,) * num_qubits)
    return np.transpose(restored, inverse).reshape(-1)


def apply_unitary_to_axes(
    vector: object,
    matrix: object,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Apply a local unitary matrix to selected state-vector axes.

    This compatibility wrapper delegates to :func:`apply_operator_to_axes`.
    The delegated implementation intentionally checks only shape and axis
    compatibility, so lower layers can reuse it for non-unitary Kraus branches.
    """
    return apply_operator_to_axes(
        vector,
        matrix,
        axes=axes,
        num_qubits=num_qubits,
    )


def expand_operator(
    matrix: object,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Expand a local qubit operator into the full Hilbert space.

    Parameters
    ----------
    matrix : object
        Local operator with shape ``(2**len(axes), 2**len(axes))``.  The
        implementation does not check that the matrix is unitary.
    axes : tuple of int
        Target axes in the operand order used by ``matrix``.
    num_qubits : int
        Positive number of qubits in the full state space.

    Returns
    -------
    ndarray of complex, shape ``(2**num_qubits, 2**num_qubits)``
        Dense operator whose action matches :func:`apply_operator_to_axes`.

    Raises
    ------
    TypeError
        If ``num_qubits`` is not exactly an ``int``.  Additional ``axes`` type
        checks are delegated to :func:`apply_unitary_to_axes`.
    ValueError
        If ``axes`` is invalid, as checked by :func:`apply_unitary_to_axes`.
    DimensionError
        If ``num_qubits`` is not positive or ``matrix`` has an incompatible
        shape.

    Notes
    -----
    The dense matrix is built column by column by applying the local operator to
    every computational-basis vector.  This is simple and exact for the small
    state spaces targeted here, but it scales as a dense ``2**num_qubits``
    operator.
    """
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be int")
    if num_qubits <= 0:
        raise DimensionError("num_qubits must be positive")

    op = as_complex_array(matrix)
    target_dim = 2 ** len(axes)
    if op.shape != (target_dim, target_dim):
        raise DimensionError("matrix shape must be (2 ** len(axes), 2 ** len(axes))")

    size = 2**num_qubits
    columns = []
    for basis_index in range(size):
        vector = np.zeros(size, dtype=COMPLEX_DTYPE)
        vector[basis_index] = 1.0
        columns.append(
            apply_unitary_to_axes(
                vector,
                op,
                axes=axes,
                num_qubits=num_qubits,
            )
        )
    return np.column_stack(columns)


__all__ = [
    "apply_operator_to_axes",
    "apply_unitary_to_axes",
    "expand_operator",
    "kron",
    "kron_all",
]
