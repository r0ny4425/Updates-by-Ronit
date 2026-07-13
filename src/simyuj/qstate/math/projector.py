from __future__ import annotations

"""Projector construction and validation helpers.

The routines here build dense ``complex128`` rank-one projectors and tensor
products in computational-basis order.  Shape and finiteness are validated at
the boundary, but most constructors do not require normalized vectors unless the
``normalize`` option is explicitly requested.
"""

import numpy as np

from ..errors import DimensionError, InvalidStateError
from .const import ATOL, COMPLEX_DTYPE
from .linalg import dagger
from .tensor import kron_all


def outer(left: object, right: object | None = None) -> np.ndarray:
    """Return the outer product :math:`|\\mathrm{left}\\rangle\\langle\\mathrm{right}|`.

    Parameters
    ----------
    left : object
        One-dimensional finite vector with shape ``(d,)``.
    right : object, optional
        One-dimensional finite vector with shape ``(d,)``.  If omitted, ``left``
        is used for both sides.

    Returns
    -------
    ndarray of complex, shape ``(len(left), len(right))``
        Dense matrix ``outer(left, conjugate(right))``.

    Raises
    ------
    DimensionError
        If either input is not a non-empty one-dimensional vector.
    InvalidStateError
        If either vector contains non-finite entries.
    """
    left_vector = _as_vector(left)
    right_vector = left_vector if right is None else _as_vector(right)
    return np.outer(left_vector, np.conjugate(right_vector)).astype(COMPLEX_DTYPE)


def vector_projector(vector: object, *, normalize: bool = False) -> np.ndarray:
    """Return the rank-one projector for a state vector.

    Parameters
    ----------
    vector : object
        One-dimensional finite vector with shape ``(d,)``.
    normalize : bool, default=False
        If ``True``, normalize the vector before forming
        :math:`|v\\rangle\\langle v|`.

    Returns
    -------
    ndarray of complex, shape (d, d)
        Dense outer product :math:`|v\\rangle\\langle v|`.

    Raises
    ------
    DimensionError
        If ``vector`` is not a non-empty one-dimensional vector.
    InvalidStateError
        If ``vector`` contains non-finite entries, or if ``normalize=True`` and
        the norm is zero or non-finite.

    Notes
    -----
    With ``normalize=False``, the implementation does not check that ``vector``
    has unit norm.  A non-normalized input produces a rank-one positive matrix
    whose trace is :math:`\\lVert v\\rVert^2` rather than a mathematical
    projector.
    """
    checked = _as_vector(vector)

    if normalize:
        norm = float(np.linalg.norm(checked))
        if norm <= 0.0 or not np.isfinite(norm):
            raise InvalidStateError("cannot normalize zero or non-finite vector")
        checked = checked / norm

    return outer(checked)


def basis_projectors(dim: int) -> tuple[np.ndarray, ...]:
    """Construct computational-basis projectors for a Hilbert space dimension.

    Parameters
    ----------
    dim : int
        Positive Hilbert-space dimension.

    Returns
    -------
    tuple of ndarray
        Projectors :math:`|0\\rangle\\langle 0|` through
        :math:`|d - 1\\rangle\\langle d - 1|` in increasing integer basis
        order.

    Raises
    ------
    TypeError
        If ``dim`` is not exactly an ``int``.
    DimensionError
        If ``dim`` is not positive.
    """
    if type(dim) is not int:
        raise TypeError("dim must be int")
    if dim <= 0:
        raise DimensionError("dim must be positive")

    projectors = []
    for index in range(dim):
        vector = np.zeros(dim, dtype=COMPLEX_DTYPE)
        vector[index] = 1.0
        projectors.append(vector_projector(vector))
    return tuple(projectors)


def computational_projectors(num_qubits: int) -> tuple[np.ndarray, ...]:
    """Construct computational-basis projectors for qubits.

    Parameters
    ----------
    num_qubits : int
        Positive number of qubits.

    Returns
    -------
    tuple of ndarray
        ``2**num_qubits`` projectors ordered by computational-basis integer
        index from ``0`` to ``2**num_qubits - 1``.

    Raises
    ------
    TypeError
        If ``num_qubits`` is not exactly an ``int``.
    DimensionError
        If ``num_qubits`` is not positive.
    """
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be int")
    if num_qubits <= 0:
        raise DimensionError("num_qubits must be positive")

    return basis_projectors(2**num_qubits)


def tensor_projectors(
    local_projectors: tuple[tuple[np.ndarray, ...], ...],
) -> tuple[np.ndarray, ...]:
    """Construct tensor products from local projector sets.

    Parameters
    ----------
    local_projectors : tuple of tuple of ndarray
        Non-empty tuple of non-empty projector sets.  Each local matrix must be
        finite and square.

    Returns
    -------
    tuple of ndarray
        Tensor products in Cartesian-product order.  Earlier projector sets are
        the more significant tensor factors.

    Raises
    ------
    TypeError
        If ``local_projectors`` or any projector set is not a tuple.
    ValueError
        If ``local_projectors`` is empty.
    DimensionError
        If any local matrix is not square.
    InvalidStateError
        If any local matrix contains non-finite entries.

    Notes
    -----
    This function checks only matrix shape and finiteness for local entries.  It
    does not call :func:`is_projector` on each local matrix.
    """
    if not isinstance(local_projectors, tuple):
        raise TypeError("local_projectors must be tuple")
    if not local_projectors:
        raise ValueError("local_projectors must be non-empty")

    result = [np.array([[1.0]], dtype=COMPLEX_DTYPE)]
    for projector_set in local_projectors:
        if not isinstance(projector_set, tuple) or not projector_set:
            raise TypeError("each projector set must be a non-empty tuple")

        next_result = []
        for prefix in result:
            for local in projector_set:
                next_result.append(kron_all((prefix, _as_square_matrix(local))))
        result = next_result
    return tuple(result)


def is_projector(matrix: object, *, atol: float = ATOL) -> bool:
    """Return whether a matrix is Hermitian and idempotent.

    Parameters
    ----------
    matrix : object
        Array-like square matrix.
    atol : float, default=ATOL
        Absolute and relative tolerance passed to ``numpy.allclose``.

    Returns
    -------
    bool
        ``True`` when ``matrix`` is close to its conjugate transpose and to
        ``matrix @ matrix``.

    Raises
    ------
    DimensionError
        If ``matrix`` is not square.
    InvalidStateError
        If ``matrix`` contains non-finite entries.
    """
    checked = _as_square_matrix(matrix)
    return bool(
        np.allclose(checked, dagger(checked), atol=atol, rtol=atol)
        and np.allclose(checked @ checked, checked, atol=atol, rtol=atol)
    )


def _as_vector(value: object) -> np.ndarray:
    vector = np.asarray(value, dtype=COMPLEX_DTYPE)
    if vector.ndim != 1:
        raise DimensionError("value must be one-dimensional vector")
    if vector.size <= 0:
        raise DimensionError("vector must be non-empty")
    if not np.all(np.isfinite(vector)):
        raise InvalidStateError("vector entries must be finite")
    return vector


def _as_square_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=COMPLEX_DTYPE)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError("value must be square matrix")
    if not np.all(np.isfinite(matrix)):
        raise InvalidStateError("matrix entries must be finite")
    return matrix


__all__ = [
    "basis_projectors",
    "computational_projectors",
    "is_projector",
    "outer",
    "tensor_projectors",
    "vector_projector",
]
