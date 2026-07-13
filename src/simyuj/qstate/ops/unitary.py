from __future__ import annotations

"""Dense unitary operation records and constructors.

``Unitary`` stores a read-only ``complex128`` matrix whose dimension is a power
of two.  The matrix acts on ``arity`` qubit operands in the operand order chosen
by the caller or by the gate factory that constructs it.
"""

from dataclasses import dataclass

import numpy as np

from ..errors import DimensionError, InvalidOperationError
from ..math.linalg import is_unitary, readonly
from .base import check_arity


def _infer_qubit_arity(size: int) -> int:
    """Infer the number of qubit operands from a matrix dimension."""
    if size <= 0:
        raise DimensionError("unitary dimension must be a power of two")
    arity = int(np.log2(size))
    if 2**arity != size:
        raise DimensionError("unitary dimension must be a power of two")
    return arity


@dataclass(frozen=True, slots=True, init=False)
class Unitary:
    """Dense qubit unitary operation.

    Parameters
    ----------
    matrix : object
        Square array-like matrix with shape ``(2**arity, 2**arity)``.  The
        matrix is coerced to ``complex128``, checked for unitarity with package
        tolerances, copied, and stored as read-only.
    name : str, default="unitary"
        Human-readable operation name.  Surrounding whitespace is stripped.
    arity : int, optional
        Number of qubit operands.  If omitted, it is inferred from the matrix
        dimension.

    Attributes
    ----------
    matrix : ndarray of complex
        Read-only dense operation matrix.
    name : str
        Non-empty operation name after stripping whitespace.
    arity : int
        Number of qubit operands consumed by the operation.

    Raises
    ------
    DimensionError
        If the matrix is not square, its dimension is not a power of two, or
        the supplied arity does not match the matrix dimension.
    InvalidOperationError
        If the matrix is not unitary within package tolerances.
    TypeError
        If ``arity`` is supplied but is not exactly an ``int``, or if ``name``
        is not a string.
    ValueError
        If ``arity`` is supplied but is not positive, or if ``name`` is empty
        after stripping whitespace.

    Examples
    --------
    >>> import numpy as np
    >>> from simyuj.qstate import QuantumStateManager, SubsystemId
    >>> from simyuj.qstate.ops.unitary import unitary
    >>> flip = unitary(np.array([[0, 1], [1, 0]]), name="X-like")
    >>> q0 = SubsystemId("q0")
    >>> manager = QuantumStateManager()
    >>> state_ref = manager.prepare("|0>", subsystems=(q0,))
    >>> _ = manager.apply(flip, targets=(q0,))
    >>> int(manager.get(state_ref).vector.argmax())
    1
    """

    matrix: np.ndarray
    name: str
    arity: int

    def __init__(
        self,
        matrix: object,
        name: str = "unitary",
        arity: int | None = None,
    ) -> None:
        """Validate and store a dense unitary matrix."""
        array = np.asarray(matrix, dtype=np.complex128)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise DimensionError("unitary matrix must be square")

        checked_arity = _infer_qubit_arity(array.shape[0])
        if arity is not None:
            checked_arity = check_arity(arity)
        if 2**checked_arity != array.shape[0]:
            raise DimensionError("unitary dimension does not match arity")
        if not is_unitary(array):
            raise InvalidOperationError("matrix must be unitary")
        if not isinstance(name, str):
            raise TypeError("name must be str")
        checked_name = name.strip()
        if not checked_name:
            raise ValueError("name must be non-empty")

        object.__setattr__(self, "matrix", readonly(array))
        object.__setattr__(self, "name", checked_name)
        object.__setattr__(self, "arity", checked_arity)


def unitary(
    matrix: object,
    *,
    name: str = "unitary",
    arity: int | None = None,
) -> Unitary:
    """Construct a ``Unitary`` from a dense matrix.

    Parameters
    ----------
    matrix : object
        Array-like matrix passed to ``Unitary``.
    name : str, default="unitary"
        Operation name.
    arity : int, optional
        Number of qubit operands.  If omitted, it is inferred from
        ``matrix.shape[0]``.

    Returns
    -------
    Unitary
        Validated dense operation record.

    Raises
    ------
    DimensionError
        If matrix shape or arity is inconsistent.
    InvalidOperationError
        If the matrix is not unitary within package tolerances.
    TypeError
        If ``name`` or ``arity`` has the wrong type.
    ValueError
        If ``name`` is empty after stripping or ``arity`` is not positive.
    """
    return Unitary(matrix=matrix, name=name, arity=arity)


def identity(arity: int = 1) -> Unitary:
    """Return an identity unitary on ``arity`` qubit operands.

    Parameters
    ----------
    arity : int, default=1
        Positive number of qubit operands.

    Returns
    -------
    Unitary
        Identity matrix with shape ``(2**arity, 2**arity)`` and name
        ``"I{arity}"``.

    Raises
    ------
    TypeError
        If ``arity`` is not exactly an ``int``.
    ValueError
        If ``arity`` is not positive.
    """
    arity = check_arity(arity)
    return Unitary(np.eye(2**arity, dtype=np.complex128), name=f"I{arity}", arity=arity)


def check_unitary(value: object) -> Unitary:
    """Return ``value`` if it is a ``Unitary``.

    Parameters
    ----------
    value : object
        Candidate operation.

    Returns
    -------
    Unitary
        The input value, unchanged.

    Raises
    ------
    TypeError
        If ``value`` is not a ``Unitary`` instance.
    """
    if not isinstance(value, Unitary):
        raise TypeError("operation must be Unitary")
    return value


__all__ = ["Unitary", "check_unitary", "identity", "unitary"]
