from __future__ import annotations

"""Single-qubit measurement bases and basis resolution.

Measurement bases are represented by two read-only ``complex128`` vectors whose
columns form a unitary matrix.  Built-in bases cover computational ``Z``, ``X``,
and ``Y`` measurements.
"""

from dataclasses import dataclass

import numpy as np

from ..errors import MeasurementError
from ..math.const import SQRT2
from ..math.linalg import dagger, is_unitary, readonly


@dataclass(frozen=True, slots=True, init=False)
class MeasurementBasis:
    """Single-qubit orthonormal measurement basis.

    Parameters
    ----------
    name : str
        Basis name.  The stored value is stripped and lower-cased.
    vectors : tuple of object
        Two array-like basis vectors.  Each vector must reshape to length two,
        and the two vectors must be orthonormal under ``is_unitary`` when used
        as matrix columns.
    labels : tuple of str
        Labels corresponding to outcome bits ``0`` and ``1``.

    Attributes
    ----------
    name : str
        Normalized basis name.
    vectors : tuple of ndarray
        Read-only ``complex128`` basis vectors.
    labels : tuple of str
        Outcome labels in bit order.
    matrix : ndarray
        Matrix whose columns are ``vectors``.
    inverse_matrix : ndarray
        Conjugate transpose of ``matrix``.

    Raises
    ------
    TypeError
        If ``name``, ``vectors``, ``labels``, or label entries have invalid
        types.
    ValueError
        If ``name`` or any label is empty, or if ``vectors`` or ``labels`` do
        not contain exactly two entries.
    MeasurementError
        If a vector cannot reshape to length two, or if the vectors are not
        orthonormal.
    """

    name: str
    vectors: tuple[np.ndarray, np.ndarray]
    labels: tuple[str, str]

    def __init__(
        self,
        name: str,
        vectors: tuple[object, object],
        labels: tuple[str, str],
    ) -> None:
        """Validate and store a two-vector measurement basis."""
        if not isinstance(name, str):
            raise TypeError("name must be str")
        checked_name = name.strip().lower()
        if not checked_name:
            raise ValueError("name must be non-empty")

        if not isinstance(vectors, tuple):
            raise TypeError("vectors must be tuple")
        if len(vectors) != 2:
            raise ValueError("vectors must contain exactly two vectors")
        if not isinstance(labels, tuple):
            raise TypeError("labels must be tuple")
        if len(labels) != 2:
            raise ValueError("labels must contain exactly two labels")
        for label in labels:
            if not isinstance(label, str):
                raise TypeError("labels entries must be str")
            if not label:
                raise ValueError("labels entries must be non-empty")

        checked_vectors = tuple(self._check_vector(vector) for vector in vectors)
        matrix = np.column_stack(checked_vectors)
        if not is_unitary(matrix):
            raise MeasurementError("basis vectors must be orthonormal")

        object.__setattr__(self, "name", checked_name)
        object.__setattr__(self, "vectors", checked_vectors)
        object.__setattr__(self, "labels", labels)

    @staticmethod
    def _check_vector(vector: object) -> np.ndarray:
        """Coerce one basis vector to read-only ``complex128`` length two."""
        try:
            array = np.asarray(vector, dtype=np.complex128).reshape(2)
        except (TypeError, ValueError):
            raise MeasurementError("vectors must reshape to length two") from None
        return readonly(array)

    @property
    def matrix(self) -> np.ndarray:
        """Return the dense basis matrix with basis vectors as columns."""
        return np.column_stack(self.vectors)

    @property
    def inverse_matrix(self) -> np.ndarray:
        """Return the conjugate transpose of ``matrix``."""
        return dagger(self.matrix)


Z_BASIS = MeasurementBasis(
    name="z",
    vectors=(np.array([1, 0]), np.array([0, 1])),
    labels=("0", "1"),
)
X_BASIS = MeasurementBasis(
    name="x",
    vectors=(np.array([1, 1]) / SQRT2, np.array([1, -1]) / SQRT2),
    labels=("+", "-"),
)
Y_BASIS = MeasurementBasis(
    name="y",
    vectors=(np.array([1, 1j]) / SQRT2, np.array([1, -1j]) / SQRT2),
    labels=("+i", "-i"),
)


def z_basis() -> MeasurementBasis:
    """Return the built-in computational ``Z`` basis."""
    return Z_BASIS


def x_basis() -> MeasurementBasis:
    """Return the built-in ``X`` basis."""
    return X_BASIS


def y_basis() -> MeasurementBasis:
    """Return the built-in ``Y`` basis."""
    return Y_BASIS


def basis_for(basis: str | MeasurementBasis) -> MeasurementBasis:
    """Resolve a basis name or existing basis object.

    Parameters
    ----------
    basis : str or MeasurementBasis
        Existing basis instance or a supported string.  Supported strings are
        ``"z"``, ``"computational"``, ``"x"``, and ``"y"`` after stripping and
        lower-casing.

    Returns
    -------
    MeasurementBasis
        The existing basis object or one of the built-in basis singletons.

    Raises
    ------
    TypeError
        If ``basis`` is neither ``str`` nor ``MeasurementBasis``.
    MeasurementError
        If a string basis name is unsupported.

    Examples
    --------
    >>> from simyuj.qstate.measure import basis_for
    >>> basis = basis_for("x")
    >>> basis.labels
    ('+', '-')
    >>> basis.matrix.shape
    (2, 2)
    """
    if isinstance(basis, MeasurementBasis):
        return basis
    if not isinstance(basis, str):
        raise TypeError("basis must be str or MeasurementBasis")
    normalized = basis.strip().lower()
    if normalized in {"z", "computational"}:
        return Z_BASIS
    if normalized == "x":
        return X_BASIS
    if normalized == "y":
        return Y_BASIS
    raise MeasurementError(f"unsupported measurement basis: {basis!r}")


__all__ = [
    "MeasurementBasis",
    "X_BASIS",
    "Y_BASIS",
    "Z_BASIS",
    "basis_for",
    "x_basis",
    "y_basis",
    "z_basis",
]
