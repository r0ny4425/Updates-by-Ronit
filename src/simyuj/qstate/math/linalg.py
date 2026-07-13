from __future__ import annotations

"""Dense linear-algebra helpers for quantum-state payloads.

The functions in this module use NumPy ``complex128`` arrays and package-level
absolute and relative tolerances.  Validation is deliberately narrow: predicate
helpers return booleans for shape/property checks, while normalizers validate
only the conditions needed before rescaling.
"""

import numpy as np

from .const import ATOL, COMPLEX_DTYPE, RTOL


def as_complex_array(value: object) -> np.ndarray:
    """Return ``value`` as a ``complex128`` NumPy array.

    Parameters
    ----------
    value : object
        Array-like input accepted by ``numpy.asarray``.

    Returns
    -------
    ndarray
        View or array with dtype ``complex128``.  A copy is made only when NumPy
        needs one for dtype conversion or array construction.
    """
    return np.asarray(value, dtype=COMPLEX_DTYPE)


def dagger(matrix: object) -> np.ndarray:
    """Return the conjugate transpose of an array-like matrix.

    Parameters
    ----------
    matrix : object
        Matrix-like input.  The implementation coerces it to ``complex128`` but
        does not validate that it is two-dimensional.

    Returns
    -------
    ndarray
        ``conjugate(as_complex_array(matrix).T)``.
    """
    array = as_complex_array(matrix)
    return np.conjugate(array.T)


def trace(matrix: object) -> complex:
    """Return the NumPy trace of a complex array as a Python ``complex``.

    Parameters
    ----------
    matrix : object
        Matrix-like input accepted by ``numpy.trace`` after ``complex128``
        coercion.

    Returns
    -------
    complex
        Trace of the input.  No square-matrix validation is performed.
    """
    return complex(np.trace(as_complex_array(matrix)))


def is_square_matrix(matrix: object) -> bool:
    """Return whether ``matrix`` is a two-dimensional square array.

    Parameters
    ----------
    matrix : object
        Array-like input coerced to ``complex128`` before checking shape.

    Returns
    -------
    bool
        ``True`` only for arrays with shape ``(d, d)``.
    """
    array = as_complex_array(matrix)
    return array.ndim == 2 and array.shape[0] == array.shape[1]


def is_hermitian(matrix: object, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    """Return whether a matrix is Hermitian within tolerance.

    Parameters
    ----------
    matrix : object
        Array-like candidate matrix.
    atol : float, default=ATOL
        Absolute tolerance passed to ``numpy.allclose``.
    rtol : float, default=RTOL
        Relative tolerance passed to ``numpy.allclose``.

    Returns
    -------
    bool
        ``True`` when the input is square and close to its conjugate transpose.
    """
    array = as_complex_array(matrix)
    if not is_square_matrix(array):
        return False
    return bool(np.allclose(array, dagger(array), atol=atol, rtol=rtol))


def is_psd(matrix: object, *, atol: float = ATOL) -> bool:
    """Return whether a matrix is Hermitian positive semidefinite.

    Parameters
    ----------
    matrix : object
        Array-like candidate matrix.
    atol : float, default=ATOL
        Negative eigenvalue tolerance.  Eigenvalues greater than or equal to
        ``-atol`` are accepted.

    Returns
    -------
    bool
        ``True`` when the matrix is Hermitian and all Hermitian eigenvalues are
        non-negative up to ``atol``.

    Notes
    -----
    Hermiticity is checked with ``is_hermitian`` using its default tolerances.
    The eigensolver is then applied with ``numpy.linalg.eigvalsh``.
    """
    array = as_complex_array(matrix)
    if not is_hermitian(array):
        return False
    eigvals = np.linalg.eigvalsh(array)
    return bool(np.all(eigvals >= -atol))


def is_unitary(matrix: object, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    """Return whether a square matrix is unitary within tolerance.

    Parameters
    ----------
    matrix : object
        Array-like candidate matrix with expected shape ``(d, d)``.
    atol : float, default=ATOL
        Absolute tolerance passed to ``numpy.allclose``.
    rtol : float, default=RTOL
        Relative tolerance passed to ``numpy.allclose``.

    Returns
    -------
    bool
        ``True`` when ``matrix.conj().T @ matrix`` is close to ``I``.
    """
    array = as_complex_array(matrix)
    if not is_square_matrix(array):
        return False
    size = array.shape[0]
    return bool(np.allclose(dagger(array) @ array, np.eye(size), atol=atol, rtol=rtol))


def normalize_vector(vector: object) -> np.ndarray:
    """Return a normalized one-dimensional complex vector.

    Parameters
    ----------
    vector : object
        Array-like vector with shape ``(d,)``.

    Returns
    -------
    ndarray
        ``complex128`` vector divided by its Euclidean norm.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or its norm is zero or non-finite.
    """
    array = as_complex_array(vector)
    if array.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("vector norm must be positive and finite")
    return array / norm


def normalize_density(matrix: object) -> np.ndarray:
    """Return a density-like matrix rescaled to unit trace.

    Parameters
    ----------
    matrix : object
        Array-like square matrix with shape ``(d, d)``.

    Returns
    -------
    ndarray
        ``complex128`` matrix divided by its trace.

    Raises
    ------
    ValueError
        If the input is not square or has a zero, near-zero, or non-finite
        trace.

    Notes
    -----
    This function does not check Hermiticity, positive semidefiniteness, power-
    of-two dimension, or whether the trace is real before normalizing.
    """
    array = as_complex_array(matrix)
    if not is_square_matrix(array):
        raise ValueError("density matrix must be square")
    tr = trace(array)
    if abs(tr) <= ATOL or not np.isfinite(tr.real) or not np.isfinite(tr.imag):
        raise ValueError("density matrix trace must be finite and non-zero")
    return array / tr


def readonly(array: np.ndarray) -> np.ndarray:
    """Return a read-only ``complex128`` copy of an array.

    Parameters
    ----------
    array : ndarray
        Input array to copy.

    Returns
    -------
    ndarray
        Independent ``complex128`` copy with ``writeable`` set to ``False``.
    """
    copy = np.array(array, dtype=COMPLEX_DTYPE, copy=True)
    copy.setflags(write=False)
    return copy


__all__ = [
    "as_complex_array",
    "dagger",
    "trace",
    "is_square_matrix",
    "is_hermitian",
    "is_psd",
    "is_unitary",
    "normalize_density",
    "normalize_vector",
    "readonly",
]
