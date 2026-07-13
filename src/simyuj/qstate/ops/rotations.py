from __future__ import annotations

"""Parameterized rotation and phase-gate constructors.

Each function converts its angle with ``float(...)``, builds the corresponding
dense matrix from ``qstate.math.matrix``, and wraps it in a ``Unitary``.
Controlled gates use operand order ``(control, target)``.
"""

from ..math import matrix as _matrix
from .unitary import Unitary


def RX(theta: float) -> Unitary:
    """Return a single-qubit X-axis rotation operation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.  The value is converted with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"RX"`` with arity one.

    Raises
    ------
    TypeError
        If ``theta`` cannot be converted to ``float``.
    ValueError
        If ``theta`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.RX(float(theta)), name="RX", arity=1)


def RY(theta: float) -> Unitary:
    """Return a single-qubit Y-axis rotation operation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.  The value is converted with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"RY"`` with arity one.

    Raises
    ------
    TypeError
        If ``theta`` cannot be converted to ``float``.
    ValueError
        If ``theta`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.RY(float(theta)), name="RY", arity=1)


def RZ(theta: float) -> Unitary:
    """Return a single-qubit Z-axis rotation operation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.  The value is converted with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"RZ"`` with arity one.

    Raises
    ------
    TypeError
        If ``theta`` cannot be converted to ``float``.
    ValueError
        If ``theta`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.RZ(float(theta)), name="RZ", arity=1)


def Phase(phi: float) -> Unitary:
    """Return a single-qubit phase operation.

    Parameters
    ----------
    phi : float
        Phase angle in radians applied to the :math:`|1\\rangle` amplitude.
        The value is converted with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"Phase"`` with arity one.

    Raises
    ------
    TypeError
        If ``phi`` cannot be converted to ``float``.
    ValueError
        If ``phi`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.Phase(float(phi)), name="Phase", arity=1)


def CPhase(phi: float) -> Unitary:
    """Return a controlled phase operation.

    Parameters
    ----------
    phi : float
        Phase angle in radians applied to the :math:`|11\\rangle` amplitude.
        The value is converted with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"CPhase"`` with arity two and operand order
        ``(control, target)``.

    Raises
    ------
    TypeError
        If ``phi`` cannot be converted to ``float``.
    ValueError
        If ``phi`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.CPhase(float(phi)), name="CPhase", arity=2)


def CRX(theta: float) -> Unitary:
    """Return a controlled X-axis rotation operation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians for the target qubit.  The value is converted
        with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"CRX"`` with arity two and operand order
        ``(control, target)``.

    Raises
    ------
    TypeError
        If ``theta`` cannot be converted to ``float``.
    ValueError
        If ``theta`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.CRX(float(theta)), name="CRX", arity=2)


def CRY(theta: float) -> Unitary:
    """Return a controlled Y-axis rotation operation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians for the target qubit.  The value is converted
        with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"CRY"`` with arity two and operand order
        ``(control, target)``.

    Raises
    ------
    TypeError
        If ``theta`` cannot be converted to ``float``.
    ValueError
        If ``theta`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.CRY(float(theta)), name="CRY", arity=2)


def CRZ(theta: float) -> Unitary:
    """Return a controlled Z-axis rotation operation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians for the target qubit.  The value is converted
        with ``float``.

    Returns
    -------
    Unitary
        Operation named ``"CRZ"`` with arity two and operand order
        ``(control, target)``.

    Raises
    ------
    TypeError
        If ``theta`` cannot be converted to ``float``.
    ValueError
        If ``theta`` conversion fails.
    InvalidOperationError
        If a non-finite angle produces a matrix that fails unitary validation.
    """
    return Unitary(_matrix.CRZ(float(theta)), name="CRZ", arity=2)


__all__ = ["RX", "RY", "RZ", "Phase", "CPhase", "CRX", "CRY", "CRZ"]
