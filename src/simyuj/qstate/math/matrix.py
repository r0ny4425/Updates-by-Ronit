from __future__ import annotations

"""Dense matrices for qubit gates and rotations.

This module exposes NumPy ``complex128`` matrices in computational-basis order.
For multi-qubit gates the first operand is the most significant basis bit and
the last operand is the least significant bit, matching the package's state
vector indexing convention.
"""

import numpy as np

from .const import COMPLEX_DTYPE, SQRT2

I2 = np.array([[1, 0], [0, 1]], dtype=COMPLEX_DTYPE)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)
H = np.array([[1, 1], [1, -1]], dtype=COMPLEX_DTYPE) / SQRT2
S = np.array([[1, 0], [0, 1j]], dtype=COMPLEX_DTYPE)
Sdg = np.array([[1, 0], [0, -1j]], dtype=COMPLEX_DTYPE)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=COMPLEX_DTYPE)
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=COMPLEX_DTYPE)

CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=COMPLEX_DTYPE,
)
CZ = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ],
    dtype=COMPLEX_DTYPE,
)
SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=COMPLEX_DTYPE,
)


def _check_num_controls(num_controls: object) -> int:
    if type(num_controls) is not int:
        raise TypeError("num_controls must be int")
    if num_controls <= 0:
        raise ValueError("num_controls must be positive")
    return num_controls


def controlled_x(num_controls: int) -> np.ndarray:
    """Construct a dense multi-controlled-X matrix.

    Parameters
    ----------
    num_controls : int
        Number of control qubits.  Must be a positive ``int``.

    Returns
    -------
    ndarray of complex, shape ``(2**(num_controls + 1), 2**(num_controls + 1))``
        Unitary matrix whose operands are ordered as controls first and target
        last: ``(control0, ..., controlN, target)``.

    Raises
    ------
    TypeError
        If ``num_controls`` is not exactly an ``int``.
    ValueError
        If ``num_controls`` is not positive.

    Notes
    -----
    The target is the least significant bit of each computational-basis index.
    The target bit flips only when all control bits are ``1``.
    """
    checked_controls = _check_num_controls(num_controls)
    arity = checked_controls + 1
    size = 2**arity

    op = np.zeros((size, size), dtype=COMPLEX_DTYPE)
    control_mask = ((1 << checked_controls) - 1) << 1

    for column in range(size):
        row = column
        if (column & control_mask) == control_mask:
            row = column ^ 1
        op[row, column] = 1.0

    return op


def controlled_z(num_controls: int) -> np.ndarray:
    """Construct a dense multi-controlled-Z matrix.

    Parameters
    ----------
    num_controls : int
        Number of control qubits.  Must be a positive ``int``.

    Returns
    -------
    ndarray of complex, shape ``(2**(num_controls + 1), 2**(num_controls + 1))``
        Diagonal unitary whose operands are ordered as controls first and target
        last: ``(control0, ..., controlN, target)``.

    Raises
    ------
    TypeError
        If ``num_controls`` is not exactly an ``int``.
    ValueError
        If ``num_controls`` is not positive.

    Notes
    -----
    The operation applies phase ``-1`` only to the all-ones basis state.
    """
    checked_controls = _check_num_controls(num_controls)
    arity = checked_controls + 1
    size = 2**arity

    op = np.eye(size, dtype=COMPLEX_DTYPE)
    op[-1, -1] = -1.0
    return op


def controlled_swap() -> np.ndarray:
    """Construct the Fredkin, or controlled-SWAP, matrix.

    Returns
    -------
    ndarray of complex, shape (8, 8)
        Three-qubit unitary with operands ordered as
        ``(control, swap_left, swap_right)``.

    Notes
    -----
    The control is the most significant basis bit.  When it is ``1``, the two
    lower-order swap operands are exchanged.
    """
    op = np.zeros((8, 8), dtype=COMPLEX_DTYPE)

    for column in range(8):
        row = column
        control = (column >> 2) & 1

        if control:
            left = (column >> 1) & 1
            right = column & 1
            row = (control << 2) | (right << 1) | left

        op[row, column] = 1.0

    return op


CCX = controlled_x(2)
CCZ = controlled_z(2)
CSWAP = controlled_swap()


def RX(theta: float) -> np.ndarray:
    """Return the single-qubit X-axis rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of complex, shape (2, 2)
        Matrix for :math:`e^{-i\\theta X / 2}`.
    """
    half = theta / 2.0
    return np.array(
        [
            [np.cos(half), -1j * np.sin(half)],
            [-1j * np.sin(half), np.cos(half)],
        ],
        dtype=COMPLEX_DTYPE,
    )


def RY(theta: float) -> np.ndarray:
    """Return the single-qubit Y-axis rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of complex, shape (2, 2)
        Matrix for :math:`e^{-i\\theta Y / 2}`.
    """
    half = theta / 2.0
    return np.array(
        [
            [np.cos(half), -np.sin(half)],
            [np.sin(half), np.cos(half)],
        ],
        dtype=COMPLEX_DTYPE,
    )


def RZ(theta: float) -> np.ndarray:
    """Return the single-qubit Z-axis rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of complex, shape (2, 2)
        Diagonal matrix for :math:`e^{-i\\theta Z / 2}`.
    """
    half = theta / 2.0
    return np.array(
        [
            [np.exp(-1j * half), 0],
            [0, np.exp(1j * half)],
        ],
        dtype=COMPLEX_DTYPE,
    )


def Phase(phi: float) -> np.ndarray:
    """Return the single-qubit phase gate matrix.

    Parameters
    ----------
    phi : float
        Phase angle in radians applied to :math:`|1\\rangle`.

    Returns
    -------
    ndarray of complex, shape (2, 2)
        Diagonal matrix :math:`\\operatorname{diag}(1, e^{i\\phi})`.
    """
    return np.array(
        [
            [1, 0],
            [0, np.exp(1j * phi)],
        ],
        dtype=COMPLEX_DTYPE,
    )


def _controlled_one(target: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=COMPLEX_DTYPE)
    matrix[2:4, 2:4] = target
    return matrix


def CPhase(phi: float) -> np.ndarray:
    """Return the controlled phase gate matrix.

    Parameters
    ----------
    phi : float
        Phase angle in radians applied to :math:`|11\\rangle`.

    Returns
    -------
    ndarray of complex, shape (4, 4)
        Controlled form of :func:`Phase`, with operand order
        ``(control, target)``.
    """
    return _controlled_one(Phase(phi))


def CRX(theta: float) -> np.ndarray:
    """Return the controlled X-axis rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of complex, shape (4, 4)
        Controlled form of :func:`RX`, with operand order
        ``(control, target)``.
    """
    return _controlled_one(RX(theta))


def CRY(theta: float) -> np.ndarray:
    """Return the controlled Y-axis rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of complex, shape (4, 4)
        Controlled form of :func:`RY`, with operand order
        ``(control, target)``.
    """
    return _controlled_one(RY(theta))


def CRZ(theta: float) -> np.ndarray:
    """Return the controlled Z-axis rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of complex, shape (4, 4)
        Controlled form of :func:`RZ`, with operand order
        ``(control, target)``.
    """
    return _controlled_one(RZ(theta))


__all__ = [
    "I2",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "CNOT",
    "CZ",
    "SWAP",
    "CCX",
    "CCZ",
    "CSWAP",
    "RX",
    "RY",
    "RZ",
    "Phase",
    "CPhase",
    "CRX",
    "CRY",
    "CRZ",
    "controlled_swap",
    "controlled_x",
    "controlled_z",
]
