from __future__ import annotations

"""Prebuilt dense qubit gates and multi-controlled gate factories.

Gate constants wrap matrices from ``qstate.math.matrix`` in ``Unitary``
records.  Multi-qubit gates use computational-basis ordering with controls
before targets; the first operand is the most significant basis bit.
"""

from ..math import matrix as _matrix
from .unitary import Unitary

I = Unitary(_matrix.I2, name="I", arity=1)  # noqa: E741
X = Unitary(_matrix.X, name="X", arity=1)
Y = Unitary(_matrix.Y, name="Y", arity=1)
Z = Unitary(_matrix.Z, name="Z", arity=1)
H = Unitary(_matrix.H, name="H", arity=1)
S = Unitary(_matrix.S, name="S", arity=1)
Sdg = Unitary(_matrix.Sdg, name="Sdg", arity=1)
T = Unitary(_matrix.T, name="T", arity=1)
Tdg = Unitary(_matrix.Tdg, name="Tdg", arity=1)
CNOT = Unitary(_matrix.CNOT, name="CNOT", arity=2)
CZ = Unitary(_matrix.CZ, name="CZ", arity=2)
SWAP = Unitary(_matrix.SWAP, name="SWAP", arity=2)
CCX = Unitary(_matrix.CCX, name="CCX", arity=3)
TOFFOLI = Unitary(_matrix.CCX, name="TOFFOLI", arity=3)
CCZ = Unitary(_matrix.CCZ, name="CCZ", arity=3)
CSWAP = Unitary(_matrix.CSWAP, name="CSWAP", arity=3)
FREDKIN = Unitary(_matrix.CSWAP, name="FREDKIN", arity=3)


def MCX(num_controls: int) -> Unitary:
    """Construct a multi-controlled-X unitary.

    Parameters
    ----------
    num_controls : int
        Positive number of control qubits.  The target operand is appended
        after these controls.

    Returns
    -------
    Unitary
        Dense unitary with arity ``num_controls + 1``.  The generated name is
        ``"CNOT"`` for one control, ``"CCX"`` for two controls, and
        ``"C{n}X"`` otherwise.

    Raises
    ------
    TypeError
        If ``num_controls`` is not exactly an ``int``.
    ValueError
        If ``num_controls`` is not positive.

    Notes
    -----
    The target is the least significant operand bit and flips only when every
    control bit is ``1``.
    """
    matrix = _matrix.controlled_x(num_controls)
    arity = int(matrix.shape[0]).bit_length() - 1
    control_count = arity - 1

    if control_count == 1:
        name = "CNOT"
    elif control_count == 2:
        name = "CCX"
    else:
        name = f"C{control_count}X"

    return Unitary(matrix, name=name, arity=arity)


def MCZ(num_controls: int) -> Unitary:
    """Construct a multi-controlled-Z unitary.

    Parameters
    ----------
    num_controls : int
        Positive number of control qubits.  The target operand is appended
        after these controls.

    Returns
    -------
    Unitary
        Dense unitary with arity ``num_controls + 1``.  The generated name is
        ``"CZ"`` for one control, ``"CCZ"`` for two controls, and
        ``"C{n}Z"`` otherwise.

    Raises
    ------
    TypeError
        If ``num_controls`` is not exactly an ``int``.
    ValueError
        If ``num_controls`` is not positive.

    Notes
    -----
    The operation applies a phase of ``-1`` only to the all-ones basis state.
    """
    matrix = _matrix.controlled_z(num_controls)
    arity = int(matrix.shape[0]).bit_length() - 1
    control_count = arity - 1

    if control_count == 1:
        name = "CZ"
    elif control_count == 2:
        name = "CCZ"
    else:
        name = f"C{control_count}Z"

    return Unitary(matrix, name=name, arity=arity)


__all__ = [
    "I",
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
    "TOFFOLI",
    "CCZ",
    "CSWAP",
    "FREDKIN",
    "MCX",
    "MCZ",
]
