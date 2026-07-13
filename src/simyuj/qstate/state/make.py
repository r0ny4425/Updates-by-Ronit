from __future__ import annotations

"""Convenience constructors for normalized ket payloads."""

from collections.abc import Sequence

import numpy as np

from ..errors import InvalidStateError
from ..math.const import SQRT2
from .ket import KetState


def zero() -> KetState:
    """Return the one-qubit computational :math:`|0\\rangle` state."""
    return KetState(np.array([1.0, 0.0], dtype=np.complex128))


def one() -> KetState:
    """Return the one-qubit computational :math:`|1\\rangle` state."""
    return KetState(np.array([0.0, 1.0], dtype=np.complex128))


def plus() -> KetState:
    """Return the one-qubit X-basis :math:`|+\\rangle` state."""
    return KetState(np.array([1.0, 1.0], dtype=np.complex128) / SQRT2)


def minus() -> KetState:
    """Return the one-qubit X-basis :math:`|-\\rangle` state."""
    return KetState(np.array([1.0, -1.0], dtype=np.complex128) / SQRT2)


def plus_i() -> KetState:
    """Return the one-qubit Y-basis :math:`|+i\\rangle` state."""
    return KetState(np.array([1.0, 1j], dtype=np.complex128) / SQRT2)


def minus_i() -> KetState:
    """Return the one-qubit Y-basis :math:`|-i\\rangle` state."""
    return KetState(np.array([1.0, -1j], dtype=np.complex128) / SQRT2)


def bell(label: object = "phi+") -> KetState:
    """Return a two-qubit Bell ket.

    Parameters
    ----------
    label : object, default="phi+"
        Bell label accepted by ``measure.bell.bell_vector``.

    Returns
    -------
    KetState
        Bell-state ket in computational-basis order.

    Examples
    --------
    >>> from simyuj.qstate.state import bell
    >>> bell("psi-").num_qubits
    2
    """
    from ..measure.bell import bell_vector

    return KetState(bell_vector(label))


def basis(bits: str | Sequence[int]) -> KetState:
    """Return a computational-basis ket from bits.

    Parameters
    ----------
    bits : str or sequence of int
        Non-empty bit string or sequence.  Strings may optionally use ``|...>``
        wrapping.  Axis ``0`` is the first bit and contributes the most
        significant basis index.

    Returns
    -------
    KetState
        Basis vector with amplitude one at the parsed bit index.

    Raises
    ------
    TypeError
        If ``bits`` is not a string or sequence.
    InvalidStateError
        If the bit sequence is empty or contains values other than integer
        ``0`` and ``1``.

    Examples
    --------
    >>> from simyuj.qstate.state import basis
    >>> state = basis("10")
    >>> state.num_qubits
    2
    >>> int(state.vector.argmax())
    2
    """
    checked = _parse_bits(bits)
    index = 0
    for bit in checked:
        index = (index << 1) | bit

    vector = np.zeros(2 ** len(checked), dtype=np.complex128)
    vector[index] = 1.0
    return KetState(vector)


def ghz(num_qubits: int) -> KetState:
    """Return an ``n``-qubit GHZ ket.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.  Must be exactly an ``int`` and at least three.

    Returns
    -------
    KetState
        State :math:`(|00\\ldots0\\rangle + |11\\ldots1\\rangle) / \\sqrt{2}`.

    Raises
    ------
    TypeError
        If ``num_qubits`` is not exactly an ``int``.
    InvalidStateError
        If ``num_qubits`` is less than three.

    Examples
    --------
    >>> from simyuj.qstate.state import ghz
    >>> ghz(3).vector.shape
    (8,)
    """
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be int")
    if num_qubits < 3:
        raise InvalidStateError("GHZ state requires at least three qubits")

    vector = np.zeros(2**num_qubits, dtype=np.complex128)
    vector[0] = 1.0 / SQRT2
    vector[-1] = 1.0 / SQRT2
    return KetState(vector)


_KET_FACTORIES = {
    "0": zero,
    "|0>": zero,
    "zero": zero,
    "1": one,
    "|1>": one,
    "one": one,
    "+": plus,
    "|+>": plus,
    "plus": plus,
    "-": minus,
    "|->": minus,
    "minus": minus,
    "+i": plus_i,
    "|+i>": plus_i,
    "plus_i": plus_i,
    "plus-i": plus_i,
    "-i": minus_i,
    "|-i>": minus_i,
    "minus_i": minus_i,
    "minus-i": minus_i,
}


def make_ket(state: object = "|0>") -> KetState:
    """Coerce a value into a ``KetState``.

    Parameters
    ----------
    state : object, default="|0>"
        Existing ``KetState``, supported named one-qubit state string, Bell
        label string, binary basis string, or normalized vector sequence.

    Returns
    -------
    KetState
        Ket payload.

    Raises
    ------
    TypeError
        If ``state`` is not a ``KetState``, string, sequence, or NumPy array.
    InvalidStateError
        If a string alias is unsupported or the vector cannot construct a valid
        ``KetState``.

    Examples
    --------
    >>> from simyuj.qstate.state import make_ket
    >>> make_ket("|+>").num_qubits
    1
    >>> make_ket("phi+").num_qubits
    2
    """
    if isinstance(state, KetState):
        return state
    if isinstance(state, str):
        key = state.strip().lower()
        factory = _KET_FACTORIES.get(key)
        if factory is not None:
            return factory()
        if _looks_bell_label(key):
            return bell(key)
        if _looks_binary_basis(key):
            return basis(key)
        raise InvalidStateError(f"unsupported ket state: {state!r}")
    if isinstance(state, Sequence) or isinstance(state, np.ndarray):
        return KetState(state)
    raise TypeError("state must be KetState, str, or sequence")


def _parse_bits(bits: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(bits, str):
        value = bits.strip()
        if value.startswith("|") and value.endswith(">"):
            value = value[1:-1]
        if not value:
            raise InvalidStateError("basis bits must be non-empty")
        parsed: list[int] = []
        for bit in value:
            if bit not in {"0", "1"}:
                raise InvalidStateError("basis bits must be 0 or 1")
            parsed.append(int(bit))
        return tuple(parsed)

    if not isinstance(bits, Sequence):
        raise TypeError("bits must be str or sequence")
    if not bits:
        raise InvalidStateError("basis bits must be non-empty")
    parsed = []
    for raw_bit in bits:
        if type(raw_bit) is not int or raw_bit not in {0, 1}:
            raise InvalidStateError("basis bits must be 0 or 1")
        parsed.append(raw_bit)
    return tuple(parsed)


def _looks_binary_basis(value: str) -> bool:
    if value.startswith("|") and value.endswith(">"):
        value = value[1:-1]
    return bool(value) and all(bit in {"0", "1"} for bit in value)


def _looks_bell_label(value: str) -> bool:
    from .bell_diag import normalize_bell_label

    try:
        normalize_bell_label(value)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "bell",
    "basis",
    "ghz",
    "make_ket",
    "minus",
    "minus_i",
    "one",
    "plus",
    "plus_i",
    "zero",
]
