from __future__ import annotations

"""Classical Pauli-frame correction bookkeeping.

Pauli frames store pending ``X`` and ``Z`` correction bits separately from the
quantum state.  Bell-label corrections use the Bell outcome bit convention from
``state.bell_diag.label_to_bits``: ``phi+ -> (0, 0)``, ``phi- -> (0, 1)``,
``psi+ -> (1, 0)``, and ``psi- -> (1, 1)``.
"""

from dataclasses import dataclass

from ..state.bell_diag import label_to_bits, normalize_bell_label


@dataclass(frozen=True, slots=True, init=False)
class PauliFrame:
    """Immutable classical Pauli-frame correction bits.

    Parameters
    ----------
    x : object
        Tuple of integer bits marking pending ``X`` corrections.
    z : object
        Tuple of integer bits marking pending ``Z`` corrections.

    Attributes
    ----------
    x, z : tuple of int
        Read-only tuple fields with equal length.  Entries must be exactly
        integer ``0`` or ``1``; boolean values are rejected by the bit check.

    Raises
    ------
    TypeError
        If either bit collection is not a tuple.
    ValueError
        If any bit is not ``0`` or ``1``, or if ``x`` and ``z`` have different
        lengths.
    """

    x: tuple[int, ...]
    z: tuple[int, ...]

    def __init__(self, x: object, z: object) -> None:
        """Validate and store Pauli-frame bit tuples."""
        checked_x = _check_bits(x, name="x")
        checked_z = _check_bits(z, name="z")
        if len(checked_x) != len(checked_z):
            raise ValueError("x and z must have the same length")

        object.__setattr__(self, "x", checked_x)
        object.__setattr__(self, "z", checked_z)

    @property
    def size(self) -> int:
        """Number of qubits represented by the frame."""
        return len(self.x)

    def compose(self, other: PauliFrame) -> PauliFrame:
        """XOR this frame with another frame of the same size.

        Parameters
        ----------
        other : PauliFrame
            Frame whose ``x`` and ``z`` bits are XORed with this frame.

        Returns
        -------
        PauliFrame
            New frame containing the bitwise composition.  The inputs are not
            mutated.

        Raises
        ------
        TypeError
            If ``other`` is not a ``PauliFrame``.
        ValueError
            If the two frames have different sizes.
        """
        if not isinstance(other, PauliFrame):
            raise TypeError("other must be PauliFrame")
        if self.size != other.size:
            raise ValueError("frames must have the same size")
        return PauliFrame(
            tuple(left ^ right for left, right in zip(self.x, other.x)),
            tuple(left ^ right for left, right in zip(self.z, other.z)),
        )

    def with_correction(self, qubit: int, *, x: int = 0, z: int = 0) -> PauliFrame:
        """Return a new frame with one qubit correction composed in.

        Parameters
        ----------
        qubit : int
            Zero-based frame index to update.
        x : int, default=0
            ``X`` correction bit to XOR into ``qubit``.
        z : int, default=0
            ``Z`` correction bit to XOR into ``qubit``.

        Returns
        -------
        PauliFrame
            New frame after composing a single-qubit correction frame.

        Raises
        ------
        TypeError
            If ``qubit`` is not exactly an ``int``.
        IndexError
            If ``qubit`` is outside ``[0, size)``.
        ValueError
            If ``x`` or ``z`` is not exactly integer ``0`` or ``1``.
        """
        checked_qubit = _check_qubit_index(qubit, self.size)
        checked_x = _check_bit(x, name="x")
        checked_z = _check_bit(z, name="z")

        x_bits = [0] * self.size
        z_bits = [0] * self.size
        x_bits[checked_qubit] = checked_x
        z_bits[checked_qubit] = checked_z
        return self.compose(PauliFrame(tuple(x_bits), tuple(z_bits)))


def identity_frame(size: int) -> PauliFrame:
    """Return a zero-correction Pauli frame.

    Parameters
    ----------
    size : int
        Number of qubits in the frame.  Zero is accepted and produces an empty
        frame.

    Returns
    -------
    PauliFrame
        Frame with all ``x`` and ``z`` bits set to ``0``.

    Raises
    ------
    TypeError
        If ``size`` is not exactly an ``int``.
    ValueError
        If ``size`` is negative.
    """
    if type(size) is not int:
        raise TypeError("size must be int")
    if size < 0:
        raise ValueError("size must be non-negative")
    return PauliFrame((0,) * size, (0,) * size)


def correction_for_bell(label: object) -> tuple[int, int]:
    """Return Pauli correction bits for a Bell outcome label.

    Parameters
    ----------
    label : object
        Bell label or alias accepted by ``normalize_bell_label``.

    Returns
    -------
    tuple of int
        ``(x, z)`` correction bits for the normalized label.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If ``label`` is empty or unsupported.
    """
    normalized = normalize_bell_label(label)
    return label_to_bits(normalized)


def update_after_bsm(
    frame: PauliFrame,
    label: object,
    *,
    qubit: int = 0,
) -> PauliFrame:
    """Apply a Bell-state measurement correction to a Pauli frame.

    Parameters
    ----------
    frame : PauliFrame
        Existing correction frame.
    label : object
        Bell outcome label whose bits are mapped to ``(x, z)``.
    qubit : int, default=0
        Frame index that receives the correction.

    Returns
    -------
    PauliFrame
        New frame with the Bell correction composed onto ``qubit``.

    Raises
    ------
    TypeError
        If ``frame`` is not a ``PauliFrame`` or if delegated label/qubit checks
        fail.
    ValueError
        If ``label`` is unsupported.
    IndexError
        If ``qubit`` is outside the frame.
    """
    if not isinstance(frame, PauliFrame):
        raise TypeError("frame must be PauliFrame")
    x, z = correction_for_bell(label)
    return frame.with_correction(qubit, x=x, z=z)


def update_after_teleport(
    frame: PauliFrame,
    label: object,
    *,
    qubit: int = 0,
) -> PauliFrame:
    """Apply a teleportation Bell-outcome correction to a frame.

    This is currently the same update rule as ``update_after_bsm``.

    Parameters
    ----------
    frame : PauliFrame
        Existing correction frame.
    label : object
        Bell outcome label whose bits are mapped to ``(x, z)``.
    qubit : int, default=0
        Frame index that receives the correction.

    Returns
    -------
    PauliFrame
        New frame with the correction composed onto ``qubit``.

    Raises
    ------
    TypeError
        If ``frame`` is not a ``PauliFrame`` or if delegated label/qubit checks
        fail.
    ValueError
        If ``label`` is unsupported.
    IndexError
        If ``qubit`` is outside the frame.
    """
    return update_after_bsm(frame, label, qubit=qubit)


def correction_for_entanglement_swap(
    left_label: object,
    right_label: object,
    bsm_label: object,
) -> tuple[int, int]:
    """Return correction bits for a Bell-pair entanglement swap.

    Parameters
    ----------
    left_label, right_label : object
        Bell labels describing the two input Bell pairs.
    bsm_label : object
        Bell-state measurement outcome for the two relay qubits.

    Returns
    -------
    tuple of int
        ``(x, z)`` bits that map the remaining endpoint pair back to
        ``phi+`` when applied to one endpoint qubit.

    Raises
    ------
    TypeError
        If any label is not a string.
    ValueError
        If any label is empty or unsupported.
    """
    left_x, left_z = correction_for_bell(left_label)
    right_x, right_z = correction_for_bell(right_label)
    bsm_x, bsm_z = correction_for_bell(bsm_label)
    return left_x ^ right_x ^ bsm_x, left_z ^ right_z ^ bsm_z


def update_after_swap(left: object, right: object) -> tuple[int, int]:
    """Combine two Bell-swap outcome corrections.

    Parameters
    ----------
    left, right : object
        Bell labels or aliases accepted by ``normalize_bell_label``.

    Returns
    -------
    tuple of int
        XOR of the two ``(x, z)`` Bell correction pairs.

    Raises
    ------
    TypeError
        If either label is not a string.
    ValueError
        If either label is empty or unsupported.
    """
    left_x, left_z = correction_for_bell(left)
    right_x, right_z = correction_for_bell(right)
    return left_x ^ right_x, left_z ^ right_z


def _check_bits(bits: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(bits, tuple):
        raise TypeError(f"{name} must be tuple")
    return tuple(
        _check_bit(bit, name=f"{name}[{index}]") for index, bit in enumerate(bits)
    )


def _check_bit(bit: object, *, name: str) -> int:
    if type(bit) is not int or bit not in {0, 1}:
        raise ValueError(f"{name} must be 0 or 1")
    return bit


def _check_qubit_index(qubit: object, size: int) -> int:
    if type(qubit) is not int:
        raise TypeError("qubit must be int")
    if qubit < 0 or qubit >= size:
        raise IndexError("qubit index out of range")
    return qubit


__all__ = [
    "PauliFrame",
    "correction_for_entanglement_swap",
    "correction_for_bell",
    "identity_frame",
    "update_after_bsm",
    "update_after_swap",
    "update_after_teleport",
]
