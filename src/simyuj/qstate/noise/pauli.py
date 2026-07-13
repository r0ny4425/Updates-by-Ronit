from __future__ import annotations

"""Pauli-noise channel constructors.

Single-qubit helpers build Kraus operators from ``I``, ``X``, ``Y``, and ``Z``.
The two-qubit constructor accepts Pauli labels in target order, for example
``"IX"`` means identity on the first target and ``X`` on the second target.
"""

from collections.abc import Mapping

import numpy as np

from ..check import check_probability
from ..math import matrix as mat
from .base import KrausChannel

_PAULI_MATRICES = {
    "I": mat.I2,
    "X": mat.X,
    "Y": mat.Y,
    "Z": mat.Z,
}

_TWO_QUBIT_PAULI_ORDER = tuple(
    left + right
    for left in ("I", "X", "Y", "Z")
    for right in ("I", "X", "Y", "Z")
    if left + right != "II"
)


def bit_flip(p: object) -> KrausChannel:
    """Return a single-qubit bit-flip channel.

    Parameters
    ----------
    p : object
        Probability of applying ``X``.  Checked by ``check_probability``.

    Returns
    -------
    KrausChannel
        One-qubit channel with operators :math:`\\sqrt{1 - p}I` and
        :math:`\\sqrt{p}X`.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by ``check_probability``.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.
    """
    p = check_probability(p, name="p")
    return KrausChannel(
        (
            np.sqrt(1.0 - p) * mat.I2,
            np.sqrt(p) * mat.X,
        ),
        name="bit_flip",
        arity=1,
    )


def phase_flip(p: object) -> KrausChannel:
    """Return a single-qubit phase-flip channel.

    Parameters
    ----------
    p : object
        Probability of applying ``Z``.  Checked by ``check_probability``.

    Returns
    -------
    KrausChannel
        One-qubit channel with operators :math:`\\sqrt{1 - p}I` and
        :math:`\\sqrt{p}Z`.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by ``check_probability``.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.
    """
    p = check_probability(p, name="p")
    return KrausChannel(
        (
            np.sqrt(1.0 - p) * mat.I2,
            np.sqrt(p) * mat.Z,
        ),
        name="phase_flip",
        arity=1,
    )


def bit_phase_flip(p: object) -> KrausChannel:
    """Return a single-qubit bit-and-phase-flip channel.

    Parameters
    ----------
    p : object
        Probability of applying ``Y``.  Checked by ``check_probability``.

    Returns
    -------
    KrausChannel
        One-qubit channel with operators :math:`\\sqrt{1 - p}I` and
        :math:`\\sqrt{p}Y`.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by ``check_probability``.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.
    """
    p = check_probability(p, name="p")
    return KrausChannel(
        (
            np.sqrt(1.0 - p) * mat.I2,
            np.sqrt(p) * mat.Y,
        ),
        name="bit_phase_flip",
        arity=1,
    )


def two_qubit_pauli_channel(probabilities: object) -> KrausChannel:
    """Return a correlated two-qubit Pauli noise channel.

    Parameters
    ----------
    probabilities : object
        Mapping from two-character Pauli labels to probabilities.  Labels are
        stripped and upper-cased.  ``"II"`` is not accepted because its
        probability is implicit.

    Returns
    -------
    KrausChannel
        Two-qubit Pauli channel.  The first operator is the implicit identity
        branch with probability ``1 - sum(probabilities.values())``.

    Raises
    ------
    TypeError
        If ``probabilities`` is not a mapping, any label is not a string, or any
        probability fails scalar probability type checks.
    ValueError
        If a label is unsupported, ``"II"`` is supplied, a normalized label is
        duplicated, a probability is invalid, or explicit probabilities sum to
        more than one.

    Notes
    -----
    Probability labels follow target order.  For example, ``"IX"`` applies
    identity to the first target and ``X`` to the second target.
    """
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping")

    checked: dict[str, float] = {}
    total = 0.0
    for raw_label, raw_probability in probabilities.items():
        if not isinstance(raw_label, str):
            raise TypeError("Pauli label must be str")

        label = raw_label.strip().upper()
        if len(label) != 2 or any(symbol not in _PAULI_MATRICES for symbol in label):
            raise ValueError(f"unsupported two-qubit Pauli label: {raw_label!r}")
        if label == "II":
            raise ValueError("II probability is implicit")
        if label in checked:
            raise ValueError(f"duplicate Pauli label: {label}")

        probability = check_probability(raw_probability, name=f"p[{label}]")
        checked[label] = probability
        total += probability

    if total > 1.0:
        raise ValueError("Pauli probabilities must sum to at most 1")

    ops = [np.sqrt(1.0 - total) * np.kron(mat.I2, mat.I2)]
    for label in _TWO_QUBIT_PAULI_ORDER:
        probability = checked.get(label, 0.0)
        if probability == 0.0:
            continue

        pauli = np.kron(_PAULI_MATRICES[label[0]], _PAULI_MATRICES[label[1]])
        ops.append(np.sqrt(probability) * pauli)

    return KrausChannel(tuple(ops), name="two_qubit_pauli", arity=2)


def pauli_channel(px: object, py: object, pz: object) -> KrausChannel:
    """Return a single-qubit Pauli noise channel.

    This implements the single-qubit Pauli channel:

    .. math::

       E(\\rho) =
       (1 - p_x - p_y - p_z)\\rho
       + p_x X\\rho X
       + p_y Y\\rho Y
       + p_z Z\\rho Z

    where ``px``, ``py``, and ``pz`` are the probabilities of applying
    Pauli ``X``, ``Y``, and ``Z`` errors, respectively. The identity branch
    has probability ``1 - px - py - pz``.

    Parameters
    ----------
    px, py, pz : object
        Probabilities of applying the Pauli ``X``, ``Y``, and ``Z`` errors.
        Each value is validated using ``check_probability`` and must represent
        a scalar probability in the interval ``[0, 1]``.

    Returns
    -------
    KrausChannel
        A one-qubit Kraus representation of the Pauli channel with Kraus
        operators:

        .. math::

           \\sqrt{1 - p_x - p_y - p_z}I,\\quad
           \\sqrt{p_x}X,\\quad
           \\sqrt{p_y}Y,\\quad
           \\sqrt{p_z}Z

    Raises
    ------
    TypeError
        If any of ``px``, ``py``, or ``pz`` is not a scalar probability
        accepted by ``check_probability``.
    ValueError
        If any probability lies outside ``[0, 1]``, or if
        ``px + py + pz > 1``.

    Notes
    -----
    This is a general Pauli channel. Special cases include:

    * Bit-flip channel: ``pauli_channel(p, 0, 0)``.
    * Bit-phase-flip channel: ``pauli_channel(0, p, 0)``.
    * Phase-flip channel: ``pauli_channel(0, 0, p)``.
    * Depolarizing channel in the mixing-probability convention:
      ``pauli_channel(p / 4, p / 4, p / 4)``.

    Examples
    --------
    >>> from simyuj.qstate.noise import pauli_channel
    >>> channel = pauli_channel(0.1, 0.2, 0.3)
    >>> channel.arity
    1
    """
    px = check_probability(px, name="px")
    py = check_probability(py, name="py")
    pz = check_probability(pz, name="pz")
    total = px + py + pz
    if total > 1.0:
        raise ValueError("Pauli probabilities must sum to at most 1")
    return KrausChannel(
        (
            np.sqrt(1.0 - total) * mat.I2,
            np.sqrt(px) * mat.X,
            np.sqrt(py) * mat.Y,
            np.sqrt(pz) * mat.Z,
        ),
        name="pauli",
        arity=1,
    )


__all__ = [
    "bit_flip",
    "bit_phase_flip",
    "pauli_channel",
    "phase_flip",
    "two_qubit_pauli_channel",
]
