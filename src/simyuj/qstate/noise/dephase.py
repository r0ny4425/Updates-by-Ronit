from __future__ import annotations

"""Dephasing-channel constructors.

The current one-qubit dephasing helpers are aliases for phase-flip Pauli noise.
The common-mode helper constructs a correlated two-qubit ``ZZ`` Pauli channel.
"""

from .pauli import phase_flip, two_qubit_pauli_channel


def dephasing(p: object):
    """Return the dephasing alias to phase-flip noise.

    Parameters
    ----------
    p : object
        Probability passed to ``phase_flip``.

    Returns
    -------
    KrausChannel
        One-qubit phase-flip channel.

    Raises
    ------
    TypeError
        If ``p`` fails scalar probability type checks.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.
    """
    return phase_flip(p)


def phase_damping(p: object):
    """Return a phase-flip alias, not a separate damping model.

    Parameters
    ----------
    p : object
        Probability passed to ``phase_flip``.

    Returns
    -------
    KrausChannel
        One-qubit phase-flip channel.

    Raises
    ------
    TypeError
        If ``p`` fails scalar probability type checks.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.
    """
    return phase_flip(p)


def common_mode_dephasing(p: object):
    """Return correlated two-qubit phase-flip dephasing.

    Parameters
    ----------
    p : object
        Probability of applying the correlated ``ZZ`` operator.

    Returns
    -------
    KrausChannel
        Two-qubit Pauli channel with an implicit identity branch and one
        explicit ``ZZ`` branch.

    Raises
    ------
    TypeError
        If ``p`` fails scalar probability type checks.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.

    Notes
    -----
    This Pauli approximation implements
    :math:`\\rho \\mapsto (1 - p)\\rho + p (Z \\otimes Z)\\rho(Z \\otimes Z)`.

    Target order follows ``two_qubit_pauli_channel``.  Since the only
    non-identity operator is ``"ZZ"``, both targets receive the same phase
    flip.
    """
    from ..check import check_probability

    p = check_probability(p, name="p")
    return two_qubit_pauli_channel({"ZZ": p})


__all__ = ["common_mode_dephasing", "dephasing", "phase_damping"]
