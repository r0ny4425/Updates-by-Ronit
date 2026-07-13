from __future__ import annotations

"""Amplitude-damping channel constructors."""

import numpy as np

from ..check import check_probability
from .base import KrausChannel


def amplitude_damping(gamma: object) -> KrausChannel:
    """Return a single-qubit amplitude-damping channel.

    Parameters
    ----------
    gamma : object
        Excited-state decay probability.  Checked by ``check_probability``.

    Returns
    -------
    KrausChannel
        One-qubit Kraus channel with two operators.  At ``gamma = 1``, the
        :math:`|1\\rangle` population fully decays to :math:`|0\\rangle`.

    Raises
    ------
    TypeError
        If ``gamma`` is not a scalar probability accepted by
        ``check_probability``.
    ValueError
        If ``gamma`` is outside ``[0, 1]`` beyond probability tolerance.

    Examples
    --------
    >>> from simyuj.qstate import QuantumStateManager, SubsystemId
    >>> from simyuj.qstate.noise import amplitude_damping
    >>> q0 = SubsystemId("q0")
    >>> manager = QuantumStateManager()
    >>> state_ref = manager.prepare("|1>", rep="density", subsystems=(q0,))
    >>> _ = manager.apply_noise(amplitude_damping(1.0), targets=(q0,))
    >>> float(manager.get(state_ref).rho[0, 0].real)
    1.0
    """
    gamma = check_probability(gamma, name="gamma")
    return KrausChannel(
        (
            np.array(
                [[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]],
                dtype=np.complex128,
            ),
            np.array(
                [[0.0, np.sqrt(gamma)], [0.0, 0.0]],
                dtype=np.complex128,
            ),
        ),
        name="amplitude_damping",
        arity=1,
    )


def generalized_amplitude_damping(
    gamma: object,
    prob: object = 1.0,
) -> KrausChannel:
    """Return a generalized amplitude-damping channel.

    Parameters
    ----------
    gamma : object
        Damping probability.  Checked by ``check_probability``.
    prob : object, default=1.0
        Stationary ground-state population.  Checked by ``check_probability``.

    Returns
    -------
    KrausChannel
        One-qubit Kraus channel with four operators.

    Raises
    ------
    TypeError
        If ``gamma`` or ``prob`` fails scalar probability type checks.
    ValueError
        If ``gamma`` or ``prob`` is outside ``[0, 1]`` beyond probability
        tolerance.

    Notes
    -----
    The stationary state is ``diag([prob, 1 - prob])``.  ``prob=1.0`` reduces
    to ordinary amplitude damping, with two additional zero operators retained
    by the current implementation.
    """
    gamma = check_probability(gamma, name="gamma")
    prob = check_probability(prob, name="prob")

    sqrt_gamma = np.sqrt(gamma)
    sqrt_1mgamma = np.sqrt(1.0 - gamma)
    sqrt_prob = np.sqrt(prob)
    sqrt_1mprob = np.sqrt(1.0 - prob)

    return KrausChannel(
        (
            sqrt_prob
            * np.array(
                [[1.0, 0.0], [0.0, sqrt_1mgamma]],
                dtype=np.complex128,
            ),
            sqrt_prob
            * np.array(
                [[0.0, sqrt_gamma], [0.0, 0.0]],
                dtype=np.complex128,
            ),
            sqrt_1mprob
            * np.array(
                [[sqrt_1mgamma, 0.0], [0.0, 1.0]],
                dtype=np.complex128,
            ),
            sqrt_1mprob
            * np.array(
                [[0.0, 0.0], [sqrt_gamma, 0.0]],
                dtype=np.complex128,
            ),
        ),
        name="generalized_amplitude_damping",
        arity=1,
    )


__all__ = ["amplitude_damping", "generalized_amplitude_damping"]
