from __future__ import annotations

"""Depolarizing-channel constructors using Pauli-channel decompositions."""

from .pauli import pauli_channel, two_qubit_pauli_channel

_TWO_QUBIT_NON_IDENTITY_PAULIS = tuple(
    left + right
    for left in ("I", "X", "Y", "Z")
    for right in ("I", "X", "Y", "Z")
    if left + right != "II"
)


def depolarizing(p: object):
    """Return a single-qubit depolarizing channel.

    This implements the single-qubit depolarizing map:

    .. math::

       \\rho \\mapsto (1 - p)\\rho + pI / 2

    where ``p`` is the mixing probability with the maximally mixed state.
    Equivalently, this channel can be written as a Pauli channel with Kraus
    operators proportional to ``I``, ``X``, ``Y``, and ``Z``:

    .. math::

       E(\\rho) =
       (1 - 3p / 4)\\rho
       + (p / 4)X\\rho X
       + (p / 4)Y\\rho Y
       + (p / 4)Z\\rho Z

    Therefore, the returned channel uses Pauli error probabilities ``p / 4``
    for each of ``X``, ``Y``, and ``Z``. The total non-identity Pauli error
    probability is ``3p / 4``.

    Parameters
    ----------
    p : object
        Mixing probability with the maximally mixed state. The value is
        validated using ``check_probability`` and must represent a scalar
        probability in the interval ``[0, 1]``.

    Returns
    -------
    KrausChannel
        A one-qubit Kraus representation of the depolarizing channel
        equivalent to :math:`\\rho \\mapsto (1 - p)\\rho + pI / 2`.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by
        ``check_probability``.
    ValueError
        If ``p`` lies outside the valid probability interval ``[0, 1]``,
        up to the tolerance used by ``check_probability``.

    Notes
    -----
    This function uses the "mixing probability" convention for the
    depolarizing parameter ``p``. Under this convention:

    * ``p = 0`` gives the identity channel.
    * ``p = 1`` gives the fully depolarizing channel:

      .. math::

         \\rho \\mapsto I / 2

    Some libraries instead define ``p`` as the total probability of applying
    a non-identity Pauli error, with probabilities ``p / 3`` for each of
    ``X``, ``Y``, and ``Z``. That is a different parameter convention. In this
    implementation, the total non-identity Pauli error probability is
    ``3p / 4``, not ``p``.

    Examples
    --------
    >>> from simyuj.qstate import QuantumStateManager, SubsystemId
    >>> from simyuj.qstate.noise import depolarizing
    >>> q0 = SubsystemId("q0")
    >>> manager = QuantumStateManager()
    >>> state_ref = manager.prepare("|0>", rep="density", subsystems=(q0,))
    >>> _ = manager.apply_noise(depolarizing(1.0), targets=(q0,))
    >>> float(manager.get(state_ref).rho[0, 0].real)
    0.5
    """
    from ..check import check_probability

    p = check_probability(p, name="p")
    return pauli_channel(p / 4.0, p / 4.0, p / 4.0)


def two_qubit_depolarizing(p: object):
    """Return a two-qubit depolarizing channel.

    This implements the two-qubit depolarizing map:

    .. math::

       \\rho \\mapsto (1 - p)\\rho + pI / 4

    where ``p`` is the mixing probability with the two-qubit maximally mixed
    state. Equivalently, this channel can be written as a two-qubit Pauli
    channel over the 15 non-identity two-qubit Pauli operators:

    .. math::

       E(\\rho) =
       (1 - 15p / 16)\\rho
       + (p / 16) \\sum_{P \\ne II} P\\rho P

    Therefore, the returned channel assigns probability ``p / 16`` to each
    non-identity two-qubit Pauli label. The total probability of applying a
    non-identity two-qubit Pauli error is ``15p / 16``.

    Parameters
    ----------
    p : object
        Mixing probability with the two-qubit maximally mixed state. The value
        is validated using ``check_probability`` and must represent a scalar
        probability in the interval ``[0, 1]``.

    Returns
    -------
    KrausChannel
        A two-qubit Kraus representation of the depolarizing channel
        equivalent to :math:`\\rho \\mapsto (1 - p)\\rho + pI / 4`.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by
        ``check_probability``.
    ValueError
        If ``p`` lies outside the valid probability interval ``[0, 1]``,
        up to the tolerance used by ``check_probability``.

    Notes
    -----
    This function uses the "mixing probability" convention for the
    depolarizing parameter ``p``. Under this convention:

    * ``p = 0`` gives the identity channel.
    * ``p = 1`` gives the fully depolarizing channel:

      .. math::

         \\rho \\mapsto I / 4

    This is not the same as defining ``p`` to be the total probability of
    applying a non-identity two-qubit Pauli error. In this implementation, the
    total non-identity Pauli error probability is ``15p / 16``.

    If one instead wants a Pauli channel where the total non-identity Pauli
    error probability is ``q``, each of the 15 non-identity two-qubit Pauli
    labels should have probability ``q / 15``. That convention corresponds to
    the depolarizing mixing parameter ``p = 16q / 15``.

    Target order follows ``two_qubit_pauli_channel``: for example, ``"IX"``
    means ``I`` on ``targets[0]`` and ``X`` on ``targets[1]``.
    """
    from ..check import check_probability

    p = check_probability(p, name="p")
    each = p / 16.0
    return two_qubit_pauli_channel(
        {label: each for label in _TWO_QUBIT_NON_IDENTITY_PAULIS}
    )


__all__ = ["depolarizing", "two_qubit_depolarizing"]
