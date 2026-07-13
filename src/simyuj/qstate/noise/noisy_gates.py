from __future__ import annotations

"""Noisy two-qubit gate channels.

The constructors in this module return Kraus channels that first apply an ideal
two-qubit gate matrix and then apply two-qubit depolarizing noise.
"""

import numpy as np

from ..check import check_probability
from ..errors import DimensionError
from ..math import matrix as mat
from .base import KrausChannel, check_kraus_channel
from .depolarize import two_qubit_depolarizing


def _post_gate_noise_channel(
    unitary: object,
    noise: object,
    *,
    name: str,
) -> KrausChannel:
    """Compose a two-qubit unitary with post-gate Kraus noise.

    Parameters
    ----------
    unitary : object
        Two-qubit unitary matrix with shape ``(4, 4)``.
    noise : object
        Two-qubit ``KrausChannel`` applied after the unitary.
    name : str
        Name assigned to the returned channel.

    Returns
    -------
    KrausChannel
        Two-qubit channel with operators ``noise_op @ unitary``.

    Raises
    ------
    DimensionError
        If ``unitary`` does not have shape ``(4, 4)`` or ``noise`` is not a
        two-qubit channel.
    TypeError
        If ``noise`` is not a ``KrausChannel``.

    Notes
    -----
    The returned channel implements
    :math:`\\rho \\mapsto \\mathcal{N}(U\\rho U^\\dagger)`.
    """
    unitary_array = np.asarray(unitary, dtype=np.complex128)
    if unitary_array.shape != (4, 4):
        raise DimensionError("two-qubit gate unitary must have shape (4, 4)")

    checked_noise = check_kraus_channel(noise)
    if checked_noise.arity != 2:
        raise DimensionError("post-gate noise must have arity 2")

    return KrausChannel(
        tuple(noise_op @ unitary_array for noise_op in checked_noise.ops),
        name=name,
        arity=2,
    )


def imperfect_cnot(p: object) -> KrausChannel:
    """Return an imperfect CNOT channel.

    Parameters
    ----------
    p : object
        Two-qubit depolarizing mixing probability after the ideal CNOT.

    Returns
    -------
    KrausChannel
        Two-qubit channel named ``"imperfect_cnot"``.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by ``check_probability``.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.

    Notes
    -----
    The channel implements an ideal CNOT followed by two-qubit depolarizing
    noise:

    .. math::

       \\rho \\mapsto D_{2,p}(\\mathrm{CNOT}\\rho\\mathrm{CNOT}^\\dagger)

    Target order is ``targets[0] = control`` and ``targets[1] = target``.  The
    depolarizing probability follows the mixing convention
    :math:`D_{2,p}(\\rho) = (1 - p)\\rho + pI / 4`.
    """
    p = check_probability(p, name="p")
    return _post_gate_noise_channel(
        mat.CNOT,
        two_qubit_depolarizing(p),
        name="imperfect_cnot",
    )


def imperfect_cz(p: object) -> KrausChannel:
    """Return an imperfect CZ channel.

    Parameters
    ----------
    p : object
        Two-qubit depolarizing mixing probability after the ideal CZ.

    Returns
    -------
    KrausChannel
        Two-qubit channel named ``"imperfect_cz"``.

    Raises
    ------
    TypeError
        If ``p`` is not a scalar probability accepted by ``check_probability``.
    ValueError
        If ``p`` is outside ``[0, 1]`` beyond probability tolerance.

    Notes
    -----
    The channel implements an ideal CZ followed by two-qubit depolarizing noise:

    .. math::

       \\rho \\mapsto D_{2,p}(\\mathrm{CZ}\\rho\\mathrm{CZ}^\\dagger)

    The depolarizing probability follows the mixing convention
    :math:`D_{2,p}(\\rho) = (1 - p)\\rho + pI / 4`.

    Target order follows the supplied target tuple:
    ``targets[0] = first CZ qubit`` and ``targets[1] = second CZ qubit``.
    """
    p = check_probability(p, name="p")
    return _post_gate_noise_channel(
        mat.CZ,
        two_qubit_depolarizing(p),
        name="imperfect_cz",
    )


__all__ = ["imperfect_cnot", "imperfect_cz"]
