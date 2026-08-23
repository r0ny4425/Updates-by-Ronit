"""Optical arithmetic on coherent-state amplitudes.

Every operation on a :class:`~simyuj.primitives.coherent_state.CoherentState`
lives here. The type itself lives in ``primitives/coherent_state.py`` and holds
no math -- that split is what makes "``Signal`` carries definitions, not
physics" true by construction, and it is forced besides: ``signal/signal.py``
imports the type, so defining it here would make ``import simyuj.signal``
circular through ``components/__init__``.

The functions are free functions rather than methods on ``CoherentState``, and
their scalar arguments are keyword-only. Of the operations this module will
eventually hold, only these two could have been methods at all: interference and
beamsplitter splitting are binary and return pairs, and temporal overlap and
click probability take no amplitude. Methods would have covered two of eight and
split the arithmetic across two layers.

This module takes no RNG and returns no random value, which makes "nothing here
samples a photon number" structural rather than a comment. Photon statistics are
integrated in closed form at detection.

Shipped when their first caller exists, not before: ``split_50_50``,
``interfere``, ``gaussian_temporal_overlap``, ``click_probability``,
``polarization_weights``, ``rotated_polarization``.
"""

from __future__ import annotations

import cmath
from math import sqrt

from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.validation import require_finite_real, require_probability


def _require_coherent_state(state: object, *, field_name: str) -> CoherentState:
    """Return `state` after confirming it is a :class:`CoherentState`."""
    if not isinstance(state, CoherentState):
        raise TypeError(f"{field_name} must be CoherentState")
    return state


def attenuated(
    state: CoherentState,
    *,
    power_transmission: float,
) -> CoherentState:
    """Return `state` attenuated by a **power** transmission.

    Parameters
    ----------
    state : CoherentState
        Amplitude entering the lossy element.
    power_transmission : float
        Fraction of optical **power** that survives, in ``[0, 1]``. For a fibre
        this is :math:`10^{-L/10}` with ``L`` the total loss in dB -- the same
        quantity that is a Bernoulli survival probability for a single photon.

    Returns
    -------
    CoherentState
        State with :math:`\\alpha \\to \\sqrt{\\eta}\\,\\alpha`, so
        :math:`\\mu \\to \\eta\\mu`.

    Raises
    ------
    TypeError
        If `state` is not a ``CoherentState``, or `power_transmission` is not
        numeric.
    ValueError
        If `power_transmission` is outside ``[0, 1]``.

    Notes
    -----
    **Power, not amplitude.** The mean photon number scales as
    :math:`\\eta`, and the amplitude as :math:`\\sqrt{\\eta}`. Passing an
    amplitude transmission here would make loss look like its own square root,
    which at 25 km of standard fibre is the difference between 0.2 and 0.45.

    ``attenuated`` rather than ``with_attenuation``: the latter does not say
    whether ``0.1`` is the loss or what survives it. The parameter name carries
    the convention.

    The phase is untouched -- attenuation is a real scaling, so
    :attr:`~simyuj.primitives.coherent_state.CoherentState.phase_rad` comes out
    of this function exactly as it went in.

    ``power_transmission = 0.0`` returns the **coherent vacuum**, a real optical
    state that still occupies its slot. It is not an absent pulse and it is not
    a dropped one.

    Values above ``1.0`` are rejected rather than clamped. Optical gain is not
    modelled anywhere in this simulator; see ``CAPABILITY_MAP.md`` section 5.

    Examples
    --------
    >>> state = CoherentState.from_mean_photon_number(0.2)
    >>> round(attenuated(state, power_transmission=0.5).mean_photon_number, 12)
    0.1
    """
    resolved = _require_coherent_state(state, field_name="state")
    eta = require_probability(power_transmission, field_name="power_transmission")
    return CoherentState(resolved.alpha * sqrt(eta))


def phase_shifted(state: CoherentState, *, phase_rad: float) -> CoherentState:
    """Return `state` with an added optical phase.

    Parameters
    ----------
    state : CoherentState
        Amplitude entering the phase element.
    phase_rad : float
        Finite phase shift in radians. May be negative.

    Returns
    -------
    CoherentState
        State with :math:`\\alpha \\to \\alpha e^{i\\theta}`, so
        :math:`|\\alpha|` and :math:`\\mu` are unchanged.

    Raises
    ------
    TypeError
        If `state` is not a ``CoherentState``, or `phase_rad` is not numeric.
    ValueError
        If `phase_rad` is ``nan`` or infinite.

    Notes
    -----
    Lossless by construction: the multiplier has unit modulus, so the mean
    photon number is invariant up to floating-point rounding.

    **:math:`\\theta = \\pi` does not give exactly** :math:`-\\alpha`.
    ``exp(1j*pi)`` is ``(-1+1.2246e-16j)``, and that residue is deliberately not
    special-cased. A hidden branch for :math:`\\pi` would make exactness look
    guaranteed, and an interferometer's dark port would then rest on an
    implementation accident rather than on its own tolerance. Compare phases and
    mean photon numbers with a tolerance, never with ``==``.

    ``phase_shifted`` and :func:`attenuated` commute analytically but **not**
    bit-exactly, because complex multiplication is not associative in floating
    point.

    Examples
    --------
    >>> import math
    >>> state = CoherentState.from_mean_photon_number(0.25)
    >>> shifted = phase_shifted(state, phase_rad=math.pi / 2)
    >>> round(shifted.mean_photon_number, 12)
    0.25
    """
    resolved = _require_coherent_state(state, field_name="state")
    theta = require_finite_real(phase_rad, field_name="phase_rad")
    return CoherentState(resolved.alpha * cmath.rect(1.0, theta))


__all__ = [
    "attenuated",
    "phase_shifted",
]
