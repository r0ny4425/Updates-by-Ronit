"""Optical arithmetic on coherent-state amplitudes.

Every operation on a :class:`~simyuj.primitives.coherent_state.CoherentState`
lives here; the type itself lives in ``primitives/coherent_state.py`` and holds
no math. This module takes no RNG and returns no random value.

Not yet shipped, pending a first caller: ``polarization_weights``,
``rotated_polarization``.

Beamsplitter convention
-----------------------

Both splitters use the **real** 50:50 matrix

.. math::

   \\frac{1}{\\sqrt{2}}\\begin{pmatrix} 1 & -1 \\\\ 1 & 1 \\end{pmatrix}

stated here once and used by :func:`split_50_50`, :func:`interfere`, and the
tests. The symmetric convention puts the interference term in
:math:`\\operatorname{Im}` where this one puts it in
:math:`\\operatorname{Re}`. **Never mix them**: a specification that writes
:math:`a_l = i\\alpha/\\sqrt2` at the first splitter *and* :math:`\\mu`
equations in :math:`\\operatorname{Re}` is internally inconsistent. See
``docs/dev/dps-design.md`` section 4.
"""

from __future__ import annotations

import cmath
from math import exp, expm1, isfinite, sqrt

from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.validation import (
    require_finite_real,
    require_non_negative_real,
    require_positive_real,
    require_probability,
)

_SQRT2 = sqrt(2.0)
"""Amplitude divisor for a 50:50 splitter, computed once."""

_OVERLAP_MODULUS_ATOL = 1e-12
"""Slack allowed above ``abs(overlap) == 1`` before it is rejected.

An overlap is an inner product of two normalized temporal modes, so
Cauchy-Schwarz caps its modulus at one. The tolerance admits a value that a
caller computed and rounded just past the cap; anything further is a bug in the
caller, not floating point.
"""


def _require_coherent_state(state: object, *, field_name: str) -> CoherentState:
    """Return `state` after confirming it is a :class:`CoherentState`."""
    if not isinstance(state, CoherentState):
        raise TypeError(f"{field_name} must be CoherentState")
    return state


def _require_overlap(value: object, *, field_name: str) -> complex:
    """Return a finite overlap of modulus at most one, as ``complex``.

    Accepts ``int``, ``float``, and ``complex``; rejects ``bool``, following
    :class:`~simyuj.primitives.coherent_state.CoherentState` on the same
    question.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise TypeError(f"{field_name} must be int, float, or complex")

    resolved = complex(value)
    if not isfinite(resolved.real) or not isfinite(resolved.imag):
        raise ValueError(f"{field_name} must have finite real and imaginary parts")

    modulus = abs(resolved)
    if modulus > 1.0 + _OVERLAP_MODULUS_ATOL:
        raise ValueError(
            f"{field_name} must have modulus at most 1, got {modulus!r}",
        )

    return resolved


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
    :math:`\\eta`, the amplitude as :math:`\\sqrt{\\eta}`. Passing an amplitude
    transmission here would make loss look like its own square root -- at 25 km
    of standard fibre, the difference between 0.2 and 0.45.

    The phase is untouched. ``power_transmission = 0.0`` returns the coherent
    vacuum, which still occupies its slot and is not an absent or dropped pulse.
    Values above ``1.0`` are rejected rather than clamped; optical gain is not
    modelled, see ``CAPABILITY_MAP.md`` section 5.

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
    ``exp(1j*pi)`` is ``(-1+1.2246e-16j)``, and that residue is not
    special-cased. Compare phases and mean photon numbers with a tolerance,
    never with ``==``. For the same reason ``phase_shifted`` and
    :func:`attenuated` commute analytically but not bit-exactly.

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


def split_50_50(state: CoherentState) -> tuple[CoherentState, CoherentState]:
    """Split `state` at a 50:50 beamsplitter with vacuum on the second input.

    Parameters
    ----------
    state : CoherentState
        Amplitude entering port 0 of the splitter.

    Returns
    -------
    tuple[CoherentState, CoherentState]
        Both output arms, each carrying
        :math:`\\alpha/\\sqrt{2}` and therefore :math:`\\mu/2`.

    Raises
    ------
    TypeError
        If `state` is not a ``CoherentState``.

    Notes
    -----
    With vacuum on the second input both outputs are equal, and **the same
    immutable object is returned twice**, so ``left is right``. Worth knowing
    before writing an identity assertion that means to test something else.

    The *lossless* splitter; compose with :func:`attenuated` for insertion loss.

    Examples
    --------
    >>> left, right = split_50_50(CoherentState.from_mean_photon_number(0.4))
    >>> round(left.mean_photon_number, 12), round(right.mean_photon_number, 12)
    (0.2, 0.2)
    >>> left is right
    True
    """
    resolved = _require_coherent_state(state, field_name="state")
    half = CoherentState(resolved.alpha / _SQRT2)
    return half, half


def gaussian_temporal_overlap(
    *,
    sigma_a_s: float,
    sigma_b_s: float,
    delta_s: float,
) -> float:
    """Return the mode overlap of two Gaussian pulses separated in time.

    Parameters
    ----------
    sigma_a_s, sigma_b_s : float
        Positive **field**-envelope standard deviations in seconds, as carried
        by ``Signal.temporal_mode_sigma_s``.
    delta_s : float
        Finite separation between the two envelope **centres**, in seconds. The
        result depends only on its magnitude.

    Returns
    -------
    float
        Overlap :math:`\\gamma` in ``(0, 1]``, where

        .. math::

           \\gamma = \\sqrt{\\frac{2\\sigma_a\\sigma_b}
                                 {\\sigma_a^2 + \\sigma_b^2}}
                     \\exp\\!\\left[-\\frac{\\Delta t^{2}}
                                          {2(\\sigma_a^2 + \\sigma_b^2)}\\right]

    Raises
    ------
    TypeError
        If any argument is not numeric, or is ``bool``.
    ValueError
        If either width is zero, negative, ``nan``, or infinite, or if
        `delta_s` is ``nan`` or infinite.

    Notes
    -----
    **Field envelope, so equal widths give a denominator of 4** --
    :math:`\\exp[-\\Delta t^2/(4\\sigma^2)]`. An *intensity*-envelope
    :math:`\\sigma` would give 8; the two differ by :math:`\\sqrt2` in the width
    and must never be mixed. This module and ``Signal.temporal_mode_sigma_s``
    are field-envelope throughout.

    **``delta_s`` is a centre-to-centre separation**, computed from a tick
    difference and nothing else -- see ``Signal.temporal_mode_sigma_s``.

    Zero width is rejected rather than special-cased; pass a small width to
    approach that limit. The prefactor is the width mismatch alone, so two
    pulses of different duration cannot interfere perfectly even when perfectly
    aligned. At large separations the exponential underflows to ``0.0``, which
    is the correct limit and not an error.

    Examples
    --------
    >>> gaussian_temporal_overlap(sigma_a_s=1e-11, sigma_b_s=1e-11, delta_s=0.0)
    1.0
    >>> round(
    ...     gaussian_temporal_overlap(
    ...         sigma_a_s=1e-11, sigma_b_s=1e-11, delta_s=2e-11
    ...     ),
    ...     12,
    ... )
    0.367879441171
    """
    sigma_a = require_positive_real(sigma_a_s, field_name="sigma_a_s")
    sigma_b = require_positive_real(sigma_b_s, field_name="sigma_b_s")
    delta = require_finite_real(delta_s, field_name="delta_s")

    variance_sum = sigma_a * sigma_a + sigma_b * sigma_b
    width_factor = sqrt(2.0 * sigma_a * sigma_b / variance_sum)
    return width_factor * exp(-(delta * delta) / (2.0 * variance_sum))


def interfere(
    short_arm: CoherentState,
    long_arm: CoherentState,
    *,
    overlap: complex = 1.0,
) -> tuple[CoherentState, CoherentState]:
    """Recombine two arms at a 50:50 beamsplitter.

    Parameters
    ----------
    short_arm, long_arm : CoherentState
        Amplitudes arriving at the two input ports. `short_arm` defines the
        reference temporal mode; see the notes.
    overlap : complex, default=1.0
        Mode overlap :math:`\\gamma` between the two arms, of modulus at most
        one. ``1.0`` means the arms occupy the same mode and interfere fully;
        ``0.0`` means they cannot interfere at all.

    Returns
    -------
    tuple[CoherentState, CoherentState]
        Outputs of port 0 and port 1. Port 0 is the **destructive** port when
        the arms are in phase.

    Raises
    ------
    TypeError
        If either arm is not a ``CoherentState``, or `overlap` is not numeric.
    ValueError
        If `overlap` is not finite, or its modulus exceeds one.

    Notes
    -----
    The long arm is split into a component along the short arm's mode, of weight
    :math:`\\gamma`, and an orthogonal remainder that cannot interfere:

    .. math::

       m = \\gamma\\,\\alpha_l, \\qquad
       a_k = \\frac{\\alpha_s \\mp m}{\\sqrt2}, \\qquad
       r = \\frac{(1 - |\\gamma|^2)\\,\\mu_l}{2}, \\qquad
       \\mu_k = |a_k|^2 + r

    Energy is conserved at every overlap: :math:`\\mu_0 + \\mu_1 = \\mu_s +
    \\mu_l`. Vacuum inputs, zero overlap and unequal amplitudes are values of
    this equation, not branches around it. See ``docs/dev/dps-design.md``
    section 4 for the derivation and the equivalent :math:`\\mu` form.

    **The result is intensity-exact and mode-truncated.** At
    :math:`|\\gamma| < 1` the field leaving a port is a superposition of two
    non-identical envelopes that no single ``(alpha, sigma)`` pair describes. The
    returned state carries the exact :math:`\\mu` *including* the orthogonal
    residual, and the phase of the interfering component only. **Neither the
    phase nor the width of an output may feed a further phase-sensitive or
    temporal-mode interference.** At :math:`|\\gamma| = 1` all of it is exact.

    **:math:`\\mu` does not round-trip bit-exactly**, because the state is
    rebuilt through ``from_mean_photon_number``: a :math:`\\mu` of ``0.2``
    returns as ``0.19999999999999998``, and an analytically dark port lands
    wherever the inputs' own rounding leaves it. Compare with a tolerance
    everywhere, and never assert a dark port is ``== 0.0``.

    Examples
    --------
    >>> arm = CoherentState.from_mean_photon_number(0.2)
    >>> dark, bright = interfere(arm, arm)
    >>> dark.mean_photon_number < 1e-30
    True
    >>> round(bright.mean_photon_number, 12)
    0.4
    """
    short = _require_coherent_state(short_arm, field_name="short_arm")
    long_ = _require_coherent_state(long_arm, field_name="long_arm")
    gamma = _require_overlap(overlap, field_name="overlap")

    mixed = gamma * long_.alpha
    # max() guards the rounding of `1 - abs(gamma)**2` at abs(gamma) == 1, where
    # it can land a few ulp below zero. The bound in _require_overlap means it
    # can never be hiding a real negative.
    residual = max(0.0, 1.0 - abs(gamma) ** 2) * long_.mean_photon_number / 2.0

    outputs = []
    for amplitude in (
        (short.alpha - mixed) / _SQRT2,
        (short.alpha + mixed) / _SQRT2,
    ):
        mean_photon_number = (
            amplitude.real * amplitude.real + amplitude.imag * amplitude.imag + residual
        )
        outputs.append(
            CoherentState.from_mean_photon_number(
                mean_photon_number,
                phase_rad=cmath.phase(amplitude),
            )
        )

    return outputs[0], outputs[1]


def click_probability(mean_photon_number: float, *, efficiency: float) -> float:
    """Return the probability that a threshold detector fires on one pulse.

    Parameters
    ----------
    mean_photon_number : float
        Non-negative :math:`\\mu` arriving in the detector's mode. ``0.0`` is
        the coherent vacuum, which is a real pulse and not an absent one.
    efficiency : float
        Detector quantum efficiency :math:`\\eta_d` in ``[0, 1]``.

    Returns
    -------
    float
        :math:`P = 1 - e^{-\\eta_d \\mu}`, in ``[0, 1]``.

    Raises
    ------
    TypeError
        If either argument is not numeric, or is ``bool``.
    ValueError
        If `mean_photon_number` is negative, ``nan`` or infinite, or
        `efficiency` is outside ``[0, 1]``.

    Notes
    -----
    **Exact, not an approximation.** Thinning a :math:`\\mathrm{Poisson}(\\mu)`
    photon number by :math:`\\eta_d` gives
    :math:`\\mathrm{Poisson}(\\eta_d\\mu)`, so this holds with no truncation and
    no small-:math:`\\mu` assumption.

    **:math:`\\eta_d` is already inside the exponent.** A caller that hands this
    value to a detector must let it *replace* that detector's own efficiency, not
    multiply it. Applying :math:`\\eta_d` twice lowers every click rate by that
    factor, silently and plausibly -- at :math:`\\eta_d = 0.2` a run yields a
    fifth of the clicks it should, nothing raises, and the result merely looks
    like a lossier link, which is the quantity a QKD run is trying to measure.

    Computed with ``expm1``, which is accurate where ``1.0 - exp(-x)`` loses
    every significant digit to cancellation.

    **The ceiling is reached.** Above :math:`\\eta_d \\mu \\approx 37` this
    returns exactly ``1.0``, and a detector reading that value clicks without
    drawing from its efficiency stream -- which a later window's stream position
    depends on.

    Examples
    --------
    >>> round(click_probability(0.0, efficiency=0.2), 12)
    0.0
    >>> round(click_probability(0.2, efficiency=1.0), 12)
    0.181269246922
    >>> round(click_probability(1e-33, efficiency=0.2), 12)
    0.0
    """
    mu = require_non_negative_real(
        mean_photon_number,
        field_name="mean_photon_number",
        type_name="numeric",
    )
    eta = require_probability(efficiency, field_name="efficiency")
    return -expm1(-eta * mu)


__all__ = [
    "attenuated",
    "click_probability",
    "gaussian_temporal_overlap",
    "interfere",
    "phase_shifted",
    "split_50_50",
]
