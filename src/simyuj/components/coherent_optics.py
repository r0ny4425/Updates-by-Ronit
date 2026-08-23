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

Shipped when their first caller exists, not before: ``click_probability``,
``polarization_weights``, ``rotated_polarization``.

Beamsplitter convention
-----------------------

Both splitters use the **real** 50:50 matrix

.. math::

   \\frac{1}{\\sqrt{2}}\\begin{pmatrix} 1 & -1 \\\\ 1 & 1 \\end{pmatrix}

stated here once and used by :func:`split_50_50`, :func:`interfere`, and the
tests. The symmetric convention
:math:`\\tfrac{1}{\\sqrt2}\\bigl(\\begin{smallmatrix}1 & i\\\\ i &
1\\end{smallmatrix}\\bigr)` is the *same physical device*: it gives an identical
port 0 and a port 1 differing only by an unobservable global :math:`i`. The two
differ in where the interference term lands -- :math:`\\operatorname{Im}` for
the symmetric one, :math:`\\operatorname{Re}` for the real one -- and
:math:`\\operatorname{Re}` is chosen so the :math:`\\mu` equations below read
the way the tests are written. **Never mix them.** A specification that writes
:math:`a_l = i\\alpha/\\sqrt2` at the first splitter *and* :math:`\\mu`
equations in :math:`\\operatorname{Re}` is internally inconsistent; the
:math:`i` does not remove interference, it moves it to
:math:`\\operatorname{Im}`.
"""

from __future__ import annotations

import cmath
from math import exp, isfinite, sqrt

from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.validation import (
    require_finite_real,
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
    With vacuum on the second input the module's real 50:50 matrix gives
    :math:`(\\alpha - 0)/\\sqrt2` and :math:`(\\alpha + 0)/\\sqrt2`, which are
    equal. **The same immutable object is returned twice**, so
    ``left is right``. That is safe because ``CoherentState`` is frozen, and it
    is worth knowing before writing an identity assertion that means to test
    something else.

    Under the symmetric convention port 1 would carry
    :math:`i\\alpha/\\sqrt2` instead. The mean photon numbers are identical;
    only a later phase-sensitive recombination could tell the two apart, which
    is exactly why the module fixes one convention -- see the module docstring.

    This is the *lossless* splitter. A real device's insertion loss is not
    modelled here; compose with :func:`attenuated` if it is needed.

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
    **Field envelope, so equal widths give a denominator of 4.** At
    :math:`\\sigma_a = \\sigma_b = \\sigma` this reduces to
    :math:`\\exp[-\\Delta t^2/(4\\sigma^2)]`. An *intensity*-envelope
    :math:`\\sigma` would give 8. The two parameterisations differ by
    :math:`\\sqrt2` in the width and must never be mixed; this module and
    ``Signal.temporal_mode_sigma_s`` are both field-envelope throughout.

    **``delta_s`` is a centre-to-centre separation.** ``Signal`` fixes the
    convention that a signal's tick is the centre of its temporal mode, so a
    caller computes this from a tick difference and nothing else -- see
    ``Signal.temporal_mode_sigma_s``.

    Keyword-only, because three floats in a row are trivially transposable and
    two of them are interchangeable while the third is not.

    **Zero width is rejected rather than special-cased.** The
    :math:`\\sigma \\to 0` limit is a different, discrete model in which overlap
    is an equality test on arrival ticks, and returning ``1.0`` or ``0.0`` from
    here would quietly answer a question this formula was not asked. A caller
    who wants that limit should pass a small width and see the exponential do
    it.

    The prefactor is the width mismatch alone: it is ``1.0`` at equal widths and
    falls away as they diverge, so two pulses of different duration cannot
    interfere perfectly even when perfectly aligned. At large separations the
    exponential underflows to ``0.0``, which is the correct limit and not an
    error.

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

    This is algebraically identical to
    :math:`\\mu_k = \\tfrac12[\\,\\mu_s + \\mu_l \\mp
    2\\operatorname{Re}(\\overline{\\alpha_s}\\alpha_l\\gamma)\\,]` but keeps the
    non-interfering light visible in the code rather than folded into an
    algebraic identity.

    **Energy is conserved at every overlap**: :math:`\\mu_0 + \\mu_1 =
    \\mu_s + \\mu_l`. That identity is what catches a convention error, so it is
    asserted on every case in the tests. It is also why the modulus of `overlap`
    is bounded rather than accepted freely -- at :math:`|\\gamma| > 1` the
    orthogonal remainder would go negative, the ``max`` clamp would hide it, and
    the two ports would carry more light than entered.

    **Vacuum inputs, zero overlap, unequal amplitudes, the first pulse of a
    train and the last are values of this equation, not branches around it.**
    Because the interference term is proportional to *both* amplitudes, a vacuum
    partner kills it at any :math:`\\gamma`, and the result is a plain 50:50
    split of whichever arm is present.

    **The result is intensity-exact and mode-truncated.** At
    :math:`|\\gamma| < 1` the field leaving a port is
    :math:`\\alpha_s f_s(t) \\mp \\gamma\\alpha_l f_l(t)`, a superposition of two
    non-identical envelopes that no single ``(alpha, sigma)`` pair describes. The
    returned state carries the exact :math:`\\mu` *including* the orthogonal
    residual, and the phase of the interfering component only. **Neither the
    phase nor the width of an output may feed a further phase-sensitive or
    temporal-mode interference.** At :math:`|\\gamma| = 1` all of it is exact.

    The asymmetry between the arguments is the mode reference, not the physics:
    :math:`\\gamma` projects the long arm onto the short arm's mode, so the
    residual is the long arm's. A caller that swaps the arguments gets the same
    two mean photon numbers, because the identity above is symmetric in
    :math:`\\mu_s` and :math:`\\mu_l`, but the port-0 phase is taken from a
    different reference.

    **:math:`\\mu` does not round-trip bit-exactly.** The residual is added after
    the amplitude is formed, so :math:`a_k` has the wrong modulus and the state
    is rebuilt through ``from_mean_photon_number``, sending :math:`\\mu` through
    ``sqrt`` and back. A :math:`\\mu` of ``0.2`` returns as
    ``0.19999999999999998``. An analytically dark port lands wherever the
    inputs' own rounding leaves it: exactly ``0.0`` for two equal real
    amplitudes, and ``1.5e-33`` when one arm arrived through the ``exp(1j*pi)``
    residue that :func:`phase_shifted` documents. Neither number is a
    guarantee. Compare with a tolerance everywhere, and never assert a dark port
    is ``== 0.0``.

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


__all__ = [
    "attenuated",
    "gaussian_temporal_overlap",
    "interfere",
    "phase_shifted",
    "split_50_50",
]
