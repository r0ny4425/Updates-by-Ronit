"""Coherent-state optical amplitude carried by pulse signals.

This module defines the classical-wave-optics description of an optical pulse.
A coherent state :math:`|\\alpha\\rangle` is fully specified by one complex
amplitude, so :class:`CoherentState` stores ``alpha`` and nothing else. Mean
photon number and phase are derived properties; they cannot drift out of sync
with the amplitude because there is nothing else to drift.

The amplitude is deliberately **not** a qstate-backed representation. ``alpha``
has no ``StateRef``, no subsystem in the quantum state manager, and no density
matrix, and quantum behaviour enters it only at photon detection. That is a
claim about the amplitude, not about the pulse: a signal carrying a
:class:`CoherentState` may also carry a qstate record describing the mode that
amplitude occupies, as a polarized weak coherent pulse does. That record is not
this type and is reached through ``Signal.state_ref``.

Scope of this module
--------------------

Only the *definition* lives here. Every optical **operation** on a coherent
amplitude -- phase shift, attenuation, beamsplitter splitting, interference,
temporal overlap -- belongs in ``simyuj.components.coherent_optics`` and takes
and returns values of this type.

Why the type lives in ``primitives``
------------------------------------

It follows :class:`~simyuj.primitives.subsystems.SubsystemHandle`: the same kind
of thing in the same relationship -- a value type carried by a field of
``Signal``, referenced by component code, and owned by neither package.
``SubsystemHandle`` is likewise not re-exported from ``simyuj.signal``; callers
import it from here.

There is also a hard constraint. ``Signal`` type-checks the value it carries, so
the type must be importable from the signal layer. Defining it under
``components`` makes ``import simyuj.signal`` circular, because
``components.__init__`` imports the channels, which import ``simyuj.signal``
while it is still initialising. ``primitives`` is below both and is safe: this
package's ``__init__`` loads submodules lazily, so importing this module does
not pull in ``primitives.messages``, which *does* depend on ``simyuj.signal``.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from math import isfinite, sqrt

from .validation import require_finite_real, require_non_negative_real


@dataclass(frozen=True, slots=True)
class CoherentState:
    """Immutable coherent-state amplitude for one optical pulse.

    Parameters
    ----------
    alpha : complex
        Complex optical amplitude. ``int`` and ``float`` values are accepted and
        converted to ``complex``; ``bool`` is rejected. Both the real and
        imaginary parts must be finite.

    Attributes
    ----------
    alpha : complex
        The stored amplitude, always exactly ``complex`` after construction.

    Raises
    ------
    TypeError
        If `alpha` is not an ``int``, ``float``, or ``complex``, or is ``bool``.
    ValueError
        If the real or imaginary part is ``nan`` or infinite.

    Notes
    -----
    ``alpha`` is the single source of truth. :attr:`mean_photon_number` and
    :attr:`phase_rad` are computed on access rather than stored, so no
    combination of fields can become inconsistent.

    There is no upper bound on the amplitude. A "weak" coherent pulse is a
    modelling convention chosen by the caller, not a constraint enforced here;
    a mean photon number of 4 is as valid as 0.1.

    The vacuum state ``alpha = 0`` is a valid coherent state, not an absent
    pulse. :attr:`phase_rad` returns ``0.0`` for it rather than raising, which
    matches ``cmath.phase(0j)``.

    Examples
    --------
    >>> state = CoherentState.from_mean_photon_number(0.25)
    >>> state.alpha
    (0.5+0j)
    >>> state.mean_photon_number
    0.25
    >>> state.phase_rad
    0.0
    """

    alpha: complex

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(
            self.alpha,
            (int, float, complex),
        ):
            raise TypeError("alpha must be int, float, or complex")

        resolved = complex(self.alpha)
        if not isfinite(resolved.real) or not isfinite(resolved.imag):
            raise ValueError("alpha must have finite real and imaginary parts")

        object.__setattr__(self, "alpha", resolved)

    @property
    def mean_photon_number(self) -> float:
        """Mean photon number :math:`\\mu = |\\alpha|^2`.

        Notes
        -----
        Computed as ``re * re + im * im`` rather than ``abs(alpha) ** 2`` to
        avoid the square-root round trip inside ``abs``.
        """
        return self.alpha.real * self.alpha.real + self.alpha.imag * self.alpha.imag

    @property
    def phase_rad(self) -> float:
        """Optical phase :math:`\\varphi = \\arg(\\alpha)` in radians.

        Returns ``0.0`` for the vacuum amplitude ``0j``.

        Notes
        -----
        This is the *total wrapped* phase. When a carrier phase and an encoding
        phase have been summed into the amplitude, neither is recoverable from
        it, and any later phase noise makes the divergence larger. Code that
        needs to know which phase was applied must read it from the preparation
        report, never from here.
        """
        return cmath.phase(self.alpha)

    @classmethod
    def from_mean_photon_number(
        cls,
        mean_photon_number: float,
        *,
        phase_rad: float = 0.0,
    ) -> "CoherentState":
        """Build a coherent state from :math:`\\mu` and :math:`\\varphi`.

        Parameters
        ----------
        mean_photon_number : float
            Finite non-negative mean photon number. Zero is valid and yields the
            vacuum amplitude. No upper bound is applied.
        phase_rad : float, default=0.0
            Finite optical phase in radians.

        Returns
        -------
        CoherentState
            State with :math:`\\alpha = \\sqrt{\\mu}\\,e^{i\\varphi}`.

        Raises
        ------
        TypeError
            If either argument is not numeric, or is ``bool``.
        ValueError
            If `mean_photon_number` is negative, ``nan``, or infinite, or if
            `phase_rad` is ``nan`` or infinite.

        Notes
        -----
        At the default ``phase_rad=0.0`` the imaginary part is exactly ``0.0``
        and the real part is exactly ``sqrt(mean_photon_number)``.

        Round-tripping through this constructor is not bit-exact in general:
        :math:`\\mu` passes through ``sqrt`` on the way in and is recovered by
        squaring, so ``0.2`` comes back as ``0.19999999999999998``. Compare
        mean photon numbers with a tolerance, never with ``==``.
        """
        resolved_mu = require_non_negative_real(
            mean_photon_number,
            field_name="mean_photon_number",
        )
        resolved_phase = require_finite_real(phase_rad, field_name="phase_rad")
        return cls(cmath.rect(sqrt(resolved_mu), resolved_phase))


__all__ = ["CoherentState"]
