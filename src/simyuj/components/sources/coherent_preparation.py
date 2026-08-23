"""Per-pulse preparation policies for a coherent optical source.

A weak coherent pulse source makes three independent classical choices for every
pulse it emits: a mean photon number :math:`\\mu`, a carrier phase
:math:`\\Theta`, and a deliberate encoding phase :math:`\\varphi_{enc}`. It then
builds one amplitude from them,

.. math::

   \\alpha = \\sqrt{\\mu}\\,e^{i(\\Theta + \\varphi_{enc})}

A fourth choice is optional and selects **which polarization mode that amplitude
occupies**. It is the only one whose value is a quantum state rather than a
number, which is why :class:`PolarizationSelection` carries a specification and
the *source*, not the selector, turns it into a qstate record. No concrete
polarization selector is shipped here: the alphabet, its noise model, and who
retires the mode record are a later change. The protocol and the record exist so
that change is additive.

Each choice is made by a small frozen strategy object supplied at construction,
following ``GateModel`` in ``detectors/primitives/gate.py``: a ``Protocol`` with
a trivial implementation, a parametric one, and an explicit-sequence one.

Four protocols, not one
-----------------------

Three of the four quantities are floats, so a single generic selector protocol
would type-check ``intensity=RandomPhaseChoice(...)`` and silently produce
:math:`\\mu \\in \\{0, \\pi\\}`. They also have different domains --
:math:`\\mu` is non-negative, a phase is any finite real -- and different
validation is different type. The fourth is not a float at all.

The carrier and encoding phases are kept apart for a physical reason, not a
stylistic one. In a differential-phase protocol the bit lives in
:math:`\\varphi_n - \\varphi_{n-1}`, so a carrier phase randomized independently
per pulse contributes :math:`\\Theta_n - \\Theta_{n-1}` and destroys the
encoding, while a carrier phase held across a block cancels exactly. One
conflated phase quantity cannot express both. Decoy-state BB84 wants the
opposite of what DPS wants here, which is why this is a policy and not a
constant.

Return shapes
-------------

Intensity, encoding phase, and polarization return a small frozen record carrying
**both the value and its position in the alphabet**, because the index is what a
protocol agent decodes and what the preparation report records. The carrier phase
returns a bare ``float``: ``PerPulseRandomCarrierPhase`` draws from a continuous
distribution and has no alphabet to index, which is exactly why
``CoherentPulsePreparationReport`` has ``intensity_index``,
``encoding_phase_index`` and ``polarization_index`` but no
``carrier_phase_index``.

Purity and the pulse counter
----------------------------

``select_*(index, rng)`` is pure: the selector receives the source's pulse
counter rather than holding a cursor, so one selector instance can drive several
sources without shared hidden state. ``index`` is **zero-based** -- the first
pulse of a run selects with ``index=0``, so ``PhaseSequence`` starts at
``phases[0]``. ``CoherentPulsePreparationReport.pulse_index`` is one-based; the
two differ by one on purpose and the report field is the one meant for humans.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Protocol

from simyuj.primitives.validation import require_finite_real, require_non_negative_real

# Reached into rather than reimplemented, following ``signal.py``'s own import of
# ``primitives.ids._require_optional_correlation_id``. A second definition would
# mean a second normalization tolerance and a second error message for one
# physical constraint, and the two would drift. This is the *same* check
# ``Signal.__post_init__`` runs -- which is the point: the source builds its
# signals with ``validation_flag=False``, so that check never fires on the
# emission hot path and the selection is where it has to happen instead.
from simyuj.signal.signal import _normalized_polarization

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG


DPS_PHASES: tuple[float, ...] = (0.0, pi)
"""The two-phase alphabet used by DPS-QKD.

Index ``0`` is phase ``0`` and index ``1`` is phase ``pi``. **That ordering is a
convention held by the caller, not by this constant.** See
:class:`RandomPhaseChoice` for why it matters.
"""


def _normalize_phases(phases: object) -> tuple[float, ...]:
    """Validate a non-empty alphabet of finite phases.

    Raises
    ------
    TypeError
        If `phases` is not a sequence, or is a ``str``/``bytes``, or contains a
        non-numeric or ``bool`` entry.
    ValueError
        If `phases` is empty or contains a non-finite entry.
    """
    if isinstance(phases, (str, bytes)) or not isinstance(phases, Sequence):
        raise TypeError("phases must be a sequence")

    resolved = tuple(
        require_finite_real(phase, field_name="phases") for phase in phases
    )
    if not resolved:
        raise ValueError("phases must be non-empty")
    return resolved


@dataclass(frozen=True, slots=True)
class IntensitySelection:
    """One selected mean photon number and its position in the alphabet.

    Parameters
    ----------
    mean_photon_number : float
        Finite non-negative mean photon number for this pulse.
    index : int
        Position of that value in the selector's alphabet. A single-valued
        selector reports ``0``. Decoy-state analysis reads this index to
        separate signal, decoy, and vacuum populations.
    """

    mean_photon_number: float
    index: int


@dataclass(frozen=True, slots=True)
class PhaseSelection:
    """One selected phase and its position in the alphabet.

    Parameters
    ----------
    phase_rad : float
        Finite phase in radians.
    index : int
        Position of that phase in the selector's alphabet. This is what the
        protocol layer decodes.

    Notes
    -----
    There is no classical label, unlike ``StateSample``. A sampler needs one
    because the prepared quantum state is opaque; a phase is fully described by
    ``phase_rad`` and located by ``index``, so a label would only restate them.
    """

    phase_rad: float
    index: int


@dataclass(frozen=True, slots=True)
class PolarizationSelection:
    """One selected polarization mode and its position in the alphabet.

    Parameters
    ----------
    jones : tuple[complex, complex]
        Jones vector ``(u_H, u_V)`` of the mode this pulse occupies, normalized
        to ``|u_H|**2 + |u_V|**2 == 1``. ``int`` and ``float`` components are
        accepted and converted, so ``(1.0, 0.0)`` is a valid horizontal state.
    index : int
        Position of that state in the selector's alphabet. This is what the
        protocol layer decodes; the report records it beside the vector for the
        same reason ``encoding_phase_rad`` sits beside ``encoding_phase_index``.

    Notes
    -----
    **This is a specification, not a record.** ``jones`` is *what to prepare*;
    the source calls ``timeline.qstate.prepare`` and owns the resulting
    ``state_ref``. The precedent is ``StateSample`` in ``qstate/sampler.py``,
    which ``SinglePhotonSource`` consumes exactly this way: the sampler is pure
    and never touches the timeline. Keeping the selector on the specification
    side of that seam is what lets one selector instance drive several sources,
    and it keeps the emit path free of a "descriptor or reference" branch.

    **The vector is validated here because the emit path cannot validate it.**
    ``Signal.__post_init__`` normalizes ``polarization``, but the source builds
    its signals with ``validation_flag=False`` and the check is skipped whole.
    Validating at selection construction costs one call per selector rather than
    one per pulse, and it fires at the point a caller can act on -- when the
    alphabet is written, not on the pulse that happens to draw the bad entry.

    There is no ``rep`` field, unlike ``StateSample``. A Jones vector is by
    construction a pure single-mode state, so ``"ket"`` is the only
    representation it can name; a field with one legal value documents nothing
    and invites a caller to set it wrongly. Partially polarized light is not a
    Jones vector at all -- it arrives as a ``NoiseModel`` on the prepared mode
    record, which ``QuantumChannel`` already applies through the density path.
    """

    jones: tuple[complex, complex]
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "jones", _normalized_polarization(self.jones))


class IntensitySelector(Protocol):
    """Protocol for per-pulse mean-photon-number selection.

    Notes
    -----
    Implementations are frozen value objects. They may consume the supplied
    deterministic RNG stream or ignore it entirely, but must never use global
    randomness. ``index`` is the source's zero-based pulse counter.
    """

    def select_intensity(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> IntensitySelection: ...


class CarrierPhaseSelector(Protocol):
    """Protocol for per-pulse carrier-phase selection.

    Notes
    -----
    Returns a bare ``float`` rather than a record: a carrier phase may come from
    a continuous distribution, which has no alphabet position to report.
    ``EmissionTimingProfile`` in ``_common.py`` returns a bare ``int`` for the
    same reason.
    """

    def select_carrier_phase(self, index: int, rng: DeterministicRNG) -> float: ...


class EncodingPhaseSelector(Protocol):
    """Protocol for per-pulse encoding-phase selection.

    Notes
    -----
    Returns a :class:`PhaseSelection` so the alphabet index travels with the
    value. A protocol agent decodes the index, never ``arg(alpha)``: the emitted
    amplitude carries the *sum* of the carrier and encoding phases, wrapped, and
    neither is recoverable from it.
    """

    def select_encoding_phase(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> PhaseSelection: ...


class PolarizationSelector(Protocol):
    """Protocol for per-pulse polarization-mode selection.

    Notes
    -----
    **No implementation of this protocol ships in this module.** The protocol,
    :class:`PolarizationSelection`, the source's ``polarization`` parameter and
    its RNG stream exist so that a decoy-BB84 alphabet is one new class here and
    no change anywhere else. Choosing that alphabet -- H/V/D/A ordering, and the
    convention a receiving agent decodes ``polarization_index`` against -- is a
    protocol decision and is deliberately not made here. ``RandomPhaseChoice``'s
    warning about silently inverted alphabets applies verbatim.

    Returns a :class:`PolarizationSelection`, which is a *specification*: the
    selector stays pure and never touches ``timeline.qstate``. The source
    prepares the record and stamps it ``SubsystemHandle(kind="mode")`` so
    ``qstate_payload_role`` reports ``"mode"`` and channel loss scales the
    amplitude instead of destroying the state.
    """

    def select_polarization(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> PolarizationSelection: ...


@dataclass(frozen=True, slots=True)
class FixedIntensity:
    """Selector that emits one mean photon number on every pulse.

    Parameters
    ----------
    mean_photon_number : float
        Finite non-negative mean photon number. Required: there is no
        physically neutral default.

    Notes
    -----
    Consumes no randomness.

    ``0.0`` is valid and means coherent vacuum -- a real optical state that
    still occupies a slot, produces a signal, and produces a report. It is not
    an absent pulse.

    There is no upper bound. "Weak" names this family of source, it is not a
    validation constraint, so ``mean_photon_number=4.0`` is accepted.
    """

    mean_photon_number: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_photon_number",
            require_non_negative_real(
                self.mean_photon_number,
                field_name="mean_photon_number",
            ),
        )

    def select_intensity(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> IntensitySelection:
        """Return the configured intensity without consuming ``rng``."""
        del index, rng
        return IntensitySelection(
            mean_photon_number=self.mean_photon_number,
            index=0,
        )


@dataclass(frozen=True, slots=True)
class FixedCarrierPhase:
    """Selector that holds one carrier phase for the whole run.

    Parameters
    ----------
    phase_rad : float, default=0.0
        Finite carrier phase in radians.

    Notes
    -----
    Consumes no randomness.

    This is the coherent-across-the-train case that differential-phase encoding
    requires: ``Theta_n - Theta_{n-1}`` is exactly ``0.0`` for every adjacent
    pair, so the differential phase carries the encoding alone.

    **It also means infinite laser coherence length.** Finite linewidth is not
    modelled anywhere in this repository; the honest model is a Wiener process
    on ``Theta``, which does not exist yet. Runs using this selector should say
    so in their report rather than imply the laser was measured.
    """

    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_rad",
            require_finite_real(self.phase_rad, field_name="phase_rad"),
        )

    def select_carrier_phase(self, index: int, rng: DeterministicRNG) -> float:
        """Return the configured carrier phase without consuming ``rng``."""
        del index, rng
        return self.phase_rad


@dataclass(frozen=True, slots=True)
class PerPulseRandomCarrierPhase:
    """Selector that draws an independent carrier phase for every pulse.

    Notes
    -----
    Consumes exactly one draw from the source's ``carrier`` RNG stream per
    pulse.

    The draw is uniform on the **half-open** interval ``[-pi, pi)``, because
    ``DeterministicRNG.uniform(a, b)`` excludes its upper bound. The literature
    usually writes this distribution as ``(-pi, pi]``; the two differ only at a
    single point of measure zero and describe the same physics, but the
    docstring states which one this implementation actually produces.

    This is what decoy-state BB84 requires: per-pulse phase randomization is
    what makes the emitted state a Poisson mixture of Fock states and the
    decoy-state analysis valid.

    **It destroys differential-phase encoding.** The differential phase picks up
    ``Theta_n - Theta_{n-1}``, itself uniform, so DPS visibility falls to zero.
    The selector is kept for decoy BB84 and because it makes that failure
    testable rather than asserted.
    """

    def select_carrier_phase(self, index: int, rng: DeterministicRNG) -> float:
        """Draw one carrier phase uniformly from ``[-pi, pi)``."""
        del index
        return float(rng.uniform(-pi, pi))


@dataclass(frozen=True, slots=True)
class FixedPhase:
    """Selector that applies one encoding phase to every pulse.

    Parameters
    ----------
    phase_rad : float, default=0.0
        Finite phase applied to every pulse.

    Notes
    -----
    Consumes no randomness. This is the source's default encoding policy, and it
    makes an unconfigured source an ordinary unmodulated laser rather than a
    transmitter for one particular protocol.

    It is ``PhaseSequence((phase_rad,), repeat=True)`` with a different name, and
    is kept as its own type for the same reason ``DeltaTiming`` and
    ``AlwaysOpenGate`` exist: "no modulation" is worth stating directly in a
    configuration.
    """

    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_rad",
            require_finite_real(self.phase_rad, field_name="phase_rad"),
        )

    def select_encoding_phase(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> PhaseSelection:
        """Return the configured phase without consuming ``rng``."""
        del index, rng
        return PhaseSelection(phase_rad=self.phase_rad, index=0)


@dataclass(frozen=True, slots=True)
class RandomPhaseChoice:
    """Selector that draws an encoding phase per pulse from a finite alphabet.

    Parameters
    ----------
    phases : Sequence[float], default=DPS_PHASES
        Non-empty alphabet of finite phases. The default is the DPS-QKD pair
        ``(0.0, pi)``.

    Notes
    -----
    Selection is uniform over the alphabet. Weighted selection is deliberately
    absent: the phase-encoded protocols nearby all draw uniformly, and
    ``StateSampler`` already exists for a weighted classical distribution.

    Consumes exactly one draw from the source's ``encoding`` RNG stream per
    pulse, so a run replays identically from a fixed ``master_seed``.

    .. warning::

       **The alphabet ordering is a shared convention, and nothing checks it.**
       The preparation report records ``encoding_phase_index``, and a protocol
       agent decodes that index against an alphabet it assumes. Construct this
       selector with ``(pi, 0.0)`` instead of ``(0.0, pi)`` and index ``0`` now
       means ``pi``: every differential bit inverts, no exception is raised, and
       the run still produces a plausible key that is wrong.

       The defence is not a field on the report -- ``encoding_phase_rad`` is
       recorded beside the index precisely so a consumer can check. It is that a
       trial builds the alphabet **once**, as a named constant, and hands the
       same constant to both the source configuration and the decoding helper.
       ``examples/dps/configs.py`` does this and says so where the constant is
       defined.
    """

    phases: Sequence[float] = DPS_PHASES

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", _normalize_phases(self.phases))

    def select_encoding_phase(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> PhaseSelection:
        """Draw one phase uniformly from the alphabet."""
        del index

        # min() guards the rng.random() == 1.0 boundary, which would otherwise
        # index one past the end of the alphabet.
        count = len(self.phases)
        chosen = min(int(float(rng.random()) * count), count - 1)
        return PhaseSelection(phase_rad=self.phases[chosen], index=chosen)


@dataclass(frozen=True, slots=True)
class PhaseSequence:
    """Selector that walks a caller-supplied encoding pattern in order.

    Parameters
    ----------
    phases : Sequence[float]
        Non-empty pattern of finite phases, consumed one per pulse.
    repeat : bool, default=False
        Whether to wrap around after the last phase.

    Notes
    -----
    Consumes no randomness. Use this to reproduce a published phase pattern or
    to make an interference test fully deterministic.

    The pattern is indexed by the source's **zero-based** pulse counter, so the
    first emitted pulse uses ``phases[0]``.

    With ``repeat=False`` an exhausted pattern raises ``RuntimeError`` rather
    than wrapping silently: a pattern shorter than the run would otherwise
    produce a plausible-looking but wrong key downstream. A run that aborts with
    a clear reason is the better outcome.

    This selector is **counter-keyed**, not time-keyed -- it advances once per
    emitted pulse. ``ScheduledGate`` in ``detectors/primitives/gate.py`` is the
    time-keyed alternative. The distinction does not matter at a source, where
    every active slot emits exactly once; it would matter downstream of anything
    lossy.
    """

    phases: Sequence[float]
    repeat: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", _normalize_phases(self.phases))
        if type(self.repeat) is not bool:
            raise TypeError("repeat must be bool")

    def select_encoding_phase(
        self,
        index: int,
        rng: DeterministicRNG,
    ) -> PhaseSelection:
        """Return the phase at ``index`` without consuming ``rng``."""
        del rng

        count = len(self.phases)
        if index >= count:
            if not self.repeat:
                raise RuntimeError(
                    f"phase sequence exhausted after {count} phases; "
                    "pass repeat=True to cycle"
                )
            index = index % count

        return PhaseSelection(phase_rad=self.phases[index], index=index)


def validate_pulse_selectors(
    *,
    intensity: object,
    carrier_phase: object,
    encoding_phase: object,
    polarization: object = None,
) -> None:
    """Validate that the preparation selectors implement their protocols.

    Parameters
    ----------
    intensity, carrier_phase, encoding_phase : object
        Candidate selector objects supplied to a coherent source.
    polarization : object, default=None
        Optional candidate polarization selector. ``None`` means no polarization
        is modelled and is not an error: the other three name a quantity every
        pulse has, while a pulse need not occupy a described mode at all.

    Raises
    ------
    TypeError
        If any supplied selector does not expose its callable selection method.

    Notes
    -----
    This is a duck-typed check rather than ``isinstance``, following
    ``validate_timing_profile`` in ``_common.py``. The protocols here are not
    ``runtime_checkable`` and the repository does not make its component
    protocols so.

    The check is construction-time only. A selector that implements the method
    but returns a bad value fails on the pulse that produces it: an invalid mean
    photon number raises from ``CoherentState.from_mean_photon_number``, and a
    wrong return type raises ``AttributeError``. Neither is guarded per pulse,
    because only a caller-written selector can reach either and the built-in
    selectors validate their alphabets at construction.

    A polarization selector is the one case where the returned value validates
    itself: :class:`PolarizationSelection` normalizes its Jones vector in
    ``__post_init__``, so an unnormalized state raises where the selector
    constructs the selection, not later inside ``timeline.qstate.prepare``.
    """
    for value, method_name, field_name in (
        (intensity, "select_intensity", "intensity"),
        (carrier_phase, "select_carrier_phase", "carrier_phase"),
        (encoding_phase, "select_encoding_phase", "encoding_phase"),
    ):
        if not callable(getattr(value, method_name, None)):
            raise TypeError(f"{field_name} must implement {method_name}(index, rng)")

    if polarization is not None and not callable(
        getattr(polarization, "select_polarization", None)
    ):
        raise TypeError(
            "polarization must implement select_polarization(index, rng)",
        )


__all__ = [
    "DPS_PHASES",
    "CarrierPhaseSelector",
    "EncodingPhaseSelector",
    "FixedCarrierPhase",
    "FixedIntensity",
    "FixedPhase",
    "IntensitySelection",
    "IntensitySelector",
    "PerPulseRandomCarrierPhase",
    "PhaseSelection",
    "PhaseSequence",
    "PolarizationSelection",
    "PolarizationSelector",
    "RandomPhaseChoice",
    "validate_pulse_selectors",
]
