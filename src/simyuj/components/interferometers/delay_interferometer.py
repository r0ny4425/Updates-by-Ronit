"""Ideal unbalanced Mach-Zehnder interferometer for coherent pulses.

``DelayInterferometer`` splits each arriving pulse across a 50:50 beamsplitter,
delays one arm by :math:`\\tau`, and recombines it on a second 50:50
beamsplitter with the *next* pulse's undelayed arm. In DPS-QKD
:math:`\\tau = T_{\\mathrm{pulse}}`, so the recombination compares the optical
phases of two adjacent slots and the phase difference appears as an intensity
difference between the two output ports.

**The optics are ideal.** No internal loss, no internal phase noise, no
imperfect splitting ratio, no photon-number sampling. Every optical imperfection
must arrive with the incoming pulses; loss and phase noise already come from the
channel. What the component adds optically is the delay, the recombination, and
the temporal-mode bookkeeping that decides how much the two contributions
actually overlap.

The receiver is one unit
------------------------

Two ``SinglePhotonDetector`` channels sit inside, one per output port, and they
are not ideal -- efficiency, dark counts, dead time, jitter and afterpulsing are
all theirs. A DPS receiver is one physical unit, thermally stabilised together,
and no protocol deploys the interferometer bare; putting detection downstream
would also invent a slot-join problem that does not exist here, since BS2
produces both outputs in one call and one slot decision follows with no
buffering and no cross-component coordination.

Each port's click probability is
:math:`P_k = 1 - e^{-\\eta_d \\mu_k}`, evaluated **independently**, so a real
double click occurs at a real rate -- unlike a readout that maps one measured
outcome to one detector, where two signal clicks are structurally impossible.
Detection is opt-in: ``detectors=None`` reproduces this component before
detection existed, exactly.

Nearest-neighbour only
----------------------

One long-arm contribution is held at a time, so each pulse pairs with its
immediate predecessor and nothing else. The intended operating regime is
:math:`\\tau \\approx T`. A :math:`\\tau` of two slot periods, which would need
pulse *k* to meet pulse *k+2*, is **not** supported: pulse *k+1* arrives first
and takes the holder. That is a scoping decision, not an oversight -- arbitrary
:math:`\\tau` needs a keyed queue and belongs in a different component. See
``CAPABILITY_MAP.md`` section 5.

Timing is observed, never corrected
-----------------------------------

BS1 acts at a pulse's actual arrival tick, which -- by the contract on
``Signal.temporal_mode_sigma_s`` -- is the centre of that pulse's temporal mode.
The overlap :math:`\\gamma` between the two contributions meeting at BS2 is
computed from their real centre-to-centre separation, so a pulse that arrives
late interferes less rather than being quietly realigned.

The component never checks :math:`\\tau` against the pulse period, because it
does not know the period: taking it as configuration would put the source's
``frequency_hz`` in two places and let them drift. A mismatch shows up as
:math:`\\gamma` collapsing on every slot, which every report and every log record
carries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.units import seconds_to_ticks, ticks_to_seconds
from simyuj.primitives.validation import (
    require_non_negative_int,
    require_optional_positive_real,
    require_positive_int,
)
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel

from ..coherent_optics import (
    click_probability,
    gaussian_temporal_overlap,
    interfere,
    split_50_50,
)
from ..connections import PortDelivery, require_connection
from ..detectors.primitives.click import ClickPatternResolver, ThresholdClickResolver
from ..detectors.primitives.gate import AlwaysOpenGate, GateModel
from ..detectors.primitives.readout import DetectorExposure
from ..detectors.primitives.reports import DetectionReport
from ..detectors.primitives.rng import DetectorRNGStreams
from ..detectors.primitives.window import (
    bind_detector_rngs,
    evaluate_detector_windows,
    normalize_detectors,
    validate_gate_model,
)
from ..detectors.single_photon import SinglePhotonDetector
from ..ports import Port, PortDirection, PortKind

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG
    from simyuj.engine.timeline import Timeline


COMPONENT_KEY = "delay_interferometer"

ACTION_INTERFERE = "interfere"
"""Ingress action: one coherent pulse arriving at BS1."""

ACTION_RESOLVE_BS2 = "resolve_bs2"
"""Self-scheduled action: a combination deferred until both arms reach BS2."""

ACTION_FLUSH_DELAY_ARM = "flush_delay_arm"
"""Self-scheduled action: a held long arm that never found a partner."""

PORT_OUT_0 = "out_0"
PORT_OUT_1 = "out_1"

PORT_DETECTION = "detection"
"""Classical egress carrying one ``DetectionReport`` per interference slot.

Separate from the ``"report"`` port, which carries ``InterferenceReport``. The
two have different consumers -- optics inspection and a protocol agent -- and an
agent that type-checks its inbox, which is the established pattern, would reject
a mixed stream.
"""

_VACUUM = CoherentState(0j)

_FLUSH_ORDERING_MARGIN_TICKS = 1
"""Ticks added to the flush deadline so ordering rests on ``time``.

The physical deadline on a held arm is ``arrival + 2 * tau``. The flush *event*
is scheduled one tick after it. See :class:`DelayInterferometer` for why; the
short version is that ``flush_priority`` alone couples this component's
correctness to a value configured on a different one.
"""


@dataclass(frozen=True, slots=True)
class ArmContribution:
    """One arm's amplitude and the tick at which it reaches BS2.

    Parameters
    ----------
    state : CoherentState
        Post-BS1 amplitude on this arm.
    bs2_tick : int
        Tick at which this contribution arrives at the second beamsplitter. For
        the short arm that is the pulse's arrival tick plus the common transit
        ``short_delay_ticks``; for the long arm it is that plus :math:`\\tau`.
        Their difference is therefore :math:`\\tau` alone, whatever the common
        transit is.
    sigma_s : float
        Field-envelope standard deviation in seconds, copied from the pulse.
    wavelength_nm : float
        Carrier wavelength copied from the pulse.
    pulse_index : int or None
        Source slot of the pulse this arm came from, or ``None`` when this
        contribution is the coherent vacuum.
    signal_id : str or None
        Identifier of the pulse this arm came from, or ``None`` for vacuum.

    Notes
    -----
    Both arms use the same record and the same ``bs2_tick`` meaning, which is
    what lets the overlap be ``abs(short.bs2_tick - long.bs2_tick)`` with no
    per-arm special case. Because a signal's tick is its envelope centre, that
    difference is directly the centre-to-centre separation
    ``gaussian_temporal_overlap`` wants.
    """

    state: CoherentState
    bs2_tick: int
    sigma_s: float
    wavelength_nm: float
    pulse_index: Optional[int]
    signal_id: Optional[str]


@dataclass(frozen=True, slots=True)
class HeldLongArm:
    """The long-arm contribution held between two pulses.

    Parameters
    ----------
    hold_id : int
        Monotonically increasing identifier. A flush event carries the
        ``hold_id`` it was scheduled for and does nothing when the holder has
        moved on, which is how a taken *or replaced* arm is recognised.
    arm : ArmContribution
        The held contribution.

    Notes
    -----
    ``hold_id`` is what makes the stale-flush check correct rather than merely
    plausible. Testing ``self._held is None`` alone would pass every time the
    holder happened to be empty and still let pulse *k*'s expired deadline
    destroy pulse *k+1*'s freshly held arm -- the holder is *present and
    different* in exactly the case that matters.
    """

    hold_id: int
    arm: ArmContribution


@dataclass(frozen=True, slots=True)
class PendingCombination:
    """Payload for a combination deferred until the long arm reaches BS2.

    Parameters
    ----------
    short, long_ : ArmContribution
        The two contributions to combine. They are carried by the event rather
        than by the component, which is what releases the holder immediately and
        keeps a second pulse arriving mid-flight from disturbing anything.
    """

    short: ArmContribution
    long_: ArmContribution


@dataclass(frozen=True, slots=True)
class DelayArmFlush:
    """Payload for the deadline on a held long arm.

    Parameters
    ----------
    hold_id : int
        Holder generation this deadline was scheduled for.
    """

    hold_id: int


@dataclass(frozen=True, slots=True)
class InterferenceReport:
    """Immutable report for one BS2 combination.

    Parameters
    ----------
    report_id : str
        Stable identifier chosen by the interferometer.
    device_id : str
        Interferometer device identifier.
    time : int
        Simulation tick at which the combination was resolved.
    interference_index : int
        One-based count of combinations this device has produced. An ``N`` pulse
        train produces ``N + 1`` of them: the first pulse's short arm meets
        vacuum, and the last pulse's long arm meets vacuum.
    signal_ids : tuple[str, ...]
        Identifiers of the contributing input pulses, short arm first. A vacuum
        arm contributes nothing, so this holds one entry on the first and last
        combinations and two in between.
    output_signal_ids : tuple[str, str]
        Identifiers of the signals emitted on ``out_0`` and ``out_1``.
    short_pulse_index, long_pulse_index : int or None
        Source slots of the two contributions. **``None`` means that arm was the
        coherent vacuum**, which is a defined absence, not a missing lookup:
        ``long_pulse_index`` is ``None`` on the first combination and
        ``short_pulse_index`` is ``None`` on the flush. When the incoming pulse
        carried no ``pulse_index`` metadata at all this is also ``None``; the
        signal ids disambiguate.
    temporal_overlap : float
        :math:`\\gamma` used for this combination.
    delta_ticks : int
        Centre-to-centre separation of the two contributions at BS2, in ticks.
    short_bs2_tick, long_bs2_tick : int
        Ticks at which the two contributions reached BS2.
    mean_photon_number_in : float
        :math:`\\mu_s + \\mu_\\ell` entering BS2.
    mean_photon_number_0, mean_photon_number_1 : float
        :math:`\\mu` leaving each output port. These sum to
        ``mean_photon_number_in``.
    meta : tuple[tuple[str, object], ...]
        Additional immutable metadata.

    Notes
    -----
    No ``state_ref``, no ``state_targets``, no sampler field. Interfering two
    coherent amplitudes creates no quantum state record, and none arrives to be
    carried through: this device refuses any signal carrying a ``state_ref``, so
    a polarized pulse never reaches BS2. Optical interference of a described mode
    is not implemented; see ``CAPABILITY_MAP.md`` section 5.

    ``interference_index`` counts this device's combinations. Join downstream
    results on ``short_pulse_index`` / ``long_pulse_index`` or on the signal ids,
    never on ``interference_index``.
    """

    report_id: str
    device_id: str
    time: int
    interference_index: int
    signal_ids: tuple[str, ...]
    output_signal_ids: tuple[str, str]
    short_pulse_index: Optional[int]
    long_pulse_index: Optional[int]
    temporal_overlap: float
    delta_ticks: int
    short_bs2_tick: int
    long_bs2_tick: int
    mean_photon_number_in: float
    mean_photon_number_0: float
    mean_photon_number_1: float
    meta: tuple[tuple[str, object], ...] = ()


def vacuum_like(other: ArmContribution) -> ArmContribution:
    """Return a coherent-vacuum contribution matching ``other``'s mode.

    Parameters
    ----------
    other : ArmContribution
        The contribution that *is* present at BS2.

    Returns
    -------
    ArmContribution
        Vacuum amplitude carrying ``other``'s BS2 tick, width, and wavelength,
        and no pulse identity.

    Notes
    -----
    This is what removes the first-pulse and last-pulse branches. The
    interference term is proportional to both amplitudes, so a vacuum arm makes
    it vanish whatever :math:`\\gamma` is; borrowing the present arm's mode keeps
    :math:`\\gamma` well defined -- a zero width would be rejected -- without
    that value affecting the result. It also collapses the output-width rule to
    one line: the output always carries the short arm's ``sigma_s``, which on a
    flush is the long arm's, through here.
    """
    return ArmContribution(
        state=_VACUUM,
        bs2_tick=other.bs2_tick,
        sigma_s=other.sigma_s,
        wavelength_nm=other.wavelength_nm,
        pulse_index=None,
        signal_id=None,
    )


def _report_ready_time(*, report: DetectionReport, fallback_time: int) -> int:
    """Tick at which a slot decision is available to a consumer.

    The latest raw click when there is one, because that is when the last
    contributing channel actually fired; otherwise the detector-window
    completion tick, because a no-click report is only known once the windows
    have closed. Mirrors ``detector_array._report_ready_time``.
    """
    if report.raw_clicks:
        return max(click.time for click in report.raw_clicks)
    return fallback_time


def _incoming_pulse_index(signal: Signal) -> Optional[int]:
    """Return the source slot from a pulse's metadata, or ``None``."""
    for key, value in signal.meta:
        if key == "pulse_index":
            return value if isinstance(value, int) else None
    return None


@dataclass(slots=True)
class DelayInterferometer(Component):
    """Recombine each coherent pulse with its predecessor across a delay.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier used in event metadata, logs, emitted
        signal identifiers, and report identifiers.
    detectors : Sequence[SinglePhotonDetector] or None
        **Required, with no default.** Exactly two detector channels, index 0
        reading ``out_0`` and index 1 reading ``out_1``, or ``None`` for an
        optics-only device that emits pulses and decides no clicks.

        A DPS receiver is one physical unit -- interferometer and detectors
        thermally stabilised together -- and no protocol deploys the
        interferometer bare, so ``None`` should be something a caller says on
        purpose rather than something a default hands them. It is a supported
        and documented choice: it reproduces this component's behaviour before
        detection existed, exactly, including declaring no RNG stream, and it is
        what makes the optics testable against exact amplitudes and exact ticks
        rather than against click statistics.
    delay_s : float or None, default=None
        The **extra** time the long arm takes, :math:`\\tau`, in seconds. This
        is the only quantity the interference depends on: the common transit is
        ``short_delay_ticks`` and cancels out of the arms' separation. Exactly
        one of ``delay_s`` and ``delay_ticks`` must be supplied.
    delay_ticks : int or None, default=None
        The same :math:`\\tau` in simulation ticks, for callers that need
        tick-exact control. Exactly one of ``delay_s`` and ``delay_ticks`` must
        be supplied; there is no override precedence, because ``delay_s`` has no
        other role here and a silent winner would only be ambiguity.
    flush_priority : int, default=10000
        Event priority for the deadline on a held long arm. Timeline ordering is
        ``(time, priority, event_id)`` with lower first, so this must stay
        **strictly above** the priority upstream deliveries arrive with -- in
        practice ``QuantumChannel.delivery_priority``, which defaults to ``0``.
        Equality is worse than inversion: the tie falls through to ``event_id``,
        making the outcome depend on which event was scheduled first rather than
        on anything physical. The default gap is wide on purpose.
    short_delay_ticks : int, default=0
        Common transit both arms take, in simulation ticks. Stating it is more
        honest than an instantaneous short arm, and it changes nothing physical:
        both arms shift together, so the separation that sets :math:`\\gamma`
        stays :math:`\\tau` alone. ``0`` reproduces an instantaneous short arm
        exactly.
    click_resolver : ClickPatternResolver, optional
        Resolver turning the slot's raw clicks into one ``DetectionReport``.
        Defaults to ``ThresholdClickResolver()``, whose ``double_click_policy``
        is where a protocol's response to a double click belongs. The *rate* is
        physics and belongs to the two ports; do not conflate them.
    detection_window_ticks : int, default=1
        Positive detector observation window opened at each BS2 resolution.
    gate_model : GateModel, optional
        Gate schedule queried per exposure. Defaults to ``AlwaysOpenGate()``.
        A gated receiver must still be open at ``last_arrival + 2 * tau + 1``,
        where the final flush lands, or accept losing that slot.

    Attributes
    ----------
    input_port : Port
        Quantum input port named ``"in"``.
    output_port_0, output_port_1 : Port
        Quantum output ports named ``"out_0"`` and ``"out_1"``. Both must be
        connected: an ideal interferometer always puts light on both, and the
        destructive port carrying nearly nothing is a result, not an absence.
        They remain wired even with detectors fitted -- they are an inspection
        point, not a deployment boundary.
    report_port : Port
        Classical output port named ``"report"``, carrying ``InterferenceReport``.
    detection_port : Port
        Classical output port named ``"detection"``, carrying
        ``DetectionReport``. Separate from ``report_port`` because the two have
        different consumers and an agent that type-checks its inbox would reject
        a mixed stream.
    reports : list[InterferenceReport]
        Stored optics reports in resolution order.
    detection_reports : list[DetectionReport]
        One slot decision per combination, in the same order. Empty when
        ``detectors is None``.
    interference_count : int
        Number of BS2 combinations resolved so far.
    held_arm_count : int
        ``1`` while a long-arm contribution is waiting, ``0`` otherwise. Never
        more: each arrival releases the holder before refilling it.

    Raises
    ------
    ValueError
        At construction, if the delay is not given exactly once or resolves to
        fewer than one tick, or if ``detectors`` is neither ``None`` nor exactly
        two channels with distinct ids. At event time, if the arriving signal
        carries no ``coherent_state``, carries a ``state_ref``, or carries no
        ``temporal_mode_sigma_s``.

    Notes
    -----
    **Beamsplitter convention.** The real 50:50 matrix, stated once at the top of
    ``components/coherent_optics.py``. BS1 with vacuum on its second input gives
    :math:`\\alpha_s = \\alpha_\\ell = \\alpha/\\sqrt 2`; BS2 gives port 0 the
    difference and port 1 the sum, so port 0 is dark and port 1 bright when the
    two arms are in phase.

    **Event shape.** A pulse arrival does BS1 and, when both contributions have
    already reached BS2, BS2 as well, in one event. When the new pulse arrives
    *before* the held arm has reached BS2 the pair is deferred to
    ``ACTION_RESOLVE_BS2`` at the long arm's BS2 tick, which is reachable
    whenever the source uses a stochastic timing profile. Deferring releases the
    holder in the same step, so a pulse arriving while a combination is pending
    interacts only with the holder's new occupant.

    **Never schedule a self-event at delay 0.** ``Timeline.pop_batch`` dispatches
    the batch already queued at a tick, so an event scheduled *during* that batch
    joins a later one and runs after it **regardless of priority** -- silently
    escaping the ordering this component relies on. Neither self-scheduled action
    here can land on its own tick. See ``docs/dev/dps-design.md`` section 6.

    **The deadline is the nearest-neighbour assumption, not a decay estimate.**
    A held arm's deadline is ``arrival + 2 * tau``: one further slot opportunity
    has passed, so it is no longer a pair candidate. This is *not* a claim that
    :math:`\\gamma` has become negligible -- at :math:`\\sigma = \\tau` the
    discarded overlap is still about ``0.78``, which a test asserts. Energy is
    conserved either way.

    The flush fires one tick after that deadline, at ``arrival + 2 * tau + 1``,
    as ordering margin rather than physics, so the last tick on which a pulse
    still pairs is ``arrival + 2 * tau + 1``. See ``docs/dev/dps-design.md``
    section 6.

    **Outputs are new optical events** and get new identities: two amplitudes go
    in and two different optical modes come out, so there is no one signal to
    preserve. Both incoming pulse indices travel in the outgoing metadata and in
    the report.

    **Outputs are intensity-exact and mode-truncated.** At :math:`|\\gamma| < 1`
    the field leaving a port is a superposition of two non-identical envelopes.
    Each emitted signal carries the exact :math:`\\mu_k`, the phase of the
    interfering component, and the short arm's ``temporal_mode_sigma_s``;
    **none of the phase or width may be used for a further phase-sensitive or
    temporal-mode interference.** At :math:`|\\gamma| = 1` the output is
    genuinely single-mode and all three are exact. The internal detectors read
    intensity alone and are unaffected -- **but only because they read intensity
    alone.** Sampling a photon's arrival *within* the envelope would depend on
    the width this truncation approximates, and is not modelled; see
    ``CAPABILITY_MAP.md`` section 5.

    **All the randomness is the detectors'.** The optics sample nothing, so at
    ``detectors=None`` ``bind`` declares no RNG stream at all. Fitting detectors
    cannot perturb any other component's draws.

    A run must extend to ``last_arrival + 2 * tau + 1`` for the final flush, and
    to any outstanding pending resolution tick, or those combinations never
    execute and the last slots are missing from the reports.
    ``Timeline.run_until_empty`` does this by construction.

    Insertion loss, arm imbalance, non-ideal splitting ratio, thermal or
    mechanical drift of the arm lengths, and polarization mismatch between the
    arms are not modelled. Neither is arrival-time sampling within the pulse
    envelope, nor polarization-resolved detection. See ``CAPABILITY_MAP.md``
    section 5.
    """

    device_id: str
    # Required, and deliberately without a default: a DPS receiver is one
    # physical unit and there is no protocol that deploys the interferometer
    # bare, so `detectors=None` should be something a caller says on purpose.
    detectors: Optional[Sequence[SinglePhotonDetector]]
    delay_s: Optional[float] = None
    delay_ticks: Optional[int] = None
    flush_priority: int = 10_000

    # Appended after `flush_priority` rather than filed with the delays, so no
    # existing keyword's positional index moves.
    short_delay_ticks: int = 0
    click_resolver: ClickPatternResolver = field(default_factory=ThresholdClickResolver)
    detection_window_ticks: int = 1
    gate_model: GateModel = field(default_factory=AlwaysOpenGate)

    input_port: Port = field(init=False)
    output_port_0: Port = field(init=False)
    output_port_1: Port = field(init=False)
    report_port: Port = field(init=False)
    detection_port: Port = field(init=False)
    reports: list[InterferenceReport] = field(init=False, default_factory=list)
    detection_reports: list[DetectionReport] = field(init=False, default_factory=list)

    _resolved_delay_ticks: int = field(init=False)
    _bound_timeline_id: Optional[int] = field(init=False, default=None)
    _held: Optional[HeldLongArm] = field(init=False, default=None)
    _hold_counter: int = field(init=False, default=0)
    _interference_count: int = field(init=False, default=0)
    _detector_rngs: dict[str, DetectorRNGStreams] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _resolver_rng: Optional["DeterministicRNG"] = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")

        if type(self.flush_priority) is not int:
            raise TypeError("flush_priority must be int")

        self._resolved_delay_ticks = self._resolve_delay()
        self.short_delay_ticks = require_non_negative_int(
            self.short_delay_ticks,
            field_name="short_delay_ticks",
        )
        self.detectors = self._resolve_detectors()

        # Validated whether or not detection is configured. An inert wrong value
        # is still a wrong value, and this matches `DetectorArray`, which
        # validates its whole receiver configuration at construction.
        validate_gate_model(self.gate_model)
        if not callable(getattr(self.click_resolver, "resolve", None)):
            raise TypeError("click_resolver must provide resolve(...)")
        if type(self.detection_window_ticks) is not int:
            raise TypeError("detection_window_ticks must be int")
        if self.detection_window_ticks <= 0:
            raise ValueError("detection_window_ticks must be positive")

        # Ports are built directly rather than through
        # `sources/_common.quantum_output_port`, which is private to the sources
        # package. `BellStateAnalyzer` builds its four ports the same way.
        self.input_port = self._port("in", PortKind.QUANTUM, PortDirection.INGRESS)
        self.output_port_0 = self._port(
            PORT_OUT_0,
            PortKind.QUANTUM,
            PortDirection.EGRESS,
        )
        self.output_port_1 = self._port(
            PORT_OUT_1,
            PortKind.QUANTUM,
            PortDirection.EGRESS,
        )
        self.report_port = self._port(
            "report",
            PortKind.CLASSICAL,
            PortDirection.EGRESS,
        )
        self.detection_port = self._port(
            PORT_DETECTION,
            PortKind.CLASSICAL,
            PortDirection.EGRESS,
        )

    def _resolve_detectors(self) -> Optional[tuple[SinglePhotonDetector, ...]]:
        """Validate the detector pair, or pass ``None`` through untouched."""
        if self.detectors is None:
            return None

        normalized = normalize_detectors(self.detectors, require_non_empty=True)

        if len(normalized) != 2:
            raise ValueError(
                f"delay interferometer {self.device_id!r} takes exactly two "
                f"detectors and got {len(normalized)}: BS2 has two output ports "
                "and each needs its own channel, index 0 reading "
                f"{PORT_OUT_0!r} and index 1 reading {PORT_OUT_1!r}; pass "
                "detectors=None for an optics-only device that emits pulses and "
                "decides no clicks"
            )

        return normalized

    def _resolve_delay(self) -> int:
        """Resolve tau to a whole number of ticks, at least one."""
        delay_s = require_optional_positive_real(self.delay_s, field_name="delay_s")

        if (delay_s is None) == (self.delay_ticks is None):
            raise ValueError(
                f"delay interferometer {self.device_id!r} needs exactly one of "
                "delay_s and delay_ticks: they describe the same long-arm "
                "delay, and neither takes precedence over the other; supply "
                "delay_s for a physical configuration or delay_ticks for "
                "tick-exact control"
            )

        if self.delay_ticks is not None:
            return require_positive_int(self.delay_ticks, field_name="delay_ticks")

        assert delay_s is not None
        resolved = int(seconds_to_ticks(delay_s))
        if resolved < 1:
            raise ValueError(
                f"delay interferometer {self.device_id!r} has delay_s="
                f"{delay_s!r}, which rounds to {resolved} ticks at the "
                "repository resolution of 1 tick == 1 ps: a zero-delay "
                "unbalanced interferometer is a balanced one, in which every "
                "pulse would interfere only with itself; use a delay of at "
                "least 1e-12 s, or pass delay_ticks directly"
            )
        return resolved

    def _port(self, name: str, kind: PortKind, direction: PortDirection) -> Port:
        return Port(
            name=name,
            owner=self,
            owner_id=self.device_id,
            port_kind=kind,
            direction=direction,
        )

    @property
    def resolved_delay_ticks(self) -> int:
        """Long-arm delay :math:`\\tau` in simulation ticks."""
        return self._resolved_delay_ticks

    @property
    def interference_count(self) -> int:
        """Number of BS2 combinations resolved so far."""
        return self._interference_count

    @property
    def held_arm_count(self) -> int:
        """``1`` while a long arm is waiting for a partner, else ``0``."""
        return 0 if self._held is None else 1

    def bind(self, context: BindingContext) -> None:
        """Bind to a timeline, declaring streams only if detectors are fitted.

        Parameters
        ----------
        context : BindingContext
            Runtime binding context that supplies the timeline.

        Raises
        ------
        TypeError
            If ``context`` is not a ``BindingContext``.
        RuntimeError
            If already bound to a different timeline.

        Notes
        -----
        Binding is idempotent for the same timeline.

        **The optics declare nothing.** BS1, the delay and BS2 are ideal by
        specification, so at ``detectors=None`` this component still requests no
        stream at all; a declared-but-never-consumed stream would be dead
        configuration and a lie in the binding log.

        With detectors fitted the sampling is theirs, not the optics'. Four
        streams per channel come from ``bind_detector_rngs`` on the four-segment
        path ``(device_id, "delay_interferometer", detector_id, role)`` -- the
        detector id is the segment that stops two identical channels sharing --
        plus one ``"resolver"`` stream, which only a ``"random"`` double-click
        policy ever draws from. They are declared eagerly because
        ``Timeline.rng`` refuses a new stream once execution has begun.

        Because stream values derive from the stream path rather than from
        creation order or stream count, fitting detectors cannot perturb any
        other component's draws.
        """
        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)

        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "delay interferometer is already bound to another timeline"
                )
            return

        if self.detectors is not None:
            self._detector_rngs = bind_detector_rngs(
                timeline=timeline,
                device_id=self.device_id,
                namespace=COMPONENT_KEY,
                detectors=tuple(self.detectors),
            )
            self._resolver_rng = timeline.rng(
                self.device_id,
                COMPONENT_KEY,
                "resolver",
            )

        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.interferometers.delay_interferometer.ready",
            "delay interferometer ready",
            meta={
                "device_id": self.device_id,
                "delay_ticks": self._resolved_delay_ticks,
                "delay_s": ticks_to_seconds(self._resolved_delay_ticks),
                "short_delay_ticks": self.short_delay_ticks,
                "flush_priority": self.flush_priority,
                "detector_count": (
                    0 if self.detectors is None else len(self.detectors)
                ),
            },
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """Handle a pulse arrival, a deferred combination, or a flush deadline.

        Parameters
        ----------
        event : Event
            ``ACTION_INTERFERE`` with a ``PortDelivery`` payload,
            ``ACTION_RESOLVE_BS2`` with a ``PendingCombination``, or
            ``ACTION_FLUSH_DELAY_ARM`` with a ``DelayArmFlush``.
        timeline : Timeline
            Timeline currently executing the event.

        Raises
        ------
        RuntimeError
            If the interferometer has not been bound before execution.
        TypeError
            If a payload has the wrong type.
        ValueError
            If the action is unsupported or the delivery targets another port.
        """
        if self._bound_timeline_id is None:
            raise RuntimeError("delay interferometer must be bound before execution")

        if event.action == ACTION_INTERFERE:
            if not isinstance(event.payload_ref, PortDelivery):
                raise TypeError("ACTION_INTERFERE payload_ref must be PortDelivery")

            delivery = event.payload_ref
            if delivery.target_port is not self.input_port:
                raise ValueError(
                    "delay interferometer received delivery on unknown port"
                )
            if not isinstance(delivery.payload, Signal):
                raise TypeError("delivery payload must be Signal")

            self._handle_arrival(
                timeline,
                signal=delivery.payload,
                event_id=event.event_id,
                action=event.action,
            )
            return

        if event.action == ACTION_RESOLVE_BS2:
            if not isinstance(event.payload_ref, PendingCombination):
                raise TypeError(
                    "ACTION_RESOLVE_BS2 payload_ref must be PendingCombination"
                )

            self._resolve(
                timeline,
                short=event.payload_ref.short,
                long_=event.payload_ref.long_,
                event_id=event.event_id,
                action=event.action,
            )
            return

        if event.action == ACTION_FLUSH_DELAY_ARM:
            if not isinstance(event.payload_ref, DelayArmFlush):
                raise TypeError(
                    "ACTION_FLUSH_DELAY_ARM payload_ref must be DelayArmFlush"
                )

            self._handle_flush(
                timeline,
                flush=event.payload_ref,
                event_id=event.event_id,
                action=event.action,
            )
            return

        raise ValueError(
            f"unsupported event action for delay interferometer: {event.action!r}"
        )

    def _handle_arrival(
        self,
        timeline: Timeline,
        *,
        signal: Signal,
        event_id: Optional[int],
        action: str,
    ) -> None:
        """Split one pulse at BS1 and combine or defer at BS2."""
        incoming, sigma_s = self._require_interferable(signal)

        arrival_tick = timeline.current_time
        pulse_index = _incoming_pulse_index(signal)
        short_state, long_state = split_50_50(incoming)

        # Both arms take the common transit; only the long arm also takes tau.
        # Every difference the physics depends on is therefore tau alone, and a
        # short_delay_ticks of 0 reproduces an instantaneous short arm exactly.
        short_bs2_tick = arrival_tick + self.short_delay_ticks
        long_bs2_tick = short_bs2_tick + self._resolved_delay_ticks

        short = ArmContribution(
            state=short_state,
            bs2_tick=short_bs2_tick,
            sigma_s=sigma_s,
            wavelength_nm=float(signal.wavelength_nm),
            pulse_index=pulse_index,
            signal_id=str(signal.id),
        )

        # Take the holder before anything else. Whichever way the combination
        # goes, the holder is free for this pulse's own long arm by the end of
        # this event, which is what makes a pulse arriving mid-flight a
        # non-event for the pair already in progress.
        held, self._held = self._held, None
        partner = vacuum_like(short) if held is None else held.arm

        # BS2 acts when the *later* arm gets there, which at
        # short_delay_ticks == 0 is the held arm and nothing else.
        resolve_tick = max(short.bs2_tick, partner.bs2_tick)

        if resolve_tick <= arrival_tick:
            self._resolve(
                timeline,
                short=short,
                long_=partner,
                event_id=event_id,
                action=action,
            )
        else:
            self._defer(timeline, short=short, long_=partner, at_tick=resolve_tick)

        self._hold_counter += 1
        self._held = HeldLongArm(
            hold_id=self._hold_counter,
            arm=ArmContribution(
                state=long_state,
                bs2_tick=long_bs2_tick,
                sigma_s=sigma_s,
                wavelength_nm=float(signal.wavelength_nm),
                pulse_index=pulse_index,
                signal_id=str(signal.id),
            ),
        )
        self._schedule_flush(timeline, arrival_tick=arrival_tick)

    def _require_interferable(self, signal: Signal) -> tuple[CoherentState, float]:
        """Reject what this device cannot interfere, before it does anything.

        One rule, three cases, the same message shape as
        ``QuantumChannel._require_coherent_transport_supported``: the device, the
        offending field and its value, ``:`` and why it does not work here, then
        ``;`` and what to change.

        This is a transform, not transport. A transform component's whole purpose
        is the transformation, so passing an uninterferable signal through would
        make the device a silent no-op for exactly the wiring mistake it should
        catch.
        """
        if signal.coherent_state is None:
            raise ValueError(
                f"delay interferometer {self.device_id!r} cannot interfere "
                f"signal {signal.id!r} with coherent_state=None: this device "
                "combines optical amplitudes and a signal without one carries "
                "nothing to combine; send a coherent pulse, from a source such "
                "as WeakCoherentPulseSource"
            )

        if signal.state_ref is not None:
            raise ValueError(
                f"delay interferometer {self.device_id!r} cannot interfere "
                f"signal {signal.id!r} with state_ref={signal.state_ref!r}: the "
                "amplitude is splittable but the qstate record travelling with "
                "it is not, and this device builds its outputs with "
                "state_ref=None, so interfering it would silently strand that "
                "record; send a pulse with no qstate record, such as one from "
                "WeakCoherentPulseSource configured without a polarization "
                "selector"
            )

        if signal.temporal_mode_sigma_s is None:
            raise ValueError(
                f"delay interferometer {self.device_id!r} cannot interfere "
                f"signal {signal.id!r} with temporal_mode_sigma_s=None: how "
                "much two contributions interfere depends on how much their "
                "envelopes overlap, and an amplitude alone does not say when "
                "the light is; set temporal_mode_sigma_s on the source, for "
                "example WeakCoherentPulseSource(temporal_mode_sigma_s=...)"
            )

        return signal.coherent_state, signal.temporal_mode_sigma_s

    def _defer(
        self,
        timeline: Timeline,
        *,
        short: ArmContribution,
        long_: ArmContribution,
        at_tick: int,
    ) -> None:
        """Schedule a combination whose arms have not both reached BS2 yet.

        ``at_tick`` is the later of the two BS2 ticks. Only reached when it
        exceeds ``timeline.current_time`` strictly, so this is never a delay-0
        self-schedule -- see the class notes for why that matters.

        No log record of its own. The scheduled event carries both signal ids and
        both BS2 ticks in its own ``meta``, and an event trace already shows it
        being queued here and executed at ``long_.bs2_tick``; a separate record
        would only restate that. ``_schedule_flush`` queues a comparable
        self-event the same way.
        """
        timeline.schedule(
            Event(
                time=at_tick,
                priority=0,
                target_ref=self,
                action=ACTION_RESOLVE_BS2,
                payload_ref=PendingCombination(short=short, long_=long_),
                source=self,
                subsystem_id="components",
                meta={
                    "device_id": self.device_id,
                    "short_signal_id": short.signal_id,
                    "long_signal_id": long_.signal_id,
                    "short_bs2_tick": short.bs2_tick,
                    "long_bs2_tick": long_.bs2_tick,
                },
            )
        )

    def _schedule_flush(self, timeline: Timeline, *, arrival_tick: int) -> None:
        """Set the nearest-neighbour deadline on the newly held long arm.

        The deadline is ``arrival + 2 * tau``; the event is scheduled one tick
        later so that a pulse landing on the deadline is ordered first by
        ``time`` rather than only by ``flush_priority``. See the class notes.
        """
        assert self._held is not None
        flush_tick = (
            arrival_tick + 2 * self._resolved_delay_ticks + _FLUSH_ORDERING_MARGIN_TICKS
        )

        timeline.schedule(
            Event(
                time=flush_tick,
                priority=self.flush_priority,
                target_ref=self,
                action=ACTION_FLUSH_DELAY_ARM,
                payload_ref=DelayArmFlush(hold_id=self._held.hold_id),
                source=self,
                subsystem_id="components",
                meta={
                    "device_id": self.device_id,
                    "hold_id": self._held.hold_id,
                    "signal_id": self._held.arm.signal_id,
                    "deadline_tick": flush_tick - _FLUSH_ORDERING_MARGIN_TICKS,
                },
            )
        )

    def _handle_flush(
        self,
        timeline: Timeline,
        *,
        flush: DelayArmFlush,
        event_id: Optional[int],
        action: str,
    ) -> None:
        """Combine an unpaired long arm with vacuum, if it is still held.

        A deadline whose ``hold_id`` no longer matches the holder is stale: the
        arm was taken by a later pulse, or the holder has been refilled. Both are
        normal, so it returns quietly.

        **The generation check is the point, not the emptiness check.** By the
        time pulse *k*'s deadline fires, pulse *k+1* has normally already taken
        that arm and left its own in the holder -- so the holder is *present and
        different*, and a guard that only tested ``self._held is None`` would
        flush pulse *k+1*'s arm against vacuum and destroy a real pairing.
        """
        held = self._held
        if held is None or held.hold_id != flush.hold_id:
            return

        self._held = None
        self._resolve(
            timeline,
            short=vacuum_like(held.arm),
            long_=held.arm,
            event_id=event_id,
            action=action,
        )

    def _resolve(
        self,
        timeline: Timeline,
        *,
        short: ArmContribution,
        long_: ArmContribution,
        event_id: Optional[int],
        action: str,
    ) -> None:
        """Run BS2 on two contributions and emit on both output ports."""
        connection_0 = require_connection(self.output_port_0)
        connection_1 = require_connection(self.output_port_1)

        delta_ticks = abs(short.bs2_tick - long_.bs2_tick)
        overlap = gaussian_temporal_overlap(
            sigma_a_s=short.sigma_s,
            sigma_b_s=long_.sigma_s,
            delta_s=ticks_to_seconds(delta_ticks),
        )
        out_0, out_1 = interfere(short.state, long_.state, overlap=overlap)

        self._interference_count += 1
        index = self._interference_count

        # Built in one loop rather than twice by hand: the two ports differ in
        # three arguments and agree on six, and two call sites could drift.
        signal_0, signal_1 = (
            self._make_output_signal(
                timeline,
                port_name=port_name,
                port_index=port_index,
                state=state,
                short=short,
                long_=long_,
                overlap=overlap,
                delta_ticks=delta_ticks,
                index=index,
            )
            for port_name, port_index, state in (
                (PORT_OUT_0, 0, out_0),
                (PORT_OUT_1, 1, out_1),
            )
        )

        mean_in = short.state.mean_photon_number + long_.state.mean_photon_number

        timeline.log(
            LogLevel.DEBUG,
            "components.interferometers.delay_interferometer.interfered",
            "pulses combined at bs2",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "interference_index": index,
                "short_signal_id": short.signal_id,
                "long_signal_id": long_.signal_id,
                "short_pulse_index": short.pulse_index,
                "long_pulse_index": long_.pulse_index,
                "short_bs2_tick": short.bs2_tick,
                "long_bs2_tick": long_.bs2_tick,
                "delta_ticks": delta_ticks,
                "temporal_overlap": overlap,
                # Where a tau/period mismatch becomes visible: the overlap
                # collapses on every slot and these two stop being
                # complementary. The device deliberately does not validate tau
                # against the pulse period, so this record is the only place a
                # mismatch shows.
                "mean_photon_number_in": mean_in,
                "mean_photon_number_0": out_0.mean_photon_number,
                "mean_photon_number_1": out_1.mean_photon_number,
            },
        )

        self._store_report(
            timeline,
            InterferenceReport(
                report_id=f"{self.device_id}:bs2:{index}",
                device_id=self.device_id,
                time=timeline.current_time,
                interference_index=index,
                signal_ids=tuple(
                    signal_id
                    for signal_id in (short.signal_id, long_.signal_id)
                    if signal_id is not None
                ),
                output_signal_ids=(str(signal_0.id), str(signal_1.id)),
                short_pulse_index=short.pulse_index,
                long_pulse_index=long_.pulse_index,
                temporal_overlap=overlap,
                delta_ticks=delta_ticks,
                short_bs2_tick=short.bs2_tick,
                long_bs2_tick=long_.bs2_tick,
                mean_photon_number_in=mean_in,
                mean_photon_number_0=out_0.mean_photon_number,
                mean_photon_number_1=out_1.mean_photon_number,
            ),
        )

        self._detect(
            timeline,
            short=short,
            long_=long_,
            out_0=out_0,
            out_1=out_1,
            overlap=overlap,
            index=index,
            signal_ids=(str(signal_0.id), str(signal_1.id)),
            event_id=event_id,
            action=action,
        )

        for connection, signal in (
            (connection_0, signal_0),
            (connection_1, signal_1),
        ):
            connection.transmit(
                signal,
                timeline,
                time=timeline.current_time,
                source=self,
                subsystem_id="components",
                meta={
                    "device_id": self.device_id,
                    "output_port": connection.source_port.name,
                    "signal_id": signal.id,
                    "interference_index": index,
                },
            )

    def _detect(
        self,
        timeline: Timeline,
        *,
        short: ArmContribution,
        long_: ArmContribution,
        out_0: CoherentState,
        out_1: CoherentState,
        overlap: float,
        index: int,
        signal_ids: tuple[str, str],
        event_id: Optional[int],
        action: str,
    ) -> None:
        """Turn the two BS2 amplitudes into one slot decision.

        Notes
        -----
        **Both ports are evaluated independently, and that is the point.** Each
        gets its own Bernoulli trial against its own
        :math:`1 - e^{-\\eta_d\\mu_k}`, so a slot with
        :math:`\\mu_0 = 0.19` and :math:`\\mu_1 = 0.01` clicks on port 0 about
        17% of the time, on port 1 about 1%, and on both about 0.17% of the
        time. That last number is a genuine double click, present at every
        :math:`\\mu`, and what the slot then *reports* is the resolver's
        double-click policy rather than physics.

        **The detector's efficiency is applied once.**
        ``click_probability`` is handed ``params.efficiency`` and returns a
        probability with it already in the exponent;
        ``signal_click_probability`` then *replaces* that efficiency inside the
        detector rather than multiplying it. Dark counts and afterpulses still
        read ``params`` on their own paths, which is why the override is of one
        term and not of the parameter record.

        **Vacuum slots are not special-cased.** The first pulse's short arm and
        the flushed long arm each meet vacuum and split ``mu/2`` both ways, so
        both ports carry equal probability and the slot cannot hold a bit. It is
        an ordinary pulse to a detector and gets an ordinary report; dropping it
        is the agent's job, on the ``None`` pulse index in the report metadata.
        """
        detectors = self.detectors
        if detectors is None:
            return

        time = timeline.current_time

        exposures = tuple(
            DetectorExposure(
                detector_id=detector.detector_id,
                signal_present=True,
                outcome_label=port_name,
                signal_click_probability=click_probability(
                    state.mean_photon_number,
                    efficiency=detector.params.efficiency,
                ),
            )
            for detector, port_name, state in (
                (detectors[0], PORT_OUT_0, out_0),
                (detectors[1], PORT_OUT_1, out_1),
            )
        )

        raw_clicks, detection_complete_time = evaluate_detector_windows(
            device_id=self.device_id,
            time=time,
            detectors=tuple(detectors),
            exposures=exposures,
            detector_rngs=self._detector_rngs,
            detection_window_ticks=self.detection_window_ticks,
            gate_model=self.gate_model,
            # No measurement was run and none could be: a threshold detector
            # reads intensity and there is no basis to name.
            measurement_label=None,
            fallback_complete_time=time,
        )

        report = self.click_resolver.resolve(
            device_id=self.device_id,
            time=time,
            # Two signals left BS2 and neither is "the" measured one, so the
            # report carries no signal id and joins to the InterferenceReport on
            # `interference_index` instead.
            signal=None,
            qstate_result=None,
            measurement_call=None,
            raw_clicks=raw_clicks,
            rng=self._resolver_rng,
        )

        report = replace(
            report,
            meta=report.meta
            + (
                ("interference_index", index),
                ("short_pulse_index", short.pulse_index),
                ("long_pulse_index", long_.pulse_index),
                ("temporal_overlap", overlap),
                ("mean_photon_number_0", out_0.mean_photon_number),
                ("mean_photon_number_1", out_1.mean_photon_number),
                # Derived floats, never the raw alpha: the JSONL sink has no
                # complex case and would fall back to repr().
                ("phase_rad_0", out_0.phase_rad),
                ("phase_rad_1", out_1.phase_rad),
                ("output_signal_ids", signal_ids),
            ),
        )

        self._store_detection_report(
            timeline,
            report=report,
            fallback_time=detection_complete_time,
            event_id=event_id,
            action=action,
        )

    def _store_detection_report(
        self,
        timeline: Timeline,
        *,
        report: DetectionReport,
        fallback_time: int,
        event_id: Optional[int],
        action: str,
    ) -> None:
        """Record one slot decision and emit it when the port is wired.

        Follows ``DetectorArray._store_report`` and its emission timing: a
        report is ready at the latest raw click it contains, or at the
        detector-window completion tick when it contains none.
        """
        self.detection_reports.append(report)

        timeline.log(
            LogLevel.DEBUG if report.success else LogLevel.TRACE,
            "components.interferometers.delay_interferometer.detected",
            "slot detected",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "report_id": report.report_id,
                "success": report.success,
                "outcome": report.outcome,
                "click_count": len(report.raw_clicks),
                "flags": report.flags,
                **dict(report.meta),
            },
        )

        connection = self.detection_port.connection
        if connection is None:
            return

        connection.transmit(
            report,
            timeline,
            time=max(
                timeline.current_time,
                _report_ready_time(report=report, fallback_time=fallback_time),
            ),
            source=self,
            subsystem_id="components",
            meta={
                "device_id": self.device_id,
                "output_port": self.detection_port.name,
                "report_id": report.report_id,
                "report_kind": "detection",
            },
        )

    def _make_output_signal(
        self,
        timeline: Timeline,
        *,
        port_name: str,
        port_index: int,
        state: CoherentState,
        short: ArmContribution,
        long_: ArmContribution,
        overlap: float,
        delta_ticks: int,
        index: int,
    ) -> Signal:
        """Build one output-port signal.

        A fresh ``Signal`` rather than a derived one: two amplitudes entered and
        two different optical modes leave, so there is no single incoming
        identity to preserve. This follows
        ``QuantumMemory._make_emitted_signal`` rather than an in-flight
        transform.

        ``temporal_mode_sigma_s`` is the short arm's -- which, through
        ``vacuum_like``, is the long arm's on a flush. It is exact whenever the
        two arms share a mode, and a stated truncation when they do not; see the
        class notes. ``emission_time`` is the resolve tick, which by the
        ``Signal`` contract is this output's own envelope centre.
        """
        return Signal(
            id=f"{self.device_id}:bs2:{index}:{port_index}",
            signal_kind=SignalKind.PULSE,
            encoding_scheme=EncodingScheme.PHASE,
            emission_time=timeline.current_time,
            origin=self.device_id,
            wavelength_nm=short.wavelength_nm,
            state_ref=None,
            coherent_state=state,
            temporal_mode_sigma_s=short.sigma_s,
            meta=(
                ("delay_interferometer_id", self.device_id),
                ("interference_index", index),
                ("output_port", port_name),
                ("short_pulse_index", short.pulse_index),
                ("long_pulse_index", long_.pulse_index),
                ("short_signal_id", short.signal_id),
                ("long_signal_id", long_.signal_id),
                ("temporal_overlap", overlap),
            ),
            timing_meta=(
                ("time_unit", "ps"),
                ("interferometer_delay_ticks", self._resolved_delay_ticks),
                ("short_bs2_tick", short.bs2_tick),
                ("long_bs2_tick", long_.bs2_tick),
                ("delta_ticks", delta_ticks),
                ("resolve_tick", timeline.current_time),
            ),
            # Trusted path: every field comes from validated component
            # configuration or from an already-validated incoming signal.
            validation_flag=False,
        )

    def _store_report(self, timeline: Timeline, report: InterferenceReport) -> None:
        """Record a report and emit it when the report port is wired.

        Mirrors ``sources.reports.store_source_report`` in shape rather than
        reusing it: that helper stamps ``report_kind="source_preparation"``,
        which would misdescribe this device, and its ``reports`` list is typed to
        the source report union.
        """
        self.reports.append(report)

        connection = self.report_port.connection
        if connection is None:
            return

        connection.transmit(
            report,
            timeline,
            time=timeline.current_time,
            source=self,
            subsystem_id="components",
            meta={
                "device_id": report.device_id,
                "report_id": report.report_id,
                "report_kind": "interference",
            },
        )


__all__ = [
    "ACTION_FLUSH_DELAY_ARM",
    "ACTION_INTERFERE",
    "ACTION_RESOLVE_BS2",
    "COMPONENT_KEY",
    "PORT_OUT_0",
    "PORT_OUT_1",
    "ArmContribution",
    "DelayArmFlush",
    "DelayInterferometer",
    "HeldLongArm",
    "InterferenceReport",
    "PendingCombination",
    "vacuum_like",
]
