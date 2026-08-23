"""Ideal unbalanced Mach-Zehnder interferometer for coherent pulses.

``DelayInterferometer`` splits each arriving pulse across a 50:50 beamsplitter,
delays one arm by :math:`\\tau`, and recombines it on a second 50:50
beamsplitter with the *next* pulse's undelayed arm. In DPS-QKD
:math:`\\tau = T_{\\mathrm{pulse}}`, so the recombination compares the optical
phases of two adjacent slots and the phase difference appears as an intensity
difference between the two output ports.

**The device is ideal.** No internal loss, no internal phase noise, no imperfect
splitting ratio, no photon-number sampling, no detector behaviour. Every
imperfection must arrive with the incoming pulses; loss and phase noise already
come from the channel. What the component adds is the delay, the recombination,
and the temporal-mode bookkeeping that decides how much the two contributions
actually overlap.

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.units import seconds_to_ticks, ticks_to_seconds
from simyuj.primitives.validation import (
    require_optional_positive_real,
    require_positive_int,
)
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel

from ..coherent_optics import gaussian_temporal_overlap, interfere, split_50_50
from ..connections import PortDelivery, require_connection
from ..ports import Port, PortDirection, PortKind

if TYPE_CHECKING:
    from simyuj.engine.timeline import Timeline


COMPONENT_KEY = "delay_interferometer"

# The action names live here, in the component module, rather than in a
# package-level constants module. `detectors/primitives/actions.py` exists
# because several detector components share a vocabulary -- DetectorArray,
# QubitReadoutDevice and BellStateAnalyzer all draw from it. This package holds
# one component, so a separate constants module would be an empty layer
# indirecting a single reader. `sources/` shows the shape of the eventual move:
# ACTION_EMIT and ACTION_START sit in `sources/_common.py` because two sources
# share them. When a second interferometer arrives, these move to an
# `interferometers/_common.py` the same way -- a rename, not a redesign.
ACTION_INTERFERE = "interfere"
"""Ingress action: one coherent pulse arriving at BS1."""

ACTION_RESOLVE_BS2 = "resolve_bs2"
"""Self-scheduled action: a combination deferred until both arms reach BS2."""

ACTION_FLUSH_DELAY_ARM = "flush_delay_arm"
"""Self-scheduled action: a held long arm that never found a partner."""

PORT_OUT_0 = "out_0"
PORT_OUT_1 = "out_1"

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
        the short arm that is the pulse's arrival tick; for the long arm it is
        the arrival tick plus :math:`\\tau`.
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
    Like the coherent source's report there is no ``state_ref``, no
    ``state_targets``, and no sampler field: interfering two coherent amplitudes
    creates no quantum state record.

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
    delay_s : float or None, default=None
        Long-arm delay :math:`\\tau` in seconds. Exactly one of ``delay_s`` and
        ``delay_ticks`` must be supplied.
    delay_ticks : int or None, default=None
        Long-arm delay in simulation ticks, for callers that need tick-exact
        control. Exactly one of ``delay_s`` and ``delay_ticks`` must be
        supplied; there is no override precedence, because ``delay_s`` has no
        other role here and a silent winner would only be ambiguity.
    flush_priority : int, default=10000
        Event priority for the deadline on a held long arm. Timeline ordering is
        ``(time, priority, event_id)`` with lower first, so this must stay
        **strictly above** the priority upstream deliveries arrive with -- in
        practice ``QuantumChannel.delivery_priority``, which defaults to ``0``.
        Equality is worse than inversion: the tie falls through to ``event_id``,
        making the outcome depend on which event was scheduled first rather than
        on anything physical. The default gap is wide on purpose.

    Attributes
    ----------
    input_port : Port
        Quantum input port named ``"in"``.
    output_port_0, output_port_1 : Port
        Quantum output ports named ``"out_0"`` and ``"out_1"``. Both must be
        connected: an ideal interferometer always puts light on both, and the
        destructive port carrying nearly nothing is a result, not an absence.
    report_port : Port
        Classical output port named ``"report"``.
    reports : list[InterferenceReport]
        Stored reports in resolution order.
    interference_count : int
        Number of BS2 combinations resolved so far.
    held_arm_count : int
        ``1`` while a long-arm contribution is waiting, ``0`` otherwise. Never
        more: each arrival releases the holder before refilling it.

    Raises
    ------
    ValueError
        At construction, if the delay is not given exactly once or resolves to
        fewer than one tick. At event time, if the arriving signal carries no
        ``coherent_state``, carries a ``state_ref``, or carries no
        ``temporal_mode_sigma_s``.

    Notes
    -----
    **Beamsplitter convention.** The real 50:50 matrix, stated once at the top of
    ``components/coherent_optics.py`` and used everywhere including the tests.
    BS1 with vacuum on its second input gives
    :math:`\\alpha_s = \\alpha_\\ell = \\alpha/\\sqrt 2`; BS2 gives port 0 the
    difference and port 1 the sum, so port 0 is dark and port 1 bright when the
    two arms are in phase.

    **One equation, no decision tree.** Every combination goes through the same
    :func:`~simyuj.components.coherent_optics.interfere` call. Vacuum inputs,
    orthogonal modes, unequal amplitudes, first pulse, and last pulse are all
    values of that equation rather than branches around it.

    **Event shape.** A pulse arrival does BS1 and, when both contributions have
    already reached BS2, BS2 as well -- in one event, which is what keeps the
    :math:`\\tau = T` case free of any same-tick ordering question, since the
    short arm of pulse *k* and the long arm of pulse *k-1* land on the same tick
    by design. When the new pulse arrives *before* the held arm has reached BS2,
    resolving immediately would emit light that has not yet arrived, so the pair
    is deferred to ``ACTION_RESOLVE_BS2`` at the long arm's BS2 tick. That is
    reachable whenever the source uses a stochastic timing profile, which
    shortens the spacing below :math:`\\tau`.

    Deferring hands the pair to the scheduled event and releases the holder in
    the same step, so a pulse arriving while a combination is pending interacts
    only with the holder's new occupant and cannot disturb it.

    **Never schedule a self-event at delay 0.** Neither self-scheduled action
    here can land on the tick that scheduled it, and that is a correctness
    requirement rather than an accident. ``Timeline.pop_batch`` collects the
    events already queued at a tick and dispatches that batch; an event
    scheduled *during* the batch, even at the same tick, joins a **later** batch
    and therefore runs after everything in the current one **regardless of
    priority**. A delay-0 self-event would silently escape the priority ordering
    this component relies on. The deferral is safe because it is only reached
    when ``partner.bs2_tick > arrival_tick``, strictly; the flush is safe
    because :math:`\\tau \\ge 1` tick makes ``arrival + 2\\tau + 1`` at least
    three ticks away.

    **The deadline is the nearest-neighbour assumption, not a decay estimate.**
    A held arm's deadline is ``arrival + 2 * tau``: one further slot opportunity
    has passed, so it is no longer a pair candidate. This is *not* a claim that
    :math:`\\gamma` has become negligible -- at :math:`\\sigma = \\tau` the
    discarded overlap is still about ``0.78``, which a test asserts so the
    assumption stays visible in the suite. Energy is conserved either way.

    **The flush event fires one tick after that deadline**, at
    ``arrival + 2 * tau + 1``. That extra tick is **ordering margin, not
    physics**. Priority alone would be sufficient today -- both contenders are
    queued long before their batch is popped, so ``(time, priority, event_id)``
    decides and ``flush_priority`` wins it. But ``flush_priority`` is only
    meaningful *relative to* ``QuantumChannel.delivery_priority``, configured on
    a different component; someone raising that for an unrelated reason would
    silently invert this device's pairing and the failure would surface as a
    wrong key rather than an error. Ordering on ``time`` is the one thing they
    cannot reconfigure. The observable consequence is that the last tick on
    which a pulse still pairs is ``arrival + 2 * tau + 1``, not
    ``arrival + 2 * tau``: a pulse landing exactly on the flush tick is
    processed first, by priority, and pairs at :math:`\\Delta t = \\tau + 1`
    ticks. One tick later it does not, and both it and the flushed arm meet
    vacuum in separate combinations.

    **Outputs are new optical events**, so they get new identities rather than
    inheriting a pulse's. Two amplitudes go in and two different optical modes
    come out; there is no one signal to preserve. Both incoming pulse indices
    travel in the outgoing metadata and in the report instead. This follows
    ``QuantumMemory._make_emitted_signal`` rather than an in-flight transform.

    **Outputs are intensity-exact and mode-truncated.** At :math:`|\\gamma| < 1`
    the field leaving a port is a superposition of two non-identical envelopes.
    Each emitted signal carries the exact :math:`\\mu_k`, the phase of the
    interfering component, and the short arm's ``temporal_mode_sigma_s``;
    **none of the phase or width may be used for a further phase-sensitive or
    temporal-mode interference.** At :math:`|\\gamma| = 1` the output is
    genuinely single-mode and all three are exact. A threshold detector reading
    intensity, which is what follows this device, is unaffected.

    **No randomness.** The device is ideal by specification, so ``bind``
    declares no RNG streams at all rather than declaring one that is never
    consumed. Stream values derive from the stream path, so adding one later
    would leave every existing replay untouched.

    A run must extend to ``last_arrival + 2 * tau + 1`` for the final flush, and
    to any outstanding pending resolution tick, or those combinations never
    execute and the last slots are missing from the reports.
    ``Timeline.run_until_empty`` does this by construction.

    Insertion loss, arm imbalance, non-ideal splitting ratio, thermal or
    mechanical drift of the arm lengths, and polarization mismatch between the
    arms are not modelled. See ``CAPABILITY_MAP.md`` section 5.
    """

    device_id: str
    delay_s: Optional[float] = None
    delay_ticks: Optional[int] = None
    flush_priority: int = 10_000

    input_port: Port = field(init=False)
    output_port_0: Port = field(init=False)
    output_port_1: Port = field(init=False)
    report_port: Port = field(init=False)
    reports: list[InterferenceReport] = field(init=False, default_factory=list)

    _resolved_delay_ticks: int = field(init=False)
    _bound_timeline_id: Optional[int] = field(init=False, default=None)
    _held: Optional[HeldLongArm] = field(init=False, default=None)
    _hold_counter: int = field(init=False, default=0)
    _interference_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")

        if type(self.flush_priority) is not int:
            raise TypeError("flush_priority must be int")

        self._resolved_delay_ticks = self._resolve_delay()

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
        """Bind to a timeline. No RNG streams are declared.

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
        Binding is idempotent for the same timeline. The component is ideal, so
        there is nothing to sample; a declared-but-never-consumed stream would be
        dead configuration and a lie in the binding log. Because stream values
        derive from the stream path rather than from creation order, adding one
        in a later revision cannot perturb any other component's draws.
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

        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.interferometers.delay_interferometer.ready",
            "delay interferometer ready",
            meta={
                "device_id": self.device_id,
                "delay_ticks": self._resolved_delay_ticks,
                "delay_s": ticks_to_seconds(self._resolved_delay_ticks),
                "flush_priority": self.flush_priority,
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

        short = ArmContribution(
            state=short_state,
            bs2_tick=arrival_tick,
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

        if partner.bs2_tick <= arrival_tick:
            self._resolve(
                timeline,
                short=short,
                long_=partner,
                event_id=event_id,
                action=action,
            )
        else:
            self._defer(timeline, short=short, long_=partner)

        self._hold_counter += 1
        self._held = HeldLongArm(
            hold_id=self._hold_counter,
            arm=ArmContribution(
                state=long_state,
                bs2_tick=arrival_tick + self._resolved_delay_ticks,
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
                f"signal {signal.id!r} with state_ref={signal.state_ref!r}: a "
                "qstate-backed photon has no optical amplitude to split and no "
                "record here to collapse, and interfering it would silently "
                "strand its state; send a coherent pulse, not a photon"
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
    ) -> None:
        """Schedule a combination whose long arm has not reached BS2 yet.

        Only reached when ``long_.bs2_tick > timeline.current_time`` strictly, so
        this is never a delay-0 self-schedule -- see the class notes for why that
        matters.

        No log record of its own. The scheduled event carries both signal ids and
        both BS2 ticks in its own ``meta``, and an event trace already shows it
        being queued here and executed at ``long_.bs2_tick``; a separate record
        would only restate that. ``_schedule_flush`` queues a comparable
        self-event the same way.
        """
        timeline.schedule(
            Event(
                time=long_.bs2_tick,
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
