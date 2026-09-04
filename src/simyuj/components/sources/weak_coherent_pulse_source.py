"""Weak coherent pulse source component.

``WeakCoherentPulseSource`` is a transmitter's complete optical preparation
device. Once per active slot it chooses a mean photon number, a carrier phase,
and an encoding phase, builds the amplitude

.. math::

   \\alpha = \\sqrt{\\mu}\\,e^{i(\\Theta + \\varphi_{enc})}

and emits a ``SignalKind.PULSE`` signal carrying that :class:`CoherentState`.
When a polarization selector is configured it also chooses **which mode that
amplitude occupies**, and prepares a one-qubit qstate record describing it.
There is no separate modulator component; see ``docs/dev/dps-design.md``
section 5.

The amplitude is **non-sampling in the quantum sense**. The source never draws a
photon number and never decides per slot whether a pulse exists. At
:math:`\\mu = 0.1` it does not produce a stream of vacuum and one-photon events;
it produces a uniform stream of identical :math:`|\\sqrt{0.1}\\rangle` pulses.
Photon statistics are a detection-time phenomenon and are integrated in closed
form at the detector.

That is a claim about :math:`\\alpha`, not about qstate. An unpolarized
configuration -- ``polarization=None``, which is the default -- touches
``timeline.qstate`` not at all, so ``timeline.qstate.size()`` stays ``0`` for the
whole run. A *polarized* pulse does carry a qubit: a Jones vector is a
two-dimensional pure state, and the source prepares one record per pulse for it.
The record describes the mode, it is not the propagating carrier, which is what
``SubsystemHandle(kind="mode")`` tells the channel -- see ``qstate_payload_role``
in ``components/quantum_targets.py``. Loss scales the amplitude; it does not roll
a survival trial against the mode.

Choosing *which* :math:`\\mu` to prepare is a classical preparation choice and is
sampled freely; that is what a decoy-state transmitter does. Sampling a photon
number is not, and nothing here calls ``rng.poisson``.

Unlike ``SinglePhotonSource`` this source has **no** ``emission_probability``
-- a laser fires every slot -- and no ``noise_models``: Kraus operators act on
qubit axes and have no representation for an optical amplitude. Noise on a
polarization mode belongs to the channel, not the source.

Not modelled
------------

Modulator insertion loss, finite extinction ratio, intensity-modulator dynamics,
laser relative intensity noise, side modes, chirp, and pulse broadening. The
temporal mode is a fixed width per source and nothing broadens it in flight;
dispersion is not modelled. Finite laser linewidth is not modelled either --
``FixedCarrierPhase`` means infinite coherence length. No polarization alphabet
ships: the seam is open and no selector implements it yet. See
``CAPABILITY_MAP.md`` section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.primitives.units import Hertz
from simyuj.primitives.validation import (
    require_optional_positive_real,
    require_positive_real,
)
from simyuj.qstate import StateRef, SubsystemId
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel

from ..connections import require_connection
from ..ports import Port, PortDirection, PortKind
from ._common import (
    ACTION_EMIT,
    ACTION_START,
    DeltaTiming,
    EmissionAttempt,
    EmissionTimingProfile,
    duration_ticks,
    emission_period_ticks,
    quantum_output_port,
    schedule_next_emission_event,
    schedule_start_event,
    start_time_tick,
    stop_time,
    validate_timing_profile_for_chained_scheduler,
)
from .coherent_preparation import (
    CarrierPhaseSelector,
    EncodingPhaseSelector,
    FixedCarrierPhase,
    FixedPhase,
    IntensitySelector,
    PolarizationSelection,
    PolarizationSelector,
    validate_pulse_selectors,
)
from .reports import CoherentPulsePreparationReport, store_source_report

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG
    from simyuj.engine.timeline import Timeline


COMPONENT_KEY = "weak_coherent_pulse_source"


@dataclass(slots=True)
class WeakCoherentPulseSource(Component):
    """
    Event-driven source that prepares and emits coherent optical pulses.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier used in event metadata, the RNG stream
        keys, and emitted signal identifiers.
    frequency_hz : Hertz
        Pulse repetition frequency. It is converted to an integer slot period in
        simulation ticks during construction.
    intensity : IntensitySelector
        Per-pulse mean-photon-number policy. Required: there is no physically
        neutral default intensity. ``FixedIntensity(mu)`` is the V1
        implementation.
    encoding_scheme : EncodingScheme, default=EncodingScheme.PHASE
        Encoding recorded on each emitted pulse signal.
    wavelength_nm : float, default=1550.0
        Carrier wavelength recorded on each emitted pulse signal.
    start_time_s : float, default=0.0
        External activation time in seconds.
    duration_s : float or None, default=None
        Positive source-active duration in seconds, or ``None`` for no stop
        time.
    timing_profile : EmissionTimingProfile, default=DeltaTiming()
        Timing profile that samples the delay between each nominal slot and the
        corresponding ``ACTION_EMIT`` event.
    carrier_phase : CarrierPhaseSelector, default=FixedCarrierPhase(0.0)
        Per-pulse carrier-phase policy. The default holds one phase for the
        whole run, which is what differential-phase encoding requires.
    encoding_phase : EncodingPhaseSelector, default=FixedPhase(0.0)
        Per-pulse encoding-phase policy. The default emits an unmodulated train:
        a source built with no phase configuration is an ordinary laser, not a
        transmitter for one particular protocol. A differential-phase trial
        passes ``RandomPhaseChoice(DPS_PHASES)`` explicitly, which is also where
        the alphabet ordering the decoder assumes becomes visible.
    polarization : PolarizationSelector or None, default=None
        Per-pulse polarization-mode policy, or ``None`` for a source that models
        no polarization at all. ``None`` is not "unpolarized light": it is the
        statement that this run does not describe the mode, and it is the only
        configuration under which the source creates no qstate record. **No
        selector implementing this protocol ships.**
    temporal_mode_sigma_s : float or None, default=None
        Positive field-envelope standard deviation of each emitted pulse, in
        seconds, defined by

        .. math::

           f(t) = (\\pi\\sigma^{2})^{-1/4}
                  \\exp\\left[-\\frac{(t-t_{0})^{2}}{2\\sigma^{2}}\\right],
           \\qquad \\int |f(t)|^{2}\\,\\mathrm{d}t = 1

        so it is the standard deviation of the **field** envelope, not of the
        intensity envelope; the intensity FWHM is
        :math:`2\\sqrt{\\ln 2}\\,\\sigma \\approx 1.665\\,\\sigma`. ``None``
        emits pulses with no temporal mode described, which is fine for anything
        that only reads amplitudes. A receiver that interferes adjacent pulses
        requires it.

    Attributes
    ----------
    output_port : Port
        Quantum output port named ``"out"``. Every emission requires this port
        to be connected.
    report_port : Port
        Classical output port named ``"report"``. When connected, pulse
        preparation reports are delivered through this local control-plane
        boundary.
    reports : list[CoherentPulsePreparationReport]
        Every preparation, in emission order, also kept locally.
    emission_period_ticks : int
        Slot period in simulation ticks.
    pulse_count : int
        Number of pulses emitted so far. The source's own counter, not
        ``timeline.qstate.size()`` -- which is ``0`` for an unpolarized run and,
        for a polarized one, counts mode records rather than pulses.

    Notes
    -----
    The emitted signal always carries its optical amplitude in
    ``Signal.coherent_state``. With no polarization configured it carries
    nothing else: ``state_ref`` is ``None`` and ``state_targets`` is empty, so
    components that require a qstate-backed signal reject it with a clear error
    rather than silently misinterpreting it. With a polarization selector it
    additionally carries one ``state_ref`` and one
    ``SubsystemHandle(kind="mode")`` describing the occupied mode.

    Signal metadata carries identity only -- ``source_device_id`` and
    ``pulse_index``. The preparation choices travel in
    ``CoherentPulsePreparationReport`` on the local control plane instead, so a
    downstream device cannot read a sender's private choice off a signal in
    flight. The report is the one place a Jones vector is recorded beside the
    qstate record; it never travels and nothing updates it, so it cannot fall
    out of step.

    ``bind`` declares five deterministic RNG streams under the component key
    ``"weak_coherent_pulse_source"``: ``"timing"``, ``"intensity"``,
    ``"carrier"``, ``"encoding"``, and ``"polarization"``. All five are declared
    unconditionally, because ``Timeline.rng`` refuses a new stream once
    execution begins.

    **Each selector draws from its own stream, and polarization must not be
    folded into ``"encoding"``.** Sharing one would make every polarized pulse
    consume a draw an unpolarized run does not, shifting the encoding-phase
    sequence and breaking replay of runs recorded before it. With the default
    selectors and no polarization the source consumes no randomness at all and
    replays identically at any ``master_seed``.

    Every active slot emits exactly one pulse; there is no emission Bernoulli.
    ``mean_photon_number = 0`` emits a real coherent-vacuum pulse -- a signal, a
    delivery and a report -- not a skipped slot, and ``pulse_count`` counts it.

    The source uses the shared chained one-event scheduler. Timing profiles must
    expose finite ``max_emission_delay_ticks`` strictly smaller than the slot
    period, so a delayed pulse cannot overrun a later nominal slot.

    ``temporal_mode_sigma_s`` is **not** converted with ``seconds_to_ticks``:
    that helper rounds to integer picoseconds, which would quantize any overlap
    computed downstream from it.
    """

    device_id: str
    frequency_hz: Hertz
    intensity: IntensitySelector

    encoding_scheme: EncodingScheme = EncodingScheme.PHASE
    wavelength_nm: float = 1550.0
    start_time_s: float = 0.0
    duration_s: float | None = None
    timing_profile: EmissionTimingProfile = field(default_factory=DeltaTiming)
    carrier_phase: CarrierPhaseSelector = field(default_factory=FixedCarrierPhase)
    encoding_phase: EncodingPhaseSelector = field(default_factory=FixedPhase)

    # Both appended rather than grouped; see docs/dev/dps-design.md section 2.
    temporal_mode_sigma_s: Optional[float] = None
    polarization: Optional[PolarizationSelector] = None

    output_port: Port = field(init=False)
    report_port: Port = field(init=False)
    reports: list[CoherentPulsePreparationReport] = field(
        init=False,
        default_factory=list,
    )
    emission_period_ticks: int = field(init=False)

    _start_time_tick: int = field(init=False)
    _duration_ticks: int | None = field(init=False, default=None)
    _pulse_count: int = field(init=False, default=0)
    _bound_timeline_id: int | None = field(init=False, default=None)
    _timing_rng: DeterministicRNG | None = field(init=False, default=None)
    _intensity_rng: DeterministicRNG | None = field(init=False, default=None)
    _carrier_rng: DeterministicRNG | None = field(init=False, default=None)
    _encoding_rng: DeterministicRNG | None = field(init=False, default=None)
    _polarization_rng: DeterministicRNG | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")
        self.emission_period_ticks = emission_period_ticks(self.frequency_hz)

        if not isinstance(self.encoding_scheme, EncodingScheme):
            raise TypeError("encoding_scheme must be EncodingScheme")

        self.wavelength_nm = require_positive_real(
            self.wavelength_nm,
            field_name="wavelength_nm",
        )
        self.temporal_mode_sigma_s = require_optional_positive_real(
            self.temporal_mode_sigma_s,
            field_name="temporal_mode_sigma_s",
        )

        self._start_time_tick = start_time_tick(self.start_time_s)
        self._duration_ticks = duration_ticks(self.duration_s)

        validate_timing_profile_for_chained_scheduler(
            timing_profile=self.timing_profile,
            emission_period_tick_count=self.emission_period_ticks,
        )
        validate_pulse_selectors(
            intensity=self.intensity,
            carrier_phase=self.carrier_phase,
            encoding_phase=self.encoding_phase,
            polarization=self.polarization,
        )

        self.output_port = quantum_output_port(
            owner=self,
            owner_id=self.device_id,
            name="out",
        )
        self.report_port = Port(
            name="report",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    @property
    def pulse_count(self) -> int:
        """Number of pulses emitted so far."""
        return self._pulse_count

    @property
    def stop_time(self) -> int | None:
        """Exclusive stop tick, or ``None`` when no duration is configured."""
        return stop_time(
            start_tick=self._start_time_tick,
            duration_tick_count=self._duration_ticks,
        )

    def bind(self, context: BindingContext) -> None:
        """
        Declare the five deterministic RNG streams before execution starts.

        Notes
        -----
        Binding is idempotent for the same timeline and rejected for a
        different one. All five streams are declared even when the configured
        selectors draw from none of them -- including ``"polarization"``,
        because ``Timeline.rng`` refuses a new stream once the timeline
        freezes and a selector could otherwise never be added.
        """
        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)
        if self._bound_timeline_id is not None and self._bound_timeline_id != (
            timeline_id
        ):
            raise RuntimeError("source is already bound to another timeline")

        self._bound_timeline_id = timeline_id
        self._timing_rng = timeline.rng(self.device_id, COMPONENT_KEY, "timing")
        self._intensity_rng = timeline.rng(self.device_id, COMPONENT_KEY, "intensity")
        self._carrier_rng = timeline.rng(self.device_id, COMPONENT_KEY, "carrier")
        self._encoding_rng = timeline.rng(self.device_id, COMPONENT_KEY, "encoding")
        self._polarization_rng = timeline.rng(
            self.device_id,
            COMPONENT_KEY,
            "polarization",
        )

    def schedule_start(self, timeline: Timeline) -> Event:
        """
        Bind the RNG streams and schedule the external source activation event.

        Returns
        -------
        Event
            Timeline-owned ``ACTION_START`` event scheduled at ``start_time_s``
            converted to simulation ticks.
        """
        self.bind(BindingContext(timeline=timeline, logger=timeline.logger))
        return schedule_start_event(
            source=self,
            timeline=timeline,
            start_tick=self._start_time_tick,
            device_id=self.device_id,
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """
        Handle source activation and pulse-emission events.

        Parameters
        ----------
        event : Event
            ``ACTION_START`` or ``ACTION_EMIT`` event targeting this source.
            ``ACTION_EMIT`` requires an :class:`EmissionAttempt` payload.
        timeline : Timeline
            Timeline that owns event ordering, scheduling, RNG streams, and
            logs.

        Notes
        -----
        ``ACTION_START`` logs activation and schedules the first slot.
        ``ACTION_EMIT`` emits the pulse for the current slot, then schedules the
        next slot from the original nominal slot so jitter does not accumulate.
        Unlike ``SinglePhotonSource`` there is no skip path: every executed slot
        emits.
        """
        if self._bound_timeline_id is None:
            raise RuntimeError("source must be bound via schedule_start() or bind()")

        if event.action == ACTION_START:
            timeline.log(
                LogLevel.INFO,
                "components.sources.weak_coherent_pulse_source.start",
                "source started",
                event_id=event.event_id,
                action=event.action,
                meta={
                    "device_id": self.device_id,
                    "frequency_hz": float(self.frequency_hz),
                    "start_tick": self._start_time_tick,
                    "stop_tick": self.stop_time,
                    "temporal_mode_sigma_s": self.temporal_mode_sigma_s,
                },
            )
            self._schedule_next_emission(
                timeline,
                emission_slot_tick=timeline.current_time,
            )
            return

        if event.action == ACTION_EMIT:
            if not isinstance(event.payload_ref, EmissionAttempt):
                raise TypeError("ACTION_EMIT payload_ref must be EmissionAttempt")

            self._emit_now(
                timeline,
                emission_slot_tick=event.payload_ref.emission_slot_tick,
                emission_delay_ticks=event.payload_ref.emission_delay_ticks,
                event_id=event.event_id,
                action=event.action,
            )
            self._schedule_next_emission(
                timeline,
                emission_slot_tick=(
                    event.payload_ref.emission_slot_tick + self.emission_period_ticks
                ),
            )
            return

        raise ValueError(f"unsupported event action for source: {event.action!r}")

    def _schedule_next_emission(
        self,
        timeline: Timeline,
        *,
        emission_slot_tick: int,
    ) -> None:
        assert self._timing_rng is not None
        schedule_next_emission_event(
            source=self,
            timeline=timeline,
            device_id=self.device_id,
            emission_slot_tick=emission_slot_tick,
            emission_period_tick_count=self.emission_period_ticks,
            start_tick=self._start_time_tick,
            duration_tick_count=self._duration_ticks,
            timing_profile=self.timing_profile,
            timing_rng=self._timing_rng,
        )

    def _prepare_polarization_mode(
        self,
        timeline: Timeline,
        *,
        polarization: PolarizationSelection | None,
        pulse_index: int,
    ) -> tuple[Optional[StateRef], tuple[SubsystemHandle, ...]]:
        """
        Turn a polarization selection into a qstate record and its handle.

        Returns
        -------
        tuple
            ``(state_ref, state_targets)`` for the signal. ``(None, ())`` when
            no polarization is configured, which is the whole of the behaviour
            for an unpolarized source: ``timeline.qstate`` is never reached.

        Notes
        -----
        **The handle is stamped ``kind="mode"``.** ``qstate_payload_role`` reads
        it to decide that channel loss must scale the amplitude rather than roll
        a survival trial against this record. ``SubsystemHandle`` defaults to
        ``"qubit"``, so forgetting the stamp would silently make the
        polarization state the carrier.

        **Nothing retires these records.** One per pulse accumulates for the
        length of the run; see ``docs/dev/dps-design.md`` section 7.1 for the
        measured ceiling. An unpolarized run allocates none.
        """
        if polarization is None:
            return None, ()

        mode_label = f"{self.device_id}:mode:{pulse_index}"
        state_ref = timeline.qstate.prepare(
            polarization.jones,
            rep="ket",
            subsystems=(SubsystemId(mode_label),),
            meta=(
                ("component", self.device_id),
                ("component_type", COMPONENT_KEY),
                ("polarization_index", polarization.index),
                ("pulse_index", pulse_index),
            ),
        )
        return state_ref, (
            SubsystemHandle(
                label=mode_label,
                kind="mode",
                index=0,
                metadata=(("qstate_subsystem", mode_label),),
            ),
        )

    def _emit_now(
        self,
        timeline: Timeline,
        *,
        emission_slot_tick: int,
        emission_delay_ticks: int,
        event_id: int | None,
        action: str,
    ) -> None:
        """
        Prepare and emit one coherent pulse.

        Notes
        -----
        ``timeline.qstate`` is touched only when a polarization selector is
        configured, and then exactly once per pulse. The output port is required
        before any selector runs, so an unconnected source raises without
        perturbing the RNG streams.

        The selectors receive the **zero-based** pulse counter; the report
        records the one-based index. Neither phase is recoverable from the
        amplitude afterwards -- everything a protocol needs is in the report.
        """
        connection = require_connection(self.output_port)

        assert self._intensity_rng is not None
        assert self._carrier_rng is not None
        assert self._encoding_rng is not None
        assert self._polarization_rng is not None

        selection_index = self._pulse_count
        intensity = self.intensity.select_intensity(
            selection_index,
            self._intensity_rng,
        )
        carrier_phase_rad = self.carrier_phase.select_carrier_phase(
            selection_index,
            self._carrier_rng,
        )
        encoding = self.encoding_phase.select_encoding_phase(
            selection_index,
            self._encoding_rng,
        )
        polarization = (
            None
            if self.polarization is None
            else self.polarization.select_polarization(
                selection_index,
                self._polarization_rng,
            )
        )

        coherent_state = CoherentState.from_mean_photon_number(
            intensity.mean_photon_number,
            phase_rad=carrier_phase_rad + encoding.phase_rad,
        )

        self._pulse_count += 1
        pulse_index = self._pulse_count

        state_ref, state_targets = self._prepare_polarization_mode(
            timeline,
            polarization=polarization,
            pulse_index=pulse_index,
        )

        signal = Signal(
            id=f"{self.device_id}:pulse:{pulse_index}",
            signal_kind=SignalKind.PULSE,
            encoding_scheme=self.encoding_scheme,
            emission_time=timeline.current_time,
            origin=self.device_id,
            wavelength_nm=float(self.wavelength_nm),
            # An amplitude is not a qubit, so an unpolarized pulse carries no
            # qstate at all and both of these are empty. A polarized one does:
            # the mode it occupies is a Jones qubit, and the "mode" role on the
            # handle is what stops the channel treating it as the carrier.
            # Either way identity lives in `id`, `origin`, and `pulse_index`.
            state_ref=state_ref,
            state_targets=state_targets,
            coherent_state=coherent_state,
            temporal_mode_sigma_s=self.temporal_mode_sigma_s,
            # Identity only. The preparation choices stay on the control plane.
            meta=(
                ("source_device_id", self.device_id),
                ("pulse_index", pulse_index),
            ),
            timing_meta=(
                ("time_unit", "ps"),
                ("emission_slot_tick", emission_slot_tick),
                ("emission_delay_ticks", emission_delay_ticks),
                ("emission_period_ticks", self.emission_period_ticks),
                ("frequency_hz", float(self.frequency_hz)),
                ("pulse_index", pulse_index),
            ),
            # Trusted emission hot path: every field comes from validated
            # component configuration.
            validation_flag=False,
        )

        timeline.log(
            LogLevel.DEBUG,
            "components.sources.weak_coherent_pulse_source.emit",
            "coherent pulse emitted",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "pulse_index": pulse_index,
                "signal_id": signal.id,
                "connection_id": connection.connection_id,
                "emission_slot_tick": emission_slot_tick,
                "emission_delay_ticks": emission_delay_ticks,
                # Derived floats only: the JSONL sink has no `complex` case.
                "mean_photon_number": coherent_state.mean_photon_number,
                "phase_rad": coherent_state.phase_rad,
                "carrier_phase_rad": carrier_phase_rad,
                "encoding_phase_rad": encoding.phase_rad,
                "encoding_phase_index": encoding.index,
                "intensity_index": intensity.index,
                # The alphabet index, never the Jones vector: `complex` has
                # no JSONL case.
                "polarization_index": (
                    None if polarization is None else polarization.index
                ),
                "state_ref": state_ref,
                "temporal_mode_sigma_s": self.temporal_mode_sigma_s,
            },
        )

        store_source_report(
            reports=self.reports,
            report_port=self.report_port,
            report=CoherentPulsePreparationReport(
                report_id=f"{self.device_id}:prep:{pulse_index}",
                device_id=self.device_id,
                time=timeline.current_time,
                pulse_index=pulse_index,
                signal_ids=(str(signal.id),),
                coherent_state=coherent_state,
                emission_slot_tick=emission_slot_tick,
                emission_delay_ticks=emission_delay_ticks,
                mean_photon_number=intensity.mean_photon_number,
                intensity_index=intensity.index,
                carrier_phase_rad=carrier_phase_rad,
                encoding_phase_rad=encoding.phase_rad,
                encoding_phase_index=encoding.index,
                polarization=None if polarization is None else polarization.jones,
                polarization_index=(
                    None if polarization is None else polarization.index
                ),
            ),
            timeline=timeline,
            source=self,
        )

        connection.transmit(
            signal,
            timeline,
            time=timeline.current_time,
            source=self,
            subsystem_id="components",
            meta={
                "source_device_id": self.device_id,
                "output_port": self.output_port.name,
                "signal_id": signal.id,
            },
        )


__all__ = ["COMPONENT_KEY", "WeakCoherentPulseSource"]
