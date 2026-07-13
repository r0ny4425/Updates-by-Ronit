"""Single-photon source component.

``SinglePhotonSource`` schedules emission attempts, creates one one-qubit
qstate for each successful attempt, wraps it in a ``SignalKind.PHOTON`` signal,
and transmits it through a quantum output port. It models zero-or-one photon
emission rather than a full optical pulse source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.primitives.units import Hertz, ticks_to_seconds
from simyuj.qstate import StateSampler, SubsystemId
from simyuj.qstate.noise import NoiseModel
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
    ExGaussianTiming,
    GaussianTiming,
    bind_source_rngs,
    duration_ticks,
    emission_period_ticks,
    is_before_stop,
    normalize_noise_models,
    quantum_output_port,
    schedule_next_emission_event,
    schedule_start_event,
    start_time_tick,
    stop_time,
    validate_source_scalars,
    validate_timing_profile_for_chained_scheduler,
)
from .reports import SourcePreparationReport, store_source_report

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG
    from simyuj.engine.timeline import Timeline


@dataclass(slots=True)
class SinglePhotonSource(Component):
    """
    Event-driven source that emits one-photon quantum signals.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier used in event metadata, RNG stream keys,
        qstate subsystem labels, and emitted signal identifiers.
    frequency_hz : Hertz
        Emission-attempt frequency. It is converted to integer simulation ticks
        during construction.
    encoding_scheme : EncodingScheme, default=EncodingScheme.POLARIZATION
        Encoding recorded on each emitted photon signal.
    emission_probability : float, default=1.0
        Bernoulli probability that an emission attempt creates and transmits a
        photon.
    wavelength_nm : float, default=1550.0
        Wavelength recorded on each emitted photon signal.
    start_time_s : float, default=0.0
        External activation time in seconds.
    duration_s : float or None, default=None
        Positive source-active duration in seconds, or ``None`` for no stop
        time.
    sampler : StateSampler or None, default=None
        One-qubit state sampler. When omitted, the source emits
        :math:`|0\\rangle` ket states with label ``"default"``.
    timing_profile : EmissionTimingProfile, default=DeltaTiming()
        Timing profile that samples the delay between each nominal slot and
        the corresponding ``ACTION_EMIT`` event.
    noise_models : Sequence[NoiseModel], default=()
        Source noise models applied to the emitted qubit for the sampled
        emission delay before transmission.

    Attributes
    ----------
    output_port : Port
        Quantum output port named ``"out"``. A successful emission requires
        this port to be connected.
    report_port : Port
        Classical output port named ``"report"``. When connected, successful
        preparation reports are delivered through this local control-plane
        boundary.
    emission_period_ticks : int
        Emission-attempt period in simulation ticks.

    Notes
    -----
    ``schedule_start`` binds three deterministic RNG streams named
    ``"emission"``, ``"state"``, and ``"timing"`` before scheduling
    ``ACTION_START``. ``ACTION_START`` schedules the first emission attempt.
    Each ``ACTION_EMIT`` consumes the emission RNG for the Bernoulli emission
    decision, uses the state RNG through ``StateSampler`` when a photon is
    created, transmits a ``SignalKind.PHOTON`` signal through ``output_port``,
    and schedules the next nominal slot.

    Each emitted photon owns a new qstate subsystem named
    ``"{device_id}:photon:{photon_index}"``. The signal carries one
    ``SubsystemHandle`` for that subsystem, sampler metadata, source indices,
    and timing metadata containing ``emission_slot_tick``,
    ``emission_delay_ticks``, ``emission_period_ticks``, and ``frequency_hz``.

    As of now, this component models idealized zero-or-one photon emission. A
    successful attempt creates exactly one photon subsystem; a skipped attempt
    creates none. It does not sample photon-number statistics, emit
    multi-photon pulses, or model weak coherent pulses.

    Source noise, when configured, treats ``emission_delay_ticks`` as internal
    dwell time and applies each noise model for ``ticks_to_seconds(delay)``.
    This component does not model optical pulse shape or hardware recovery
    effects.

    The source uses a chained one-event scheduler. Timing profiles must expose
    finite ``max_emission_delay_ticks`` values strictly smaller than the
    emission period, so a delayed event cannot overrun a later nominal slot.
    """

    device_id: str
    frequency_hz: Hertz

    encoding_scheme: EncodingScheme = EncodingScheme.POLARIZATION
    emission_probability: float = 1.0
    wavelength_nm: float = 1550.0
    start_time_s: float = 0.0
    duration_s: float | None = None
    sampler: StateSampler | None = None
    timing_profile: EmissionTimingProfile = field(default_factory=DeltaTiming)
    noise_models: Sequence[NoiseModel] = field(default_factory=tuple)

    output_port: Port = field(init=False)
    report_port: Port = field(init=False)
    reports: list[SourcePreparationReport] = field(init=False, default_factory=list)
    emission_period_ticks: int = field(init=False)

    _start_time_tick: int = field(init=False)
    _duration_ticks: int | None = field(init=False, default=None)
    _attempt_count: int = field(init=False, default=0)
    _emitted_count: int = field(init=False, default=0)
    _bound_timeline_id: int | None = field(init=False, default=None)
    _emission_rng: DeterministicRNG | None = field(init=False, default=None)
    _state_rng: DeterministicRNG | None = field(init=False, default=None)
    _timing_rng: DeterministicRNG | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")
        self.emission_period_ticks = emission_period_ticks(self.frequency_hz)
        validate_source_scalars(
            encoding_scheme=self.encoding_scheme,
            emission_probability=self.emission_probability,
            wavelength_nm=self.wavelength_nm,
        )
        self._start_time_tick = start_time_tick(self.start_time_s)
        self._duration_ticks = duration_ticks(self.duration_s)

        if self.sampler is None:
            self.sampler = StateSampler(
                states=("|0>",),
                probabilities=(1.0,),
                rep="ket",
                labels=("default",),
            )
        elif not isinstance(self.sampler, StateSampler):
            raise TypeError("sampler must be StateSampler or None")

        if self.sampler.num_qubits != 1:
            raise ValueError("SinglePhotonSource requires a one-qubit sampler")

        validate_timing_profile_for_chained_scheduler(
            timing_profile=self.timing_profile,
            emission_period_tick_count=self.emission_period_ticks,
        )
        self.noise_models = normalize_noise_models(
            self.noise_models,
            field_name="SinglePhotonSource.noise_models",
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
    def stop_time(self) -> int | None:
        return stop_time(
            start_tick=self._start_time_tick,
            duration_tick_count=self._duration_ticks,
        )

    def _is_before_stop(self, time: int) -> bool:
        return is_before_stop(
            time,
            start_tick=self._start_time_tick,
            duration_tick_count=self._duration_ticks,
        )

    def bind(self, context: BindingContext) -> None:
        """
        Declare deterministic RNG streams before timeline execution starts.

        Notes
        -----
        Binding is idempotent for the same timeline and rejected for a
        different timeline. The source uses separate emission, state, and
        timing streams so fixed seeds and fixed configuration reproduce the
        same emission decisions, sampled states, and timing delays.
        """
        (
            self._bound_timeline_id,
            self._emission_rng,
            self._state_rng,
            self._timing_rng,
        ) = bind_source_rngs(
            source=self,
            context=context,
            device_id=self.device_id,
            component_key="single_photon_source",
            bound_timeline_id=self._bound_timeline_id,
        )

    def schedule_start(self, timeline: Timeline) -> Event:
        """
        Bind RNG streams and schedule the external source activation event.

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
        Handle source activation and emission events.

        Parameters
        ----------
        event : Event
            ``ACTION_START`` or ``ACTION_EMIT`` event targeting this source.
            ``ACTION_EMIT`` requires an :class:`EmissionAttempt` payload.
        timeline : Timeline
            Timeline that owns event ordering, scheduling, RNG streams, qstate
            storage, and logs.

        Notes
        -----
        ``ACTION_START`` logs activation and schedules the first emission slot.
        ``ACTION_EMIT`` emits or skips the current attempt, then schedules the
        next slot based on the original nominal slot. Successful emissions
        require ``output_port`` to be connected; skipped attempts return before
        checking the connection.
        """
        if self._bound_timeline_id is None:
            raise RuntimeError("source must be bound via schedule_start() or bind()")

        if event.action == ACTION_START:
            timeline.log(
                LogLevel.INFO,
                "components.sources.single_photon_source.start",
                "source started",
                event_id=event.event_id,
                action=event.action,
                meta={
                    "device_id": self.device_id,
                    "frequency_hz": float(self.frequency_hz),
                    "start_tick": self._start_time_tick,
                    "stop_tick": self.stop_time,
                    "emission_probability": float(self.emission_probability),
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
        Emit or skip a single scheduled photon attempt.

        Notes
        -----
        A successful attempt prepares a one-qubit qstate subsystem, optionally
        applies configured source noise, builds a ``SignalKind.PHOTON`` signal,
        logs the emission, and transmits the signal through the connected
        output port. A failed Bernoulli draw logs a skip and leaves qstate and
        port delivery unchanged.
        """
        self._attempt_count += 1

        assert self._emission_rng is not None
        if self._emission_rng.random() >= float(self.emission_probability):
            timeline.log(
                LogLevel.TRACE,
                "components.sources.single_photon_source.skip",
                "emission skipped",
                event_id=event_id,
                action=action,
                meta={
                    "device_id": self.device_id,
                    "attempt_index": self._attempt_count,
                    "emission_slot_tick": emission_slot_tick,
                    "emission_delay_ticks": emission_delay_ticks,
                },
            )
            return

        connection = require_connection(self.output_port)

        assert self._state_rng is not None
        assert self.sampler is not None

        sample = self.sampler.sample(rng=self._state_rng)

        self._emitted_count += 1
        photon_index = self._emitted_count
        subsystem_label = f"{self.device_id}:photon:{photon_index}"
        subsystem = SubsystemId(subsystem_label)

        state_ref = timeline.qstate.prepare(
            sample.state,
            rep=sample.rep,
            subsystems=(subsystem,),
            meta=(
                ("component", self.device_id),
                ("component_type", "single_photon_source"),
                ("sampler_index", sample.index),
                ("sampler_label", sample.label),
                ("attempt_index", self._attempt_count),
                ("photon_index", photon_index),
            ),
        )

        if self.noise_models:
            # Physical model:
            # emission_delay_ticks is interpreted as the internal dwell time between
            # nominal photon creation at emission_slot_tick and release from the
            # source. Source noise is therefore applied for this dwell duration.
            #
            # If emission_delay_ticks is intended to model only timestamp jitter,
            # do not use it as dwell time. Use a zero-delay timing profile or
            # introduce a separate source_noise_duration_s field.
            state_ref = timeline.qstate.apply_noise_models(
                self.noise_models,
                targets=(subsystem,),
                duration_s=ticks_to_seconds(emission_delay_ticks),
            )

        signal = Signal(
            id=f"{self.device_id}:photon:{photon_index}",
            signal_kind=SignalKind.PHOTON,
            encoding_scheme=self.encoding_scheme,
            emission_time=timeline.current_time,
            origin=self.device_id,
            wavelength_nm=float(self.wavelength_nm),
            state_ref=state_ref,
            state_targets=(
                SubsystemHandle(
                    label=subsystem_label,
                    kind="qubit",
                    index=0,
                    metadata=(("qstate_subsystem", subsystem_label),),
                ),
            ),
            meta=(
                ("source_device_id", self.device_id),
                ("attempt_index", self._attempt_count),
                ("photon_index", photon_index),
                ("sampler_index", sample.index),
                ("sampler_label", sample.label),
            ),
            timing_meta=(
                ("time_unit", "ps"),
                ("emission_slot_tick", emission_slot_tick),
                ("emission_delay_ticks", emission_delay_ticks),
                ("emission_period_ticks", self.emission_period_ticks),
                ("frequency_hz", float(self.frequency_hz)),
            ),
            # Trusted emission hot path: fields come from validated component
            # config, qstate output, and locally constructed subsystem handles.
            validation_flag=False,
        )

        timeline.log(
            LogLevel.DEBUG,
            "components.sources.single_photon_source.emit",
            "photon emitted",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "attempt_index": self._attempt_count,
                "photon_index": photon_index,
                "signal_id": signal.id,
                "connection_id": connection.connection_id,
                "emission_slot_tick": emission_slot_tick,
                "emission_delay_ticks": emission_delay_ticks,
                "sampler_label": sample.label,
            },
        )

        store_source_report(
            reports=self.reports,
            report_port=self.report_port,
            report=SourcePreparationReport(
                report_id=f"{self.device_id}:prep:{self._attempt_count}",
                device_id=self.device_id,
                time=timeline.current_time,
                attempt_index=self._attempt_count,
                emission_index=photon_index,
                signal_ids=(str(signal.id),),
                sampler_index=sample.index,
                sampler_label=sample.label,
                state_ref=state_ref,
                state_targets=(subsystem,),
                emission_slot_tick=emission_slot_tick,
                emission_delay_ticks=emission_delay_ticks,
                meta=(("photon_index", photon_index),),
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


__all__ = [
    "ACTION_EMIT",
    "ACTION_START",
    "DeltaTiming",
    "EmissionTimingProfile",
    "ExGaussianTiming",
    "GaussianTiming",
    "SinglePhotonSource",
]
