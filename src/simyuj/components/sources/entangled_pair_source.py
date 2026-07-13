"""Entangled-pair source component.

``EntangledPairSource`` schedules emission attempts, creates one shared
two-qubit qstate for successful attempts, and transmits left/right
``SignalKind.ENTANGLED_MEMBER`` signals through separate quantum output ports.
It is an idealized pair source with explicit timing, RNG, and qstate ownership
contracts.
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
class EntangledPairSource(Component):
    """
    Two-port source that emits one shared two-qubit qstate as two signals.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier used in event metadata, RNG stream keys,
        qstate subsystem labels, and emitted signal identifiers.
    frequency_hz : Hertz
        Emission-attempt frequency. It is converted to integer simulation ticks
        during construction.
    encoding_scheme : EncodingScheme, default=EncodingScheme.POLARIZATION
        Encoding recorded on each emitted member signal.
    emission_probability : float, default=1.0
        Bernoulli probability that an emission attempt creates and transmits a
        pair.
    wavelength_nm : float, default=1550.0
        Wavelength recorded on each emitted member signal.
    start_time_s : float, default=0.0
        External activation time in seconds.
    duration_s : float or None, default=None
        Positive source-active duration in seconds, or ``None`` for no stop
        time.
    sampler : StateSampler or None, default=None
        Two-qubit state sampler. When omitted, the source emits ``phi+`` ket
        states with label ``"phi+"``.
    timing_profile : EmissionTimingProfile, default=DeltaTiming()
        Timing profile that samples the delay between each nominal slot and
        the corresponding ``ACTION_EMIT`` event.
    pair_noise_models : Sequence[NoiseModel], default=()
        Noise models applied jointly to the two emitted subsystems.
    left_noise_models : Sequence[NoiseModel], default=()
        Noise models applied to the left subsystem after pair noise.
    right_noise_models : Sequence[NoiseModel], default=()
        Noise models applied to the right subsystem after pair noise.

    Attributes
    ----------
    left_output_port, right_output_port : Port
        Quantum output ports named ``"left"`` and ``"right"``. A successful
        pair emission requires both ports to be connected.
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
    decision, uses the state RNG through ``StateSampler`` when a pair is
    created, transmits two ``SignalKind.ENTANGLED_MEMBER`` signals, and
    schedules the next nominal slot.

    Each emitted pair owns two qstate subsystems named
    ``"{device_id}:pair:{pair_index}:left"`` and
    ``"{device_id}:pair:{pair_index}:right"`` under one shared state reference.
    The left and right signals each carry one ``SubsystemHandle`` into that
    shared state, plus correlation metadata with ``pair_index`` and
    ``member``. ``bsa_pair_id`` is included in signal metadata for downstream
    Bell-state-analysis style components.

    Pair delivery is all-or-nothing at this component boundary. A successful
    attempt requires both output ports to be connected before either member is
    transmitted.

    Source noise, when configured, treats ``emission_delay_ticks`` as internal
    dwell time. Pair noise is applied to both subsystems first, then left and
    right noise are applied to their respective subsystems for the same
    duration. This component does not model optical pulse shape, multi-pair
    emission, or hardware recovery effects.

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
    pair_noise_models: Sequence[NoiseModel] = field(default_factory=tuple)
    left_noise_models: Sequence[NoiseModel] = field(default_factory=tuple)
    right_noise_models: Sequence[NoiseModel] = field(default_factory=tuple)

    left_output_port: Port = field(init=False)
    right_output_port: Port = field(init=False)
    report_port: Port = field(init=False)
    reports: list[SourcePreparationReport] = field(init=False, default_factory=list)
    emission_period_ticks: int = field(init=False)

    _start_time_tick: int = field(init=False)
    _duration_ticks: int | None = field(init=False, default=None)
    _attempt_count: int = field(init=False, default=0)
    _emitted_pair_count: int = field(init=False, default=0)
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
                states=("phi+",),
                probabilities=(1.0,),
                rep="ket",
                labels=("phi+",),
            )
        elif not isinstance(self.sampler, StateSampler):
            raise TypeError("sampler must be StateSampler or None")

        if self.sampler.num_qubits != 2:
            raise ValueError("EntangledPairSource requires a two-qubit sampler")

        validate_timing_profile_for_chained_scheduler(
            timing_profile=self.timing_profile,
            emission_period_tick_count=self.emission_period_ticks,
        )
        self.pair_noise_models = normalize_noise_models(
            self.pair_noise_models,
            field_name="pair_noise_models",
        )
        self.left_noise_models = normalize_noise_models(
            self.left_noise_models,
            field_name="left_noise_models",
        )
        self.right_noise_models = normalize_noise_models(
            self.right_noise_models,
            field_name="right_noise_models",
        )
        self.left_output_port = quantum_output_port(
            owner=self,
            owner_id=self.device_id,
            name="left",
        )
        self.right_output_port = quantum_output_port(
            owner=self,
            owner_id=self.device_id,
            name="right",
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
        different timeline. Separate emission, state, and timing streams keep
        emission decisions, sampled two-qubit states, and timing delays
        reproducible under fixed seeds and configuration.
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
            component_key="entangled_pair_source",
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
        Handle source activation and pair-emission events.

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
        ``ACTION_EMIT`` emits or skips the current pair attempt, then schedules
        the next slot from the original nominal slot. Successful emissions
        require both output ports to be connected; skipped attempts return
        before checking either connection.
        """
        if self._bound_timeline_id is None:
            raise RuntimeError("source must be bound via schedule_start() or bind()")

        if event.action == ACTION_START:
            timeline.log(
                LogLevel.INFO,
                "components.sources.entangled_pair_source.start",
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
        Emit or skip a single scheduled entangled-pair attempt.

        Notes
        -----
        A successful attempt prepares one two-qubit qstate with left and right
        subsystem labels, applies configured pair and per-member noise, builds
        two correlated member signals that share the same state reference, logs
        the emission, and transmits each signal through its connected output
        port. A failed Bernoulli draw logs a skip and leaves qstate and port
        delivery unchanged.
        """
        self._attempt_count += 1

        assert self._emission_rng is not None
        if self._emission_rng.random() >= float(self.emission_probability):
            timeline.log(
                LogLevel.TRACE,
                "components.sources.entangled_pair_source.skip",
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

        left_connection = require_connection(self.left_output_port)
        right_connection = require_connection(self.right_output_port)

        assert self._state_rng is not None
        assert self.sampler is not None

        sample = self.sampler.sample(rng=self._state_rng)

        self._emitted_pair_count += 1
        pair_index = self._emitted_pair_count
        left_label = f"{self.device_id}:pair:{pair_index}:left"
        right_label = f"{self.device_id}:pair:{pair_index}:right"
        left_subsystem = SubsystemId(left_label)
        right_subsystem = SubsystemId(right_label)

        state_ref = timeline.qstate.prepare(
            sample.state,
            rep=sample.rep,
            subsystems=(left_subsystem, right_subsystem),
            meta=(
                ("component", self.device_id),
                ("component_type", "entangled_pair_source"),
                ("sampler_index", sample.index),
                ("sampler_label", sample.label),
                ("attempt_index", self._attempt_count),
                ("pair_index", pair_index),
            ),
        )

        # Physical model:
        # emission_delay_ticks is interpreted as the internal dwell time between
        # nominal pair creation at emission_slot_tick and release from the source.
        # Source noise is therefore applied for this dwell duration.
        #
        # If emission_delay_ticks is intended to model only timestamp jitter, do
        # not use it as dwell time. Use a zero-delay timing profile or introduce
        # a separate source_noise_duration_s field.
        duration_s = ticks_to_seconds(emission_delay_ticks)
        if self.pair_noise_models:
            state_ref = timeline.qstate.apply_noise_models(
                self.pair_noise_models,
                targets=(left_subsystem, right_subsystem),
                duration_s=duration_s,
            )
        if self.left_noise_models:
            state_ref = timeline.qstate.apply_noise_models(
                self.left_noise_models,
                targets=(left_subsystem,),
                duration_s=duration_s,
            )
        if self.right_noise_models:
            state_ref = timeline.qstate.apply_noise_models(
                self.right_noise_models,
                targets=(right_subsystem,),
                duration_s=duration_s,
            )

        timing_meta = (
            ("time_unit", "ps"),
            ("emission_slot_tick", emission_slot_tick),
            ("emission_delay_ticks", emission_delay_ticks),
            ("emission_period_ticks", self.emission_period_ticks),
            ("frequency_hz", float(self.frequency_hz)),
        )
        common_meta = (
            ("source_device_id", self.device_id),
            ("attempt_index", self._attempt_count),
            ("pair_index", pair_index),
            ("bsa_pair_id", pair_index),
            ("sampler_index", sample.index),
            ("sampler_label", sample.label),
        )

        left_signal = self._make_member_signal(
            member="left",
            pair_index=pair_index,
            subsystem_label=left_label,
            subsystem_index=0,
            state_ref=state_ref,
            timeline_time=timeline.current_time,
            common_meta=common_meta,
            timing_meta=timing_meta,
        )
        right_signal = self._make_member_signal(
            member="right",
            pair_index=pair_index,
            subsystem_label=right_label,
            subsystem_index=1,
            state_ref=state_ref,
            timeline_time=timeline.current_time,
            common_meta=common_meta,
            timing_meta=timing_meta,
        )

        timeline.log(
            LogLevel.DEBUG,
            "components.sources.entangled_pair_source.emit",
            "pair emitted",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "attempt_index": self._attempt_count,
                "pair_index": pair_index,
                "left_signal_id": left_signal.id,
                "right_signal_id": right_signal.id,
                "left_connection_id": left_connection.connection_id,
                "right_connection_id": right_connection.connection_id,
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
                emission_index=pair_index,
                signal_ids=(str(left_signal.id), str(right_signal.id)),
                sampler_index=sample.index,
                sampler_label=sample.label,
                state_ref=state_ref,
                state_targets=(left_subsystem, right_subsystem),
                emission_slot_tick=emission_slot_tick,
                emission_delay_ticks=emission_delay_ticks,
                meta=(("pair_index", pair_index),),
            ),
            timeline=timeline,
            source=self,
        )

        left_connection.transmit(
            left_signal,
            timeline,
            time=timeline.current_time,
            source=self,
            subsystem_id="components",
            meta={
                "source_device_id": self.device_id,
                "output_port": self.left_output_port.name,
                "signal_id": left_signal.id,
                "pair_index": pair_index,
                "member": "left",
            },
        )
        right_connection.transmit(
            right_signal,
            timeline,
            time=timeline.current_time,
            source=self,
            subsystem_id="components",
            meta={
                "source_device_id": self.device_id,
                "output_port": self.right_output_port.name,
                "signal_id": right_signal.id,
                "pair_index": pair_index,
                "member": "right",
            },
        )

    def _make_member_signal(
        self,
        *,
        member: str,
        pair_index: int,
        subsystem_label: str,
        subsystem_index: int,
        state_ref: int,
        timeline_time: int,
        common_meta: tuple[tuple[str, object], ...],
        timing_meta: tuple[tuple[str, object], ...],
    ) -> Signal:
        """
        Build one signal for a member of an emitted entangled pair.

        Notes
        -----
        The returned signal represents one accessible subsystem of a shared
        two-qubit qstate. The two member signals use the same ``state_ref`` and
        ``correlation_id`` but carry different subsystem handles and
        ``member`` metadata.
        """
        return Signal(
            id=f"{self.device_id}:pair:{pair_index}:{member}",
            signal_kind=SignalKind.ENTANGLED_MEMBER,
            encoding_scheme=self.encoding_scheme,
            emission_time=timeline_time,
            origin=self.device_id,
            wavelength_nm=float(self.wavelength_nm),
            correlation_id=pair_index,
            correlation_meta=(
                ("pair_index", pair_index),
                ("member", member),
            ),
            state_ref=state_ref,
            state_targets=(
                SubsystemHandle(
                    label=subsystem_label,
                    kind="qubit",
                    index=subsystem_index,
                    metadata=(
                        ("qstate_subsystem", subsystem_label),
                        ("pair_index", pair_index),
                        ("member", member),
                    ),
                ),
            ),
            meta=common_meta + (("member", member),),
            timing_meta=timing_meta,
            # Trusted emission hot path: fields come from validated component
            # config, qstate output, and locally constructed subsystem handles.
            validation_flag=False,
        )


__all__ = ["EntangledPairSource"]
