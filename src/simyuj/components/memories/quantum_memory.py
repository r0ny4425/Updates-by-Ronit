"""Event-driven quantum memory component.

``QuantumMemory`` stores qstate-backed photon signals in physical memory
positions, exposes quantum input/output ports plus a classical notice port, and
routes all memory operations through timeline events. It owns position lifecycle
records while qstate math remains in the timeline's qstate manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from simyuj.components.detectors.primitives.readout import run_qubit_readout
from simyuj.components.quantum_targets import qstate_targets_from_signal
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import validate_meta
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.primitives.units import ticks_to_seconds
from simyuj.primitives.validation import (
    validate_non_negative_int,
    validate_positive_int,
    validate_probability,
)
from simyuj.qstate import SubsystemId
from simyuj.qstate.ops import Unitary
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel

from ..connections import PortDelivery, require_connection
from ..ports import Port, PortDirection, PortKind
from .noise import (
    MemoryNoiseModels,
    MemoryNoiseModelsInput,
    normalize_memory_noise_models,
)
from .position import (
    MemoryPositionRecord,
    MemoryPositionStatus,
    emitted_photon_subsystem_id,
    memory_subsystem_id,
)
from .reporting import log_memory_report
from .reports import (
    MemoryAbsorbReport,
    MemoryDiscardReport,
    MemoryEmitReport,
    MemoryExpireReport,
    MemoryMeasurementReport,
    MemoryMetaUpdateReport,
    MemoryOperatorReport,
    MemoryReport,
)
from .requests import (
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    MemoryExpireRequest,
    MemoryMeasureRequest,
    MemoryUpdateMetaRequest,
)

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG


MEMORY_ABSORB = "memory_absorb"
"""Absorb a signal into memory.

Payload is ``MemoryAbsorbRequest`` or a quantum ``PortDelivery`` carrying a
``Signal``.
"""

MEMORY_EMIT = "memory_emit"
"""Emit one occupied position as an outgoing photon signal.

Payload is ``MemoryEmitRequest``.
"""

MEMORY_APPLY_OPERATOR = "memory_apply_operator"
"""Apply an operator to ordered occupied memory positions.

Payload is ``MemoryApplyOperatorRequest``.
"""

MEMORY_MEASURE = "memory_measure"
"""Measure ordered occupied memory positions.

Payload is ``MemoryMeasureRequest``.
"""

MEMORY_DISCARD = "memory_discard"
"""Discard one occupied memory position.

Payload is ``MemoryDiscardRequest``.
"""

MEMORY_EXPIRE = "memory_expire"
"""Expire one occupied position when its occupancy token still matches.

Payload is ``MemoryExpireRequest``. The component schedules this action after a
successful absorb when ``storage_lifetime_ticks`` is configured.
"""

MEMORY_UPDATE_META = "memory_update_meta"
"""Update classical metadata for one occupied memory position.

Payload is ``MemoryUpdateMetaRequest``.
"""

_MEMORY_ABSORB_COMPLETE = "_memory_absorb_complete"
_MEMORY_EMIT_COMPLETE = "_memory_emit_complete"
_MEMORY_APPLY_OPERATOR_COMPLETE = "_memory_apply_operator_complete"
_MEMORY_MEASURE_COMPLETE = "_memory_measure_complete"

_HANDLER_BY_ACTION = {
    MEMORY_ABSORB: "_handle_absorb",
    MEMORY_EMIT: "_handle_emit",
    MEMORY_APPLY_OPERATOR: "_handle_apply_operator",
    MEMORY_MEASURE: "_handle_measure",
    MEMORY_DISCARD: "_handle_discard",
    MEMORY_EXPIRE: "_handle_expire",
    MEMORY_UPDATE_META: "_handle_update_meta",
    _MEMORY_ABSORB_COMPLETE: "_handle_absorb_complete",
    _MEMORY_EMIT_COMPLETE: "_handle_emit_complete",
    _MEMORY_APPLY_OPERATOR_COMPLETE: "_handle_apply_operator_complete",
    _MEMORY_MEASURE_COMPLETE: "_handle_measure_complete",
}


@dataclass(frozen=True, slots=True)
class _AbsorbCompletion:
    """Internal delayed-absorb completion payload guarded by occupancy token."""

    request: MemoryAbsorbRequest
    position: int
    occupancy_token: int


@dataclass(frozen=True, slots=True)
class _EmitCompletion:
    """Internal delayed-emit completion payload guarded by occupancy token."""

    request: MemoryEmitRequest
    occupancy_token: int


@dataclass(frozen=True, slots=True)
class _ApplyOperatorCompletion:
    """Internal delayed-operator completion payload guarded by occupancy tokens."""

    request: MemoryApplyOperatorRequest
    occupancy_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _MeasureCompletion:
    """Internal delayed-measure completion payload guarded by occupancy tokens."""

    request: MemoryMeasureRequest
    occupancy_tokens: tuple[int, ...]


@dataclass(slots=True)
class QuantumMemory(Component):
    """
    Event-driven quantum memory with physical position lifecycle records.

    Parameters
    ----------
    memory_id : str
        Non-empty component identifier used for ports, reports, logs, RNG
        streams, and qstate subsystem labels.
    num_positions : int
        Positive number of physical memory positions.
    noise_models : MemoryNoiseModelsInput, default=None
        Storage-noise configuration normalized to one ordered chain per
        position.
    absorb_delay_ticks, emit_delay_ticks : int, default=0
        Non-negative operation delays. Nonzero values mark the affected
        position busy and schedule an internal completion event.
    operator_delay_ticks, measure_delay_ticks : int, default=0
        Non-negative delays for operator and measurement operations.
    recovery_ticks : int, default=0
        Recovery interval applied after a stored quantum carrier is removed.
    storage_lifetime_ticks : int or None, default=None
        Optional lifetime from successful absorb to scheduled expiry.
    absorb_success_probability : float, default=1.0
        Bernoulli probability for successful absorption.
    emit_success_probability : float, default=1.0
        Bernoulli probability for successful emission.
    readout_model : object or None, default=None
        Readout model passed to the detector readout primitive during memory
        measurement.
    notice_priority, delivery_priority : int, default=0
        Timeline priorities used for notice reports, output signals, internal
        completions, and expiry scheduling.

    Attributes
    ----------
    input_port : Port
        Quantum ingress port named ``"in"``.
    output_port : Port
        Quantum egress port named ``"out"``.
    notice_port : Port
        Classical egress port named ``"notice"`` for memory reports.
    positions : tuple[MemoryPositionRecord, ...]
        Immutable snapshots of every physical memory position.
    reports : list[MemoryReport]
        Stored memory reports in handling order. Operator reports are stored
        only when the classical notice port is connected.

    Notes
    -----
    Public operations are explicit events: ``MEMORY_ABSORB``, ``MEMORY_EMIT``,
    ``MEMORY_APPLY_OPERATOR``, ``MEMORY_MEASURE``, ``MEMORY_DISCARD``,
    ``MEMORY_EXPIRE``, and ``MEMORY_UPDATE_META``. Delayed absorb, emit,
    operator, and measure operations schedule private completion actions and
    use occupancy tokens to ignore stale completions after reuse or discard.

    Absorb relabels the incoming photon subsystem to the stable
    ``memory:{memory_id}:position:{position}`` subsystem. Emit relabels that
    memory subsystem to a unique emitted-photon subsystem before transmitting a
    new ``Signal``. Discard, expiry, and destructive measurement remove memory
    subsystems from qstate and clear the position into recovery.

    Storage noise is applied lazily to occupied positions before operations
    that touch them. Absorb and emit success use timeline-owned RNG streams;
    probabilities of ``0.0`` and ``1.0`` do not consume RNG.

    Delayed absorb keeps the incoming signal pending and relabels or discards
    the photon subsystem only at completion. Delayed operator and measurement
    operations apply pending storage noise before entering the busy state; no
    additional storage noise is accrued during that operation delay.
    """

    memory_id: str
    num_positions: int
    noise_models: MemoryNoiseModelsInput = None

    absorb_delay_ticks: int = 0
    emit_delay_ticks: int = 0
    operator_delay_ticks: int = 0
    measure_delay_ticks: int = 0
    recovery_ticks: int = 0

    storage_lifetime_ticks: int | None = None

    absorb_success_probability: float = 1.0
    emit_success_probability: float = 1.0

    readout_model: object | None = None

    notice_priority: int = 0
    delivery_priority: int = 0

    input_port: Port = field(init=False)
    output_port: Port = field(init=False)
    notice_port: Port = field(init=False)

    positions: tuple[MemoryPositionRecord, ...] = field(init=False)
    reports: list[MemoryReport] = field(init=False, default_factory=list)
    emit_counter: int = field(init=False, default=0)

    _bound_timeline_id: int | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _absorb_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _emit_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _measurement_choice_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _qstate_measurement_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _readout_model_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _position_noise_models: MemoryNoiseModels = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_config()
        self._position_noise_models = normalize_memory_noise_models(
            self.noise_models,
            num_positions=self.num_positions,
        )
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.memory_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.memory_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )
        self.notice_port = Port(
            name="notice",
            owner=self,
            owner_id=self.memory_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )
        self.positions = tuple(
            MemoryPositionRecord(position=position)
            for position in range(self.num_positions)
        )

    def bind(self, context: BindingContext) -> None:
        """
        Bind the memory to one timeline and declare deterministic RNG streams.

        Parameters
        ----------
        context : BindingContext
            Binding context containing the timeline.

        Notes
        -----
        Binding is idempotent for the same timeline and rejects another
        timeline. Streams are declared for absorb, emit, measurement selection,
        qstate measurement, and readout-model sampling before event execution.
        """

        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)

        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "quantum memory is already bound to another timeline"
                )
            return

        self._absorb_rng = timeline.rng(self.memory_id, "quantum_memory", "absorb")
        self._emit_rng = timeline.rng(self.memory_id, "quantum_memory", "emit")
        self._measurement_choice_rng = timeline.rng(
            self.memory_id,
            "quantum_memory",
            "measurement_choice",
        )
        self._qstate_measurement_rng = timeline.rng(
            self.memory_id,
            "quantum_memory",
            "qstate_measurement",
        )
        self._readout_model_rng = timeline.rng(
            self.memory_id,
            "quantum_memory",
            "readout_model",
        )
        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.memories.quantum_memory.ready",
            "quantum memory ready",
            meta={
                "memory_id": self.memory_id,
                "num_positions": self.num_positions,
                "storage_lifetime_ticks": self.storage_lifetime_ticks,
                "recovery_ticks": self.recovery_ticks,
                "absorb_delay_ticks": self.absorb_delay_ticks,
                "emit_delay_ticks": self.emit_delay_ticks,
                "operator_delay_ticks": self.operator_delay_ticks,
                "measure_delay_ticks": self.measure_delay_ticks,
                "absorb_success_probability": self.absorb_success_probability,
                "emit_success_probability": self.emit_success_probability,
            },
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """
        Dispatch one memory event to its operation handler.

        Parameters
        ----------
        event : Event
            Scheduled event targeted at this memory.
        timeline : Timeline
            Bound timeline executing the event.

        Notes
        -----
        The handler is selected from the event action. Public actions expect the
        corresponding memory request payload, except ``MEMORY_ABSORB`` also
        accepts a quantum ``PortDelivery`` carrying a ``Signal``.
        """

        self._validate_event_context(event=event, timeline=timeline)

        try:
            handler_name = _HANDLER_BY_ACTION[event.action]
        except KeyError as exc:
            raise ValueError(
                f"unsupported event action for QuantumMemory: {event.action!r}"
            ) from exc

        handler = getattr(self, handler_name)
        handler(
            event.payload_ref,
            timeline,
            event_id=event.event_id,
            action=event.action,
        )

    def _validate_config(self) -> None:
        ensure_nonempty_id(self.memory_id, field_name="memory_id")
        validate_positive_int(self.num_positions, field_name="num_positions")
        validate_non_negative_int(
            self.absorb_delay_ticks,
            field_name="absorb_delay_ticks",
        )
        validate_non_negative_int(
            self.emit_delay_ticks,
            field_name="emit_delay_ticks",
        )
        validate_non_negative_int(
            self.operator_delay_ticks,
            field_name="operator_delay_ticks",
        )
        validate_non_negative_int(
            self.measure_delay_ticks,
            field_name="measure_delay_ticks",
        )
        validate_non_negative_int(
            self.recovery_ticks,
            field_name="recovery_ticks",
        )
        if self.storage_lifetime_ticks is not None:
            validate_non_negative_int(
                self.storage_lifetime_ticks,
                field_name="storage_lifetime_ticks",
            )
        validate_probability(
            self.absorb_success_probability,
            field_name="absorb_success_probability",
        )
        validate_probability(
            self.emit_success_probability,
            field_name="emit_success_probability",
        )
        _plain_int(self.notice_priority, field_name="notice_priority")
        _plain_int(self.delivery_priority, field_name="delivery_priority")

    def _validate_event_context(
        self,
        *,
        event: Event,
        timeline: Timeline,
    ) -> None:
        if not isinstance(event, Event):
            raise TypeError("event must be Event")
        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")
        if self._bound_timeline_id is None:
            raise RuntimeError("quantum memory must be bound before handling events")
        if self._bound_timeline_id != id(timeline):
            raise RuntimeError("quantum memory received event for another timeline")
        if event.target_ref is not self:
            raise ValueError("event target_ref must be this QuantumMemory")

    def _handle_absorb(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._absorb_request_from_payload(payload, timeline)
        if request.memory_id != self.memory_id:
            raise ValueError("absorb request memory_id does not match this memory")
        position = self._select_absorb_position(
            request.position,
            time=timeline.current_time,
        )
        old_record = self.positions[position]
        occupancy_token = old_record.occupancy_token + 1
        if self.absorb_delay_ticks > 0:
            self._replace_position(
                position,
                replace(
                    old_record,
                    status=MemoryPositionStatus.ABSORBING,
                    stored_signal=request.signal,
                    occupancy_token=occupancy_token,
                    meta=request.meta,
                ),
            )
            self._schedule_completion(
                timeline=timeline,
                action=_MEMORY_ABSORB_COMPLETE,
                delay_ticks=self.absorb_delay_ticks,
                payload=_AbsorbCompletion(
                    request=request,
                    position=position,
                    occupancy_token=occupancy_token,
                ),
            )
            return

        self._complete_absorb(
            request=request,
            position=position,
            occupancy_token=occupancy_token,
            timeline=timeline,
            event_id=event_id,
            action=action,
        )

    def _handle_absorb_complete(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        if not isinstance(payload, _AbsorbCompletion):
            raise TypeError("absorb completion payload must be _AbsorbCompletion")
        if not self._completion_is_current(
            positions=(payload.position,),
            tokens=(payload.occupancy_token,),
            status=MemoryPositionStatus.ABSORBING,
        ):
            self._log_stale_completion(
                timeline,
                operation="absorb",
                positions=(payload.position,),
                occupancy_tokens=(payload.occupancy_token,),
                expected_status=MemoryPositionStatus.ABSORBING,
                event_id=event_id,
                action=action,
            )
            return
        self._complete_absorb(
            request=payload.request,
            position=payload.position,
            occupancy_token=payload.occupancy_token,
            timeline=timeline,
            event_id=event_id,
            action=action,
        )

    def _complete_absorb(
        self,
        *,
        request: MemoryAbsorbRequest,
        position: int,
        occupancy_token: int,
        timeline: Timeline,
        event_id: int | None,
        action: str,
    ) -> None:
        time = timeline.current_time
        memory_subsystem = memory_subsystem_id(self.memory_id, position)
        photon_subsystem = self._single_photon_subsystem(request.signal)
        old_record = self.positions[position]

        if not self._sample_success(
            probability=self.absorb_success_probability,
            rng=self._absorb_rng,
            field_name="absorb_success_probability",
        ):
            timeline.qstate.discard(targets=(photon_subsystem,))
            self._clear_failed_absorb_position(
                position=position,
                timeline=timeline,
                record=old_record,
                occupancy_token=occupancy_token,
                meta=(
                    ("request_id", request.request_id),
                    ("photon_subsystem", str(photon_subsystem)),
                    *request.meta,
                ),
            )
            self._store_absorb_failed_report(
                request=request,
                position=position,
                occupancy_token=occupancy_token,
                photon_subsystem=photon_subsystem,
                timeline=timeline,
                event_id=event_id,
                action=action,
            )
            return

        timeline.qstate.relabel_subsystem(photon_subsystem, memory_subsystem)

        expires_at = self._expires_at(time)
        self._replace_position(
            position,
            replace(
                old_record,
                status=MemoryPositionStatus.OCCUPIED,
                memory_subsystem=memory_subsystem,
                stored_signal=request.signal,
                stored_time=time,
                last_noise_update_time=time,
                expires_at=expires_at,
                occupancy_token=occupancy_token,
                ready_at=time,
                meta=request.meta,
            ),
        )

        if expires_at is not None:
            self._schedule_expiry(
                timeline=timeline,
                request=request,
                position=position,
                expires_at=expires_at,
                occupancy_token=occupancy_token,
            )

        self._store_report(
            MemoryAbsorbReport(
                report_id=f"{self.memory_id}:absorb:{time}:{len(self.reports)}",
                memory_id=self.memory_id,
                time=time,
                success=True,
                position=position,
                input_signal_id=request.signal.id,
                memory_subsystem=memory_subsystem,
                status="occupied",
                session_id=request.session_id,
                meta=(
                    ("request_id", request.request_id),
                    ("occupancy_token", occupancy_token),
                    ("photon_subsystem", str(photon_subsystem)),
                    *request.meta,
                ),
            ),
            timeline,
            event_id=event_id,
            action=action,
        )

    def _handle_emit(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._require_emit_request(payload)
        if request.memory_id != self.memory_id:
            raise ValueError("emit request memory_id does not match this memory")

        self._check_position_index(request.position)
        record = self.positions[request.position]
        if record.status is not MemoryPositionStatus.OCCUPIED:
            raise ValueError("memory position is not occupied")
        if record.memory_subsystem is None:
            raise ValueError("occupied memory position has no memory_subsystem")

        if self.emit_delay_ticks > 0:
            self._mark_positions_status(
                (request.position,),
                MemoryPositionStatus.EMITTING,
            )
            self._schedule_completion(
                timeline=timeline,
                action=_MEMORY_EMIT_COMPLETE,
                delay_ticks=self.emit_delay_ticks,
                payload=_EmitCompletion(
                    request=request,
                    occupancy_token=record.occupancy_token,
                ),
            )
            return

        self._complete_emit(
            request=request,
            occupancy_token=record.occupancy_token,
            timeline=timeline,
            expected_status=MemoryPositionStatus.OCCUPIED,
            event_id=event_id,
            action=action,
        )

    def _handle_emit_complete(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        if not isinstance(payload, _EmitCompletion):
            raise TypeError("emit completion payload must be _EmitCompletion")
        self._complete_emit(
            request=payload.request,
            occupancy_token=payload.occupancy_token,
            timeline=timeline,
            expected_status=MemoryPositionStatus.EMITTING,
            event_id=event_id,
            action=action,
        )

    def _complete_emit(
        self,
        *,
        request: MemoryEmitRequest,
        occupancy_token: int,
        timeline: Timeline,
        expected_status: MemoryPositionStatus,
        event_id: int | None,
        action: str,
    ) -> None:
        if self._skip_stale_completion(
            timeline,
            operation="emit",
            positions=(request.position,),
            tokens=(occupancy_token,),
            expected_status=expected_status,
            event_id=event_id,
            action=action,
        ):
            return

        self._apply_pending_storage_noise(
            request.position,
            timeline,
            allowed_statuses=(
                MemoryPositionStatus.OCCUPIED,
                MemoryPositionStatus.EMITTING,
            ),
        )
        record_after_noise = self.positions[request.position]

        time = timeline.current_time
        memory_subsystem = record_after_noise.memory_subsystem
        if memory_subsystem is None:
            return

        if not self._sample_success(
            probability=self.emit_success_probability,
            rng=self._emit_rng,
            field_name="emit_success_probability",
        ):
            restored = record_after_noise
            if record_after_noise.status is MemoryPositionStatus.EMITTING:
                restored = replace(
                    record_after_noise,
                    status=MemoryPositionStatus.OCCUPIED,
                )
                self._replace_position(request.position, restored)
            self._store_emit_failed_report(
                request=request,
                record=restored,
                timeline=timeline,
                event_id=event_id,
                action=action,
            )
            return

        output_connection = require_connection(self.output_port)
        output_subsystem = emitted_photon_subsystem_id(
            self.memory_id,
            request.position,
            self.emit_counter,
        )
        self.emit_counter += 1

        timeline.qstate.relabel_subsystem(memory_subsystem, output_subsystem)
        state_ref = timeline.qstate.state_of(output_subsystem)

        output_signal = self._make_emitted_signal(
            request=request,
            record=record_after_noise,
            time=time,
            state_ref=state_ref,
            output_subsystem=output_subsystem,
        )

        output_connection.transmit(
            output_signal,
            timeline,
            time=time,
            priority=self.delivery_priority,
            source=self,
            subsystem_id="components",
            meta={
                "memory_id": self.memory_id,
                "position": request.position,
                "output_signal_id": output_signal.id,
                "output_subsystem": str(output_subsystem),
            },
        )

        ready_at = self._clear_position_after_quantum_removal(
            position=request.position,
            timeline=timeline,
            record=record_after_noise,
            meta=(
                ("last_operation", "emit"),
                ("request_id", request.request_id),
                ("output_signal_id", output_signal.id),
                ("output_subsystem", str(output_subsystem)),
                *request.meta,
            ),
        )

        self._store_report(
            MemoryEmitReport(
                report_id=f"{self.memory_id}:emit:{time}:{len(self.reports)}",
                memory_id=self.memory_id,
                time=time,
                success=True,
                position=request.position,
                memory_subsystem=memory_subsystem,
                output_signal_id=output_signal.id,
                output_subsystem=output_subsystem,
                status="emitted",
                session_id=request.session_id,
                meta=(
                    ("request_id", request.request_id),
                    ("occupancy_token", occupancy_token),
                    ("ready_at", ready_at),
                    ("recovery_ticks", self.recovery_ticks),
                    *request.meta,
                ),
            ),
            timeline,
            event_id=event_id,
            action=action,
        )

    def _handle_apply_operator(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._require_apply_operator_request(payload)
        if request.memory_id != self.memory_id:
            raise ValueError("operator request memory_id does not match this memory")
        tokens = self._tokens_for_positions(request.positions)
        for position in request.positions:
            self._apply_pending_storage_noise(position, timeline)

        if self.operator_delay_ticks > 0:
            # Storage noise has already been advanced to the request time; the
            # pending operator window models operation latency, not extra
            # passive storage.
            self._mark_positions_status(
                request.positions,
                MemoryPositionStatus.APPLYING_OPERATOR,
            )
            self._schedule_completion(
                timeline=timeline,
                action=_MEMORY_APPLY_OPERATOR_COMPLETE,
                delay_ticks=self.operator_delay_ticks,
                payload=_ApplyOperatorCompletion(
                    request=request,
                    occupancy_tokens=tokens,
                ),
            )
            return

        self._complete_apply_operator(
            request=request,
            occupancy_tokens=tokens,
            timeline=timeline,
            expected_status=MemoryPositionStatus.OCCUPIED,
            event_id=event_id,
            action=action,
        )

    def _handle_apply_operator_complete(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        if not isinstance(payload, _ApplyOperatorCompletion):
            raise TypeError(
                "apply operator completion payload must be _ApplyOperatorCompletion"
            )
        self._complete_apply_operator(
            request=payload.request,
            occupancy_tokens=payload.occupancy_tokens,
            timeline=timeline,
            expected_status=MemoryPositionStatus.APPLYING_OPERATOR,
            event_id=event_id,
            action=action,
        )

    def _complete_apply_operator(
        self,
        *,
        request: MemoryApplyOperatorRequest,
        occupancy_tokens: tuple[int, ...],
        timeline: Timeline,
        expected_status: MemoryPositionStatus,
        event_id: int | None,
        action: str,
    ) -> None:
        if self._skip_stale_completion(
            timeline,
            operation="operator",
            positions=request.positions,
            tokens=occupancy_tokens,
            expected_status=expected_status,
            event_id=event_id,
            action=action,
        ):
            return
        targets = self._memory_targets_for_positions(
            request.positions,
            allowed_status=expected_status,
        )
        timeline.qstate.apply(cast(Unitary, request.operator), targets=targets)
        if expected_status is MemoryPositionStatus.APPLYING_OPERATOR:
            self._mark_positions_status(
                request.positions,
                MemoryPositionStatus.OCCUPIED,
            )

        if self.notice_port.connection is not None:
            self._store_report(
                MemoryOperatorReport(
                    report_id=(
                        f"{self.memory_id}:operator:{timeline.current_time}:"
                        f"{len(self.reports)}"
                    ),
                    memory_id=self.memory_id,
                    time=timeline.current_time,
                    success=True,
                    positions=request.positions,
                    memory_subsystems=targets,
                    status="applied",
                    session_id=request.session_id,
                    meta=(
                        ("request_id", request.request_id),
                        ("occupancy_tokens", occupancy_tokens),
                        *request.meta,
                    ),
                ),
                timeline,
                event_id=event_id,
                action=action,
            )

    def _memory_targets_for_positions(
        self,
        positions: tuple[int, ...],
        *,
        allowed_status: MemoryPositionStatus = MemoryPositionStatus.OCCUPIED,
    ):
        targets = []
        for position in positions:
            self._check_position_index(position)
            record = self.positions[position]
            if record.status is not allowed_status:
                raise ValueError("memory position is not in the required status")
            if record.memory_subsystem is None:
                raise ValueError("memory position has no memory_subsystem")
            targets.append(record.memory_subsystem)
        return tuple(targets)

    def _tokens_for_positions(self, positions: tuple[int, ...]) -> tuple[int, ...]:
        tokens = []
        for position in positions:
            self._check_position_index(position)
            record = self.positions[position]
            if record.status is not MemoryPositionStatus.OCCUPIED:
                raise ValueError("memory position is not occupied")
            if record.memory_subsystem is None:
                raise ValueError("memory position has no memory_subsystem")
            tokens.append(record.occupancy_token)
        return tuple(tokens)

    def _handle_measure(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._require_measure_request(payload)
        if request.memory_id != self.memory_id:
            raise ValueError("measure request memory_id does not match this memory")
        tokens = self._tokens_for_positions(request.positions)
        for position in request.positions:
            self._apply_pending_storage_noise(position, timeline)

        if self.measure_delay_ticks > 0:
            # Storage noise has already been advanced to the request time; the
            # pending measurement window models operation latency, not extra
            # passive storage.
            self._mark_positions_status(
                request.positions,
                MemoryPositionStatus.MEASURING,
            )
            self._schedule_completion(
                timeline=timeline,
                action=_MEMORY_MEASURE_COMPLETE,
                delay_ticks=self.measure_delay_ticks,
                payload=_MeasureCompletion(
                    request=request,
                    occupancy_tokens=tokens,
                ),
            )
            return

        self._complete_measure(
            request=request,
            occupancy_tokens=tokens,
            timeline=timeline,
            expected_status=MemoryPositionStatus.OCCUPIED,
            event_id=event_id,
            action=action,
        )

    def _handle_measure_complete(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        if not isinstance(payload, _MeasureCompletion):
            raise TypeError("measure completion payload must be _MeasureCompletion")
        self._complete_measure(
            request=payload.request,
            occupancy_tokens=payload.occupancy_tokens,
            timeline=timeline,
            expected_status=MemoryPositionStatus.MEASURING,
            event_id=event_id,
            action=action,
        )

    def _complete_measure(
        self,
        *,
        request: MemoryMeasureRequest,
        occupancy_tokens: tuple[int, ...],
        timeline: Timeline,
        expected_status: MemoryPositionStatus,
        event_id: int | None,
        action: str,
    ) -> None:
        if self._skip_stale_completion(
            timeline,
            operation="measure",
            positions=request.positions,
            tokens=occupancy_tokens,
            expected_status=expected_status,
            event_id=event_id,
            action=action,
        ):
            return
        targets = self._memory_targets_for_positions(
            request.positions,
            allowed_status=expected_status,
        )
        detection_report = run_qubit_readout(
            device_id=self.memory_id,
            time=timeline.current_time,
            targets=targets,
            measurement=request.measurement,
            collapse=request.collapse,
            qstate=timeline.qstate,
            measurement_choice_rng=self._measurement_choice_rng,
            qstate_rng=self._qstate_measurement_rng,
            readout_rng=self._readout_model_rng,
            readout_model=self.readout_model,
            detector_meta=(
                ("memory_id", self.memory_id),
                ("positions", request.positions),
                ("request_id", request.request_id),
                *request.meta,
            ),
            report_id=(
                f"{self.memory_id}:readout:{timeline.current_time}:"
                f"{len(self.reports)}:{request.request_id}"
            ),
        )

        cleared_positions: tuple[int, ...] = ()
        if request.destructive:
            for position, target in zip(request.positions, targets, strict=True):
                record = self.positions[position]
                timeline.qstate.discard(targets=(target,))
                self._clear_position_after_quantum_removal(
                    position=position,
                    timeline=timeline,
                    record=record,
                    meta=(
                        ("last_operation", "destructive_measure"),
                        ("request_id", request.request_id),
                        *request.meta,
                    ),
                )
            cleared_positions = request.positions
        elif expected_status is MemoryPositionStatus.MEASURING:
            self._mark_positions_status(
                request.positions,
                MemoryPositionStatus.OCCUPIED,
            )

        self._store_report(
            MemoryMeasurementReport(
                report_id=f"{self.memory_id}:measure:{timeline.current_time}:"
                f"{len(self.reports)}",
                memory_id=self.memory_id,
                time=timeline.current_time,
                success=detection_report.success,
                positions=request.positions,
                memory_subsystems=targets,
                detection_report=detection_report,
                destructive=request.destructive,
                cleared_positions=cleared_positions,
                status="measured" if detection_report.success else "no_outcome",
                session_id=request.session_id,
                meta=(
                    ("request_id", request.request_id),
                    ("occupancy_tokens", occupancy_tokens),
                    *request.meta,
                ),
            ),
            timeline,
            event_id=event_id,
            action=action,
        )

    def _handle_discard(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._require_discard_request(payload)
        if request.memory_id != self.memory_id:
            raise ValueError("discard request memory_id does not match this memory")
        self._discard_position(
            position=request.position,
            timeline=timeline,
            status="discarded",
            reason=request.reason,
            session_id=request.session_id,
            request_id=request.request_id,
            meta=request.meta,
            event_id=event_id,
            action=action,
        )

    def _handle_expire(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._require_expire_request(payload)
        if request.memory_id != self.memory_id:
            raise ValueError("expire request memory_id does not match this memory")
        self._check_position_index(request.position)

        record = self.positions[request.position]
        if record.occupancy_token != request.occupancy_token:
            self._log_stale_expiry(
                timeline,
                request=request,
                reason="stale_occupancy_token",
                event_id=event_id,
                action=action,
            )
            return
        if record.status is not MemoryPositionStatus.OCCUPIED:
            self._log_stale_expiry(
                timeline,
                request=request,
                reason=f"not_occupied:{record.status.value}",
                event_id=event_id,
                action=action,
            )
            return

        self._discard_position(
            position=request.position,
            timeline=timeline,
            status="expired",
            reason="expired",
            session_id=request.session_id,
            request_id=request.request_id,
            meta=request.meta,
            report_kind="expire",
            event_id=event_id,
            action=action,
        )

    def _handle_update_meta(
        self,
        payload: object,
        timeline: Timeline,
        *,
        event_id: int | None,
        action: str,
    ) -> None:
        request = self._require_update_meta_request(payload)
        if request.memory_id != self.memory_id:
            raise ValueError(
                "metadata update request memory_id does not match this memory"
            )
        self._check_position_index(request.position)
        record = self.positions[request.position]

        if record.status is not MemoryPositionStatus.OCCUPIED:
            self._store_meta_update_report(
                request=request,
                timeline=timeline,
                record=record,
                success=False,
                status=f"not_occupied:{record.status.value}",
                event_id=event_id,
                action=action,
            )
            return

        if record.memory_subsystem is None:
            self._store_meta_update_report(
                request=request,
                timeline=timeline,
                record=record,
                success=False,
                status="missing_memory_subsystem",
                event_id=event_id,
                action=action,
            )
            return

        if (
            request.expected_occupancy_token is not None
            and record.occupancy_token != request.expected_occupancy_token
        ):
            self._store_meta_update_report(
                request=request,
                timeline=timeline,
                record=record,
                success=False,
                status="stale_occupancy_token",
                meta=(
                    (
                        "expected_occupancy_token",
                        request.expected_occupancy_token,
                    ),
                ),
                event_id=event_id,
                action=action,
            )
            return

        new_record = self._update_position_meta(
            position=request.position,
            updates=request.updates,
            remove_keys=request.remove_keys,
        )
        self._store_meta_update_report(
            request=request,
            timeline=timeline,
            record=new_record,
            success=True,
            status="updated",
            event_id=event_id,
            action=action,
        )

    def _absorb_request_from_payload(
        self,
        payload: object,
        timeline: Timeline,
    ) -> MemoryAbsorbRequest:
        if isinstance(payload, MemoryAbsorbRequest):
            return payload

        if isinstance(payload, PortDelivery):
            if payload.target_port is not self.input_port:
                raise ValueError(
                    "absorb delivery target_port must be memory input_port"
                )
            if not isinstance(payload.payload, Signal):
                raise TypeError("absorb delivery payload must be Signal")
            return MemoryAbsorbRequest(
                request_id=(
                    f"{self.memory_id}:absorb:{timeline.current_time}:"
                    f"{len(self.reports)}"
                ),
                memory_id=self.memory_id,
                signal=payload.payload,
                position=None,
                meta=(
                    ("connection_id", payload.connection_id),
                    ("source_port", payload.source_port.name),
                    ("target_port", payload.target_port.name),
                ),
            )

        raise TypeError(
            "MEMORY_ABSORB payload must be MemoryAbsorbRequest or PortDelivery"
        )

    def _require_emit_request(self, payload: object) -> MemoryEmitRequest:
        if not isinstance(payload, MemoryEmitRequest):
            raise TypeError("MEMORY_EMIT payload must be MemoryEmitRequest")
        return payload

    def _require_apply_operator_request(
        self,
        payload: object,
    ) -> MemoryApplyOperatorRequest:
        if not isinstance(payload, MemoryApplyOperatorRequest):
            raise TypeError(
                "MEMORY_APPLY_OPERATOR payload must be MemoryApplyOperatorRequest"
            )
        return payload

    def _require_measure_request(self, payload: object) -> MemoryMeasureRequest:
        if not isinstance(payload, MemoryMeasureRequest):
            raise TypeError("MEMORY_MEASURE payload must be MemoryMeasureRequest")
        return payload

    def _require_discard_request(self, payload: object) -> MemoryDiscardRequest:
        if not isinstance(payload, MemoryDiscardRequest):
            raise TypeError("MEMORY_DISCARD payload must be MemoryDiscardRequest")
        return payload

    def _require_expire_request(self, payload: object) -> MemoryExpireRequest:
        if not isinstance(payload, MemoryExpireRequest):
            raise TypeError("MEMORY_EXPIRE payload must be MemoryExpireRequest")
        return payload

    def _require_update_meta_request(
        self,
        payload: object,
    ) -> MemoryUpdateMetaRequest:
        if not isinstance(payload, MemoryUpdateMetaRequest):
            raise TypeError(
                "MEMORY_UPDATE_META payload must be MemoryUpdateMetaRequest"
            )
        return payload

    def _apply_pending_storage_noise(
        self,
        position: int,
        timeline: Timeline,
        *,
        allowed_statuses: tuple[MemoryPositionStatus, ...] = (
            MemoryPositionStatus.OCCUPIED,
        ),
    ) -> None:
        self._check_position_index(position)
        record = self.positions[position]
        if record.status not in allowed_statuses:
            return
        if record.memory_subsystem is None:
            return

        if record.last_noise_update_time is None:
            return

        duration_ticks = timeline.current_time - record.last_noise_update_time
        if duration_ticks <= 0:
            return

        duration_s = ticks_to_seconds(duration_ticks)

        noise_models = self._position_noise_models[position]
        if noise_models:
            timeline.qstate.apply_noise_models(
                noise_models,
                targets=(record.memory_subsystem,),
                duration_s=duration_s,
            )
            timeline.log(
                LogLevel.TRACE,
                "components.memories.quantum_memory.storage_noise",
                "memory storage noise applied",
                meta={
                    "memory_id": self.memory_id,
                    "position": position,
                    "memory_subsystem": str(record.memory_subsystem),
                    "duration_ticks": duration_ticks,
                    "noise_model_count": len(noise_models),
                },
            )

        self._replace_position(
            position,
            replace(record, last_noise_update_time=timeline.current_time),
        )

    def _sample_success(
        self,
        *,
        probability: float,
        rng: DeterministicRNG | None,
        field_name: str,
    ) -> bool:
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        if rng is None:
            raise RuntimeError(f"{field_name} RNG is not bound")
        return rng.random() < float(probability)

    def _make_emitted_signal(
        self,
        *,
        request: MemoryEmitRequest,
        record: MemoryPositionRecord,
        time: int,
        state_ref: int,
        output_subsystem,
    ) -> Signal:
        stored_signal = record.stored_signal
        # Fallbacks are used only if an older or synthetic record lacks the
        # absorbed signal metadata normally preserved by successful absorb.
        encoding_scheme = (
            EncodingScheme.POLARIZATION
            if stored_signal is None
            else stored_signal.encoding_scheme
        )
        wavelength_nm = 1550.0 if stored_signal is None else stored_signal.wavelength_nm
        signal_id = f"{self.memory_id}:emit:{self.emit_counter - 1}"
        return Signal(
            id=signal_id,
            signal_kind=SignalKind.PHOTON,
            encoding_scheme=encoding_scheme,
            emission_time=time,
            origin=self.memory_id,
            wavelength_nm=wavelength_nm,
            correlation_id=(
                None if stored_signal is None else stored_signal.correlation_id
            ),
            correlation_meta=(
                None if stored_signal is None else stored_signal.correlation_meta
            ),
            state_ref=state_ref,
            state_targets=(
                SubsystemHandle(
                    label=str(output_subsystem),
                    kind="qubit",
                    index=0,
                    metadata=(("qstate_subsystem", str(output_subsystem)),),
                ),
            ),
            protocol_params=(
                () if stored_signal is None else stored_signal.protocol_params
            ),
            meta=(
                ("memory_id", self.memory_id),
                ("position", request.position),
                ("request_id", request.request_id),
                *request.meta,
            ),
            timing_meta=(
                ("memory_emit_time", time),
                ("memory_stored_time", record.stored_time),
                ("memory_storage_ticks", _storage_ticks(record.stored_time, time)),
            ),
            # Trusted memory-emission hot path: values are preserved from an
            # absorbed signal or produced by validated memory/qstate state.
            validation_flag=False,
        )

    def _select_absorb_position(self, requested: int | None, *, time: int) -> int:
        if requested is not None:
            self._check_position_index(requested)
            record = self.positions[requested]
            if record.status is not MemoryPositionStatus.EMPTY:
                raise ValueError("requested memory position is not empty")
            if time < record.ready_at:
                raise ValueError("requested memory position is still recovering")
            return requested

        for record in self.positions:
            if self._is_position_available(record, time):
                return record.position

        raise RuntimeError("no available memory position")

    def _is_position_available(
        self,
        record: MemoryPositionRecord,
        time: int,
    ) -> bool:
        return record.status is MemoryPositionStatus.EMPTY and time >= record.ready_at

    def _single_photon_subsystem(self, signal: Signal):
        targets = qstate_targets_from_signal(signal)
        if len(targets) != 1:
            raise ValueError("memory absorb requires exactly one qstate target")
        return targets[0]

    def _replace_position(
        self,
        position: int,
        record: MemoryPositionRecord,
    ) -> None:
        self.positions = (
            *self.positions[:position],
            record,
            *self.positions[position + 1 :],
        )

    def _mark_positions_status(
        self,
        positions: tuple[int, ...],
        status: MemoryPositionStatus,
    ) -> None:
        for position in positions:
            self._replace_position(
                position,
                replace(self.positions[position], status=status),
            )

    def _update_position_meta(
        self,
        *,
        position: int,
        updates: tuple[tuple[str, object], ...],
        remove_keys: tuple[str, ...],
    ) -> MemoryPositionRecord:
        self._check_position_index(position)
        record = self.positions[position]
        new_record = replace(
            record,
            meta=_merge_meta(record.meta, updates=updates, remove_keys=remove_keys),
        )
        self._replace_position(position, new_record)
        return new_record

    def _store_meta_update_report(
        self,
        *,
        request: MemoryUpdateMetaRequest,
        timeline: Timeline,
        record: MemoryPositionRecord,
        success: bool,
        status: str,
        meta: tuple[tuple[str, object], ...] = (),
        event_id: int | None = None,
        action: str | None = None,
    ) -> None:
        self._store_report(
            MemoryMetaUpdateReport(
                report_id=(
                    f"{self.memory_id}:meta_update:"
                    f"{timeline.current_time}:{len(self.reports)}"
                ),
                memory_id=self.memory_id,
                time=timeline.current_time,
                success=success,
                position=request.position,
                memory_subsystem=record.memory_subsystem,
                occupancy_token=record.occupancy_token,
                status=status,
                updated_keys=(
                    tuple(key for key, _ in request.updates) if success else ()
                ),
                removed_keys=request.remove_keys,
                session_id=request.session_id,
                meta=(
                    ("request_id", request.request_id),
                    *meta,
                    *request.meta,
                ),
            ),
            timeline,
            event_id=event_id,
            action=action,
        )

    def _clear_failed_absorb_position(
        self,
        *,
        position: int,
        timeline: Timeline,
        record: MemoryPositionRecord,
        occupancy_token: int,
        meta: tuple[tuple[str, object], ...] = (),
    ) -> None:
        self._replace_position(
            position,
            replace(
                record,
                status=MemoryPositionStatus.EMPTY,
                memory_subsystem=None,
                stored_signal=None,
                stored_time=None,
                last_noise_update_time=None,
                expires_at=None,
                occupancy_token=occupancy_token,
                ready_at=timeline.current_time,
                meta=(
                    *meta,
                    ("last_operation", "absorb_failed"),
                ),
            ),
        )

    def _store_absorb_failed_report(
        self,
        *,
        request: MemoryAbsorbRequest,
        position: int,
        occupancy_token: int,
        photon_subsystem: SubsystemId,
        timeline: Timeline,
        event_id: int | None,
        action: str,
    ) -> None:
        self._store_report(
            MemoryAbsorbReport(
                report_id=(
                    f"{self.memory_id}:absorb:"
                    f"{timeline.current_time}:{len(self.reports)}"
                ),
                memory_id=self.memory_id,
                time=timeline.current_time,
                success=False,
                position=position,
                input_signal_id=request.signal.id,
                memory_subsystem=None,
                status="absorb_failed",
                session_id=request.session_id,
                meta=(
                    ("request_id", request.request_id),
                    ("occupancy_token", occupancy_token),
                    ("photon_subsystem", str(photon_subsystem)),
                    (
                        "absorb_success_probability",
                        self.absorb_success_probability,
                    ),
                    *request.meta,
                ),
            ),
            timeline,
            event_id=event_id,
            action=action,
        )

    def _store_emit_failed_report(
        self,
        *,
        request: MemoryEmitRequest,
        record: MemoryPositionRecord,
        timeline: Timeline,
        event_id: int | None,
        action: str,
    ) -> None:
        self._store_report(
            MemoryEmitReport(
                report_id=(
                    f"{self.memory_id}:emit:"
                    f"{timeline.current_time}:{len(self.reports)}"
                ),
                memory_id=self.memory_id,
                time=timeline.current_time,
                success=False,
                position=request.position,
                memory_subsystem=record.memory_subsystem,
                output_signal_id=None,
                output_subsystem=None,
                status="emit_failed",
                session_id=request.session_id,
                meta=(
                    ("request_id", request.request_id),
                    ("occupancy_token", record.occupancy_token),
                    ("emit_success_probability", self.emit_success_probability),
                    *request.meta,
                ),
            ),
            timeline,
            event_id=event_id,
            action=action,
        )

    def _clear_position_after_quantum_removal(
        self,
        *,
        position: int,
        timeline: Timeline,
        record: MemoryPositionRecord,
        meta: tuple[tuple[str, object], ...] = (),
    ) -> int:
        ready_at = timeline.current_time + self.recovery_ticks
        self._replace_position(
            position,
            replace(
                record,
                status=MemoryPositionStatus.EMPTY,
                memory_subsystem=None,
                stored_signal=None,
                stored_time=None,
                last_noise_update_time=None,
                expires_at=None,
                occupancy_token=record.occupancy_token + 1,
                ready_at=ready_at,
                meta=(
                    *meta,
                    ("ready_at", ready_at),
                    ("recovery_ticks", self.recovery_ticks),
                ),
            ),
        )
        return ready_at

    def _discard_position(
        self,
        *,
        position: int,
        timeline: Timeline,
        status: str,
        reason: str,
        session_id: str | None,
        request_id: str,
        meta: tuple[tuple[str, object], ...],
        report_kind: str = "discard",
        event_id: int | None = None,
        action: str | None = None,
    ) -> None:
        self._check_position_index(position)
        record = self.positions[position]
        if record.status is not MemoryPositionStatus.OCCUPIED:
            raise ValueError("memory position is not occupied")
        if record.memory_subsystem is None:
            raise ValueError("memory position has no memory_subsystem")

        memory_subsystem = record.memory_subsystem
        timeline.qstate.discard(targets=(memory_subsystem,))
        ready_at = self._clear_position_after_quantum_removal(
            position=position,
            timeline=timeline,
            record=record,
            meta=(
                ("last_operation", status),
                ("reason", reason),
                ("request_id", request_id),
                *meta,
            ),
        )

        report_meta = (
            ("request_id", request_id),
            ("occupancy_token", record.occupancy_token),
            ("ready_at", ready_at),
            ("recovery_ticks", self.recovery_ticks),
            *meta,
        )
        report: MemoryReport
        if report_kind == "expire":
            report = MemoryExpireReport(
                report_id=f"{self.memory_id}:expire:{timeline.current_time}:"
                f"{len(self.reports)}",
                memory_id=self.memory_id,
                time=timeline.current_time,
                success=True,
                position=position,
                memory_subsystem=memory_subsystem,
                status=status,
                reason=reason,
                session_id=session_id,
                meta=report_meta,
            )
        else:
            report = MemoryDiscardReport(
                report_id=f"{self.memory_id}:discard:{timeline.current_time}:"
                f"{len(self.reports)}",
                memory_id=self.memory_id,
                time=timeline.current_time,
                success=True,
                position=position,
                memory_subsystem=memory_subsystem,
                status=status,
                reason=reason,
                session_id=session_id,
                meta=report_meta,
            )

        self._store_report(report, timeline, event_id=event_id, action=action)

    def _expires_at(self, stored_time: int) -> int | None:
        if self.storage_lifetime_ticks is None:
            return None
        return stored_time + self.storage_lifetime_ticks

    def _schedule_expiry(
        self,
        *,
        timeline: Timeline,
        request: MemoryAbsorbRequest,
        position: int,
        expires_at: int,
        occupancy_token: int,
    ) -> None:
        timeline.schedule(
            Event(
                time=expires_at,
                priority=self.delivery_priority,
                target_ref=self,
                action=MEMORY_EXPIRE,
                payload_ref=MemoryExpireRequest(
                    request_id=f"{request.request_id}:expire",
                    memory_id=self.memory_id,
                    position=position,
                    occupancy_token=occupancy_token,
                    session_id=request.session_id,
                    meta=request.meta,
                ),
                source=self,
                subsystem_id="components",
                meta={
                    "memory_id": self.memory_id,
                    "position": position,
                    "occupancy_token": occupancy_token,
                },
            )
        )

    def _schedule_completion(
        self,
        *,
        timeline: Timeline,
        action: str,
        delay_ticks: int,
        payload: object,
    ) -> None:
        completion_time = timeline.current_time + delay_ticks
        scheduled = timeline.schedule(
            Event(
                time=completion_time,
                priority=self.delivery_priority,
                target_ref=self,
                action=action,
                payload_ref=payload,
                source=self,
                subsystem_id="components",
                meta={
                    "memory_id": self.memory_id,
                    "completion_action": action,
                },
            )
        )
        operation, positions, occupancy_tokens, request_id = _completion_log_context(
            payload
        )
        timeline.log(
            LogLevel.TRACE,
            "components.memories.quantum_memory.pending",
            "memory operation pending",
            event_id=scheduled.event_id,
            action=action,
            meta={
                "memory_id": self.memory_id,
                "request_id": request_id,
                "operation": operation,
                "positions": positions,
                "occupancy_tokens": occupancy_tokens,
                "status": _pending_status_for_completion(action),
                "delay_ticks": delay_ticks,
                "completion_time": completion_time,
            },
        )

    def _completion_is_current(
        self,
        *,
        positions: tuple[int, ...],
        tokens: tuple[int, ...],
        status: MemoryPositionStatus,
    ) -> bool:
        if len(positions) != len(tokens):
            return False
        for position, token in zip(positions, tokens, strict=True):
            self._check_position_index(position)
            record = self.positions[position]
            if record.occupancy_token != token:
                return False
            if record.status is not status:
                return False
        return True

    def _skip_stale_completion(
        self,
        timeline: Timeline,
        *,
        operation: str,
        positions: tuple[int, ...],
        tokens: tuple[int, ...],
        expected_status: MemoryPositionStatus,
        event_id: int | None,
        action: str,
    ) -> bool:
        if self._completion_is_current(
            positions=positions,
            tokens=tokens,
            status=expected_status,
        ):
            return False
        self._log_stale_completion(
            timeline,
            operation=operation,
            positions=positions,
            occupancy_tokens=tokens,
            expected_status=expected_status,
            event_id=event_id,
            action=action,
        )
        return True

    def _log_stale_completion(
        self,
        timeline: Timeline,
        *,
        operation: str,
        positions: tuple[int, ...],
        occupancy_tokens: tuple[int, ...],
        expected_status: MemoryPositionStatus,
        event_id: int | None,
        action: str,
    ) -> None:
        timeline.log(
            LogLevel.TRACE,
            "components.memories.quantum_memory.stale_skip",
            "stale memory completion skipped",
            event_id=event_id,
            action=action,
            meta={
                "memory_id": self.memory_id,
                "operation": operation,
                "positions": positions,
                "occupancy_tokens": occupancy_tokens,
                "expected_status": expected_status.value,
                "reason": "stale_completion",
            },
        )

    def _log_stale_expiry(
        self,
        timeline: Timeline,
        *,
        request: MemoryExpireRequest,
        reason: str,
        event_id: int | None,
        action: str,
    ) -> None:
        timeline.log(
            LogLevel.TRACE,
            "components.memories.quantum_memory.stale_skip",
            "stale memory expiry skipped",
            event_id=event_id,
            action=action,
            meta={
                "memory_id": self.memory_id,
                "operation": "expire",
                "positions": (request.position,),
                "occupancy_tokens": (request.occupancy_token,),
                "expected_status": MemoryPositionStatus.OCCUPIED.value,
                "reason": reason,
            },
        )

    def _store_report(
        self,
        report: MemoryReport,
        timeline: Timeline,
        *,
        event_id: int | None = None,
        action: str | None = None,
    ) -> None:
        self.reports.append(report)
        log_memory_report(
            report,
            timeline,
            positions=self.positions,
            event_id=event_id,
            action=action,
        )
        connection = self.notice_port.connection
        if connection is None:
            return
        connection.transmit(
            report,
            timeline,
            time=timeline.current_time,
            priority=self.notice_priority,
            source=self,
            subsystem_id="components",
            meta={
                "memory_id": self.memory_id,
                "notice_port": self.notice_port.name,
                "report_id": report.report_id,
            },
        )

    def _check_position_index(self, position: int) -> None:
        if type(position) is not int:
            raise TypeError("position must be int")
        if position < 0 or position >= self.num_positions:
            raise ValueError("position is out of range")


def _plain_int(value: object, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int")
    return value


def _completion_log_context(
    payload: object,
) -> tuple[str, tuple[int, ...], tuple[int, ...], str]:
    if isinstance(payload, _AbsorbCompletion):
        return (
            "absorb",
            (payload.position,),
            (payload.occupancy_token,),
            payload.request.request_id,
        )

    if isinstance(payload, _EmitCompletion):
        return (
            "emit",
            (payload.request.position,),
            (payload.occupancy_token,),
            payload.request.request_id,
        )

    if isinstance(payload, _ApplyOperatorCompletion):
        return (
            "operator",
            payload.request.positions,
            payload.occupancy_tokens,
            payload.request.request_id,
        )

    if isinstance(payload, _MeasureCompletion):
        return (
            "measure",
            payload.request.positions,
            payload.occupancy_tokens,
            payload.request.request_id,
        )

    raise TypeError("unsupported memory completion payload")


def _pending_status_for_completion(action: str) -> str:
    if action == _MEMORY_ABSORB_COMPLETE:
        return MemoryPositionStatus.ABSORBING.value
    if action == _MEMORY_EMIT_COMPLETE:
        return MemoryPositionStatus.EMITTING.value
    if action == _MEMORY_APPLY_OPERATOR_COMPLETE:
        return MemoryPositionStatus.APPLYING_OPERATOR.value
    if action == _MEMORY_MEASURE_COMPLETE:
        return MemoryPositionStatus.MEASURING.value
    raise ValueError(f"unsupported completion action: {action!r}")


def _merge_meta(
    existing: tuple[tuple[str, object], ...],
    *,
    updates: tuple[tuple[str, object], ...],
    remove_keys: tuple[str, ...],
) -> tuple[tuple[str, object], ...]:
    """Merge metadata; remove existing keys first, then append updates."""
    validate_meta(existing)
    validate_meta(updates)
    _validate_unique_keys(updates, field_name="updates")
    if len(set(remove_keys)) != len(remove_keys):
        raise ValueError("remove_keys contains duplicate keys")

    remove = set(remove_keys)
    update_keys = {key for key, _ in updates}

    merged = [
        (key, value)
        for key, value in existing
        if key not in remove and key not in update_keys
    ]
    merged.extend(updates)

    result = tuple(merged)
    validate_meta(result)
    return result


def _validate_unique_keys(
    meta: tuple[tuple[str, object], ...],
    *,
    field_name: str,
) -> None:
    seen: set[str] = set()
    for key, _ in meta:
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate key {key!r}")
        seen.add(key)


def _storage_ticks(stored_time: int | None, current_time: int) -> int | None:
    if stored_time is None:
        return None
    return current_time - stored_time


__all__ = [
    "MEMORY_ABSORB",
    "MEMORY_APPLY_OPERATOR",
    "MEMORY_DISCARD",
    "MEMORY_EMIT",
    "MEMORY_EXPIRE",
    "MEMORY_MEASURE",
    "MEMORY_UPDATE_META",
    "QuantumMemory",
]
