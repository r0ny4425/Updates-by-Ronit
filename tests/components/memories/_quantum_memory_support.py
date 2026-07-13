from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.components import Port, PortDelivery
from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MEMORY_EXPIRE,
    MEMORY_MEASURE,
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    MemoryExpireRequest,
    MemoryMeasureRequest,
    MemoryPositionStatus,
    QuantumMemory,
    memory_subsystem_id,
)
from simyuj.components.ports import PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.qstate.noise import depolarizing
from simyuj.qstate.ops import X
from simyuj.signal import EncodingScheme, Signal, SignalKind


@dataclass(slots=True)
class QuantumSource(Component):
    device_id: str = "source"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("source should not receive events")


@dataclass(slots=True)
class NoticeSink(Component):
    device_id: str = "notice-sink"
    received: list[PortDelivery] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append(event.payload_ref)


@dataclass(slots=True)
class QuantumSink(Component):
    device_id: str = "quantum-sink"
    received: list[PortDelivery] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append(event.payload_ref)


class RecordingNoiseModel:
    name = "recording"
    arity = 1

    def __init__(self) -> None:
        self.durations: list[float] = []

    def resolve(self, *, duration_s: float):
        self.durations.append(duration_s)
        return depolarizing(0.0)


_UNSET = object()


def _event(memory: QuantumMemory, action: str) -> Event:
    return Event(
        time=0,
        target_ref=memory,
        action=action,
        payload_ref=None,
    )


def _signal(
    timeline: Timeline,
    signal_id: str = "s1",
    *,
    state: str = "|0>",
) -> Signal:
    subsystem = SubsystemId(f"photon:{signal_id}")
    state_ref = timeline.qstate.prepare(state, subsystems=(subsystem,))
    return Signal(
        id=signal_id,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=timeline.current_time,
        origin="source",
        state_ref=state_ref,
        state_targets=(
            SubsystemHandle(
                label=str(subsystem),
                kind="qubit",
                index=0,
                metadata=(("qstate_subsystem", str(subsystem)),),
            ),
        ),
    )


def _absorb_position(
    memory: QuantumMemory,
    timeline: Timeline,
    *,
    position: int | None = None,
    signal_id: str = "s1",
    state: str = "|0>",
    request_id: str = "absorb-1",
) -> Signal:
    signal = _signal(timeline, signal_id, state=state)
    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id=request_id,
                memory_id=memory.memory_id,
                signal=signal,
                position=position,
            ),
        ),
        timeline,
    )
    return signal


def _count_reports(memory: QuantumMemory, report_type: type[object]) -> int:
    return sum(1 for report in memory.reports if isinstance(report, report_type))


def _assert_position_fields(
    memory: QuantumMemory,
    position: int,
    *,
    status: MemoryPositionStatus,
    memory_subsystem: object = _UNSET,
    stored_signal: object = _UNSET,
    stored_time: object = _UNSET,
    last_noise_update_time: object = _UNSET,
    expires_at: object = _UNSET,
    occupancy_token: object = _UNSET,
    ready_at: object = _UNSET,
) -> None:
    record = memory.positions[position]
    assert record.status is status
    if memory_subsystem is not _UNSET:
        assert record.memory_subsystem == memory_subsystem
    if stored_signal is not _UNSET:
        assert record.stored_signal is stored_signal
    if stored_time is not _UNSET:
        assert record.stored_time == stored_time
    if last_noise_update_time is not _UNSET:
        assert record.last_noise_update_time == last_noise_update_time
    if expires_at is not _UNSET:
        assert record.expires_at == expires_at
    if occupancy_token is not _UNSET:
        assert record.occupancy_token == occupancy_token
    if ready_at is not _UNSET:
        assert record.ready_at == ready_at


def _conflict_event(
    memory: QuantumMemory,
    timeline: Timeline,
    action: str,
) -> Event:
    payload: object
    if action == MEMORY_ABSORB:
        payload = MemoryAbsorbRequest(
            request_id="conflict-absorb",
            memory_id=memory.memory_id,
            signal=_signal(timeline, "conflict"),
            position=0,
        )
    elif action == MEMORY_EMIT:
        payload = MemoryEmitRequest(
            request_id="conflict-emit",
            memory_id=memory.memory_id,
            position=0,
        )
    elif action == MEMORY_APPLY_OPERATOR:
        payload = MemoryApplyOperatorRequest(
            request_id="conflict-apply",
            memory_id=memory.memory_id,
            positions=(0,),
            operator=X,
        )
    elif action == MEMORY_MEASURE:
        payload = MemoryMeasureRequest(
            request_id="conflict-measure",
            memory_id=memory.memory_id,
            positions=(0,),
            destructive=False,
        )
    elif action == MEMORY_DISCARD:
        payload = MemoryDiscardRequest(
            request_id="conflict-discard",
            memory_id=memory.memory_id,
            position=0,
        )
    elif action == MEMORY_EXPIRE:
        payload = MemoryExpireRequest(
            request_id="conflict-expire",
            memory_id=memory.memory_id,
            position=0,
            occupancy_token=memory.positions[0].occupancy_token,
        )
    else:
        raise AssertionError(f"unsupported conflict action: {action}")

    return Event(
        time=timeline.current_time,
        target_ref=memory,
        action=action,
        payload_ref=payload,
    )


def _start_busy_position(
    memory: QuantumMemory,
    timeline: Timeline,
    busy_kind: str,
) -> tuple[SubsystemId, int, SubsystemId | None]:
    memory_subsystem = memory_subsystem_id(memory.memory_id, 0)
    photon_subsystem = SubsystemId("photon:busy")
    signal = _signal(timeline, "busy")
    state_ref = signal.state_ref
    assert state_ref is not None

    if busy_kind == "absorbing":
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_ABSORB,
                payload_ref=MemoryAbsorbRequest(
                    request_id="busy-absorb",
                    memory_id=memory.memory_id,
                    signal=signal,
                    position=0,
                ),
            ),
            timeline,
        )
        return photon_subsystem, state_ref, memory_subsystem

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="busy-seed",
                memory_id=memory.memory_id,
                signal=signal,
                position=0,
            ),
        ),
        timeline,
    )

    if busy_kind == "emitting":
        event = Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="busy-emit",
                memory_id=memory.memory_id,
                position=0,
            ),
        )
    elif busy_kind == "applying":
        event = Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="busy-apply",
                memory_id=memory.memory_id,
                positions=(0,),
                operator=X,
            ),
        )
    elif busy_kind == "measuring":
        event = Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="busy-measure",
                memory_id=memory.memory_id,
                positions=(0,),
                destructive=False,
            ),
        )
    else:
        raise AssertionError(f"unsupported busy kind: {busy_kind}")

    memory.handle_event(event, timeline)
    return memory_subsystem, state_ref, None
